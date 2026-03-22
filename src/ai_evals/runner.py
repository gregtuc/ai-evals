"""Experiment runner - the core orchestration logic.

Supports arbitrary experimental conditions (baseline, controls, treatments,
ablations) with async execution for throughput.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import shutil
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from ai_evals.config import ExperimentConfig, EvalTask
from ai_evals.eval_loader import load_tasks_from_config
from ai_evals.models import get_model
from ai_evals.results import ResultStore, RunResult
from ai_evals.scorers import get_scorer

console = Console()

# Approximate per-million-token pricing (input, output) for cost estimation
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "o1": (15.0, 60.0),
    "o3-mini": (1.10, 4.40),
}


class ExperimentRunner:
    def __init__(
        self,
        config_path: str | Path,
        evals_dir: str | Path = "evals",
        output_base: str | Path = "results",
        dir_name: str | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = ExperimentConfig.from_yaml(self.config_path)
        self.evals_dir = Path(evals_dir)
        self.output_base = Path(output_base)

        # Compute config hash for traceability
        config_bytes = self.config_path.read_bytes()
        self.config_hash = hashlib.sha256(config_bytes).hexdigest()[:16]

        # Set up output directory
        if dir_name:
            self.output_dir = self.output_base / dir_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.output_base / f"{timestamp}_{self.config.name}"
        self.store = ResultStore(self.output_dir)

        # Copy config into output dir for reproducibility
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.config_path, self.output_dir / "experiment_config.yaml")

    def run(self, dry_run: bool = False, resume_dir: Path | None = None) -> Path:
        """Run the full experiment (sync wrapper). Returns the output directory path."""
        return asyncio.run(self.run_async(dry_run=dry_run, resume_dir=resume_dir))

    async def run_async(self, dry_run: bool = False, resume_dir: Path | None = None) -> Path:
        """Run the full experiment with async concurrency."""
        if resume_dir:
            self.output_dir = resume_dir
            self.store = ResultStore(self.output_dir)

        book = self.config.book
        conditions = self.config.conditions
        condition_names = list(conditions.keys())

        console.print(f'\n[bold]Book:[/bold] "{book.title}" by {book.author}')
        cond_summary = ", ".join(
            f"{name} ({cond.role})" for name, cond in conditions.items()
        )
        console.print(f"[bold]Conditions:[/bold] {cond_summary}")

        if self.config.temperature == 0.0 and self.config.runs_per_task > 1:
            console.print(
                "[yellow]Warning: temperature=0.0 with multiple runs. "
                "Results will be near-identical across runs, making statistical "
                "tests unreliable. Consider temperature=0.3.[/yellow]"
            )

        if self.config.runs_per_task > 1 and self.config.temperature > 0:
            console.print(
                f"[dim]Note: {self.config.runs_per_task} runs per task will be averaged "
                f"per-task before statistical analysis. Multiple runs reduce within-task "
                f"noise but do not increase effective sample size — the unit of analysis "
                f"is the task, not the individual run.[/dim]"
            )

        # Load tasks (from benchmark or local YAML)
        tasks = load_tasks_from_config(self.config, evals_dir=self.evals_dir)
        categories = sorted(set(t.category for t in tasks))
        source = f"benchmark:{self.config.benchmark.name}" if self.config.benchmark else "local YAML"
        console.print(f"[bold]Loaded {len(tasks)} eval tasks[/bold] from {source}")
        console.print(f"  Categories ({len(categories)}): {', '.join(categories)}")
        if self.config.benchmark and self.config.benchmark.domain_matched:
            matched = self.config.benchmark.domain_matched
            mismatched = [c for c in categories if c not in matched]
            console.print(f"  [green]Domain-matched:[/green] {', '.join(matched)}")
            console.print(f"  [dim]Domain-mismatched (negative control):[/dim] {', '.join(mismatched)}")

        if not tasks:
            console.print("[red]No eval tasks found! Check your eval_categories.[/red]")
            return self.output_dir

        # Initialize models
        models = [get_model(mc) for mc in self.config.models]
        console.print(f"[bold]Models:[/bold] {', '.join(m.name for m in models)}")

        total_calls = len(models) * len(tasks) * self.config.runs_per_task * len(conditions)
        console.print(f"[bold]Total API calls:[/bold] {total_calls}")
        console.print(f"[bold]Temperature:[/bold] {self.config.temperature}")
        console.print(f"[bold]Concurrency:[/bold] {self.config.concurrency}")

        if dry_run:
            _print_dry_run_summary(self.config, tasks, models)
            return self.output_dir

        # Get completed keys for resume
        completed = self.store.get_completed_keys()
        if completed:
            console.print(f"[yellow]Resuming: {len(completed)} results already completed[/yellow]")

        # Seed RNG for reproducible condition ordering
        rng = random.Random(self.config.seed)

        # Build work items: list of (model, task, condition_name, run_number)
        work_items = []
        for model in models:
            for task in tasks:
                for run_num in range(1, self.config.runs_per_task + 1):
                    # Randomize condition order per (task, run) group
                    shuffled = condition_names.copy()
                    rng.shuffle(shuffled)
                    for cond_name in shuffled:
                        key = (task.id, model.name, cond_name, run_num)
                        if key not in completed:
                            work_items.append((model, task, cond_name, run_num))

        if not work_items:
            console.print("[green]All work already completed (resume).[/green]")
            _print_summary_table(self.store.load_all())
            return self.output_dir

        console.print(f"[bold]Remaining API calls:[/bold] {len(work_items)}")

        # Execute with async concurrency
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def run_one(model, task, cond_name, run_num):
            cond_config = conditions[cond_name]
            scorer = get_scorer(task.scorer)

            # Build template variables
            template_vars = {"task_prompt": task.prompt, "task_category": task.category}
            if cond_config.book_vars:
                book_source = cond_config.book_override or book
                template_vars["book_title"] = book_source.title
                template_vars["book_author"] = book_source.author

            prompt_text = cond_config.template.format(**template_vars)

            async with semaphore:
                response = await model.async_complete(
                    messages=[{"role": "user", "content": prompt_text}],
                    system=cond_config.system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=self.config.seed,
                )

            score_result = scorer.score(response.content, task)

            result = RunResult(
                experiment_name=self.config.name,
                run_id=str(uuid.uuid4()),
                task_id=task.id,
                task_category=task.category,
                model=response.model,
                condition=cond_name,
                condition_role=cond_config.role,
                run_number=run_num,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                response=response.content,
                score=score_result.score,
                score_passed=score_result.passed,
                scorer_details=score_result.details,
                task_metadata=task.metadata,
                timestamp=datetime.now(timezone.utc).isoformat(),
                config_hash=self.config_hash,
            )
            self.store.append(result)
            return result

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            progress_task = progress.add_task("Running evals...", total=len(work_items))

            async def run_with_progress(model, task, cond_name, run_num):
                try:
                    result = await run_one(model, task, cond_name, run_num)
                    progress.advance(progress_task)
                    return result
                except Exception as e:
                    progress.advance(progress_task)
                    console.print(
                        f"[red]Error: {model.name}/{task.id}/{cond_name}/run{run_num}: {e}[/red]"
                    )
                    return None

            results = await asyncio.gather(
                *(run_with_progress(m, t, c, r) for m, t, c, r in work_items)
            )

        succeeded = sum(1 for r in results if r is not None)
        failed = len(results) - succeeded
        console.print(f"\n[bold green]Experiment complete![/bold green] ({succeeded} succeeded, {failed} failed)")
        console.print(f"Results saved to: {self.output_dir}")

        _print_summary_table(self.store.load_all())

        return self.output_dir


def _print_summary_table(results: list[RunResult]) -> None:
    """Print a quick summary table at the end of an experiment."""
    if not results:
        return

    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in results:
        groups[(r.model, r.task_category, r.condition)].append(r.score)

    table = Table(title="Results Summary")
    table.add_column("Model")
    table.add_column("Category")
    table.add_column("Condition")
    table.add_column("Mean Score", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("N", justify="right")

    for key in sorted(groups.keys()):
        model, cat, cond = key
        scores = groups[key]
        mean = sum(scores) / len(scores)
        pass_rate = sum(1 for s in scores if s >= 0.7) / len(scores)
        table.add_row(model, cat, cond, f"{mean:.3f}", f"{pass_rate:.0%}", str(len(scores)))

    console.print(table)


def _print_dry_run_summary(
    config: ExperimentConfig,
    tasks: list[EvalTask],
    models,
) -> None:
    """Print what would happen without making API calls, including cost estimate."""
    console.print("\n[bold yellow]DRY RUN — no API calls will be made[/bold yellow]\n")

    n_conditions = len(config.conditions)
    total_calls = len(models) * len(tasks) * config.runs_per_task * n_conditions

    cond_desc = ", ".join(
        f"{name} ({cond.role})" for name, cond in config.conditions.items()
    )

    console.print(f"  Experiment: {config.name}")
    console.print(f'  Book: "{config.book.title}" by {config.book.author}')
    console.print(f"  Models: {len(models)}")
    console.print(f"  Tasks: {len(tasks)}")
    console.print(f"  Conditions: {n_conditions} ({cond_desc})")
    console.print(f"  Runs per task: {config.runs_per_task}")
    console.print(f"  Temperature: {config.temperature}")
    console.print(f"  Concurrency: {config.concurrency}")
    console.print(f"  Total API calls: {total_calls}")

    # Cost estimation
    est_input_tokens = 300  # rough estimate for MCQ prompts
    est_output_tokens = 200  # MCQ answers are short
    console.print(f"\n  [bold]Cost Estimate[/bold] (assuming ~{est_input_tokens} input, ~{est_output_tokens} output tokens/call):")

    total_cost = 0.0
    for mc in config.models:
        model_calls = len(tasks) * config.runs_per_task * n_conditions
        input_price, output_price = MODEL_PRICING.get(mc.model, (5.0, 15.0))
        input_cost = (model_calls * est_input_tokens / 1_000_000) * input_price
        output_cost = (model_calls * est_output_tokens / 1_000_000) * output_price
        model_cost = input_cost + output_cost
        total_cost += model_cost
        console.print(
            f"    {mc.provider}/{mc.model}: ~${model_cost:.2f} "
            f"({model_calls * est_input_tokens:,} in / {model_calls * est_output_tokens:,} out tokens)"
        )

    console.print(f"    [bold]Total: ~${total_cost:.2f}[/bold]")
    if any(mc.model not in MODEL_PRICING for mc in config.models):
        console.print(f"    [dim](Unknown models use fallback pricing of $5/$15 per M tokens)[/dim]")

    _print_power_preview(tasks, config)


def _print_power_preview(tasks: list[EvalTask], config: ExperimentConfig) -> None:
    """Print a priori power analysis: what effect sizes can be detected at each sample size."""
    from collections import Counter

    from ai_evals.analysis import _minimum_detectable_effect

    cat_counts = Counter(t.category for t in tasks)
    if not cat_counts:
        return

    console.print(f"\n  [bold]Power Analysis[/bold] (a priori, 80% power, alpha=0.05):")

    for cat in sorted(cat_counts.keys()):
        count = cat_counts[cat]
        min_d = _minimum_detectable_effect(count)
        if min_d is not None:
            if min_d >= 0.8:
                interp = "only large effects"
            elif min_d >= 0.5:
                interp = "medium+ effects"
            elif min_d >= 0.2:
                interp = "small+ effects"
            else:
                interp = "very small effects"
            console.print(f"    {cat} (n={count}): min |d| = {min_d:.2f} ({interp})")

    if config.benchmark and config.benchmark.domain_matched:
        categories = sorted(cat_counts.keys())
        n_matched = sum(cat_counts[c] for c in config.benchmark.domain_matched if c in cat_counts)
        n_mismatched = sum(cat_counts[c] for c in categories if c not in config.benchmark.domain_matched)
        console.print(f"\n    Domain specificity test:")
        console.print(f"      Matched tasks: {n_matched}, Mismatched tasks: {n_mismatched}")
        if n_matched > 0 and n_mismatched > 0:
            # For the domain specificity Welch's t-test, power depends on
            # both group sizes. Use the smaller group as the conservative estimate.
            min_n = min(n_matched, n_mismatched)
            min_d = _minimum_detectable_effect(min_n)
            if min_d is not None:
                console.print(
                    f"      Min detectable domain-specificity effect: |d| = {min_d:.2f} "
                    f"(based on smaller group, n={min_n})"
                )

    # Sample size recommendations
    from ai_evals.analysis import _recommended_sample_size

    for target_d, label in [(0.2, "small"), (0.5, "medium")]:
        rec_n = _recommended_sample_size(target_d)
        if rec_n is not None:
            console.print(
                f"\n    [dim]To detect a {label} effect (d={target_d}) at 80% power: "
                f"need {rec_n} tasks per category[/dim]"
            )


def run_multi_seed(
    config_path: str | Path,
    n_seeds: int,
    evals_dir: str | Path = "evals",
    output_base: str | Path = "results",
    concurrency_override: int | None = None,
) -> Path:
    """Run experiment N times with different task-sampling seeds.

    Each seed produces a different random sample of tasks from the benchmark,
    addressing single-seed fragility. Results are saved in per-seed
    subdirectories with a manifest for cross-seed analysis.

    Returns the parent directory containing all seed runs.
    """
    config_path = Path(config_path)
    config = ExperimentConfig.from_yaml(config_path)
    base_seed = config.seed or 42

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = Path(output_base) / f"{timestamp}_{config.name}_multiseed"
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Save original config to parent dir
    shutil.copy2(config_path, parent_dir / "experiment_config.yaml")

    seeds = [base_seed + i for i in range(n_seeds)]
    seed_dirs: list[str] = []

    for i, seed in enumerate(seeds):
        console.print(f"\n[bold]{'=' * 60}[/bold]")
        console.print(f"[bold]Seed {seed} (replication {i + 1}/{n_seeds})[/bold]")
        console.print(f"[bold]{'=' * 60}[/bold]")

        # Write per-seed config
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        config_data["seed"] = seed

        seed_config_path = parent_dir / f"config_seed_{seed}.yaml"
        with open(seed_config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        runner = ExperimentRunner(
            config_path=seed_config_path,
            evals_dir=evals_dir,
            output_base=str(parent_dir),
            dir_name=f"seed_{seed}",
        )
        if concurrency_override is not None:
            runner.config.concurrency = concurrency_override

        result_dir = runner.run()
        seed_dirs.append(str(result_dir))

    # Save manifest
    manifest = {
        "type": "multi_seed",
        "n_seeds": n_seeds,
        "seeds": seeds,
        "base_seed": base_seed,
        "config_name": config.name,
        "directories": seed_dirs,
        "timestamp": timestamp,
    }
    with open(parent_dir / "multi_seed_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"\n[bold green]Multi-seed experiment complete![/bold green]")
    console.print(f"  Seeds: {seeds}")
    console.print(f"  Results: {parent_dir}")
    console.print(f"\n  Analyze with: ai-evals analyze {parent_dir}")

    return parent_dir
