"""Experiment runner - the core orchestration logic."""

from __future__ import annotations

import hashlib
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ai_evals.config import ConditionConfig, ExperimentConfig, EvalTask
from ai_evals.eval_loader import load_tasks_from_config
from ai_evals.models import get_model
from ai_evals.models.base import Model
from ai_evals.results import ResultStore, RunResult
from ai_evals.scorers import get_scorer

console = Console()

# The three experimental conditions
CONDITIONS = ["baseline", "control", "primed"]


class ExperimentRunner:
    def __init__(
        self,
        config_path: str | Path,
        evals_dir: str | Path = "evals",
        output_base: str | Path = "results",
    ) -> None:
        self.config_path = Path(config_path)
        self.config = ExperimentConfig.from_yaml(self.config_path)
        self.evals_dir = Path(evals_dir)
        self.output_base = Path(output_base)

        # Compute config hash for traceability
        config_bytes = self.config_path.read_bytes()
        self.config_hash = hashlib.sha256(config_bytes).hexdigest()[:16]

        # Set up output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_base / f"{timestamp}_{self.config.name}"
        self.store = ResultStore(self.output_dir)

        # Copy config into output dir for reproducibility
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.config_path, self.output_dir / "experiment_config.yaml")

    def run(self, dry_run: bool = False, resume_dir: Path | None = None) -> Path:
        """Run the full experiment. Returns the output directory path."""
        if resume_dir:
            self.output_dir = resume_dir
            self.store = ResultStore(self.output_dir)

        book = self.config.book
        console.print(f'\n[bold]Book:[/bold] "{book.title}" by {book.author}')
        console.print(f"[bold]Conditions:[/bold] baseline, control (elaborate no-book), primed (with book)")

        if self.config.temperature == 0.0 and self.config.runs_per_task > 1:
            console.print(
                "[yellow]Warning: temperature=0.0 with multiple runs. "
                "Results will be near-identical across runs, making statistical "
                "tests unreliable. Consider temperature=0.3.[/yellow]"
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
        models: list[Model] = [get_model(mc) for mc in self.config.models]
        console.print(f"[bold]Models:[/bold] {', '.join(m.name for m in models)}")

        # 3 conditions instead of 2
        total_calls = len(models) * len(tasks) * self.config.runs_per_task * len(CONDITIONS)
        console.print(f"[bold]Total API calls:[/bold] {total_calls}")
        console.print(f"[bold]Temperature:[/bold] {self.config.temperature}")

        if dry_run:
            _print_dry_run_summary(self.config, tasks, models)
            return self.output_dir

        # Get completed keys for resume
        completed = self.store.get_completed_keys()
        if completed:
            console.print(f"[yellow]Resuming: {len(completed)} results already completed[/yellow]")

        # Build condition configs lookup
        condition_configs: dict[str, ConditionConfig] = {
            "baseline": self.config.baseline,
            "control": self.config.control,
            "primed": self.config.primed,
        }

        # Seed RNG for reproducible condition ordering
        rng = random.Random(self.config.seed)

        done = 0
        for model in models:
            for task in tasks:
                scorer = get_scorer(task.scorer)

                for run_num in range(1, self.config.runs_per_task + 1):
                    # Randomize condition order to prevent ordering effects
                    shuffled_conditions = CONDITIONS.copy()
                    rng.shuffle(shuffled_conditions)

                    for condition in shuffled_conditions:
                        key = (task.id, model.name, condition, run_num)
                        if key not in completed:
                            cond_config = condition_configs[condition]

                            # Format prompt based on condition
                            template_vars = {"task_prompt": task.prompt}
                            if condition == "primed":
                                template_vars["book_title"] = book.title
                                template_vars["book_author"] = book.author

                            result = _run_single(
                                model=model,
                                task=task,
                                condition=condition,
                                prompt_text=cond_config.template.format(**template_vars),
                                system_prompt=cond_config.system_prompt,
                                scorer=scorer,
                                run_number=run_num,
                                experiment_name=self.config.name,
                                config_hash=self.config_hash,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                                seed=self.config.seed,
                            )
                            self.store.append(result)
                        done += 1

                    _print_progress(done, total_calls, task, model, run_num)

        console.print(f"\n[bold green]Experiment complete![/bold green]")
        console.print(f"Results saved to: {self.output_dir}")

        _print_summary_table(self.store.load_all())

        return self.output_dir


def _run_single(
    *,
    model: Model,
    task: EvalTask,
    condition: str,
    prompt_text: str,
    system_prompt: str,
    scorer,
    run_number: int,
    experiment_name: str,
    config_hash: str,
    temperature: float,
    max_tokens: int,
    seed: int | None,
) -> RunResult:
    """Execute a single eval run and return the result."""
    response = model.complete(
        messages=[{"role": "user", "content": prompt_text}],
        system=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )

    score_result = scorer.score(response.content, task)

    return RunResult(
        experiment_name=experiment_name,
        run_id=str(uuid.uuid4()),
        task_id=task.id,
        task_category=task.category,
        model=response.model,
        condition=condition,
        run_number=run_number,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        response=response.content,
        score=score_result.score,
        score_passed=score_result.passed,
        scorer_details=score_result.details,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
    )


def _print_progress(done: int, total: int, task: EvalTask, model: Model, run_num: int) -> None:
    pct = done / total * 100
    console.print(
        f"  [{pct:5.1f}%] {model.name} | {task.category}/{task.id} | run {run_num}",
        highlight=False,
    )


def _print_summary_table(results: list[RunResult]) -> None:
    """Print a quick summary table at the end of an experiment."""
    if not results:
        return

    from collections import defaultdict

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
    models: list[Model],
) -> None:
    """Print what would happen without making API calls."""
    console.print("\n[bold yellow]DRY RUN — no API calls will be made[/bold yellow]\n")

    total_calls = len(models) * len(tasks) * config.runs_per_task * len(CONDITIONS)

    console.print(f"  Experiment: {config.name}")
    console.print(f'  Book: "{config.book.title}" by {config.book.author}')
    console.print(f"  Models: {len(models)}")
    console.print(f"  Tasks: {len(tasks)}")
    console.print(f"  Conditions: {len(CONDITIONS)} (baseline, control, primed)")
    console.print(f"  Runs per task: {config.runs_per_task}")
    console.print(f"  Temperature: {config.temperature}")
    console.print(f"  Total API calls: {total_calls}")
