"""CLI interface for ai-evals."""

from __future__ import annotations

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """AI Evals: Test whether re-priming LLMs with books improves task performance."""


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--evals-dir", default="evals", help="Directory containing eval YAML files")
@click.option("--output-dir", default="results", help="Base directory for results")
@click.option("--dry-run", is_flag=True, help="Show what would run without making API calls")
@click.option("--resume", type=click.Path(exists=True), default=None,
              help="Resume from an existing results directory")
@click.option("--concurrency", type=int, default=None,
              help="Override concurrency from config (number of parallel API calls)")
@click.option("--multi-seed", type=int, default=None,
              help="Run N replications with different task-sampling seeds for robustness")
def run(config_path: str, evals_dir: str, output_dir: str, dry_run: bool,
        resume: str | None, concurrency: int | None, multi_seed: int | None) -> None:
    """Run an experiment from a config file.

    Example: ai-evals run configs/biology_campbell_mmlu_pro.yaml
    Multi-seed: ai-evals run configs/biology_campbell_mmlu_pro.yaml --multi-seed 5
    """
    from pathlib import Path

    if multi_seed:
        if multi_seed < 2:
            console.print("[red]--multi-seed requires N >= 2[/red]")
            raise SystemExit(1)
        if dry_run:
            # For dry-run, just show what a single run would look like
            from ai_evals.runner import ExperimentRunner
            runner = ExperimentRunner(
                config_path=config_path, evals_dir=evals_dir, output_base=output_dir,
            )
            if concurrency is not None:
                runner.config.concurrency = concurrency
            runner.run(dry_run=True)
            base_seed = runner.config.seed or 42
            seeds = [base_seed + i for i in range(multi_seed)]
            console.print(f"\n[bold]Multi-seed:[/bold] {multi_seed} replications with seeds {seeds}")
            console.print(f"[bold]Total API calls across all seeds:[/bold] multiply above by {multi_seed}")
            return

        from ai_evals.runner import run_multi_seed
        result_dir = run_multi_seed(
            config_path=config_path,
            n_seeds=multi_seed,
            evals_dir=evals_dir,
            output_base=output_dir,
            concurrency_override=concurrency,
        )
        console.print(f"\nResults: {result_dir}")
        return

    from ai_evals.runner import ExperimentRunner

    runner = ExperimentRunner(
        config_path=config_path,
        evals_dir=evals_dir,
        output_base=output_dir,
    )

    if concurrency is not None:
        runner.config.concurrency = concurrency

    resume_dir = Path(resume) if resume else None
    result_dir = runner.run(dry_run=dry_run, resume_dir=resume_dir)
    console.print(f"\nResults: {result_dir}")


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["table", "markdown", "json"]),
              default="table", help="Output format")
@click.option("--treatment", default=None,
              help="Focus domain specificity analysis on a specific treatment condition")
def analyze(results_dir: str, fmt: str, treatment: str | None) -> None:
    """Analyze experiment results with domain specificity analysis.

    Automatically detects multi-seed structure and domain-matched categories
    from the saved experiment config.

    Example: ai-evals analyze results/20240101_120000_my_experiment/
    """
    from pathlib import Path

    results_path = Path(results_dir)

    # Check for multi-seed structure
    manifest_path = results_path / "multi_seed_manifest.json"
    if manifest_path.exists():
        from ai_evals.analysis import CrossSeedAnalyzer
        from ai_evals.config import ExperimentConfig

        cross = CrossSeedAnalyzer(results_path)
        cross.print_cross_seed_report()

        # Also run domain specificity across seeds if config available
        config_path = results_path / "experiment_config.yaml"
        if config_path.exists():
            try:
                config = ExperimentConfig.from_yaml(config_path)
                if config.benchmark and config.benchmark.domain_matched:
                    cross.domain_specificity_replication(
                        config.benchmark.domain_matched,
                        treatment_condition=treatment,
                    )
            except Exception:
                pass

        # Print per-seed details for the first seed as a representative
        first_seed = cross.seeds[0]
        console.print(f"\n[bold]Detailed report for seed {first_seed} (representative):[/bold]")
        first_analyzer = cross.analyzers[first_seed]
        if fmt == "table":
            first_analyzer.auto_print_report()
            first_analyzer.print_power_report()
        elif fmt == "markdown":
            console.print(first_analyzer.to_markdown())
        elif fmt == "json":
            import json
            console.print(json.dumps(first_analyzer.summary(), indent=2))
        return

    # Single-seed analysis
    from ai_evals.analysis import ExperimentAnalyzer

    analyzer = ExperimentAnalyzer(results_dir)

    if fmt == "table":
        analyzer.auto_print_report()
        if treatment:
            # Also run focused domain specificity for specified treatment
            from ai_evals.config import ExperimentConfig
            config_path = results_path / "experiment_config.yaml"
            if config_path.exists():
                config = ExperimentConfig.from_yaml(config_path)
                if config.benchmark and config.benchmark.domain_matched:
                    analyzer.print_domain_report(
                        config.benchmark.domain_matched,
                        treatment_condition=treatment,
                    )
        analyzer.print_power_report()
    elif fmt == "markdown":
        console.print(analyzer.to_markdown())
    elif fmt == "json":
        import json
        console.print(json.dumps(analyzer.summary(), indent=2))


@cli.command("list-evals")
@click.option("--evals-dir", default="evals", help="Directory containing eval YAML files")
@click.option("--category", default=None, help="Filter by category")
@click.option("--benchmark", default=None,
              type=click.Choice(["mmlu_pro", "gpqa_diamond", "hle"]),
              help="List tasks from a HuggingFace benchmark instead of local YAML")
def list_evals(evals_dir: str, category: str | None, benchmark: str | None) -> None:
    """List available eval tasks (local YAML or HuggingFace benchmark)."""
    if benchmark:
        from ai_evals.benchmark_loader import load_benchmark
        tasks = load_benchmark(benchmark, sample_per_category=3)
        console.print(f"[dim]Showing sample (3 per category) from {benchmark}[/dim]\n")
    else:
        from ai_evals.eval_loader import load_evals
        categories = [category] if category else None
        tasks = load_evals(evals_dir, categories=categories)

    if not tasks:
        console.print("[yellow]No eval tasks found.[/yellow]")
        return

    table = Table(title="Available Eval Tasks")
    table.add_column("ID")
    table.add_column("Category")
    table.add_column("Name", max_width=50)
    table.add_column("Scorer")

    for task in tasks:
        table.add_row(task.id, task.category, task.name, task.scorer.type)

    console.print(table)

    # Show category summary
    from collections import Counter
    if benchmark:
        # For benchmarks, show full category counts
        full_tasks = load_benchmark(benchmark)
        cat_counts = Counter(t.category for t in full_tasks)
        console.print(f"\nTotal tasks in {benchmark}: {len(full_tasks)}")
        console.print("Categories:")
        for cat, count in sorted(cat_counts.items()):
            console.print(f"  {cat}: {count}")
    else:
        console.print(f"\nTotal: {len(tasks)} tasks")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--evals-dir", default="evals", help="Directory containing eval YAML files")
def validate(config_path: str, evals_dir: str) -> None:
    """Validate an experiment config without running it.

    Example: ai-evals validate configs/biology_campbell_mmlu_pro.yaml
    """
    from ai_evals.config import ExperimentConfig
    from ai_evals.eval_loader import load_tasks_from_config
    from pathlib import Path

    try:
        config = ExperimentConfig.from_yaml(config_path)
        console.print(f"[green]Config valid:[/green] {config.name}")
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise SystemExit(1)

    # Show conditions
    console.print(f"\n[bold]Conditions ({len(config.conditions)}):[/bold]")
    for name, cond in config.conditions.items():
        book_info = ""
        if cond.book_vars:
            if cond.book_override:
                book_info = f' [book: "{cond.book_override.title}"]'
            else:
                book_info = f' [book: "{config.book.title}"]'
        console.print(f"  {name} ({cond.role}){book_info}")

    # Load tasks
    try:
        tasks = load_tasks_from_config(config, evals_dir=Path(evals_dir))
        categories = sorted(set(t.category for t in tasks))

        if config.benchmark:
            console.print(f"\n[green]Benchmark:[/green] {config.benchmark.name} ({len(tasks)} tasks)")
            console.print(f"  Categories ({len(categories)}): {', '.join(categories)}")
            console.print(f"  [green]Domain-matched:[/green] {', '.join(config.benchmark.domain_matched)}")
            mismatched = [c for c in categories if c not in config.benchmark.domain_matched]
            console.print(f"  [dim]Domain-mismatched:[/dim] {', '.join(mismatched)}")
            if config.benchmark.sample_per_category:
                console.print(f"  Sample: {config.benchmark.sample_per_category} per category")
        else:
            console.print(f"\n[green]Found {len(tasks)} local eval tasks[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load tasks: {e}[/yellow]")
        tasks = []

    console.print(f'\n[green]Book:[/green] "{config.book.title}" by {config.book.author}')
    console.print(f"Models: {', '.join(f'{m.provider}/{m.model}' for m in config.models)}")
    console.print(f"Runs per task: {config.runs_per_task}")
    console.print(f"Temperature: {config.temperature}")
    console.print(f"Concurrency: {config.concurrency}")

    if tasks:
        total = len(config.models) * len(tasks) * config.runs_per_task * len(config.conditions)
        console.print(f"Total API calls: {total}")

    if tasks:
        from ai_evals.runner import _print_power_preview
        _print_power_preview(tasks, config)

    if config.temperature == 0.0 and config.runs_per_task > 1:
        console.print(
            "[yellow]Warning: temperature=0.0 with multiple runs produces near-identical "
            "results. Statistical tests will be unreliable. Consider temperature=0.3.[/yellow]"
        )


@cli.command("pre-register")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--evals-dir", default="evals", help="Directory containing eval YAML files")
@click.option("--output", default=None, help="Output file path (default: <config_name>_preregistration.yaml)")
def pre_register(config_path: str, evals_dir: str, output: str | None) -> None:
    """Generate a pre-registration document from an experiment config.

    Creates a timestamped, hash-verified analysis plan that commits to
    hypotheses, sample sizes, statistical tests, and alpha levels BEFORE
    data collection. This prevents post-hoc rationalization and p-hacking.

    Example: ai-evals pre-register configs/biology_campbell_mmlu_pro.yaml
    """
    import hashlib
    from collections import Counter
    from datetime import datetime, timezone
    from pathlib import Path

    import yaml

    from ai_evals.analysis import _minimum_detectable_effect, _recommended_sample_size
    from ai_evals.config import ExperimentConfig
    from ai_evals.eval_loader import load_tasks_from_config

    try:
        config = ExperimentConfig.from_yaml(config_path)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise SystemExit(1)

    # Load tasks for sample size info
    try:
        tasks = load_tasks_from_config(config, evals_dir=Path(evals_dir))
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load tasks: {e}[/yellow]")
        tasks = []

    # Build pre-registration document
    timestamp = datetime.now(timezone.utc).isoformat()
    config_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()

    cat_counts = Counter(t.category for t in tasks) if tasks else {}
    matched = config.benchmark.domain_matched if config.benchmark else []
    mismatched = [c for c in sorted(cat_counts.keys()) if c not in matched]

    n_matched = sum(cat_counts.get(c, 0) for c in matched)
    n_mismatched = sum(cat_counts.get(c, 0) for c in mismatched)

    # Power analysis
    power_info = {}
    for cat, count in sorted(cat_counts.items()):
        min_d = _minimum_detectable_effect(count)
        if min_d is not None:
            power_info[cat] = {"n_tasks": count, "min_detectable_d": round(min_d, 3)}

    # Sample size recommendations
    recommendations = {}
    for target_d, label in [(0.2, "small"), (0.3, "small-medium"), (0.5, "medium")]:
        rec_n = _recommended_sample_size(target_d)
        if rec_n is not None:
            recommendations[label] = {"target_d": target_d, "required_n_per_category": rec_n}

    conditions_info = {}
    for name, cond in config.conditions.items():
        info = {"role": cond.role, "book_vars": cond.book_vars}
        if cond.book_override:
            info["book_override"] = {"title": cond.book_override.title, "author": cond.book_override.author}
        conditions_info[name] = info

    prereg = {
        "pre_registration": {
            "title": f"Pre-registration: {config.name}",
            "generated": timestamp,
            "config_hash": config_hash,
            "config_file": str(config_path),
        },
        "hypotheses": {
            "primary": (
                f'Mentioning "{config.book.title}" by {config.book.author} will produce '
                f"a larger improvement in {', '.join(matched)} MMLU-Pro questions compared "
                f"to non-matched domains (domain-specific effect)."
            ),
            "secondary": [
                "Treatment (primed) will outperform baseline across matched domains.",
                "Treatment will NOT systematically outperform control in mismatched domains.",
                "The control condition (elaborate framing without book) will not show "
                "domain-specific improvement patterns.",
            ],
            "null_hypothesis": (
                "There is no difference in treatment effect size between matched and "
                "mismatched domains (the book reference does not produce domain-specific effects)."
            ),
        },
        "design": {
            "book": {"title": config.book.title, "author": config.book.author},
            "conditions": conditions_info,
            "benchmark": config.benchmark.name if config.benchmark else "local",
            "domain_matched_categories": matched,
            "domain_mismatched_categories": mismatched,
            "runs_per_task": config.runs_per_task,
            "temperature": config.temperature,
            "seed": config.seed,
            "models": [f"{m.provider}/{m.model}" for m in config.models],
        },
        "sample_sizes": {
            "total_tasks": len(tasks),
            "tasks_per_category": dict(sorted(cat_counts.items())),
            "matched_tasks": n_matched,
            "mismatched_tasks": n_mismatched,
            "total_api_calls": (
                len(config.models) * len(tasks) * config.runs_per_task * len(config.conditions)
                if tasks else 0
            ),
        },
        "analysis_plan": {
            "unit_of_analysis": "Task (scores averaged across runs before statistical testing)",
            "primary_test": {
                "name": "Domain specificity",
                "method": "Mixed-effects linear model: effect ~ is_matched + (1|category)",
                "fallback": "Welch's t-test on task-level effects (if statsmodels unavailable)",
                "direction": "One-sided (matched > mismatched)",
                "alpha": 0.05,
            },
            "secondary_tests": {
                "pairwise_comparisons": {
                    "method": "Paired t-test (n>=20) or Wilcoxon signed-rank (n<20)",
                    "binary_data": "McNemar's exact test (single-run binary outcomes only)",
                    "correction": "Holm-Bonferroni across all pairwise comparisons",
                    "alpha": 0.05,
                },
                "effect_sizes": "Cohen's d with bootstrap 95% confidence intervals",
                "power": "Post-hoc power analysis via non-central t-distribution",
            },
            "measurement_audit": {
                "extraction_methods": "Chi-square independence test across conditions",
                "response_lengths": "Per-condition mean/median with divergence warnings (>20%)",
            },
            "stopping_rule": (
                "No interim analyses. All data collected before any analysis. "
                "No early stopping based on partial results."
            ),
            "exclusion_criteria": (
                "Tasks where MCQ extraction fails (method='none') are scored as 0. "
                "No post-hoc exclusion of tasks or categories."
            ),
        },
        "power_analysis": {
            "per_category": power_info,
            "sample_size_recommendations": recommendations,
        },
        "limitations": [
            "Single-seed design: results depend on task sampling. Use --multi-seed for replication.",
            "Temperature-based variance is not true independence between runs.",
            "Prompt sensitivity: small template changes can affect results.",
            "Behavioral test only: cannot determine mechanism (attention, training distribution, etc.).",
        ],
    }

    # Compute integrity hash of the document
    prereg_yaml = yaml.dump(prereg, default_flow_style=False, sort_keys=False)
    integrity_hash = hashlib.sha256(prereg_yaml.encode()).hexdigest()
    prereg["integrity_hash"] = f"sha256:{integrity_hash}"

    # Final YAML with hash
    final_yaml = yaml.dump(prereg, default_flow_style=False, sort_keys=False)

    # Write output
    if output is None:
        output = f"{config.name}_preregistration.yaml"
    with open(output, "w") as f:
        f.write(final_yaml)

    console.print(f"[green]Pre-registration saved:[/green] {output}")
    console.print(f"  Config hash: {config_hash[:16]}...")
    console.print(f"  Integrity hash: sha256:{integrity_hash[:16]}...")
    console.print(f"  Hypothesis: {prereg['hypotheses']['primary'][:100]}...")
    console.print(f"  Primary test: {prereg['analysis_plan']['primary_test']['method']}")
    console.print(f"  Alpha: {prereg['analysis_plan']['primary_test']['alpha']}")
    if tasks:
        console.print(f"  Tasks: {len(tasks)} ({n_matched} matched, {n_mismatched} mismatched)")
    console.print(
        f"\n[dim]Commit this file before running the experiment to establish "
        f"your analysis plan. The integrity hash verifies it was not modified.[/dim]"
    )


if __name__ == "__main__":
    cli()
