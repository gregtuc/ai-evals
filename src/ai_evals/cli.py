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
def run(config_path: str, evals_dir: str, output_dir: str, dry_run: bool, resume: str | None) -> None:
    """Run an experiment from a config file.

    Example: ai-evals run configs/biology_campbell_mmlu_pro.yaml
    """
    from pathlib import Path

    from ai_evals.runner import ExperimentRunner

    runner = ExperimentRunner(
        config_path=config_path,
        evals_dir=evals_dir,
        output_base=output_dir,
    )

    resume_dir = Path(resume) if resume else None
    result_dir = runner.run(dry_run=dry_run, resume_dir=resume_dir)
    console.print(f"\nResults: {result_dir}")


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["table", "markdown", "json"]),
              default="table", help="Output format")
def analyze(results_dir: str, fmt: str) -> None:
    """Analyze experiment results with domain specificity analysis.

    Automatically detects domain-matched categories from the saved experiment config.

    Example: ai-evals analyze results/20240101_120000_my_experiment/
    """
    from ai_evals.analysis import ExperimentAnalyzer

    analyzer = ExperimentAnalyzer(results_dir)

    if fmt == "table":
        analyzer.auto_print_report()
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

    # Load tasks
    try:
        tasks = load_tasks_from_config(config, evals_dir=Path(evals_dir))
        categories = sorted(set(t.category for t in tasks))

        if config.benchmark:
            console.print(f"[green]Benchmark:[/green] {config.benchmark.name} ({len(tasks)} tasks)")
            console.print(f"  Categories ({len(categories)}): {', '.join(categories)}")
            console.print(f"  [green]Domain-matched:[/green] {', '.join(config.benchmark.domain_matched)}")
            mismatched = [c for c in categories if c not in config.benchmark.domain_matched]
            console.print(f"  [dim]Domain-mismatched:[/dim] {', '.join(mismatched)}")
            if config.benchmark.sample_per_category:
                console.print(f"  Sample: {config.benchmark.sample_per_category} per category")
        else:
            console.print(f"[green]Found {len(tasks)} local eval tasks[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load tasks: {e}[/yellow]")
        tasks = []

    console.print(f'\n[green]Book:[/green] "{config.book.title}" by {config.book.author}')
    console.print(f"Models: {', '.join(f'{m.provider}/{m.model}' for m in config.models)}")
    console.print(f"Conditions: 3 (baseline, control, primed)")
    console.print(f"Runs per task: {config.runs_per_task}")
    console.print(f"Temperature: {config.temperature}")

    if tasks:
        total = len(config.models) * len(tasks) * config.runs_per_task * 3
        console.print(f"Total API calls: {total}")

    if config.temperature == 0.0 and config.runs_per_task > 1:
        console.print(
            "[yellow]Warning: temperature=0.0 with multiple runs produces near-identical "
            "results. Statistical tests will be unreliable. Consider temperature=0.3.[/yellow]"
        )


if __name__ == "__main__":
    cli()
