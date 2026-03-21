"""Load eval task definitions from YAML files or HuggingFace benchmarks."""

from __future__ import annotations

from pathlib import Path

import yaml

from ai_evals.config import EvalTask, ExperimentConfig, ScorerConfig


def load_eval_file(path: str | Path) -> list[EvalTask]:
    """Load eval tasks from a single YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    category = data["category"]
    tasks = []
    for task_data in data["tasks"]:
        scorer_data = task_data.pop("scorer")
        tasks.append(
            EvalTask(
                category=category,
                scorer=ScorerConfig(**scorer_data),
                **task_data,
            )
        )
    return tasks


def load_evals(
    evals_dir: str | Path,
    categories: list[str] | None = None,
) -> list[EvalTask]:
    """Load all eval tasks from a directory, optionally filtered by category."""
    evals_dir = Path(evals_dir)
    all_tasks = []

    for yaml_file in sorted(evals_dir.glob("*.yaml")):
        tasks = load_eval_file(yaml_file)
        if categories:
            tasks = [t for t in tasks if t.category in categories]
        all_tasks.extend(tasks)

    return all_tasks


def load_tasks_from_config(
    config: ExperimentConfig,
    evals_dir: Path = Path("evals"),
) -> list[EvalTask]:
    """Unified task loader: dispatches to benchmark or local YAML based on config."""
    if config.benchmark:
        from ai_evals.benchmark_loader import load_benchmark

        return load_benchmark(
            name=config.benchmark.name,
            sample_per_category=config.benchmark.sample_per_category,
            seed=config.seed or 42,
        )
    else:
        return load_evals(evals_dir, categories=config.eval_categories)
