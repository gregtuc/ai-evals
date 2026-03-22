"""RunResult model and JSONL-based result storage."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class RunResult(BaseModel):
    experiment_name: str
    run_id: str
    task_id: str
    task_category: str
    model: str
    condition: str
    condition_role: str = ""  # "baseline", "control", "treatment"; default for old data compat
    run_number: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    response: str
    score: float
    score_passed: bool
    scorer_details: dict = Field(default_factory=dict)
    task_metadata: dict = Field(default_factory=dict)
    timestamp: str  # ISO 8601
    config_hash: str


class ResultStore:
    """Append-only JSONL result store with resume support."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.jsonl"

    def append(self, result: RunResult) -> None:
        with open(self.results_file, "a") as f:
            f.write(result.model_dump_json() + "\n")

    def load_all(self) -> list[RunResult]:
        if not self.results_file.exists():
            return []
        results = []
        with open(self.results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(RunResult(**json.loads(line)))
        return results

    def get_completed_keys(self) -> set[tuple[str, str, str, int]]:
        """Return set of (task_id, model, condition, run_number) already completed."""
        keys = set()
        for result in self.load_all():
            keys.add((result.task_id, result.model, result.condition, result.run_number))
        return keys
