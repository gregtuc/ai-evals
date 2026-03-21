"""Base scorer protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ai_evals.config import EvalTask


@dataclass
class ScoreResult:
    score: float  # 0.0 to 1.0
    passed: bool
    details: dict = field(default_factory=dict)


class Scorer(Protocol):
    def score(self, response: str, task: EvalTask) -> ScoreResult: ...
