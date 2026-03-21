"""Contains scorer - checks if the response contains expected strings."""

from __future__ import annotations

from ai_evals.config import EvalTask
from ai_evals.scorers.base import ScoreResult


class ContainsScorer:
    def __init__(self, expected: str | list[str] | None) -> None:
        if expected is None:
            raise ValueError("ContainsScorer requires 'expected' in scorer config")
        self.expected = [expected] if isinstance(expected, str) else expected

    def score(self, response: str, task: EvalTask) -> ScoreResult:
        response_lower = response.lower()
        found = [exp for exp in self.expected if exp.lower() in response_lower]
        fraction = len(found) / len(self.expected)

        return ScoreResult(
            score=fraction,
            passed=fraction == 1.0,
            details={
                "expected": self.expected,
                "found": found,
                "missing": [e for e in self.expected if e not in found],
            },
        )
