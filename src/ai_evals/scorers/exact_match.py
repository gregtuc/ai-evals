"""Exact match scorer - checks if the response contains the expected answer.

More robust than a strict equality check: normalizes whitespace, case,
and common formatting variations before matching.
"""

from __future__ import annotations

import re

from ai_evals.config import EvalTask
from ai_evals.scorers.base import ScoreResult


class ExactMatchScorer:
    def __init__(self, expected: str | list[str] | None) -> None:
        if expected is None:
            raise ValueError("ExactMatchScorer requires 'expected' in scorer config")
        self.expected = expected if isinstance(expected, str) else expected[0]

    def score(self, response: str, task: EvalTask) -> ScoreResult:
        normalized_response = _normalize(response)
        normalized_expected = _normalize(self.expected)

        # Try several matching strategies
        matched = False

        # 1. Direct containment after normalization
        if normalized_expected in normalized_response:
            matched = True

        # 2. Check if response starts with the expected answer
        #    (model might give answer then explanation)
        if not matched:
            first_line = normalized_response.split("\n")[0].strip()
            if normalized_expected in first_line:
                matched = True

        # 3. For single-word/short expected values, check word boundaries
        if not matched and len(normalized_expected.split()) <= 3:
            pattern = r'\b' + re.escape(normalized_expected) + r'\b'
            if re.search(pattern, normalized_response):
                matched = True

        return ScoreResult(
            score=1.0 if matched else 0.0,
            passed=matched,
            details={
                "expected": self.expected,
                "response_start": response[:200],
                "matched": matched,
            },
        )


def _normalize(text: str) -> str:
    """Normalize for comparison: lowercase, collapse whitespace, strip punctuation edges."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing punctuation (quotes, periods, etc.)
    text = text.strip("\"'`.,!;:")
    return text
