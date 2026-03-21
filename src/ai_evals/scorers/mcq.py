"""MCQ scorer - extracts a letter answer from the response and checks it.

Models often respond with "The answer is B" or "B) option text" instead of
just "B". This scorer handles all common formats robustly.
"""

from __future__ import annotations

import re

from ai_evals.config import EvalTask
from ai_evals.scorers.base import ScoreResult


class MCQScorer:
    def __init__(self, expected: str | list[str] | None) -> None:
        if expected is None:
            raise ValueError("MCQScorer requires 'expected' in scorer config")
        raw = expected if isinstance(expected, str) else expected[0]
        self.expected = raw.strip().upper()

    def score(self, response: str, task: EvalTask) -> ScoreResult:
        extracted = _extract_letter(response)
        matched = extracted == self.expected

        return ScoreResult(
            score=1.0 if matched else 0.0,
            passed=matched,
            details={
                "expected": self.expected,
                "extracted": extracted,
                "response_start": response[:200],
                "matched": matched,
            },
        )


def _extract_letter(response: str) -> str | None:
    """Extract the answer letter from an MCQ response.

    Handles formats like:
    - "B"
    - "The answer is B"
    - "B) some text"
    - "Answer: B"
    - "(B)"
    """
    text = response.strip()

    # If the entire response is just a single letter
    if len(text) == 1 and text.upper() in "ABCDEFGHIJ":
        return text.upper()

    # Look for common patterns
    patterns = [
        # "The answer is B", "answer: B", "Answer is: B"
        r'(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-Ja-j])\)?',
        # "(B)" or "[B]"
        r'[\(\[]\s*([A-Ja-j])\s*[\)\]]',
        # "B)" or "B." at the start of a line
        r'(?:^|\n)\s*([A-Ja-j])\s*[).]',
        # Standalone letter with word boundary
        r'\b([A-Ja-j])\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Last resort: first capital letter in the response
    match = re.search(r'[A-J]', text)
    if match:
        return match.group(0)

    return None
