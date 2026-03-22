"""MCQ scorer - extracts a letter answer from the response and checks it.

Models often respond with "The answer is B" or "B) option text" instead of
just "B". This scorer handles all common formats robustly, prioritizing
the end of the response where models typically state their final answer.
"""

from __future__ import annotations

import re

from ai_evals.config import EvalTask
from ai_evals.scorers.base import ScoreResult

# Structured patterns for letter extraction, tried in order
_ANSWER_PATTERNS = [
    # "The answer is B", "answer: B", "Answer is: B"
    r'(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-Ja-j])\)?',
    # "(B)" or "[B]"
    r'[\(\[]\s*([A-Ja-j])\s*[\)\]]',
    # "B)" or "B." at the start of a line
    r'(?:^|\n)\s*([A-Ja-j])\s*[).]',
]


class MCQScorer:
    def __init__(self, expected: str | list[str] | None) -> None:
        if expected is None:
            raise ValueError("MCQScorer requires 'expected' in scorer config")
        raw = expected if isinstance(expected, str) else expected[0]
        self.expected = raw.strip().upper()

    def score(self, response: str, task: EvalTask) -> ScoreResult:
        extracted, method = _extract_letter(response)
        matched = extracted == self.expected

        return ScoreResult(
            score=1.0 if matched else 0.0,
            passed=matched,
            details={
                "expected": self.expected,
                "extracted": extracted,
                "extraction_method": method,
                "response_start": response[:200],
                "matched": matched,
            },
        )


def _try_patterns(text: str) -> str | None:
    """Try structured answer patterns against text. Returns letter or None."""
    for pattern in _ANSWER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _extract_letter(response: str) -> tuple[str | None, str]:
    """Extract the answer letter from an MCQ response.

    Returns (letter, extraction_method) for audit trail.

    Priority order:
    1. Single-letter response
    2. Last non-empty line (models often put final answer last)
    3. Full text with structured patterns
    4. End-of-text standalone letter
    """
    text = response.strip()

    # Stage 1: entire response is a single letter
    if len(text) == 1 and text.upper() in "ABCDEFGHIJ":
        return text.upper(), "single_letter"

    # Stage 2: check last non-empty line first (models typically conclude with answer)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        last_line = lines[-1]
        # Single letter on last line
        if len(last_line) == 1 and last_line.upper() in "ABCDEFGHIJ":
            return last_line.upper(), "last_line_single"
        result = _try_patterns(last_line)
        if result:
            return result, "last_line_pattern"

    # Stage 3: full text scan with structured patterns
    result = _try_patterns(text)
    if result:
        return result, "full_scan_pattern"

    # Stage 4: standalone letter at end of text (letter followed only by punctuation/whitespace)
    match = re.search(r'\b([A-Ja-j])\s*[.!?)]*\s*$', text)
    if match:
        return match.group(1).upper(), "end_of_text"

    return None, "none"
