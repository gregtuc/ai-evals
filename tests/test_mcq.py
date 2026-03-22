"""Tests for MCQ scorer letter extraction."""

from __future__ import annotations

import pytest

from ai_evals.scorers.mcq import _extract_letter, MCQScorer
from ai_evals.config import EvalTask, ScorerConfig


class TestExtractLetter:
    """Test the letter extraction logic with various response formats."""

    def test_single_letter(self):
        assert _extract_letter("B") == ("B", "single_letter")

    def test_single_letter_lowercase(self):
        assert _extract_letter("b") == ("B", "single_letter")

    def test_answer_is_pattern(self):
        letter, method = _extract_letter("The answer is B")
        assert letter == "B"

    def test_answer_colon_pattern(self):
        letter, method = _extract_letter("Answer: C")
        assert letter == "C"

    def test_parenthesized(self):
        letter, method = _extract_letter("I think (B) is correct")
        assert letter == "B"

    def test_last_line_priority(self):
        """Answer on last line should be found even if earlier text has other letters."""
        response = "Let me think about this. The Cell Biology topic is relevant.\nA is wrong.\nThe answer is B"
        letter, method = _extract_letter(response)
        assert letter == "B"

    def test_last_line_single_letter(self):
        response = "After careful analysis of all options...\nB"
        letter, method = _extract_letter(response)
        assert letter == "B"
        assert method == "last_line_single"

    def test_verbose_response_correct_extraction(self):
        """Should extract B, not C from 'Correct' or other words."""
        response = "I think the Correct answer involves Brownian motion, so the answer is B"
        letter, method = _extract_letter(response)
        assert letter == "B"

    def test_explanation_then_answer(self):
        response = "This is a complex question about DNA replication. After considering all factors, A"
        letter, method = _extract_letter(response)
        assert letter == "A"

    def test_no_letter_found(self):
        letter, method = _extract_letter("I don't know the answer to this question.")
        assert letter is None
        assert method == "none"

    def test_letter_with_period(self):
        response = "Based on my analysis, the correct answer is D."
        letter, method = _extract_letter(response)
        assert letter == "D"

    def test_multiple_mentions_last_wins(self):
        """When answer changes, last line should take priority."""
        response = "Initially I thought A, but reconsidering...\nThe answer is C"
        letter, method = _extract_letter(response)
        assert letter == "C"

    def test_letter_j_supported(self):
        """MMLU-Pro has 10 choices (A-J)."""
        letter, method = _extract_letter("The answer is J")
        assert letter == "J"

    def test_end_of_text_fallback(self):
        """A letter at the end of text should be caught by end_of_text pattern."""
        response = "After reviewing all the options carefully, I'll go with D"
        letter, method = _extract_letter(response)
        assert letter == "D"


class TestMCQScorer:
    def test_correct_answer(self):
        scorer = MCQScorer(expected="B")
        task = EvalTask(
            id="test", category="test", name="test",
            prompt="Q?", scorer=ScorerConfig(type="mcq", expected="B"),
        )
        result = scorer.score("The answer is B", task)
        assert result.passed is True
        assert result.score == 1.0
        assert result.details["extraction_method"] != "none"

    def test_wrong_answer(self):
        scorer = MCQScorer(expected="B")
        task = EvalTask(
            id="test", category="test", name="test",
            prompt="Q?", scorer=ScorerConfig(type="mcq", expected="B"),
        )
        result = scorer.score("The answer is A", task)
        assert result.passed is False
        assert result.score == 0.0

    def test_expected_from_list(self):
        scorer = MCQScorer(expected=["C"])
        assert scorer.expected == "C"

    def test_none_expected_raises(self):
        with pytest.raises(ValueError):
            MCQScorer(expected=None)
