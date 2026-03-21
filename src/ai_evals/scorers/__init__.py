"""Scorer registry."""

from __future__ import annotations

from ai_evals.config import ScorerConfig
from ai_evals.scorers.base import Scorer


def get_scorer(config: ScorerConfig, **kwargs) -> Scorer:
    """Instantiate a Scorer by config."""
    if config.type == "exact_match":
        from ai_evals.scorers.exact_match import ExactMatchScorer

        return ExactMatchScorer(expected=config.expected)
    elif config.type == "contains":
        from ai_evals.scorers.contains import ContainsScorer

        return ContainsScorer(expected=config.expected)
    elif config.type == "llm_judge":
        from ai_evals.scorers.llm_judge import LLMJudgeScorer

        return LLMJudgeScorer(rubric=config.rubric, judge_model_config=config.judge_model)
    elif config.type == "mcq":
        from ai_evals.scorers.mcq import MCQScorer

        return MCQScorer(expected=config.expected)
    elif config.type == "code_execution":
        from ai_evals.scorers.code_execution import CodeExecutionScorer

        return CodeExecutionScorer(
            expected=config.expected, timeout_seconds=config.timeout_seconds
        )
    else:
        raise ValueError(f"Unknown scorer type: {config.type}")
