"""LLM-as-judge scorer - uses a separate model to evaluate response quality."""

from __future__ import annotations

import json
import re

from ai_evals.config import EvalTask, ModelConfig
from ai_evals.models import get_model
from ai_evals.scorers.base import ScoreResult

_JUDGE_SYSTEM = """You are an expert evaluator. You will be given a task prompt and a response to evaluate.
Score the response on a scale of 0-10 based on the rubric provided.

You MUST respond with valid JSON in this exact format:
{"score": <0-10>, "reasoning": "<brief explanation>"}"""

_JUDGE_TEMPLATE = """## Task Prompt
{task_prompt}

## Response to Evaluate
{response}

## Rubric
{rubric}

Score this response 0-10 based on the rubric. Respond with JSON only."""


class LLMJudgeScorer:
    def __init__(
        self,
        rubric: str | None = None,
        judge_model_config: ModelConfig | None = None,
    ) -> None:
        if rubric is None:
            raise ValueError("LLMJudgeScorer requires 'rubric' in scorer config")
        self.rubric = rubric
        self.judge_model_config = judge_model_config or ModelConfig(
            provider="anthropic", model="claude-sonnet-4-20250514"
        )
        self._judge = get_model(self.judge_model_config)

    def score(self, response: str, task: EvalTask) -> ScoreResult:
        prompt = _JUDGE_TEMPLATE.format(
            task_prompt=task.prompt,
            response=response,
            rubric=self.rubric,
        )

        judge_response = self._judge.complete(
            messages=[{"role": "user", "content": prompt}],
            system=_JUDGE_SYSTEM,
            temperature=0.0,
            max_tokens=256,
        )

        parsed = _parse_judge_response(judge_response.content)
        normalized_score = parsed["score"] / 10.0

        return ScoreResult(
            score=normalized_score,
            passed=normalized_score >= 0.7,
            details={
                "raw_score": parsed["score"],
                "reasoning": parsed["reasoning"],
                "judge_model": self._judge.name,
            },
        )


def _parse_judge_response(text: str) -> dict:
    """Extract score and reasoning from judge response."""
    # Try direct JSON parse first
    try:
        data = json.loads(text.strip())
        return {"score": int(data["score"]), "reasoning": data.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Try to find JSON in the response
    match = re.search(r"\{[^}]+\}", text)
    if match:
        try:
            data = json.loads(match.group())
            return {"score": int(data["score"]), "reasoning": data.get("reasoning", "")}
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fallback: try to find a number
    match = re.search(r"\b(\d+)\s*/?\s*10\b", text)
    if match:
        return {"score": int(match.group(1)), "reasoning": text[:200]}

    raise ValueError(f"Could not parse judge response: {text[:200]}")
