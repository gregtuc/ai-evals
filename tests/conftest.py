"""Shared test fixtures."""

from __future__ import annotations

import pytest

from ai_evals.config import EvalTask, ScorerConfig
from ai_evals.models.base import ModelResponse


class MockModel:
    """A mock model for testing that returns a fixed response."""

    def __init__(self, response: str = "test response") -> None:
        self._response = response
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "mock-model"

    @property
    def max_context_tokens(self) -> int:
        return 100_000

    def complete(self, messages, *, system=None, temperature=0.0, max_tokens=4096, seed=None):
        self.calls.append({
            "messages": messages,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        })
        return ModelResponse(
            content=self._response,
            model="mock-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=10.0,
        )


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def sample_task():
    return EvalTask(
        id="test_001",
        category="test",
        name="Test task",
        prompt="What is 2 + 2?",
        scorer=ScorerConfig(type="exact_match", expected="4"),
    )
