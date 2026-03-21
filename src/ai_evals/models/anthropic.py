"""Anthropic (Claude) model implementation."""

from __future__ import annotations

import time

import anthropic

from ai_evals.models.base import ModelResponse

# Context window sizes for known models
_CONTEXT_SIZES = {
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
}
_DEFAULT_CONTEXT = 200_000


class AnthropicModel:
    def __init__(self, model: str) -> None:
        self._model = model
        self._client = anthropic.Anthropic()

    @property
    def name(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        return _CONTEXT_SIZES.get(self._model, _DEFAULT_CONTEXT)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        start = time.perf_counter()
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return ModelResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            raw=response.model_dump(),
        )
