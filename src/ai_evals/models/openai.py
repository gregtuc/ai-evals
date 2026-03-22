"""OpenAI model implementation."""

from __future__ import annotations

import time

import openai

from ai_evals.models.base import ModelResponse

_CONTEXT_SIZES = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
}
_DEFAULT_CONTEXT = 128_000


class OpenAIModel:
    def __init__(self, model: str, max_retries: int = 3) -> None:
        self._model = model
        self._client = openai.OpenAI(max_retries=max_retries)
        self._async_client = openai.AsyncOpenAI(max_retries=max_retries)

    @property
    def name(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        return _CONTEXT_SIZES.get(self._model, _DEFAULT_CONTEXT)

    def _build_messages(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> list[dict[str, str]]:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        return full_messages

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> dict:
        kwargs: dict = {
            "model": self._model,
            "messages": self._build_messages(messages, system),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed
        return kwargs

    def _parse_response(self, response, latency_ms: float) -> ModelResponse:
        usage = response.usage
        return ModelResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            raw=response.model_dump(),
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        kwargs = self._build_kwargs(
            messages, system=system, temperature=temperature,
            max_tokens=max_tokens, seed=seed,
        )
        start = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        return self._parse_response(response, latency_ms)

    async def async_complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        kwargs = self._build_kwargs(
            messages, system=system, temperature=temperature,
            max_tokens=max_tokens, seed=seed,
        )
        start = time.perf_counter()
        response = await self._async_client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        return self._parse_response(response, latency_ms)
