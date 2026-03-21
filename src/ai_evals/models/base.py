"""Base model protocol and response dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ModelResponse:
    """Raw response from an LLM API call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict = field(default_factory=dict)


class Model(Protocol):
    """Provider-agnostic LLM interface."""

    @property
    def name(self) -> str: ...

    @property
    def max_context_tokens(self) -> int: ...

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse: ...
