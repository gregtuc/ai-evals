"""Model registry: provider-agnostic model instantiation."""

from __future__ import annotations

from ai_evals.config import ModelConfig
from ai_evals.models.base import Model


def get_model(config: ModelConfig, **kwargs) -> Model:
    """Instantiate a Model by provider config."""
    if config.provider == "anthropic":
        from ai_evals.models.anthropic import AnthropicModel

        return AnthropicModel(model=config.model, **kwargs)
    elif config.provider == "openai":
        from ai_evals.models.openai import OpenAIModel

        return OpenAIModel(model=config.model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
