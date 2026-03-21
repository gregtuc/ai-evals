"""Pydantic models for all configuration and data structures."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    provider: Literal["anthropic", "openai"]
    model: str  # e.g. "claude-sonnet-4-20250514", "gpt-4o"


class BookConfig(BaseModel):
    title: str
    author: str
    description: str = ""  # optional extra context about the book


class BenchmarkConfig(BaseModel):
    """Config for using a HuggingFace benchmark dataset."""

    name: Literal["mmlu_pro", "gpqa_diamond", "hle"]
    domain_matched: list[str]  # categories that match the book's domain
    sample_per_category: int | None = None  # subsample for cost control (None = all)


class ConditionConfig(BaseModel):
    """Config for a single experimental condition's prompt template."""

    system_prompt: str = "You are a helpful assistant."
    template: str = "{task_prompt}"


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    models: list[ModelConfig]
    book: BookConfig
    # Three conditions: baseline (bare prompt), control (elaborate but no book),
    # primed (elaborate with book reference)
    baseline: ConditionConfig = ConditionConfig(
        template="{task_prompt}",
    )
    control: ConditionConfig = ConditionConfig(
        template=(
            "Before answering, think carefully and draw on your deep expertise "
            "and knowledge in this domain to inform your answer.\n\n{task_prompt}"
        ),
    )
    primed: ConditionConfig = ConditionConfig(
        template=(
            "Before answering, recall everything you know from the book "
            '"{book_title}" by {book_author}. '
            "Draw on the concepts, frameworks, and knowledge from that book "
            "to inform your answer.\n\n{task_prompt}"
        ),
    )
    # Task source: use benchmark OR local eval_categories (at least one required)
    benchmark: BenchmarkConfig | None = None
    eval_categories: list[str] = Field(default_factory=list)
    runs_per_task: int = Field(default=5, ge=1)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    seed: int | None = 42
    max_tokens: int = Field(default=4096, ge=1)

    @model_validator(mode="after")
    def check_task_source(self) -> ExperimentConfig:
        if not self.benchmark and not self.eval_categories:
            raise ValueError("Must specify either 'benchmark' or 'eval_categories'")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ScorerConfig(BaseModel):
    type: Literal["exact_match", "contains", "mcq", "llm_judge", "code_execution"]
    expected: str | list[str] | None = None
    rubric: str | None = None  # for llm_judge
    judge_model: ModelConfig | None = None
    timeout_seconds: int = 30  # for code_execution


class EvalTask(BaseModel):
    id: str
    category: str
    name: str
    prompt: str
    scorer: ScorerConfig
    metadata: dict = Field(default_factory=dict)
