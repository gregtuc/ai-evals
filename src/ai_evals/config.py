"""Pydantic models for all configuration and data structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    provider: Literal["anthropic", "openai"]
    model: str  # e.g. "claude-sonnet-4-20250514", "gpt-4o"
    max_retries: int = 3


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
    """Config for a single experimental condition's prompt template.

    Each condition has a role that drives automatic pair discovery in analysis:
    - "baseline": bare prompt, no framing
    - "control": elaborate framing without book reference
    - "treatment": elaborate framing with book reference (the intervention being tested)
    """

    role: Literal["baseline", "control", "treatment"] = "control"
    system_prompt: str = "You are a helpful assistant."
    template: str = "{task_prompt}"
    book_vars: bool = False  # whether to inject {book_title}, {book_author}
    book_override: BookConfig | None = None  # for ablation conditions using a different book


# Default conditions matching the original 3-condition design.
# IMPORTANT: Control and treatment templates are structurally parallel.
# They differ ONLY in the knowledge source reference ("in this domain" vs
# "from the book X by Y") to avoid confounding instruction style with the
# book reference intervention.
DEFAULT_CONDITIONS: dict[str, dict[str, Any]] = {
    "baseline": {
        "role": "baseline",
        "template": "{task_prompt}",
        "book_vars": False,
    },
    "control": {
        "role": "control",
        "template": (
            "Before answering, draw on your deep expertise and knowledge "
            "in {task_category}. Use the concepts, frameworks, and knowledge "
            "you have in this area to inform your answer.\n\n{task_prompt}"
        ),
        "book_vars": False,
    },
    "primed": {
        "role": "treatment",
        "template": (
            "Before answering, draw on your deep expertise and knowledge "
            'from the book "{book_title}" by {book_author}. Use the concepts, '
            "frameworks, and knowledge from that book "
            "to inform your answer.\n\n{task_prompt}"
        ),
        "book_vars": True,
    },
}


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    models: list[ModelConfig]
    book: BookConfig
    conditions: dict[str, ConditionConfig] = Field(default_factory=dict)
    # Task source: use benchmark OR local eval_categories (at least one required)
    benchmark: BenchmarkConfig | None = None
    eval_categories: list[str] = Field(default_factory=list)
    runs_per_task: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of runs per task per condition. Multiple runs are averaged "
            "per-task before analysis (the task is the unit of analysis, not "
            "individual runs). More runs reduce within-task noise but do not "
            "increase the effective sample size for statistical tests."
        ),
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    seed: int | None = 42
    max_tokens: int = Field(default=4096, ge=1)
    concurrency: int = Field(default=10, ge=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_conditions(cls, data: Any) -> Any:
        """Backward compatibility: convert old 3-field format to conditions dict."""
        if not isinstance(data, dict):
            return data

        # If conditions is already specified, nothing to migrate
        if "conditions" in data:
            return data

        # Check for old-format keys (baseline, control, primed as top-level)
        legacy_keys = {"baseline", "control", "primed"}
        found_legacy = {k for k in legacy_keys if k in data}

        if found_legacy:
            # Old format detected: pop the legacy keys and build conditions dict
            conditions = {}
            for key in legacy_keys:
                if key in data:
                    legacy_val = data.pop(key)
                    # Legacy ConditionConfig only had system_prompt and template
                    cond = dict(DEFAULT_CONDITIONS[key])
                    if isinstance(legacy_val, dict):
                        cond.update(legacy_val)
                    conditions[key] = cond
                else:
                    # Use default for missing legacy keys
                    conditions[key] = dict(DEFAULT_CONDITIONS[key])
            data["conditions"] = conditions
        else:
            # No conditions specified at all: use all defaults
            data["conditions"] = {k: dict(v) for k, v in DEFAULT_CONDITIONS.items()}

        return data

    @model_validator(mode="after")
    def check_task_source(self) -> ExperimentConfig:
        if not self.benchmark and not self.eval_categories:
            raise ValueError("Must specify either 'benchmark' or 'eval_categories'")
        return self

    @model_validator(mode="after")
    def check_condition_roles(self) -> ExperimentConfig:
        """Ensure at least one baseline and one treatment role exist."""
        roles = {c.role for c in self.conditions.values()}
        if "baseline" not in roles:
            raise ValueError("At least one condition must have role='baseline'")
        if "treatment" not in roles:
            raise ValueError("At least one condition must have role='treatment'")
        return self

    @model_validator(mode="after")
    def check_book_vars(self) -> ExperimentConfig:
        """Ensure book variables are only used in conditions that declare book_vars."""
        for name, cond in self.conditions.items():
            has_book_placeholders = "{book_title}" in cond.template or "{book_author}" in cond.template
            if has_book_placeholders and not cond.book_vars:
                raise ValueError(
                    f"Condition '{name}' uses {{book_title}}/{{book_author}} in template "
                    f"but book_vars is False. Set book_vars: true."
                )
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
