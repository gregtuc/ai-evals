"""Tests for config.py: flexible conditions, backward compat, validation."""

from __future__ import annotations

import pytest
import yaml

from ai_evals.config import (
    DEFAULT_CONDITIONS,
    BookConfig,
    ConditionConfig,
    ExperimentConfig,
    ModelConfig,
)


def _base_config(**overrides):
    """Build a minimal valid config dict."""
    data = {
        "name": "test-experiment",
        "models": [{"provider": "anthropic", "model": "claude-sonnet-4-20250514"}],
        "book": {"title": "Test Book", "author": "Test Author"},
        "benchmark": {"name": "mmlu_pro", "domain_matched": ["biology"]},
    }
    data.update(overrides)
    return data


class TestBackwardCompatibility:
    """Old 3-field format should still work via auto-migration."""

    def test_old_format_with_no_conditions_key_gets_defaults(self):
        """Config with no conditions/baseline/control/primed gets default 3 conditions."""
        config = ExperimentConfig(**_base_config())
        assert "baseline" in config.conditions
        assert "control" in config.conditions
        assert "primed" in config.conditions
        assert config.conditions["baseline"].role == "baseline"
        assert config.conditions["control"].role == "control"
        assert config.conditions["primed"].role == "treatment"

    def test_old_format_with_legacy_keys(self):
        """Config with top-level baseline/control/primed keys migrates correctly."""
        data = _base_config(
            baseline={"template": "custom baseline: {task_prompt}"},
            control={"template": "custom control: {task_prompt}"},
            primed={"template": "custom primed: {book_title} {book_author} {task_prompt}",
                     "book_vars": True},
        )
        config = ExperimentConfig(**data)
        assert config.conditions["baseline"].template == "custom baseline: {task_prompt}"
        assert config.conditions["baseline"].role == "baseline"
        assert config.conditions["primed"].book_vars is True

    def test_old_format_partial_override(self):
        """Only overriding some legacy keys still fills in defaults for the rest."""
        data = _base_config(
            baseline={"template": "custom: {task_prompt}"},
        )
        config = ExperimentConfig(**data)
        assert "control" in config.conditions
        assert "primed" in config.conditions
        assert config.conditions["baseline"].template == "custom: {task_prompt}"

    def test_new_format_takes_precedence(self):
        """If conditions key is present, legacy keys are NOT used."""
        data = _base_config(
            conditions={
                "baseline": {"role": "baseline", "template": "{task_prompt}"},
                "my_treatment": {
                    "role": "treatment",
                    "template": "{book_title} {task_prompt}",
                    "book_vars": True,
                },
            }
        )
        config = ExperimentConfig(**data)
        assert len(config.conditions) == 2
        assert "my_treatment" in config.conditions
        assert "control" not in config.conditions


class TestConditionValidation:
    def test_missing_baseline_role_raises(self):
        """Must have at least one condition with role=baseline."""
        data = _base_config(
            conditions={
                "only_treatment": {
                    "role": "treatment",
                    "template": "{book_title} {task_prompt}",
                    "book_vars": True,
                },
            }
        )
        with pytest.raises(ValueError, match="baseline"):
            ExperimentConfig(**data)

    def test_missing_treatment_role_raises(self):
        """Must have at least one condition with role=treatment."""
        data = _base_config(
            conditions={
                "baseline": {"role": "baseline", "template": "{task_prompt}"},
                "control": {"role": "control", "template": "{task_prompt}"},
            }
        )
        with pytest.raises(ValueError, match="treatment"):
            ExperimentConfig(**data)

    def test_book_vars_template_mismatch_raises(self):
        """Template with {book_title} but book_vars=False should raise."""
        data = _base_config(
            conditions={
                "baseline": {"role": "baseline", "template": "{task_prompt}"},
                "bad": {
                    "role": "treatment",
                    "template": "{book_title} {task_prompt}",
                    "book_vars": False,
                },
            }
        )
        with pytest.raises(ValueError, match="book_vars"):
            ExperimentConfig(**data)

    def test_valid_ablation_config(self):
        """Full ablation config with 5 conditions should validate."""
        data = _base_config(
            conditions={
                "baseline": {"role": "baseline", "template": "{task_prompt}"},
                "control": {"role": "control", "template": "Think carefully.\n{task_prompt}"},
                "primed": {
                    "role": "treatment",
                    "template": "Book: {book_title} by {book_author}\n{task_prompt}",
                    "book_vars": True,
                },
                "irrelevant_book": {
                    "role": "treatment",
                    "template": "Book: {book_title} by {book_author}\n{task_prompt}",
                    "book_vars": True,
                    "book_override": {"title": "Other Book", "author": "Other Author"},
                },
                "fake_book": {
                    "role": "treatment",
                    "template": "Book: {book_title} by {book_author}\n{task_prompt}",
                    "book_vars": True,
                    "book_override": {"title": "Fake Book", "author": "Fake Author"},
                },
            }
        )
        config = ExperimentConfig(**data)
        assert len(config.conditions) == 5
        assert config.conditions["irrelevant_book"].book_override.title == "Other Book"


class TestModelConfig:
    def test_max_retries_default(self):
        mc = ModelConfig(provider="anthropic", model="test")
        assert mc.max_retries == 3

    def test_max_retries_custom(self):
        mc = ModelConfig(provider="anthropic", model="test", max_retries=5)
        assert mc.max_retries == 5


class TestExperimentConfigFields:
    def test_concurrency_default(self):
        config = ExperimentConfig(**_base_config())
        assert config.concurrency == 10

    def test_concurrency_custom(self):
        config = ExperimentConfig(**_base_config(concurrency=50))
        assert config.concurrency == 50

    def test_task_source_validation(self):
        """Must have either benchmark or eval_categories."""
        data = _base_config()
        del data["benchmark"]
        with pytest.raises(ValueError, match="benchmark.*eval_categories"):
            ExperimentConfig(**data)


class TestFromYaml:
    def test_load_existing_config(self, tmp_path):
        """Round-trip: write YAML, load it, verify."""
        config_data = _base_config(
            conditions={
                "baseline": {"role": "baseline", "template": "{task_prompt}"},
                "test_treatment": {
                    "role": "treatment",
                    "template": "{book_title} {task_prompt}",
                    "book_vars": True,
                },
            }
        )
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = ExperimentConfig.from_yaml(config_file)
        assert config.name == "test-experiment"
        assert "test_treatment" in config.conditions


class TestTemplateParallelism:
    """Verify control and treatment DEFAULT_CONDITIONS are structurally parallel.

    The templates must differ ONLY in the knowledge source reference to avoid
    confounding instruction style with the book-reference intervention.
    """

    def test_shared_structural_markers(self):
        """Both templates share the same structural phrases."""
        control = DEFAULT_CONDITIONS["control"]["template"]
        primed = DEFAULT_CONDITIONS["primed"]["template"]

        shared_phrases = [
            "Before answering",
            "draw on your deep expertise and knowledge",
            "concepts, frameworks, and knowledge",
            "to inform your answer",
            "{task_prompt}",
        ]
        for phrase in shared_phrases:
            assert phrase in control, f"Control missing: {phrase}"
            assert phrase in primed, f"Primed missing: {phrase}"

    def test_control_has_no_book_vars(self):
        """Control template must not reference book variables."""
        control = DEFAULT_CONDITIONS["control"]["template"]
        assert "{book_title}" not in control
        assert "{book_author}" not in control
        assert DEFAULT_CONDITIONS["control"]["book_vars"] is False

    def test_primed_has_book_vars(self):
        """Primed template must reference book variables."""
        primed = DEFAULT_CONDITIONS["primed"]["template"]
        assert "{book_title}" in primed
        assert "{book_author}" in primed
        assert DEFAULT_CONDITIONS["primed"]["book_vars"] is True

    def test_control_uses_task_category(self):
        """Control should reference {task_category} for equal specificity with treatment."""
        control = DEFAULT_CONDITIONS["control"]["template"]
        assert "{task_category}" in control

    def test_primed_references_book(self):
        """Primed should reference a specific book."""
        primed = DEFAULT_CONDITIONS["primed"]["template"]
        assert "from the book" in primed

    def test_control_and_treatment_equal_specificity(self):
        """Both templates should name a specific source, not just 'this domain'."""
        control = DEFAULT_CONDITIONS["control"]["template"]
        primed = DEFAULT_CONDITIONS["primed"]["template"]
        # Control names the category; treatment names a book — both are specific
        assert "{task_category}" in control
        assert "{book_title}" in primed
        # Neither should use vague "this domain"
        assert "in this domain" not in control
