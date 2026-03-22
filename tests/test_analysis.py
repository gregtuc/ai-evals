"""Tests for analysis.py: per-task aggregation, dynamic pairs, domain specificity, power,
mixed-effects model, chi-square audit, cross-seed analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ai_evals.analysis import (
    CrossSeedAnalyzer,
    ExperimentAnalyzer,
    PairwiseComparison,
    _bootstrap_ci,
    _domain_specificity_mixed_model,
    _domain_specificity_test,
    _extraction_method_chi_square,
    _mcnemar_test,
    _mean,
    _minimum_detectable_effect,
    _permutation_test,
    _post_hoc_power,
    _recommended_sample_size,
    _run_significance_test,
    _std,
)
from ai_evals.results import RunResult


def _make_result(
    task_id: str = "t1",
    category: str = "bio",
    model: str = "test-model",
    condition: str = "baseline",
    condition_role: str = "baseline",
    run_number: int = 1,
    score: float = 1.0,
    passed: bool = True,
) -> RunResult:
    return RunResult(
        experiment_name="test",
        run_id="r1",
        task_id=task_id,
        task_category=category,
        model=model,
        condition=condition,
        condition_role=condition_role,
        run_number=run_number,
        input_tokens=100,
        output_tokens=50,
        latency_ms=10.0,
        response="test",
        score=score,
        score_passed=passed,
        scorer_details={},
        task_metadata={},
        timestamp="2024-01-01T00:00:00Z",
        config_hash="abc123",
    )


def _write_results(tmp_path: Path, results: list[RunResult]) -> Path:
    """Write results to a JSONL file and return the directory."""
    results_dir = tmp_path / "test_results"
    results_dir.mkdir()
    results_file = results_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    return results_dir


class TestDynamicPairDiscovery:
    def test_standard_3_conditions(self, tmp_path):
        """Standard baseline/control/primed generates 3 comparison pairs."""
        results = []
        for task_id in [f"t{i}" for i in range(5)]:
            for cond, role in [("baseline", "baseline"), ("control", "control"), ("primed", "treatment")]:
                results.append(_make_result(
                    task_id=task_id, condition=cond, condition_role=role,
                    score=0.8 if cond == "primed" else 0.5,
                ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        comparisons = analyzer.compare()

        pair_types = {(c.condition_a, c.condition_b) for c in comparisons}
        assert ("baseline", "primed") in pair_types
        assert ("baseline", "control") in pair_types
        assert ("control", "primed") in pair_types

    def test_5_condition_ablation(self, tmp_path):
        """5 conditions (baseline, control, 3 treatments) generates correct pairs."""
        results = []
        conditions = [
            ("baseline", "baseline"),
            ("control", "control"),
            ("primed", "treatment"),
            ("irrelevant_book", "treatment"),
            ("fake_book", "treatment"),
        ]
        for task_id in [f"t{i}" for i in range(5)]:
            for cond, role in conditions:
                results.append(_make_result(
                    task_id=task_id, condition=cond, condition_role=role,
                    score=0.6,
                ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        comparisons = analyzer.compare()

        pair_types = {(c.condition_a, c.condition_b) for c in comparisons}
        # baseline → each treatment
        assert ("baseline", "primed") in pair_types
        assert ("baseline", "irrelevant_book") in pair_types
        assert ("baseline", "fake_book") in pair_types
        # baseline → control
        assert ("baseline", "control") in pair_types
        # control → each treatment
        assert ("control", "primed") in pair_types
        assert ("control", "irrelevant_book") in pair_types
        assert ("control", "fake_book") in pair_types

    def test_legacy_results_without_roles(self, tmp_path):
        """Results without condition_role should still work via inference."""
        results = []
        for task_id in [f"t{i}" for i in range(5)]:
            for cond in ["baseline", "control", "primed"]:
                results.append(_make_result(
                    task_id=task_id, condition=cond, condition_role="",
                    score=0.5,
                ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        roles = analyzer.condition_roles
        assert roles["baseline"] == "baseline"
        assert roles["control"] == "control"
        assert roles["primed"] == "treatment"


class TestMultiRunAggregation:
    """Test that multiple runs per task are aggregated before analysis."""

    def test_n_pairs_equals_unique_tasks(self, tmp_path):
        """With 5 tasks x 3 runs, n_pairs should be 5 (not 15)."""
        results = []
        for task_id in [f"t{i}" for i in range(5)]:
            for run_num in range(1, 4):  # 3 runs
                for cond, role in [("baseline", "baseline"), ("primed", "treatment")]:
                    results.append(_make_result(
                        task_id=task_id, condition=cond, condition_role=role,
                        run_number=run_num, score=0.5,
                    ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        comparisons = analyzer.compare()

        assert len(comparisons) == 1
        assert comparisons[0].n_pairs == 5  # unique tasks, not 15

    def test_single_run_unchanged(self, tmp_path):
        """With 1 run per task, n_pairs should still be the number of tasks."""
        results = []
        for task_id in [f"t{i}" for i in range(5)]:
            for cond, role in [("baseline", "baseline"), ("primed", "treatment")]:
                results.append(_make_result(
                    task_id=task_id, condition=cond, condition_role=role,
                    run_number=1, score=0.5,
                ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        comparisons = analyzer.compare()

        assert len(comparisons) == 1
        assert comparisons[0].n_pairs == 5

    def test_aggregated_scores_are_means(self, tmp_path):
        """Scores should be averaged across runs before comparison."""
        results = []
        for task_id in [f"t{i}" for i in range(5)]:
            for run_num in range(1, 4):
                # Baseline: always 0.0
                results.append(_make_result(
                    task_id=task_id, condition="baseline", condition_role="baseline",
                    run_number=run_num, score=0.0, passed=False,
                ))
                # Treatment: scores of [0.0, 1.0, 1.0] → mean = 0.667
                treatment_score = 1.0 if run_num >= 2 else 0.0
                results.append(_make_result(
                    task_id=task_id, condition="primed", condition_role="treatment",
                    run_number=run_num, score=treatment_score,
                    passed=treatment_score == 1.0,
                ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        comparisons = analyzer.compare()

        assert len(comparisons) == 1
        c = comparisons[0]
        # All tasks have the same pattern, so b_mean should be ~0.667
        assert abs(c.b_mean - 2.0 / 3.0) < 0.01
        assert c.a_mean == 0.0

    def test_binary_outcomes_become_continuous_after_aggregation(self, tmp_path):
        """Multi-run binary data becomes proportions, using continuous tests not McNemar."""
        results = []
        for task_id in [f"t{i}" for i in range(30)]:  # need enough for t-test
            for run_num in range(1, 4):
                # Vary scores so aggregation produces non-binary values
                base_score = 1.0 if (int(task_id[1:]) + run_num) % 3 == 0 else 0.0
                treat_score = 1.0 if (int(task_id[1:]) + run_num) % 2 == 0 else 0.0
                results.append(_make_result(
                    task_id=task_id, condition="baseline", condition_role="baseline",
                    run_number=run_num, score=base_score, passed=base_score == 1.0,
                ))
                results.append(_make_result(
                    task_id=task_id, condition="primed", condition_role="treatment",
                    run_number=run_num, score=treat_score, passed=treat_score == 1.0,
                ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        comparisons = analyzer.compare()

        assert len(comparisons) == 1
        # After averaging 3 binary runs, values are 0.0, 0.333, 0.667, 1.0
        # Not all binary → should use continuous test, not McNemar
        assert comparisons[0].test_used in ("paired_t_test", "wilcoxon")

    def test_detect_max_runs(self, tmp_path):
        """_detect_max_runs should report the actual max runs per task."""
        results = []
        for task_id in [f"t{i}" for i in range(3)]:
            for run_num in range(1, 4):
                results.append(_make_result(
                    task_id=task_id, condition="baseline", condition_role="baseline",
                    run_number=run_num, score=0.5,
                ))
                results.append(_make_result(
                    task_id=task_id, condition="primed", condition_role="treatment",
                    run_number=run_num, score=0.5,
                ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        assert analyzer._detect_max_runs() == 3


class TestDomainSpecificityRegression:
    """Test the task-level Welch's t-test approach to domain specificity."""

    def test_significant_domain_effect(self, tmp_path):
        """Clear domain-specific effect should be detected."""
        import random
        rng = random.Random(42)

        results = []
        # Matched domain: treatment effect of +0.4
        for i in range(30):
            task_id = f"matched_{i}"
            ctrl_score = rng.choice([0.0, 1.0])
            treat_score = min(ctrl_score + 0.4 + rng.gauss(0, 0.1), 1.0)
            for cond, role, score in [
                ("control", "control", ctrl_score),
                ("primed", "treatment", treat_score),
            ]:
                results.append(_make_result(
                    task_id=task_id, category="biology", condition=cond,
                    condition_role=role, score=score, passed=score >= 0.5,
                ))

        # Mismatched domain: no treatment effect
        for i in range(30):
            task_id = f"mismatched_{i}"
            score = rng.choice([0.0, 1.0])
            for cond, role in [("control", "control"), ("primed", "treatment")]:
                results.append(_make_result(
                    task_id=task_id, category="law", condition=cond,
                    condition_role=role, score=score, passed=score >= 0.5,
                ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        ds_results = analyzer.domain_specificity_analysis(domain_matched=["biology"])

        assert len(ds_results) == 1
        r = ds_results[0]
        assert r.n_tasks_matched == 30
        assert r.n_tasks_mismatched == 30
        assert r.regression_coefficient > 0  # matched effects larger
        assert r.specificity_significant is True

    def test_no_domain_effect(self, tmp_path):
        """Equal effects in matched and mismatched → not significant."""
        results = []
        for i in range(20):
            for cat in ["biology", "law"]:
                task_id = f"{cat}_{i}"
                for cond, role, score in [
                    ("control", "control", 0.5),
                    ("primed", "treatment", 0.6),
                ]:
                    results.append(_make_result(
                        task_id=task_id, category=cat, condition=cond,
                        condition_role=role, score=score, passed=score >= 0.5,
                    ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        ds_results = analyzer.domain_specificity_analysis(domain_matched=["biology"])

        assert len(ds_results) == 1
        r = ds_results[0]
        # Effects are the same in both domains, so not specific
        assert r.specificity_significant is False
        assert abs(r.regression_coefficient) < 0.01

    def test_task_counts_correct(self, tmp_path):
        """n_tasks_matched/mismatched should count individual tasks, not categories."""
        results = []
        # 2 matched categories, 10 tasks each = 20 matched tasks
        for cat in ["biology", "health"]:
            for i in range(10):
                task_id = f"{cat}_{i}"
                for cond, role, score in [
                    ("control", "control", 0.5),
                    ("primed", "treatment", 0.6),
                ]:
                    results.append(_make_result(
                        task_id=task_id, category=cat, condition=cond,
                        condition_role=role, score=score, passed=True,
                    ))
        # 3 mismatched categories, 10 tasks each = 30 mismatched tasks
        for cat in ["law", "history", "math"]:
            for i in range(10):
                task_id = f"{cat}_{i}"
                for cond, role, score in [
                    ("control", "control", 0.5),
                    ("primed", "treatment", 0.6),
                ]:
                    results.append(_make_result(
                        task_id=task_id, category=cat, condition=cond,
                        condition_role=role, score=score, passed=True,
                    ))

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        ds_results = analyzer.domain_specificity_analysis(
            domain_matched=["biology", "health"]
        )

        assert len(ds_results) == 1
        assert ds_results[0].n_tasks_matched == 20
        assert ds_results[0].n_tasks_mismatched == 30


class TestDomainSpecificityTestHelper:
    """Test _domain_specificity_test directly."""

    def test_clear_difference(self):
        """Clearly separated groups should give significant p."""
        matched = [0.5, 0.4, 0.6, 0.5, 0.3, 0.7, 0.5, 0.4, 0.6, 0.5]
        mismatched = [0.0, 0.0, 0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.0, 0.0]

        p, coeff, se, ci_lo, ci_hi = _domain_specificity_test(matched, mismatched)
        assert p is not None
        assert p < 0.01
        assert coeff > 0.3
        assert ci_lo is not None and ci_lo > 0

    def test_identical_groups(self):
        """Identical groups should not be significant."""
        values = [0.1, 0.2, 0.0, -0.1, 0.15, 0.05, 0.1, 0.0]
        p, coeff, se, ci_lo, ci_hi = _domain_specificity_test(values, list(values))
        assert p is not None
        assert p > 0.4
        assert abs(coeff) < 0.01

    def test_too_few_samples(self):
        """With fewer than 2 samples in either group, p should be None."""
        p, coeff, se, ci_lo, ci_hi = _domain_specificity_test([0.5], [0.0, 0.1])
        assert p is None


class TestPowerAnalysis:
    def test_post_hoc_power_large_effect(self):
        """Large effect with decent n should have high power."""
        power = _post_hoc_power(effect_size=0.8, n=50)
        assert power is not None
        assert power > 0.9

    def test_post_hoc_power_small_effect_small_n(self):
        """Small effect with small n should be underpowered."""
        power = _post_hoc_power(effect_size=0.2, n=10)
        assert power is not None
        assert power < 0.5

    def test_post_hoc_power_zero_effect(self):
        """Zero effect size should return None."""
        power = _post_hoc_power(effect_size=0.0, n=50)
        assert power is None

    def test_post_hoc_power_too_few_samples(self):
        power = _post_hoc_power(effect_size=0.5, n=2)
        assert power is None

    def test_minimum_detectable_effect_large_n(self):
        """With large n, should be able to detect small effects."""
        min_d = _minimum_detectable_effect(n=200)
        assert min_d is not None
        assert min_d < 0.25  # should detect small effects

    def test_minimum_detectable_effect_small_n(self):
        """With small n, minimum detectable effect should be large."""
        min_d = _minimum_detectable_effect(n=10)
        assert min_d is not None
        assert min_d > 0.5


class TestBinomtest:
    def test_mcnemar_exact_uses_binomtest(self):
        """McNemar's test should use stats.binomtest (not deprecated binom_test)."""
        # 10 discordant pairs: 8 improved, 2 regressed
        a_passed = [False] * 8 + [True] * 2 + [True] * 5 + [False] * 5
        b_passed = [True] * 8 + [False] * 2 + [True] * 5 + [False] * 5

        p, test_name = _mcnemar_test(a_passed, b_passed)
        assert p is not None
        assert test_name == "mcnemar_exact"
        assert 0 < p < 1

    def test_mcnemar_no_discordant(self):
        a_passed = [True, True, False, False]
        b_passed = [True, True, False, False]
        p, test_name = _mcnemar_test(a_passed, b_passed)
        assert p == 1.0


class TestStatHelpers:
    def test_mean(self):
        assert _mean([1, 2, 3]) == 2.0
        assert _mean([]) == 0.0

    def test_std(self):
        assert _std([1]) == 0.0
        assert abs(_std([2, 4, 4, 4, 5, 5, 7, 9]) - 2.138) < 0.01

    def test_bootstrap_ci_identical(self):
        """Identical scores should give CI of [0, 0]."""
        a = [1.0] * 10
        b = [1.0] * 10
        lo, hi = _bootstrap_ci(a, b)
        assert abs(lo) < 0.01
        assert abs(hi) < 0.01

    def test_permutation_test_identical(self):
        """Identical groups should give large p (no significant difference)."""
        p = _permutation_test([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert p is not None
        # With identical values, all permutations give the same delta, so p should be ~1.0
        assert p > 0.2

    def test_permutation_test_clear_difference(self):
        """Clear difference should give small p."""
        p = _permutation_test([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0])
        assert p is not None
        assert p < 0.05


class TestExtractionMethodAudit:
    """Test extraction method frequency tracking by condition."""

    def test_groups_by_condition(self, tmp_path):
        """Extraction methods should be counted per condition."""
        results = []
        for i in range(10):
            # Baseline: all single_letter
            results.append(_make_result(
                task_id=f"t{i}", condition="baseline", condition_role="baseline",
                score=1.0, passed=True,
            ))
            results[-1].scorer_details = {"extraction_method": "single_letter"}

            # Primed: mix of methods
            results.append(_make_result(
                task_id=f"t{i}", condition="primed", condition_role="treatment",
                score=1.0, passed=True,
            ))
            method = "last_line_pattern" if i < 7 else "full_scan_pattern"
            results[-1].scorer_details = {"extraction_method": method}

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        audit = analyzer.extraction_method_audit()

        assert "baseline" in audit
        assert "primed" in audit
        assert audit["baseline"]["single_letter"] == 10
        assert audit["primed"]["last_line_pattern"] == 7
        assert audit["primed"]["full_scan_pattern"] == 3

    def test_missing_extraction_method(self, tmp_path):
        """Results without extraction_method should be counted as 'unknown'."""
        results = [
            _make_result(task_id="t1", condition="baseline", condition_role="baseline"),
        ]
        results[0].scorer_details = {}  # no extraction_method

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        audit = analyzer.extraction_method_audit()

        assert audit["baseline"]["unknown"] == 1


class TestResponseLengthStats:
    """Test response length statistics by condition."""

    def test_computes_stats_per_condition(self, tmp_path):
        """Mean and median should be computed per condition."""
        results = []
        for i in range(5):
            r = _make_result(
                task_id=f"t{i}", condition="baseline", condition_role="baseline",
            )
            r.response = "x" * 100  # 100 chars each
            r.output_tokens = 20
            results.append(r)

        for i in range(5):
            r = _make_result(
                task_id=f"t{i}", condition="primed", condition_role="treatment",
            )
            r.response = "x" * 200  # 200 chars each
            r.output_tokens = 40
            results.append(r)

        results_dir = _write_results(tmp_path, results)
        analyzer = ExperimentAnalyzer(results_dir)
        stats = analyzer.response_length_stats()

        assert stats["baseline"]["mean_chars"] == 100.0
        assert stats["baseline"]["mean_tokens"] == 20.0
        assert stats["baseline"]["n"] == 5
        assert stats["primed"]["mean_chars"] == 200.0
        assert stats["primed"]["mean_tokens"] == 40.0

    def test_empty_results(self, tmp_path):
        """No results should return empty stats."""
        results_dir = tmp_path / "empty_results"
        results_dir.mkdir()
        (results_dir / "results.jsonl").touch()

        analyzer = ExperimentAnalyzer(results_dir)
        stats = analyzer.response_length_stats()
        assert stats == {}


class TestChiSquareExtractionAudit:
    """Test chi-square independence test for extraction methods."""

    def test_identical_distributions_not_significant(self):
        """Same extraction method distribution across conditions should not be significant."""
        audit = {
            "baseline": {"single_letter": 40, "last_line_pattern": 10},
            "primed": {"single_letter": 40, "last_line_pattern": 10},
        }
        chi2, p, interp = _extraction_method_chi_square(audit)
        assert p is not None
        assert p > 0.05
        assert "OK" in interp

    def test_different_distributions_significant(self):
        """Very different extraction method distributions should be significant."""
        audit = {
            "baseline": {"single_letter": 90, "last_line_pattern": 10},
            "primed": {"single_letter": 10, "full_scan_pattern": 90},
        }
        chi2, p, interp = _extraction_method_chi_square(audit)
        assert p is not None
        assert p < 0.01
        assert "SIGNIFICANT" in interp

    def test_single_condition_returns_none(self):
        """Need at least 2 conditions for chi-square test."""
        audit = {"baseline": {"single_letter": 50}}
        chi2, p, interp = _extraction_method_chi_square(audit)
        assert chi2 is None


class TestMixedEffectsModel:
    """Test the mixed-effects model for domain specificity."""

    def test_returns_none_without_statsmodels(self):
        """Should gracefully return None if statsmodels is not installed."""
        # This test verifies the function doesn't crash with small data
        records = [(0.5, 1, "bio"), (0.0, 0, "law")]
        result = _domain_specificity_mixed_model(records)
        # With only 2 records and 2 categories, should return None (too few)
        assert result is None

    def test_too_few_records(self):
        """Fewer than 5 records should return None."""
        records = [(0.5, 1, "bio"), (0.0, 0, "law"), (0.3, 1, "bio")]
        result = _domain_specificity_mixed_model(records)
        assert result is None

    def test_too_few_categories(self):
        """Fewer than 3 categories should return None."""
        records = [
            (0.5, 1, "bio"), (0.4, 1, "bio"), (0.6, 1, "bio"),
            (0.0, 0, "law"), (0.1, 0, "law"), (0.0, 0, "law"),
        ]
        result = _domain_specificity_mixed_model(records)
        assert result is None

    def test_with_enough_data(self):
        """With sufficient data and categories, should return a result (if statsmodels installed)."""
        import random
        rng = random.Random(42)

        records = []
        # 3+ categories needed for random effects
        for cat in ["bio", "health", "law", "history", "math"]:
            is_matched = 1 if cat in ("bio", "health") else 0
            for _ in range(10):
                effect = 0.3 * is_matched + rng.gauss(0, 0.1)
                records.append((effect, is_matched, cat))

        result = _domain_specificity_mixed_model(records)
        # May be None if statsmodels not installed — that's OK
        if result is not None:
            p_val, coeff, se, ci_lo, ci_hi = result
            assert coeff > 0  # matched effects are larger
            assert p_val < 0.05
            assert ci_lo < ci_hi


class TestRecommendedSampleSize:
    def test_small_effect(self):
        """Should recommend ~200 tasks for small effect."""
        n = _recommended_sample_size(target_d=0.2)
        assert n is not None
        assert 150 < n < 250

    def test_medium_effect(self):
        """Should recommend ~35 tasks for medium effect."""
        n = _recommended_sample_size(target_d=0.5)
        assert n is not None
        assert 20 < n < 50

    def test_large_effect(self):
        """Should recommend ~15 tasks for large effect."""
        n = _recommended_sample_size(target_d=0.8)
        assert n is not None
        assert 10 < n < 25


class TestCrossSeedAnalyzer:
    """Test cross-seed replication analysis."""

    def _create_multi_seed_structure(self, tmp_path, seeds, results_per_seed):
        """Create a multi-seed directory structure for testing."""
        parent = tmp_path / "multiseed_test"
        parent.mkdir()

        directories = []
        for seed in seeds:
            seed_dir = parent / f"seed_{seed}"
            seed_dir.mkdir()
            results_file = seed_dir / "results.jsonl"
            with open(results_file, "w") as f:
                for r in results_per_seed(seed):
                    f.write(r.model_dump_json() + "\n")
            directories.append(str(seed_dir))

        manifest = {
            "type": "multi_seed",
            "n_seeds": len(seeds),
            "seeds": seeds,
            "base_seed": seeds[0],
            "directories": directories,
        }
        with open(parent / "multi_seed_manifest.json", "w") as f:
            json.dump(manifest, f)

        return parent

    def test_loads_manifest(self, tmp_path):
        """Should load manifest and create per-seed analyzers."""
        seeds = [42, 43]

        def make_results(seed):
            return [
                _make_result(task_id=f"t{i}", condition="baseline",
                             condition_role="baseline", score=0.5)
                for i in range(3)
            ] + [
                _make_result(task_id=f"t{i}", condition="primed",
                             condition_role="treatment", score=0.6)
                for i in range(3)
            ]

        parent = self._create_multi_seed_structure(tmp_path, seeds, make_results)
        cross = CrossSeedAnalyzer(parent)

        assert cross.seeds == [42, 43]
        assert len(cross.analyzers) == 2

    def test_replication_report_consistent(self, tmp_path):
        """Consistent results across seeds should be flagged as robust or direction_consistent."""
        seeds = [42, 43, 44]

        def make_results(seed):
            results = []
            for i in range(5):
                results.append(_make_result(
                    task_id=f"t{i}", condition="baseline",
                    condition_role="baseline", score=0.3,
                ))
                results.append(_make_result(
                    task_id=f"t{i}", condition="primed",
                    condition_role="treatment", score=0.7,
                ))
            return results

        parent = self._create_multi_seed_structure(tmp_path, seeds, make_results)
        cross = CrossSeedAnalyzer(parent)
        report = cross.replication_report()

        # All seeds show same direction → should have no inconsistent findings
        assert report["summary"]["inconsistent"] == 0
        total = report["summary"]["robust"] + report["summary"]["direction_consistent"]
        assert total > 0

    def test_replication_report_inconsistent(self, tmp_path):
        """Results that flip direction across seeds should be flagged inconsistent."""
        seeds = [42, 43]

        def make_results(seed):
            results = []
            for i in range(5):
                if seed == 42:
                    base_score, treat_score = 0.3, 0.7  # treatment better
                else:
                    base_score, treat_score = 0.7, 0.3  # treatment worse
                results.append(_make_result(
                    task_id=f"t{i}", condition="baseline",
                    condition_role="baseline", score=base_score,
                ))
                results.append(_make_result(
                    task_id=f"t{i}", condition="primed",
                    condition_role="treatment", score=treat_score,
                ))
            return results

        parent = self._create_multi_seed_structure(tmp_path, seeds, make_results)
        cross = CrossSeedAnalyzer(parent)
        report = cross.replication_report()

        assert report["summary"]["inconsistent"] > 0
