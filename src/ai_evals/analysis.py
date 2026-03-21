"""Statistical analysis for comparing baseline vs control vs primed conditions.

Implements:
- Cohen's d effect size
- Paired t-test and Wilcoxon signed-rank for continuous scores
- McNemar's test for binary (pass/fail) outcomes
- Holm-Bonferroni multiple comparison correction
- Bootstrap confidence intervals
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ai_evals.config import ExperimentConfig
from ai_evals.results import ResultStore, RunResult

console = Console()


@dataclass
class DomainSpecificityResult:
    """Result of comparing priming effect sizes between matched and mismatched domains."""

    model: str
    matched_categories: list[str]
    mismatched_categories: list[str]
    matched_mean_effect: float  # mean Cohen's d across matched categories
    mismatched_mean_effect: float  # mean Cohen's d across mismatched categories
    specificity_delta: float  # matched - mismatched (positive = book helps more in matched domain)
    specificity_p_value: float | None  # permutation test
    specificity_significant: bool
    matched_mean_score_diff: float  # raw score improvement in matched
    mismatched_mean_score_diff: float  # raw score improvement in mismatched


@dataclass
class PairwiseComparison:
    model: str
    category: str
    condition_a: str  # e.g. "baseline"
    condition_b: str  # e.g. "primed"
    a_mean: float
    a_std: float
    b_mean: float
    b_std: float
    mean_difference: float
    cohens_d: float
    ci_lower: float  # 95% bootstrap CI for mean difference
    ci_upper: float
    p_value: float | None
    p_value_corrected: float | None  # after Holm-Bonferroni
    test_used: str
    n_pairs: int
    significant: bool  # corrected p < 0.05
    a_pass_rate: float
    b_pass_rate: float


class ExperimentAnalyzer:
    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)
        self.store = ResultStore(self.results_dir)
        self._results: list[RunResult] | None = None

    @property
    def results(self) -> list[RunResult]:
        if self._results is None:
            self._results = self.store.load_all()
        return self._results

    def compare(self) -> list[PairwiseComparison]:
        """Run all pairwise comparisons: baseline→primed, baseline→control, control→primed.

        Applies Holm-Bonferroni correction across all comparisons.
        """
        # Group scores by (model, category, condition) -> list of (task_id, run_num, score, passed)
        grouped: dict[tuple[str, str, str], list[tuple[str, int, float, bool]]] = defaultdict(list)
        for r in self.results:
            grouped[(r.model, r.task_category, r.condition)].append(
                (r.task_id, r.run_number, r.score, r.score_passed)
            )

        # Determine which conditions exist
        conditions_present = sorted({k[2] for k in grouped.keys()})

        # Define comparison pairs (order matters: a is the "worse" expected condition)
        pairs_to_compare = []
        if "baseline" in conditions_present and "primed" in conditions_present:
            pairs_to_compare.append(("baseline", "primed"))
        if "baseline" in conditions_present and "control" in conditions_present:
            pairs_to_compare.append(("baseline", "control"))
        if "control" in conditions_present and "primed" in conditions_present:
            pairs_to_compare.append(("control", "primed"))

        # Get unique (model, category) groups
        model_cats = sorted({(k[0], k[1]) for k in grouped.keys()})

        raw_comparisons: list[PairwiseComparison] = []

        for model, category in model_cats:
            for cond_a, cond_b in pairs_to_compare:
                a_data = grouped.get((model, category, cond_a), [])
                b_data = grouped.get((model, category, cond_b), [])

                if not a_data or not b_data:
                    continue

                # Build paired data: match on (task_id, run_number)
                a_by_key = {(tid, rn): (score, passed) for tid, rn, score, passed in a_data}
                b_by_key = {(tid, rn): (score, passed) for tid, rn, score, passed in b_data}
                common_keys = sorted(set(a_by_key.keys()) & set(b_by_key.keys()))

                if len(common_keys) < 3:
                    continue

                a_scores = [a_by_key[k][0] for k in common_keys]
                b_scores = [b_by_key[k][0] for k in common_keys]
                a_passed = [a_by_key[k][1] for k in common_keys]
                b_passed = [b_by_key[k][1] for k in common_keys]

                a_mean = _mean(a_scores)
                b_mean = _mean(b_scores)
                a_std = _std(a_scores)
                b_std = _std(b_scores)
                diff = b_mean - a_mean

                pooled_std = math.sqrt((a_std**2 + b_std**2) / 2) if (a_std + b_std) > 0 else 0
                cohens_d = diff / pooled_std if pooled_std > 0 else 0.0

                # Bootstrap CI for mean difference
                ci_lower, ci_upper = _bootstrap_ci(a_scores, b_scores)

                # Choose appropriate test
                # Use McNemar's for binary data, parametric/nonparametric for continuous
                all_binary = all(s in (0.0, 1.0) for s in a_scores + b_scores)
                if all_binary:
                    p_value, test_used = _mcnemar_test(a_passed, b_passed)
                else:
                    p_value, test_used = _run_significance_test(a_scores, b_scores)

                a_pass_rate = sum(a_passed) / len(a_passed)
                b_pass_rate = sum(b_passed) / len(b_passed)

                raw_comparisons.append(PairwiseComparison(
                    model=model,
                    category=category,
                    condition_a=cond_a,
                    condition_b=cond_b,
                    a_mean=a_mean,
                    a_std=a_std,
                    b_mean=b_mean,
                    b_std=b_std,
                    mean_difference=diff,
                    cohens_d=cohens_d,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    p_value=p_value,
                    p_value_corrected=None,  # filled below
                    test_used=test_used,
                    n_pairs=len(common_keys),
                    significant=False,  # filled below
                    a_pass_rate=a_pass_rate,
                    b_pass_rate=b_pass_rate,
                ))

        # Apply Holm-Bonferroni correction
        _apply_holm_bonferroni(raw_comparisons)

        return raw_comparisons

    def summary(self) -> dict:
        comparisons = self.compare()
        total = len(self.results)
        conditions = defaultdict(int)
        for r in self.results:
            conditions[r.condition] += 1

        return {
            "total_results": total,
            "runs_by_condition": dict(conditions),
            "comparisons": len(comparisons),
            "significant_improvements": sum(
                1 for c in comparisons if c.significant and c.mean_difference > 0
            ),
            "significant_regressions": sum(
                1 for c in comparisons if c.significant and c.mean_difference < 0
            ),
            "no_significant_change": sum(
                1 for c in comparisons if not c.significant
            ),
        }

    def print_report(self) -> None:
        """Print the full analysis report."""
        comparisons = self.compare()
        summary = self.summary()

        console.print(f"\n[bold]Analysis Report[/bold]")
        console.print(f"Results directory: {self.results_dir}")
        console.print(f"Total runs: {summary['total_results']}")
        for cond, count in summary["runs_by_condition"].items():
            console.print(f"  {cond}: {count}")

        if not comparisons:
            console.print("[yellow]No comparisons available.[/yellow]")
            return

        # Group comparisons by pair type
        for pair_label in [("baseline", "control"), ("baseline", "primed"), ("control", "primed")]:
            pair_comps = [c for c in comparisons
                         if c.condition_a == pair_label[0] and c.condition_b == pair_label[1]]
            if not pair_comps:
                continue

            table = Table(title=f"\n{pair_label[0].title()} vs {pair_label[1].title()}")
            table.add_column("Model")
            table.add_column("Category")
            table.add_column(pair_label[0].title(), justify="right")
            table.add_column(pair_label[1].title(), justify="right")
            table.add_column("Diff", justify="right")
            table.add_column("95% CI", justify="right")
            table.add_column("Cohen's d", justify="right")
            table.add_column("p (corrected)", justify="right")
            table.add_column("N", justify="right")

            for c in pair_comps:
                diff_color = "green" if c.mean_difference > 0 else "red" if c.mean_difference < 0 else ""
                p_str = f"{c.p_value_corrected:.4f}" if c.p_value_corrected is not None else "N/A"
                sig = " *" if c.significant else ""

                d_abs = abs(c.cohens_d)
                if d_abs >= 0.8:
                    d_label = "L"
                elif d_abs >= 0.5:
                    d_label = "M"
                elif d_abs >= 0.2:
                    d_label = "S"
                else:
                    d_label = "~0"

                diff_str = f"{c.mean_difference:+.3f}"
                if diff_color:
                    diff_str = f"[{diff_color}]{diff_str}[/{diff_color}]"

                table.add_row(
                    c.model,
                    c.category,
                    f"{c.a_mean:.3f} ({c.a_pass_rate:.0%})",
                    f"{c.b_mean:.3f} ({c.b_pass_rate:.0%})",
                    diff_str,
                    f"[{c.ci_lower:+.3f}, {c.ci_upper:+.3f}]",
                    f"{c.cohens_d:+.2f} ({d_label})",
                    f"{p_str}{sig}",
                    str(c.n_pairs),
                )

            console.print(table)

        # Key comparison: is primed > control? (isolates book effect)
        book_effects = [c for c in comparisons
                       if c.condition_a == "control" and c.condition_b == "primed"]
        if book_effects:
            console.print("\n[bold]Key Question: Does the book reference add value beyond just 'think carefully'?[/bold]")
            for c in book_effects:
                direction = "YES" if c.significant and c.mean_difference > 0 else "NO" if c.significant else "INCONCLUSIVE"
                console.print(
                    f"  {c.model} / {c.category}: [bold]{direction}[/bold] "
                    f"(diff={c.mean_difference:+.3f}, d={c.cohens_d:+.2f}, "
                    f"p={c.p_value_corrected:.4f})" if c.p_value_corrected is not None else
                    f"  {c.model} / {c.category}: [bold]{direction}[/bold] "
                    f"(diff={c.mean_difference:+.3f}, d={c.cohens_d:+.2f}, p=N/A)"
                )

        console.print(f"\n[dim]* = significant at p<0.05 after Holm-Bonferroni correction[/dim]")
        console.print(f"[dim]Scores show mean (pass rate). CI = bootstrap 95% confidence interval.[/dim]")
        console.print(f"[dim]Cohen's d: ~0=negligible, S=small(0.2), M=medium(0.5), L=large(0.8)[/dim]")

    def to_markdown(self) -> str:
        comparisons = self.compare()
        lines = [
            "| Model | Category | Comparison | A Mean | B Mean | Diff | Cohen's d | p (corrected) | Sig | N |",
            "|-------|----------|------------|--------|--------|------|-----------|---------------|-----|---|",
        ]
        for c in comparisons:
            p_str = f"{c.p_value_corrected:.4f}" if c.p_value_corrected is not None else "N/A"
            sig = "yes" if c.significant else "no"
            lines.append(
                f"| {c.model} | {c.category} | "
                f"{c.condition_a}→{c.condition_b} | "
                f"{c.a_mean:.3f} | {c.b_mean:.3f} | "
                f"{c.mean_difference:+.3f} | {c.cohens_d:+.2f} | "
                f"{p_str} | {sig} | {c.n_pairs} |"
            )
        return "\n".join(lines)

    def domain_specificity_analysis(
        self, domain_matched: list[str]
    ) -> list[DomainSpecificityResult]:
        """Analyze whether the book priming effect is domain-specific.

        Compares the control→primed effect size in domain-matched categories
        vs domain-mismatched categories. If the effect is larger in matched
        categories, the book reference is doing something domain-specific
        (not just acting as a generic "try harder" prompt).
        """
        comparisons = self.compare()

        # Filter to control→primed only (this isolates book effect)
        book_effects = [
            c for c in comparisons
            if c.condition_a == "control" and c.condition_b == "primed"
        ]

        if not book_effects:
            return []

        # Group by model
        models = sorted(set(c.model for c in book_effects))
        results = []

        for model in models:
            model_comps = [c for c in book_effects if c.model == model]
            matched_comps = [c for c in model_comps if c.category in domain_matched]
            mismatched_comps = [c for c in model_comps if c.category not in domain_matched]

            if not matched_comps or not mismatched_comps:
                continue

            matched_effects = [c.cohens_d for c in matched_comps]
            mismatched_effects = [c.cohens_d for c in mismatched_comps]
            matched_diffs = [c.mean_difference for c in matched_comps]
            mismatched_diffs = [c.mean_difference for c in mismatched_comps]

            matched_mean = _mean(matched_effects)
            mismatched_mean = _mean(mismatched_effects)
            delta = matched_mean - mismatched_mean

            # Permutation test: is the matched mean effect significantly larger?
            p_value = _permutation_test(matched_effects, mismatched_effects)

            results.append(DomainSpecificityResult(
                model=model,
                matched_categories=sorted(set(c.category for c in matched_comps)),
                mismatched_categories=sorted(set(c.category for c in mismatched_comps)),
                matched_mean_effect=matched_mean,
                mismatched_mean_effect=mismatched_mean,
                specificity_delta=delta,
                specificity_p_value=p_value,
                specificity_significant=p_value is not None and p_value < 0.05,
                matched_mean_score_diff=_mean(matched_diffs),
                mismatched_mean_score_diff=_mean(mismatched_diffs),
            ))

        return results

    def print_domain_report(self, domain_matched: list[str]) -> None:
        """Print domain specificity analysis."""
        results = self.domain_specificity_analysis(domain_matched)

        if not results:
            console.print("[yellow]No domain specificity data available.[/yellow]")
            return

        console.print("\n[bold]Domain Specificity Analysis[/bold]")
        console.print(f"  Matched domains: {', '.join(domain_matched)}")
        console.print(f"  Question: Is the book priming effect larger in matched domains?\n")

        table = Table(title="Control → Primed Effect by Domain Match")
        table.add_column("Model")
        table.add_column("Matched Effect (d)", justify="right")
        table.add_column("Mismatched Effect (d)", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Score Diff (matched)", justify="right")
        table.add_column("Score Diff (mismatched)", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Specific?", justify="center")

        for r in results:
            delta_color = "green" if r.specificity_delta > 0 else "red"
            p_str = f"{r.specificity_p_value:.4f}" if r.specificity_p_value is not None else "N/A"

            table.add_row(
                r.model,
                f"{r.matched_mean_effect:+.3f}",
                f"{r.mismatched_mean_effect:+.3f}",
                f"[{delta_color}]{r.specificity_delta:+.3f}[/{delta_color}]",
                f"{r.matched_mean_score_diff:+.3f}",
                f"{r.mismatched_mean_score_diff:+.3f}",
                p_str,
                "[green]YES[/green]" if r.specificity_significant else "NO",
            )

        console.print(table)

        for r in results:
            if r.specificity_significant and r.specificity_delta > 0:
                console.print(
                    f"\n  [bold green]{r.model}:[/bold green] Book priming IS domain-specific. "
                    f"The effect is {r.specificity_delta:+.3f}d larger in matched domains."
                )
            elif r.specificity_delta > 0:
                console.print(
                    f"\n  [bold yellow]{r.model}:[/bold yellow] Trend toward domain specificity "
                    f"(delta={r.specificity_delta:+.3f}d) but not statistically significant."
                )
            else:
                console.print(
                    f"\n  [bold]{r.model}:[/bold] No evidence of domain-specific effect "
                    f"(delta={r.specificity_delta:+.3f}d)."
                )

    def auto_print_report(self) -> None:
        """Print full report, including domain analysis if config is available."""
        self.print_report()

        # Try to load experiment config for domain analysis
        config_path = self.results_dir / "experiment_config.yaml"
        if config_path.exists():
            try:
                config = ExperimentConfig.from_yaml(config_path)
                if config.benchmark and config.benchmark.domain_matched:
                    self.print_domain_report(config.benchmark.domain_matched)
            except Exception:
                pass


# --- Statistical helpers ---

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _run_significance_test(a: list[float], b: list[float]) -> tuple[float | None, str]:
    """Run appropriate significance test for continuous paired data."""
    n = len(a)
    if n < 3:
        return None, "too_few_samples"

    try:
        from scipy import stats

        if n >= 20:
            stat, p = stats.ttest_rel(b, a)
            return p, "paired_t_test"
        else:
            diffs = [bi - ai for ai, bi in zip(a, b)]
            if all(d == 0 for d in diffs):
                return 1.0, "wilcoxon (all_equal)"
            stat, p = stats.wilcoxon(b, a)
            return p, "wilcoxon"
    except ImportError:
        return None, "scipy_not_installed"
    except Exception:
        return None, "test_failed"


def _mcnemar_test(a_passed: list[bool], b_passed: list[bool]) -> tuple[float | None, str]:
    """McNemar's test for paired binary outcomes."""
    n = len(a_passed)
    if n < 3:
        return None, "too_few_samples"

    try:
        # Count discordant pairs
        # b improved (a=fail, b=pass) vs b regressed (a=pass, b=fail)
        b_improved = sum(1 for a, b in zip(a_passed, b_passed) if not a and b)
        b_regressed = sum(1 for a, b in zip(a_passed, b_passed) if a and not b)

        if b_improved + b_regressed == 0:
            return 1.0, "mcnemar (no_discordant)"

        from scipy import stats

        # Use exact binomial test for small counts, chi-square for large
        if b_improved + b_regressed < 25:
            p = stats.binom_test(b_improved, b_improved + b_regressed, 0.5)
            return p, "mcnemar_exact"
        else:
            # McNemar chi-square with continuity correction
            chi2 = (abs(b_improved - b_regressed) - 1) ** 2 / (b_improved + b_regressed)
            p = 1 - stats.chi2.cdf(chi2, df=1)
            return p, "mcnemar_chi2"
    except ImportError:
        return None, "scipy_not_installed"
    except Exception:
        return None, "test_failed"


def _bootstrap_ci(
    a: list[float], b: list[float], n_bootstrap: int = 10000, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for mean difference (b - a)."""
    n = len(a)
    if n < 3:
        diff = _mean(b) - _mean(a)
        return diff, diff

    rng = random.Random(42)
    diffs = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        a_sample = [a[i] for i in indices]
        b_sample = [b[i] for i in indices]
        diffs.append(_mean(b_sample) - _mean(a_sample))

    diffs.sort()
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    return diffs[lower_idx], diffs[upper_idx]


def _apply_holm_bonferroni(comparisons: list[PairwiseComparison]) -> None:
    """Apply Holm-Bonferroni correction in-place."""
    # Get comparisons with valid p-values
    valid = [(i, c) for i, c in enumerate(comparisons) if c.p_value is not None]
    if not valid:
        return

    m = len(valid)

    # Sort by raw p-value ascending
    valid.sort(key=lambda x: x[1].p_value)

    for rank, (idx, comp) in enumerate(valid):
        corrected = comp.p_value * (m - rank)
        corrected = min(corrected, 1.0)
        comparisons[idx].p_value_corrected = corrected
        comparisons[idx].significant = corrected < 0.05


def _permutation_test(
    matched: list[float], mismatched: list[float], n_permutations: int = 10000
) -> float | None:
    """Permutation test: is the mean of `matched` significantly greater than `mismatched`?

    Tests whether domain-matched effect sizes are larger than domain-mismatched
    effect sizes by randomly reassigning group labels.
    """
    if len(matched) < 1 or len(mismatched) < 1:
        return None

    observed_delta = _mean(matched) - _mean(mismatched)
    combined = matched + mismatched
    n_matched = len(matched)

    rng = random.Random(42)
    count_ge = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_matched = combined[:n_matched]
        perm_mismatched = combined[n_matched:]
        perm_delta = _mean(perm_matched) - _mean(perm_mismatched)
        if perm_delta >= observed_delta:
            count_ge += 1

    return count_ge / n_permutations
