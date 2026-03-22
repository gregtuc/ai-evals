"""Statistical analysis for comparing experimental conditions.

Supports arbitrary conditions with role-based automatic pair discovery:
- baseline vs treatment pairs
- baseline vs control pairs
- control vs treatment pairs (isolates treatment effect)

Implements:
- Per-task aggregation across runs (task is the unit of analysis)
- Cohen's d effect size with post-hoc power analysis
- Paired t-test and Wilcoxon signed-rank for continuous scores
- McNemar's test for binary (pass/fail) outcomes
- Holm-Bonferroni multiple comparison correction
- Bootstrap confidence intervals
- Mixed-effects model for domain specificity (with Welch's t-test fallback)
- Cross-seed replication analysis
"""

from __future__ import annotations

import json
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
    """Result of comparing treatment effect sizes between matched and mismatched domains."""

    model: str
    treatment_condition: str  # which treatment condition this is for
    matched_categories: list[str]
    mismatched_categories: list[str]
    matched_mean_effect: float  # mean Cohen's d across matched categories (descriptive)
    mismatched_mean_effect: float  # mean Cohen's d across mismatched categories (descriptive)
    specificity_delta: float  # task-level: mean(matched) - mean(mismatched) effects
    specificity_p_value: float | None  # Welch's t-test (one-sided)
    specificity_significant: bool
    matched_mean_score_diff: float  # raw score improvement in matched (category-level)
    mismatched_mean_score_diff: float  # raw score improvement in mismatched (category-level)
    n_tasks_matched: int = 0  # number of individual tasks in matched domains
    n_tasks_mismatched: int = 0  # number of individual tasks in mismatched domains
    regression_coefficient: float = 0.0  # mean(matched_effects) - mean(mismatched_effects)
    regression_se: float | None = None
    regression_ci_lower: float | None = None
    regression_ci_upper: float | None = None
    # Mixed-effects model results (accounts for within-category clustering)
    mixed_model_used: bool = False
    mixed_model_coefficient: float | None = None
    mixed_model_p_value: float | None = None
    mixed_model_se: float | None = None
    mixed_model_ci_lower: float | None = None
    mixed_model_ci_upper: float | None = None


@dataclass
class PairwiseComparison:
    model: str
    category: str
    condition_a: str  # e.g. "baseline"
    condition_b: str  # e.g. "primed"
    a_role: str  # e.g. "baseline"
    b_role: str  # e.g. "treatment"
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
    n_pairs: int  # number of unique tasks (unit of analysis)
    significant: bool  # corrected p < 0.05
    a_pass_rate: float
    b_pass_rate: float
    power: float | None = None  # post-hoc achieved power


class ExperimentAnalyzer:
    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)
        self.store = ResultStore(self.results_dir)
        self._results: list[RunResult] | None = None
        self._condition_roles: dict[str, str] | None = None

    @property
    def results(self) -> list[RunResult]:
        if self._results is None:
            self._results = self.store.load_all()
        return self._results

    @property
    def condition_roles(self) -> dict[str, str]:
        """Map condition name -> role, from results or saved config."""
        if self._condition_roles is None:
            self._condition_roles = self._discover_roles()
        return self._condition_roles

    def _discover_roles(self) -> dict[str, str]:
        """Discover condition roles from results data or saved config."""
        # First try: get roles from results themselves
        roles: dict[str, str] = {}
        for r in self.results:
            if r.condition_role:
                roles[r.condition] = r.condition_role

        if roles:
            return roles

        # Fallback: load from saved experiment config
        config_path = self.results_dir / "experiment_config.yaml"
        if config_path.exists():
            try:
                config = ExperimentConfig.from_yaml(config_path)
                return {name: cond.role for name, cond in config.conditions.items()}
            except Exception:
                pass

        # Last resort: infer from legacy condition names
        conditions = {r.condition for r in self.results}
        inferred = {}
        for c in conditions:
            if c == "baseline":
                inferred[c] = "baseline"
            elif c in ("control",):
                inferred[c] = "control"
            else:
                inferred[c] = "treatment"
        return inferred

    def _build_comparison_pairs(self) -> list[tuple[str, str]]:
        """Build comparison pairs based on condition roles.

        Generates:
        - (baseline, treatment) pairs
        - (baseline, control) pairs
        - (control, treatment) pairs — isolates treatment effect
        """
        roles = self.condition_roles
        conditions_present = {r.condition for r in self.results}

        baselines = sorted(c for c in conditions_present if roles.get(c) == "baseline")
        controls = sorted(c for c in conditions_present if roles.get(c) == "control")
        treatments = sorted(c for c in conditions_present if roles.get(c) == "treatment")

        pairs = []
        for b in baselines:
            for t in treatments:
                pairs.append((b, t))
            for c in controls:
                pairs.append((b, c))
        for c in controls:
            for t in treatments:
                pairs.append((c, t))

        return pairs

    def _aggregate_per_task(
        self,
    ) -> dict[tuple[str, str, str], dict[str, tuple[float, float]]]:
        """Aggregate scores per task across runs, making the task the unit of analysis.

        Multiple runs of the same task at non-zero temperature are not independent
        observations — they're the same question with stochastic variation. Averaging
        per-task prevents pseudo-replication from inflating statistical significance.

        Returns:
            Dict mapping (model, category, condition) -> {task_id: (mean_score, pass_proportion)}
        """
        # Collect all (score, passed) per (model, task_id, condition)
        raw: dict[tuple[str, str, str], list[tuple[float, bool]]] = defaultdict(list)
        task_categories: dict[str, str] = {}

        for r in self.results:
            raw[(r.model, r.task_id, r.condition)].append((r.score, r.score_passed))
            task_categories[r.task_id] = r.task_category

        # Average per task, then re-group by (model, category, condition)
        grouped: dict[tuple[str, str, str], dict[str, tuple[float, float]]] = defaultdict(dict)

        for (model, task_id, condition), scores_passed in raw.items():
            mean_score = _mean([s for s, _ in scores_passed])
            pass_prop = _mean([float(p) for _, p in scores_passed])
            category = task_categories[task_id]
            grouped[(model, category, condition)][task_id] = (mean_score, pass_prop)

        return grouped

    def _detect_max_runs(self) -> int:
        """Detect the maximum number of runs per task in the results."""
        runs_per: dict[tuple[str, str, str], int] = defaultdict(int)
        for r in self.results:
            runs_per[(r.model, r.task_id, r.condition)] += 1
        return max(runs_per.values()) if runs_per else 1

    def compare(self) -> list[PairwiseComparison]:
        """Run all pairwise comparisons based on condition roles.

        Scores are averaged per-task across runs before comparison, so the
        unit of analysis is the task (not individual runs). This prevents
        pseudo-replication from inflating significance when runs_per_task > 1.

        Applies Holm-Bonferroni correction across all comparisons.
        """
        # Aggregate scores per task across runs
        aggregated = self._aggregate_per_task()

        pairs_to_compare = self._build_comparison_pairs()
        roles = self.condition_roles

        # Get unique (model, category) groups
        model_cats = sorted({(k[0], k[1]) for k in aggregated.keys()})

        raw_comparisons: list[PairwiseComparison] = []

        for model, category in model_cats:
            for cond_a, cond_b in pairs_to_compare:
                a_tasks = aggregated.get((model, category, cond_a), {})
                b_tasks = aggregated.get((model, category, cond_b), {})

                if not a_tasks or not b_tasks:
                    continue

                # Match on task_id (runs already aggregated per task)
                common_tasks = sorted(set(a_tasks.keys()) & set(b_tasks.keys()))

                if len(common_tasks) < 3:
                    continue

                a_scores = [a_tasks[t][0] for t in common_tasks]
                b_scores = [b_tasks[t][0] for t in common_tasks]
                a_pass_props = [a_tasks[t][1] for t in common_tasks]
                b_pass_props = [b_tasks[t][1] for t in common_tasks]

                a_mean_val = _mean(a_scores)
                b_mean_val = _mean(b_scores)
                a_std_val = _std(a_scores)
                b_std_val = _std(b_scores)
                diff = b_mean_val - a_mean_val

                pooled_std = (
                    math.sqrt((a_std_val**2 + b_std_val**2) / 2)
                    if (a_std_val + b_std_val) > 0
                    else 0
                )
                cohens_d = diff / pooled_std if pooled_std > 0 else 0.0

                # Bootstrap CI for mean difference
                ci_lower, ci_upper = _bootstrap_ci(a_scores, b_scores)

                # Choose appropriate test.
                # After per-task aggregation with multiple runs, binary scores become
                # proportions (e.g. 0.333, 0.667), routing to continuous tests.
                # McNemar's is only used when all values are strictly 0 or 1
                # (single-run binary data).
                all_binary = all(s in (0.0, 1.0) for s in a_scores + b_scores)
                if all_binary:
                    a_passed = [s == 1.0 for s in a_scores]
                    b_passed = [s == 1.0 for s in b_scores]
                    p_value, test_used = _mcnemar_test(a_passed, b_passed)
                else:
                    p_value, test_used = _run_significance_test(a_scores, b_scores)

                a_pass_rate = _mean(a_pass_props)
                b_pass_rate = _mean(b_pass_props)

                # Post-hoc power analysis (n = unique tasks, the true sample size)
                power = _post_hoc_power(cohens_d, len(common_tasks))

                raw_comparisons.append(PairwiseComparison(
                    model=model,
                    category=category,
                    condition_a=cond_a,
                    condition_b=cond_b,
                    a_role=roles.get(cond_a, "unknown"),
                    b_role=roles.get(cond_b, "unknown"),
                    a_mean=a_mean_val,
                    a_std=a_std_val,
                    b_mean=b_mean_val,
                    b_std=b_std_val,
                    mean_difference=diff,
                    cohens_d=cohens_d,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    p_value=p_value,
                    p_value_corrected=None,  # filled below
                    test_used=test_used,
                    n_pairs=len(common_tasks),
                    significant=False,  # filled below
                    a_pass_rate=a_pass_rate,
                    b_pass_rate=b_pass_rate,
                    power=power,
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
        """Print the full analysis report with dynamic condition pair discovery."""
        comparisons = self.compare()
        summary = self.summary()

        console.print(f"\n[bold]Analysis Report[/bold]")
        console.print(f"Results directory: {self.results_dir}")
        console.print(f"Total runs: {summary['total_results']}")
        for cond, count in sorted(summary["runs_by_condition"].items()):
            role = self.condition_roles.get(cond, "?")
            console.print(f"  {cond} ({role}): {count}")

        # Detect and report multi-run aggregation
        max_runs = self._detect_max_runs()
        if max_runs > 1:
            console.print(
                f"\n[dim]Note: Scores averaged across {max_runs} runs per task. "
                f"The task is the unit of analysis (N = unique tasks, not runs). "
                f"Temperature-based variance does not constitute independence.[/dim]"
            )

        if not comparisons:
            console.print("[yellow]No comparisons available.[/yellow]")
            return

        # Discover unique pair types dynamically
        pair_types = sorted(set((c.condition_a, c.condition_b) for c in comparisons))

        for cond_a, cond_b in pair_types:
            pair_comps = [c for c in comparisons
                         if c.condition_a == cond_a and c.condition_b == cond_b]
            if not pair_comps:
                continue

            role_a = self.condition_roles.get(cond_a, "?")
            role_b = self.condition_roles.get(cond_b, "?")
            table = Table(title=f"\n{cond_a} ({role_a}) vs {cond_b} ({role_b})")
            table.add_column("Model")
            table.add_column("Category")
            table.add_column(cond_a, justify="right")
            table.add_column(cond_b, justify="right")
            table.add_column("Diff", justify="right")
            table.add_column("95% CI", justify="right")
            table.add_column("Cohen's d", justify="right")
            table.add_column("p (corr)", justify="right")
            table.add_column("Power", justify="right")
            table.add_column("N (tasks)", justify="right")

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

                power_str = f"{c.power:.2f}" if c.power is not None else "N/A"
                if c.power is not None and c.power < 0.8:
                    power_str = f"[yellow]{power_str}[/yellow]"

                table.add_row(
                    c.model,
                    c.category,
                    f"{c.a_mean:.3f} ({c.a_pass_rate:.0%})",
                    f"{c.b_mean:.3f} ({c.b_pass_rate:.0%})",
                    diff_str,
                    f"[{c.ci_lower:+.3f}, {c.ci_upper:+.3f}]",
                    f"{c.cohens_d:+.2f} ({d_label})",
                    f"{p_str}{sig}",
                    power_str,
                    str(c.n_pairs),
                )

            console.print(table)

        # Key question: for each (control, treatment) pair, does the treatment add value?
        roles = self.condition_roles
        treatment_effects = [c for c in comparisons
                           if roles.get(c.condition_a) == "control"
                           and roles.get(c.condition_b) == "treatment"]
        if treatment_effects:
            console.print("\n[bold]Key Question: Does each treatment add value beyond the control?[/bold]")
            for c in treatment_effects:
                direction = "YES" if c.significant and c.mean_difference > 0 else "NO" if c.significant else "INCONCLUSIVE"
                p_str = f"p={c.p_value_corrected:.4f}" if c.p_value_corrected is not None else "p=N/A"
                console.print(
                    f"  {c.model} / {c.category} / {c.condition_a}→{c.condition_b}: "
                    f"[bold]{direction}[/bold] "
                    f"(diff={c.mean_difference:+.3f}, d={c.cohens_d:+.2f}, {p_str})"
                )

        self._print_power_warnings(comparisons)

        console.print(f"\n[dim]* = significant at p<0.05 after Holm-Bonferroni correction[/dim]")
        console.print(f"[dim]Scores show mean (pass rate). CI = bootstrap 95% confidence interval.[/dim]")
        console.print(f"[dim]Cohen's d: ~0=negligible, S=small(0.2), M=medium(0.5), L=large(0.8)[/dim]")
        console.print(f"[dim]N (tasks) = unique tasks (unit of analysis). Power < 0.80 in yellow.[/dim]")

    def _print_power_warnings(self, comparisons: list[PairwiseComparison]) -> None:
        """Print warnings for underpowered comparisons."""
        underpowered = [c for c in comparisons if c.power is not None and c.power < 0.8 and not c.significant]
        if underpowered:
            console.print(f"\n[yellow][bold]Power Warning:[/bold] {len(underpowered)} comparisons are underpowered (power < 0.80).[/yellow]")
            console.print("[yellow]Non-significant results in these comparisons may be due to insufficient sample size, not absence of effect.[/yellow]")
            # Show what effect sizes could be detected
            for c in underpowered[:5]:  # show first 5
                min_d = _minimum_detectable_effect(c.n_pairs)
                if min_d is not None:
                    console.print(
                        f"  [yellow]{c.condition_a}→{c.condition_b} / {c.category}: "
                        f"n={c.n_pairs} tasks, power={c.power:.2f}, min detectable d={min_d:.2f}[/yellow]"
                    )

    def print_power_report(self) -> None:
        """Print dedicated power analysis summary."""
        comparisons = self.compare()
        if not comparisons:
            console.print("[yellow]No comparisons available for power analysis.[/yellow]")
            return

        console.print("\n[bold]Power Analysis Report[/bold]")
        console.print("Minimum detectable effect sizes at 80% power (alpha=0.05):\n")

        # Group by n_pairs to show power at different sample sizes
        n_values = sorted(set(c.n_pairs for c in comparisons))
        table = Table(title="Detectable Effect Sizes by Sample Size")
        table.add_column("N (tasks)", justify="right")
        table.add_column("Min |d| for 80% power", justify="right")
        table.add_column("Interpretation")

        for n in n_values:
            min_d = _minimum_detectable_effect(n)
            if min_d is not None:
                if min_d >= 0.8:
                    interp = "Can only detect large effects"
                elif min_d >= 0.5:
                    interp = "Can detect medium+ effects"
                elif min_d >= 0.2:
                    interp = "Can detect small+ effects"
                else:
                    interp = "Can detect very small effects"
                table.add_row(str(n), f"{min_d:.3f}", interp)

        console.print(table)

    def to_markdown(self) -> str:
        comparisons = self.compare()
        lines = [
            "| Model | Category | Comparison | A Mean | B Mean | Diff | Cohen's d | p (corrected) | Power | Sig | N (tasks) |",
            "|-------|----------|------------|--------|--------|------|-----------|---------------|-------|-----|-----------|",
        ]
        for c in comparisons:
            p_str = f"{c.p_value_corrected:.4f}" if c.p_value_corrected is not None else "N/A"
            sig = "yes" if c.significant else "no"
            power_str = f"{c.power:.2f}" if c.power is not None else "N/A"
            lines.append(
                f"| {c.model} | {c.category} | "
                f"{c.condition_a}→{c.condition_b} | "
                f"{c.a_mean:.3f} | {c.b_mean:.3f} | "
                f"{c.mean_difference:+.3f} | {c.cohens_d:+.2f} | "
                f"{p_str} | {power_str} | {sig} | {c.n_pairs} |"
            )
        return "\n".join(lines)

    def domain_specificity_analysis(
        self,
        domain_matched: list[str],
        treatment_condition: str | None = None,
    ) -> list[DomainSpecificityResult]:
        """Analyze whether a treatment effect is domain-specific.

        Uses task-level treatment effects (treatment_score - control_score per task)
        and compares matched-domain tasks vs mismatched-domain tasks using Welch's
        t-test. This operates on individual tasks (not category-level summaries),
        providing far greater statistical power — especially when there are few
        matched categories (e.g. 2) but many tasks per category (e.g. 50).

        Args:
            domain_matched: Categories that match the treatment's domain.
            treatment_condition: Specific treatment to analyze. If None, analyzes all.
        """
        # Get category-level comparisons for descriptive stats
        comparisons = self.compare()
        roles = self.condition_roles

        # Get task-level aggregated scores
        aggregated = self._aggregate_per_task()

        # Identify control and treatment conditions
        conditions_present = {r.condition for r in self.results}
        controls = sorted(c for c in conditions_present if roles.get(c) == "control")
        treatments = sorted(c for c in conditions_present if roles.get(c) == "treatment")

        if treatment_condition:
            treatments = [t for t in treatments if t == treatment_condition]

        if not controls or not treatments:
            return []

        models = sorted({r.model for r in self.results})

        results = []
        for model in models:
            for control in controls:
                for treatment in treatments:
                    # Compute task-level effects: treatment_score - control_score
                    matched_effects: list[float] = []
                    mismatched_effects: list[float] = []
                    # Per-task records for mixed-effects model
                    task_effect_records: list[tuple[float, int, str]] = []

                    categories = sorted({
                        k[1] for k in aggregated
                        if k[0] == model and k[2] in (control, treatment)
                    })

                    for category in categories:
                        ctrl_tasks = aggregated.get((model, category, control), {})
                        treat_tasks = aggregated.get((model, category, treatment), {})
                        common = sorted(set(ctrl_tasks.keys()) & set(treat_tasks.keys()))

                        for task_id in common:
                            effect = treat_tasks[task_id][0] - ctrl_tasks[task_id][0]
                            is_matched = 1 if category in domain_matched else 0
                            task_effect_records.append((effect, is_matched, category))
                            if is_matched:
                                matched_effects.append(effect)
                            else:
                                mismatched_effects.append(effect)

                    if not matched_effects or not mismatched_effects:
                        continue

                    # Primary: mixed-effects model (accounts for category clustering)
                    mm_result = _domain_specificity_mixed_model(task_effect_records)

                    # Fallback: Welch's t-test on task-level effects
                    p_value, coefficient, se, ci_lo, ci_hi = _domain_specificity_test(
                        matched_effects, mismatched_effects
                    )

                    # Category-level descriptive stats from compare()
                    cat_comps = [
                        c for c in comparisons
                        if c.model == model
                        and c.condition_a == control
                        and c.condition_b == treatment
                    ]
                    matched_comps = [c for c in cat_comps if c.category in domain_matched]
                    mismatched_comps = [c for c in cat_comps if c.category not in domain_matched]

                    matched_cat_d = (
                        _mean([c.cohens_d for c in matched_comps]) if matched_comps else 0.0
                    )
                    mismatched_cat_d = (
                        _mean([c.cohens_d for c in mismatched_comps])
                        if mismatched_comps else 0.0
                    )
                    matched_cat_diff = (
                        _mean([c.mean_difference for c in matched_comps])
                        if matched_comps else 0.0
                    )
                    mismatched_cat_diff = (
                        _mean([c.mean_difference for c in mismatched_comps])
                        if mismatched_comps else 0.0
                    )

                    # Use mixed model as primary if available, Welch's as fallback
                    if mm_result is not None:
                        primary_p = mm_result[0]
                        primary_sig = bool(primary_p is not None and primary_p < 0.05)
                    else:
                        primary_p = float(p_value) if p_value is not None else None
                        primary_sig = bool(p_value is not None and p_value < 0.05)

                    results.append(DomainSpecificityResult(
                        model=model,
                        treatment_condition=treatment,
                        matched_categories=sorted(set(
                            c.category for c in matched_comps
                        )) if matched_comps else sorted(
                            c for c in categories if c in domain_matched
                        ),
                        mismatched_categories=sorted(set(
                            c.category for c in mismatched_comps
                        )) if mismatched_comps else sorted(
                            c for c in categories if c not in domain_matched
                        ),
                        matched_mean_effect=matched_cat_d,
                        mismatched_mean_effect=mismatched_cat_d,
                        specificity_delta=float(coefficient),
                        specificity_p_value=primary_p,
                        specificity_significant=primary_sig,
                        matched_mean_score_diff=matched_cat_diff,
                        mismatched_mean_score_diff=mismatched_cat_diff,
                        n_tasks_matched=len(matched_effects),
                        n_tasks_mismatched=len(mismatched_effects),
                        regression_coefficient=coefficient,
                        regression_se=se,
                        regression_ci_lower=ci_lo,
                        regression_ci_upper=ci_hi,
                        mixed_model_used=mm_result is not None,
                        mixed_model_coefficient=mm_result[1] if mm_result else None,
                        mixed_model_p_value=mm_result[0] if mm_result else None,
                        mixed_model_se=mm_result[2] if mm_result else None,
                        mixed_model_ci_lower=mm_result[3] if mm_result else None,
                        mixed_model_ci_upper=mm_result[4] if mm_result else None,
                    ))

        return results

    def print_domain_report(
        self, domain_matched: list[str], treatment_condition: str | None = None
    ) -> None:
        """Print domain specificity analysis."""
        results = self.domain_specificity_analysis(domain_matched, treatment_condition)

        if not results:
            console.print("[yellow]No domain specificity data available.[/yellow]")
            return

        console.print("\n[bold]Domain Specificity Analysis[/bold]")
        console.print(f"  Matched domains: {', '.join(domain_matched)}")
        any_mixed = any(r.mixed_model_used for r in results)
        if any_mixed:
            console.print(f"  Primary method: Mixed-effects model: effect ~ is_matched + (1|category)")
            console.print(f"  Fallback: Welch's t-test on task-level treatment effects")
        else:
            console.print(f"  Method: Welch's t-test on task-level treatment effects")
            console.print(f"  [dim]Install statsmodels for mixed-effects model (accounts for category clustering)[/dim]")
        console.print(f"  Question: Is the treatment effect larger in matched domains?\n")

        table = Table(title="Control \u2192 Treatment Effect by Domain Match")
        table.add_column("Model")
        table.add_column("Treatment")
        table.add_column("Matched d", justify="right")
        table.add_column("Mismatched d", justify="right")
        table.add_column("N (matched)", justify="right")
        table.add_column("N (mismatched)", justify="right")
        table.add_column("Effect [95% CI]", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Specific?", justify="center")

        for r in results:
            delta_color = "green" if r.specificity_delta > 0 else "red"
            p_str = (
                f"{r.specificity_p_value:.4f}"
                if r.specificity_p_value is not None
                else "N/A"
            )

            if r.regression_ci_lower is not None and r.regression_ci_upper is not None:
                effect_str = (
                    f"[{delta_color}]{r.regression_coefficient:+.4f}[/{delta_color}] "
                    f"[{r.regression_ci_lower:+.4f}, {r.regression_ci_upper:+.4f}]"
                )
            else:
                effect_str = (
                    f"[{delta_color}]{r.regression_coefficient:+.4f}[/{delta_color}]"
                )

            table.add_row(
                r.model,
                r.treatment_condition,
                f"{r.matched_mean_effect:+.3f}",
                f"{r.mismatched_mean_effect:+.3f}",
                str(r.n_tasks_matched),
                str(r.n_tasks_mismatched),
                effect_str,
                p_str,
                "[green]YES[/green]" if r.specificity_significant else "NO",
            )

        console.print(table)

        for r in results:
            if r.specificity_significant and r.specificity_delta > 0:
                console.print(
                    f"\n  [bold green]{r.model}/{r.treatment_condition}:[/bold green] "
                    f"Treatment IS domain-specific. "
                    f"Matched-domain tasks improved {r.regression_coefficient:+.4f} more "
                    f"than mismatched (n={r.n_tasks_matched}+{r.n_tasks_mismatched} tasks, "
                    f"p={r.specificity_p_value:.4f})."
                )
            elif r.specificity_delta > 0:
                p_display = (
                    f"{r.specificity_p_value:.4f}"
                    if r.specificity_p_value is not None
                    else "N/A"
                )
                console.print(
                    f"\n  [bold yellow]{r.model}/{r.treatment_condition}:[/bold yellow] "
                    f"Trend toward domain specificity "
                    f"(delta={r.regression_coefficient:+.4f}) but not statistically "
                    f"significant (n={r.n_tasks_matched}+{r.n_tasks_mismatched} tasks, "
                    f"p={p_display})."
                )
            else:
                console.print(
                    f"\n  [bold]{r.model}/{r.treatment_condition}:[/bold] "
                    f"No evidence of domain-specific effect "
                    f"(delta={r.regression_coefficient:+.4f})."
                )

        # Show mixed model details if used
        for r in results:
            if r.mixed_model_used and r.mixed_model_coefficient is not None:
                mm_ci = ""
                if r.mixed_model_ci_lower is not None:
                    mm_ci = f" [{r.mixed_model_ci_lower:+.4f}, {r.mixed_model_ci_upper:+.4f}]"
                mm_p = f"{r.mixed_model_p_value:.4f}" if r.mixed_model_p_value is not None else "N/A"
                welch_p = ""
                if p_value is not None:
                    welch_p = f", Welch's t-test p={float(p_value):.4f}"
                console.print(
                    f"\n  [dim]{r.treatment_condition} mixed-effects: "
                    f"coeff={r.mixed_model_coefficient:+.4f}{mm_ci}, "
                    f"p={mm_p}{welch_p}[/dim]"
                )

        console.print(
            f"\n[dim]Matched/Mismatched d = category-level Cohen's d (descriptive).[/dim]"
        )
        console.print(
            f"[dim]Effect = mean task-level score difference (matched - mismatched).[/dim]"
        )
        if any_mixed:
            console.print(
                f"[dim]p-value from mixed-effects model (effect ~ is_matched + (1|category)).[/dim]"
            )
        else:
            console.print(
                f"[dim]p-value from one-sided Welch's t-test on task-level effects.[/dim]"
            )

    def extraction_method_audit(self) -> dict[str, dict[str, int]]:
        """Report MCQ extraction method frequency by condition.

        If treatment prompts cause models to respond differently (e.g., more
        verbose, less structured), the MCQ scorer may use different extraction
        paths. Systematic differences are a potential measurement confound.

        Returns:
            {condition_name: {extraction_method: count}}
        """
        audit: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in self.results:
            method = r.scorer_details.get("extraction_method", "unknown")
            audit[r.condition][method] += 1
        return {k: dict(v) for k, v in audit.items()}

    def response_length_stats(self) -> dict[str, dict[str, float]]:
        """Report response length statistics by condition.

        Systematic differences in response length across conditions suggest
        the prompts elicit different response styles, which could confound
        scoring.

        Returns:
            {condition_name: {"mean_chars": float, "median_chars": float,
                              "mean_tokens": float, "n": int}}
        """
        by_condition: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for r in self.results:
            by_condition[r.condition].append((len(r.response), r.output_tokens))

        stats: dict[str, dict[str, float]] = {}
        for condition, lengths in by_condition.items():
            chars = sorted(c for c, _ in lengths)
            tokens = [t for _, t in lengths]
            n = len(chars)
            median_chars = chars[n // 2] if n else 0
            stats[condition] = {
                "mean_chars": _mean([float(c) for c in chars]),
                "median_chars": float(median_chars),
                "mean_tokens": _mean([float(t) for t in tokens]),
                "n": n,
            }
        return stats

    def print_measurement_audit(self) -> None:
        """Print measurement bias diagnostics: extraction methods and response lengths."""
        # --- Extraction method audit ---
        audit = self.extraction_method_audit()
        if not audit:
            return

        console.print("\n[bold]Measurement Bias Audit[/bold]")

        # Build extraction method table
        all_methods = sorted({m for methods in audit.values() for m in methods})
        table = Table(title="MCQ Extraction Methods by Condition")
        table.add_column("Condition")
        for method in all_methods:
            table.add_column(method, justify="right")
        table.add_column("Total", justify="right")

        warnings = []
        for condition in sorted(audit.keys()):
            methods = audit[condition]
            total = sum(methods.values())
            row = [condition]
            for method in all_methods:
                count = methods.get(method, 0)
                pct = count / total * 100 if total else 0
                row.append(f"{count} ({pct:.0f}%)")
            row.append(str(total))
            table.add_row(*row)

            # Check for high "none" extraction rate
            none_count = methods.get("none", 0)
            if total > 0 and none_count / total > 0.15:
                warnings.append(
                    f"[yellow]Warning: {condition} has {none_count/total:.0%} "
                    f"'none' extractions (scorer may be unreliable)[/yellow]"
                )

        console.print(table)
        for w in warnings:
            console.print(w)

        # --- Chi-square independence test ---
        chi2, chi2_p, chi2_interp = _extraction_method_chi_square(audit)
        if chi2 is not None:
            p_color = "red" if chi2_p < 0.05 else "green"
            console.print(
                f"\n  Extraction method independence: chi2={chi2:.2f}, "
                f"[{p_color}]p={chi2_p:.4f}[/{p_color}] — {chi2_interp}"
            )

        # --- Response length audit ---
        length_stats = self.response_length_stats()
        if not length_stats:
            return

        table = Table(title="Response Length by Condition")
        table.add_column("Condition")
        table.add_column("Mean Chars", justify="right")
        table.add_column("Median Chars", justify="right")
        table.add_column("Mean Tokens", justify="right")
        table.add_column("N", justify="right")

        mean_chars_values = [s["mean_chars"] for s in length_stats.values()]
        overall_mean = _mean(mean_chars_values) if mean_chars_values else 0

        for condition in sorted(length_stats.keys()):
            s = length_stats[condition]
            table.add_row(
                condition,
                f"{s['mean_chars']:.0f}",
                f"{s['median_chars']:.0f}",
                f"{s['mean_tokens']:.0f}",
                str(s["n"]),
            )

        console.print(table)

        # Warn if lengths diverge substantially
        if overall_mean > 0:
            for condition, s in length_stats.items():
                divergence = abs(s["mean_chars"] - overall_mean) / overall_mean
                if divergence > 0.20:
                    console.print(
                        f"[yellow]Warning: {condition} response length "
                        f"({s['mean_chars']:.0f} chars) diverges {divergence:.0%} "
                        f"from mean ({overall_mean:.0f}) — possible prompt-induced "
                        f"style difference[/yellow]"
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

        self.print_measurement_audit()


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
    """McNemar's test for paired binary outcomes.

    Only appropriate when data is strictly binary (single-run experiments).
    With multi-run data, per-task averaging produces proportions, which are
    routed to continuous tests instead.
    """
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
            result = stats.binomtest(b_improved, b_improved + b_regressed, 0.5)
            return result.pvalue, "mcnemar_exact"
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

    Note: This is a general-purpose utility kept for backward compatibility.
    For domain specificity analysis, _domain_specificity_test (Welch's t-test
    on task-level effects) is preferred as it provides much greater power when
    categories contain many tasks.
    """
    if len(matched) < 1 or len(mismatched) < 1:
        return None

    observed_delta = _mean(matched) - _mean(mismatched)
    combined = matched + mismatched
    n_matched = len(matched)

    rng = random.Random(42)
    count_ge = 0
    for _ in range(n_permutations):
        shuffled = list(combined)  # copy to avoid mutating across iterations
        rng.shuffle(shuffled)
        perm_matched = shuffled[:n_matched]
        perm_mismatched = shuffled[n_matched:]
        perm_delta = _mean(perm_matched) - _mean(perm_mismatched)
        if perm_delta >= observed_delta:
            count_ge += 1

    return count_ge / n_permutations


def _domain_specificity_mixed_model(
    task_records: list[tuple[float, int, str]],
) -> tuple[float, float, float, float, float] | None:
    """Test domain specificity with a mixed-effects linear model.

    Model: effect ~ is_matched + (1|category)

    This accounts for within-category clustering that Welch's t-test ignores.
    Tasks within the same category share topic structure and difficulty,
    violating the independence assumption of the t-test.

    Args:
        task_records: List of (effect, is_matched, category) per task.

    Returns:
        (p_value, coefficient, se, ci_lower, ci_upper) or None if unavailable.
    """
    if len(task_records) < 5:
        return None

    # Need at least 2 categories to fit random effects
    categories = set(r[2] for r in task_records)
    if len(categories) < 3:
        return None

    try:
        import pandas as pd
        import statsmodels.formula.api as smf

        df = pd.DataFrame(task_records, columns=["effect", "is_matched", "category"])

        model = smf.mixedlm("effect ~ is_matched", data=df, groups=df["category"])
        result = model.fit(reml=True)

        coeff = result.params["is_matched"]
        p_val = result.pvalues["is_matched"]
        se = result.bse["is_matched"]

        ci = result.conf_int().loc["is_matched"]
        ci_lo, ci_hi = float(ci.iloc[0]), float(ci.iloc[1])

        return float(p_val), float(coeff), float(se), ci_lo, ci_hi
    except ImportError:
        return None
    except Exception:
        return None


def _extraction_method_chi_square(
    audit: dict[str, dict[str, int]],
) -> tuple[float | None, float | None, str]:
    """Chi-square test of independence for extraction method distributions.

    Tests whether MCQ extraction methods differ significantly across conditions.
    Systematic differences indicate the prompts are changing response format,
    which is a potential measurement confound.

    Returns:
        (chi2_statistic, p_value, interpretation)
    """
    if len(audit) < 2:
        return None, None, "Need at least 2 conditions"

    try:
        from scipy import stats

        # Build contingency table: conditions × methods
        all_methods = sorted({m for methods in audit.values() for m in methods})
        conditions = sorted(audit.keys())

        table = []
        for cond in conditions:
            row = [audit[cond].get(m, 0) for m in all_methods]
            table.append(row)

        chi2, p_value, dof, expected = stats.chi2_contingency(table)

        if p_value < 0.01:
            interp = "SIGNIFICANT: extraction methods differ across conditions (potential confound)"
        elif p_value < 0.05:
            interp = "MARGINAL: extraction methods may differ across conditions"
        else:
            interp = "OK: no significant difference in extraction methods across conditions"

        return float(chi2), float(p_value), interp
    except ImportError:
        return None, None, "scipy not installed"
    except Exception as e:
        return None, None, f"test failed: {e}"


def _recommended_sample_size(
    target_d: float = 0.2, power: float = 0.80, alpha: float = 0.05
) -> int | None:
    """Compute required sample size per group to detect a given effect size.

    Uses binary search over sample sizes.
    """
    try:
        for n in range(5, 10000):
            achieved = _post_hoc_power(target_d, n, alpha)
            if achieved is not None and achieved >= power:
                return n
        return None
    except Exception:
        return None


def _domain_specificity_test(
    matched_effects: list[float],
    mismatched_effects: list[float],
) -> tuple[float | None, float, float | None, float | None, float | None]:
    """Test whether matched-domain task effects are larger than mismatched.

    Uses Welch's t-test on task-level treatment effects (one-sided: matched > mismatched).
    This operates on individual task effects rather than category-level summaries,
    providing far greater power when categories contain many tasks (e.g. 100 matched
    tasks + 600 mismatched tasks vs 2 matched categories + 12 mismatched categories).

    Returns: (p_value, coefficient, standard_error, ci_lower, ci_upper)
        coefficient = mean(matched_effects) - mean(mismatched_effects)
    """
    coefficient = _mean(matched_effects) - _mean(mismatched_effects)

    if len(matched_effects) < 2 or len(mismatched_effects) < 2:
        return None, coefficient, None, None, None

    try:
        from scipy import stats

        result = stats.ttest_ind(
            matched_effects,
            mismatched_effects,
            equal_var=False,  # Welch's t-test
            alternative="greater",  # one-sided: matched > mismatched
        )
        p_value = result.pvalue

        # Compute SE and CI
        se_matched = _std(matched_effects) / math.sqrt(len(matched_effects))
        se_mismatched = _std(mismatched_effects) / math.sqrt(len(mismatched_effects))
        se = math.sqrt(se_matched**2 + se_mismatched**2)

        # Welch-Satterthwaite degrees of freedom for 95% CI
        if se > 0:
            num = (se_matched**2 + se_mismatched**2) ** 2
            denom = (
                se_matched**4 / (len(matched_effects) - 1)
                + se_mismatched**4 / (len(mismatched_effects) - 1)
            )
            df = num / denom if denom > 0 else 1
            t_crit = stats.t.ppf(0.975, df=df)  # two-sided 95% CI
            ci_lower = coefficient - t_crit * se
            ci_upper = coefficient + t_crit * se
        else:
            ci_lower = coefficient
            ci_upper = coefficient

        return p_value, coefficient, se, ci_lower, ci_upper
    except ImportError:
        return None, coefficient, None, None, None
    except Exception:
        return None, coefficient, None, None, None


def _post_hoc_power(effect_size: float, n: int, alpha: float = 0.05) -> float | None:
    """Compute achieved power for a two-sided paired t-test.

    Uses the non-central t-distribution to compute the probability of
    rejecting H0 given the observed effect size and sample size.
    """
    if n < 3 or effect_size == 0:
        return None

    try:
        from scipy import stats

        df = n - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df=df)
        ncp = abs(effect_size) * math.sqrt(n)  # non-centrality parameter
        # Power = P(reject H0) = P(|T| > t_crit) under non-central t
        power = (
            1 - stats.nct.cdf(t_crit, df=df, nc=ncp)
            + stats.nct.cdf(-t_crit, df=df, nc=ncp)
        )
        return power
    except ImportError:
        return None
    except Exception:
        return None


def _minimum_detectable_effect(
    n: int, power: float = 0.80, alpha: float = 0.05
) -> float | None:
    """Find the minimum Cohen's d detectable at given power and sample size.

    Uses binary search over effect sizes.
    """
    if n < 3:
        return None

    try:
        from scipy import stats

        # Binary search for the effect size that gives target power
        lo, hi = 0.01, 5.0
        for _ in range(50):
            mid = (lo + hi) / 2
            achieved = _post_hoc_power(mid, n, alpha)
            if achieved is None:
                return None
            if achieved < power:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    except Exception:
        return None


# --- Cross-seed replication analysis ---


class CrossSeedAnalyzer:
    """Analyze consistency of results across multiple task-sampling seeds.

    When an experiment is run with --multi-seed, each seed produces a different
    random sample of tasks. This analyzer checks whether conclusions replicate
    across samples, addressing single-seed fragility.
    """

    def __init__(self, parent_dir: str | Path) -> None:
        self.parent_dir = Path(parent_dir)
        self.manifest = self._load_manifest()
        self.analyzers: dict[int, ExperimentAnalyzer] = {}
        for seed, directory in zip(self.manifest["seeds"], self.manifest["directories"]):
            self.analyzers[seed] = ExperimentAnalyzer(directory)

    def _load_manifest(self) -> dict:
        manifest_path = self.parent_dir / "multi_seed_manifest.json"
        with open(manifest_path) as f:
            return json.load(f)

    @property
    def seeds(self) -> list[int]:
        return self.manifest["seeds"]

    def per_seed_summaries(self) -> dict[int, dict]:
        """Get summary statistics per seed."""
        return {seed: analyzer.summary() for seed, analyzer in self.analyzers.items()}

    def cross_seed_comparisons(self) -> dict[int, list[PairwiseComparison]]:
        """Run pairwise comparisons for each seed."""
        return {seed: analyzer.compare() for seed, analyzer in self.analyzers.items()}

    def replication_report(self) -> dict:
        """Assess which findings replicate across seeds.

        A finding "replicates" if its direction is consistent across all seeds.
        A finding is "robust" if it is significant in the majority of seeds.
        """
        all_comparisons = self.cross_seed_comparisons()

        # Group by (model, category, condition_a, condition_b) across seeds
        finding_key = lambda c: (c.model, c.category, c.condition_a, c.condition_b)
        findings: dict[tuple, list[tuple[int, PairwiseComparison]]] = defaultdict(list)

        for seed, comps in all_comparisons.items():
            for c in comps:
                findings[finding_key(c)].append((seed, c))

        report: dict[str, list] = {
            "robust": [],
            "direction_consistent": [],
            "inconsistent": [],
            "summary": {},
        }

        for key, seed_comps in findings.items():
            model, category, cond_a, cond_b = key
            directions = [c.mean_difference > 0 for _, c in seed_comps]
            sig_count = sum(1 for _, c in seed_comps if c.significant)
            mean_diffs = [c.mean_difference for _, c in seed_comps]
            mean_ds = [c.cohens_d for _, c in seed_comps]

            entry = {
                "model": model,
                "category": category,
                "comparison": f"{cond_a} -> {cond_b}",
                "seeds_tested": len(seed_comps),
                "seeds_significant": sig_count,
                "direction_positive": sum(directions),
                "direction_negative": len(directions) - sum(directions),
                "mean_diff_range": (min(mean_diffs), max(mean_diffs)),
                "mean_diff_avg": _mean(mean_diffs),
                "cohens_d_avg": _mean(mean_ds),
            }

            all_same_direction = all(directions) or not any(directions)
            majority_sig = sig_count > len(seed_comps) / 2

            if all_same_direction and majority_sig:
                report["robust"].append(entry)
            elif all_same_direction:
                report["direction_consistent"].append(entry)
            else:
                report["inconsistent"].append(entry)

        report["summary"] = {
            "total_findings": len(findings),
            "robust": len(report["robust"]),
            "direction_consistent": len(report["direction_consistent"]),
            "inconsistent": len(report["inconsistent"]),
            "seeds": self.seeds,
        }

        return report

    def print_cross_seed_report(self) -> None:
        """Print cross-seed replication analysis."""
        report = self.replication_report()
        summary = report["summary"]

        console.print(f"\n[bold]Cross-Seed Replication Analysis[/bold]")
        console.print(f"  Seeds tested: {summary['seeds']}")
        console.print(f"  Total findings: {summary['total_findings']}")
        console.print(
            f"  [green]Robust (consistent direction + majority significant):[/green] "
            f"{summary['robust']}"
        )
        console.print(
            f"  [yellow]Direction consistent (but not majority significant):[/yellow] "
            f"{summary['direction_consistent']}"
        )
        console.print(
            f"  [red]Inconsistent (direction flips across seeds):[/red] "
            f"{summary['inconsistent']}"
        )

        if report["robust"]:
            table = Table(title="\nRobust Findings (replicate across seeds)")
            table.add_column("Model")
            table.add_column("Category")
            table.add_column("Comparison")
            table.add_column("Seeds Sig", justify="right")
            table.add_column("Avg Diff", justify="right")
            table.add_column("Avg d", justify="right")
            table.add_column("Diff Range", justify="right")

            for entry in report["robust"]:
                lo, hi = entry["mean_diff_range"]
                table.add_row(
                    entry["model"],
                    entry["category"],
                    entry["comparison"],
                    f"{entry['seeds_significant']}/{entry['seeds_tested']}",
                    f"{entry['mean_diff_avg']:+.3f}",
                    f"{entry['cohens_d_avg']:+.2f}",
                    f"[{lo:+.3f}, {hi:+.3f}]",
                )
            console.print(table)

        if report["inconsistent"]:
            table = Table(title="\nInconsistent Findings (seed-dependent)")
            table.add_column("Model")
            table.add_column("Category")
            table.add_column("Comparison")
            table.add_column("+/-", justify="right")
            table.add_column("Diff Range", justify="right")

            for entry in report["inconsistent"]:
                lo, hi = entry["mean_diff_range"]
                table.add_row(
                    entry["model"],
                    entry["category"],
                    entry["comparison"],
                    f"{entry['direction_positive']}/{entry['direction_negative']}",
                    f"[{lo:+.3f}, {hi:+.3f}]",
                )
            console.print(table)

        if summary["inconsistent"] > 0:
            console.print(
                f"\n[yellow]Warning: {summary['inconsistent']} findings change direction "
                f"across seeds. These results are seed-dependent and should not be "
                f"interpreted as reliable.[/yellow]"
            )

    def domain_specificity_replication(
        self, domain_matched: list[str], treatment_condition: str | None = None
    ) -> None:
        """Check if domain specificity findings replicate across seeds."""
        console.print(f"\n[bold]Domain Specificity Across Seeds[/bold]")

        all_results: dict[int, list[DomainSpecificityResult]] = {}
        for seed, analyzer in self.analyzers.items():
            all_results[seed] = analyzer.domain_specificity_analysis(
                domain_matched, treatment_condition
            )

        if not any(all_results.values()):
            console.print("[yellow]No domain specificity data available.[/yellow]")
            return

        table = Table(title="Domain Specificity by Seed")
        table.add_column("Seed")
        table.add_column("Treatment")
        table.add_column("Coefficient", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Significant?", justify="center")
        table.add_column("N matched", justify="right")
        table.add_column("N mismatched", justify="right")

        coefficients = []
        for seed in self.seeds:
            for r in all_results.get(seed, []):
                p_str = f"{r.specificity_p_value:.4f}" if r.specificity_p_value is not None else "N/A"
                sig_str = "[green]YES[/green]" if r.specificity_significant else "NO"
                table.add_row(
                    str(seed),
                    r.treatment_condition,
                    f"{r.regression_coefficient:+.4f}",
                    p_str,
                    sig_str,
                    str(r.n_tasks_matched),
                    str(r.n_tasks_mismatched),
                )
                coefficients.append(r.regression_coefficient)

        console.print(table)

        if coefficients:
            all_positive = all(c > 0 for c in coefficients)
            all_negative = all(c < 0 for c in coefficients)
            if all_positive:
                console.print(
                    f"\n  [green]Direction consistent: all {len(coefficients)} coefficients "
                    f"are positive (matched > mismatched).[/green]"
                )
            elif all_negative:
                console.print(
                    f"\n  [yellow]Direction consistent but negative: matched domains "
                    f"show LESS improvement than mismatched.[/yellow]"
                )
            else:
                console.print(
                    f"\n  [red]Direction INCONSISTENT across seeds. The domain specificity "
                    f"finding is seed-dependent and unreliable.[/red]"
                )
