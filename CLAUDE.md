# AI Evals

## What This Is
A framework for testing whether mentioning book references before tasks produces domain-specific performance changes on established academic benchmarks.

## Hypothesis
Mentioning a domain-relevant book (title + author only) before a task may produce measurable, domain-specific performance improvements. The experiment tests whether such improvements exist and whether they are specific to the book's domain, ruling out general prompt-enhancement effects.

## Experimental Design
**Flexible conditions** with role-based analysis. Default 3 conditions, expandable with ablation controls:
1. **Baseline** (role: baseline) — bare task prompt
2. **Control** (role: control) — equally specific framing referencing the task category, no book
3. **Primed** (role: treatment) — structurally parallel framing referencing a specific book

**Ablation conditions** for rigorous testing:
4. **Irrelevant Book** (role: treatment) — same framing, wrong-domain book
5. **Fake Book** (role: treatment) — same framing, made-up book/author

Each condition has a `role` (baseline/control/treatment) that drives automatic pair discovery in analysis. The analysis compares all (baseline, treatment), (baseline, control), and (control, treatment) pairs.

**Domain specificity** is the key metric. Run ALL categories of a benchmark with ONE book:
- Domain-matched categories (e.g., biology for a biology book) should improve
- Domain-mismatched categories (e.g., law, history) should NOT improve
- If primed > control only in matched domains, the book reference has a real, specific effect
- Ablation controls (irrelevant_book, fake_book) should NOT show domain-specific patterns

**Statistical rigor:**
- Per-task aggregation across runs (task is the unit of analysis, not individual runs)
- Temperature 0.3 with multiple runs to reduce within-task noise
- Condition order randomized per task
- Holm-Bonferroni multiple comparison correction
- McNemar's test for binary outcomes (single-run), paired t-test/Wilcoxon for continuous
- Bootstrap 95% CIs
- Mixed-effects model for domain specificity: effect ~ is_matched + (1|category), with Welch's t-test fallback
- A priori power analysis (dry-run/validate) and post-hoc with minimum detectable effect sizes and sample size recommendations
- Measurement bias audit: extraction method chi-square test, response length distributions by condition
- Multi-seed replication (--multi-seed N) with cross-seed consistency analysis
- Pre-registration support (ai-evals pre-register) for hash-verified analysis plans

## Benchmarks (HuggingFace)
- **MMLU-Pro** (`TIGER-Lab/MMLU-Pro`) — 12K tasks, 10-choice MCQ, 14 categories
- **GPQA Diamond** (`Idavidrein/gpqa`) — 198 graduate-level science MCQs (sampled per subdomain)
- **Humanity's Last Exam** (`cais/hle`) — 2,500 expert-level questions
- Local YAML evals in `evals/` still supported as fallback

## Architecture
- `src/ai_evals/`
  - `config.py` — Pydantic models with flexible condition system (role-based), backward-compat migration
  - `benchmark_loader.py` — HuggingFace dataset → EvalTask conversion (per-category sampling)
  - `eval_loader.py` — Unified task loading (benchmark or local YAML)
  - `models/` — Provider abstraction (Anthropic, OpenAI) with sync/async and SDK retry support
  - `scorers/` — exact_match, contains, mcq (last-line priority), llm_judge, code_execution
  - `runner.py` — Async experiment runner with configurable concurrency, cost estimation, and multi-seed support
  - `analysis.py` — Per-task aggregation, role-based pair discovery, mixed-effects domain specificity, power analysis, measurement bias audit, cross-seed replication analysis
  - `results.py` — JSONL storage with resume support, condition roles, task metadata
  - `cli.py` — Click CLI with --concurrency, --treatment, --multi-seed flags, pre-register command

## Commands
```bash
pip install -e ".[dev]"
pip install -e ".[stats]"  # optional: mixed-effects models

# Run a benchmark experiment (async with concurrency)
ai-evals run configs/biology_campbell_mmlu_pro.yaml
ai-evals run configs/biology_campbell_mmlu_pro.yaml --dry-run
ai-evals run configs/biology_campbell_mmlu_pro_ablation.yaml --concurrency 20

# Multi-seed replication (addresses single-seed fragility)
ai-evals run configs/biology_campbell_mmlu_pro.yaml --multi-seed 5

# Analyze (auto-detects conditions, domain-matched categories, and multi-seed structure)
ai-evals analyze results/<dir>/
ai-evals analyze results/<dir>/ --treatment primed

# Pre-registration (generate analysis plan before data collection)
ai-evals pre-register configs/biology_campbell_mmlu_pro.yaml

# List benchmark tasks and categories
ai-evals list-evals --benchmark mmlu_pro

# Validate config (includes power analysis and sample size recommendations)
ai-evals validate configs/biology_campbell_mmlu_pro_ablation.yaml

# Run tests
python -m pytest tests/ -v
```

## Config Format

### Simple (backward-compatible, auto-generates 3 default conditions)
```yaml
name: "biology-campbell-mmlu-pro"
models:
  - provider: anthropic
    model: claude-sonnet-4-20250514
book:
  title: "Campbell Biology"
  author: "Lisa A. Urry et al."
benchmark:
  name: mmlu_pro
  domain_matched: [biology, health]
  sample_per_category: 50
runs_per_task: 3
temperature: 0.3
```

### Ablation (explicit conditions with roles and book overrides)
```yaml
name: "biology-campbell-ablation"
models:
  - provider: anthropic
    model: claude-sonnet-4-20250514
book:
  title: "Campbell Biology"
  author: "Lisa A. Urry et al."
conditions:
  baseline:
    role: baseline
    template: "{task_prompt}"
  control:
    role: control
    template: "Draw on your deep expertise and knowledge in {task_category}...\n{task_prompt}"
  primed:
    role: treatment
    book_vars: true
    template: 'Draw on your deep expertise and knowledge from "{book_title}" by {book_author}...\n{task_prompt}'
  irrelevant_book:
    role: treatment
    book_vars: true
    book_override:
      title: "The Feynman Lectures on Physics"
      author: "Richard P. Feynman"
    template: 'Draw on your deep expertise and knowledge from "{book_title}" by {book_author}...\n{task_prompt}'
  fake_book:
    role: treatment
    book_vars: true
    book_override:
      title: "Advanced Principles of Cellular Dynamics"
      author: "Dr. Robert J. Thornfield"
    template: 'Draw on your deep expertise and knowledge from "{book_title}" by {book_author}...\n{task_prompt}'
benchmark:
  name: mmlu_pro
  domain_matched: [biology, health]
  sample_per_category: 50
runs_per_task: 3
temperature: 0.3
concurrency: 10
```
