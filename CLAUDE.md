# AI Evals

## What This Is
A framework for testing whether "re-priming" LLMs with book references improves performance on domain-relevant tasks, using established academic benchmarks.

## Hypothesis
Books ingested during training are dormant knowledge. By reminding the model of a domain-relevant book (title + author only) before a task, we may activate latent knowledge and see measurable, domain-specific improvements.

## Experimental Design
**Three conditions** per task to isolate the book-specific effect:
1. **Baseline** — bare task prompt
2. **Control** — elaborate "think carefully, use your expertise" framing, no book
3. **Primed** — same elaborate framing referencing a specific book

**Domain specificity** is the key metric. Run ALL categories of a benchmark with ONE book:
- Domain-matched categories (e.g., biology for a biology book) should improve
- Domain-mismatched categories (e.g., law, history) should NOT improve
- If primed > control only in matched domains, the book reference has a real, specific effect

**Statistical rigor:**
- Temperature 0.3 for genuine variance across runs
- Condition order randomized per task
- Holm-Bonferroni multiple comparison correction
- McNemar's test for binary outcomes, paired t-test/Wilcoxon for continuous
- Bootstrap 95% CIs
- Permutation test for domain specificity

## Benchmarks (HuggingFace)
- **MMLU-Pro** (`TIGER-Lab/MMLU-Pro`) — 12K tasks, 10-choice MCQ, 14 categories
- **GPQA Diamond** (`Idavidrein/gpqa`) — 198 graduate-level science MCQs
- **Humanity's Last Exam** (`cais/hle`) — 2,500 expert-level questions
- Local YAML evals in `evals/` still supported as fallback

## Architecture
- `src/ai_evals/`
  - `config.py` — Pydantic models (ExperimentConfig, BenchmarkConfig, etc.)
  - `benchmark_loader.py` — HuggingFace dataset → EvalTask conversion
  - `eval_loader.py` — Unified task loading (benchmark or local YAML)
  - `models/` — Provider abstraction (Anthropic, OpenAI)
  - `scorers/` — exact_match, contains, mcq, llm_judge, code_execution
  - `runner.py` — 3-condition experiment loop with randomized ordering
  - `analysis.py` — Per-category stats + domain specificity analysis
  - `results.py` — JSONL storage with resume support
  - `cli.py` — Click CLI

## Commands
```bash
pip install -e ".[dev]"

# Run a benchmark experiment
ai-evals run configs/biology_campbell_mmlu_pro.yaml
ai-evals run configs/biology_campbell_mmlu_pro.yaml --dry-run

# Analyze (auto-detects domain-matched categories from saved config)
ai-evals analyze results/<dir>/

# List benchmark tasks and categories
ai-evals list-evals --benchmark mmlu_pro

# Validate config
ai-evals validate configs/biology_campbell_mmlu_pro.yaml
```

## Config Format (Benchmark)
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
  sample_per_category: 50  # cost control
runs_per_task: 3
temperature: 0.3
```
