# ai-evals

**Does mentioning a book before a question improve LLM performance in that book's domain?**

Large language models are trained on millions of books. This project tests a behavioral question: if you mention a specific book title and author before a task, does the model perform better on questions in that book's domain — and *only* in that domain?

Not by feeding it the book. Just by saying: *"Draw on your knowledge from Campbell Biology by Urry et al."*

## The Experiment

Each experiment runs a full academic benchmark (e.g., [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) with 12,000+ questions across 14 domains) under multiple conditions with role-based analysis.

### Default Conditions

| Condition | Role | Prompt | Purpose |
|-----------|------|--------|---------|
| **Baseline** | baseline | Raw question, no framing | Ground truth performance |
| **Control** | control | "Draw on your expertise in this domain..." | Isolate the effect of elaborate prompting |
| **Primed** | treatment | "Draw on your knowledge from [Book] by [Author]..." | Test book-specific effect |

The control condition is critical — its prompt is structurally parallel to the treatment, differing only in whether it references a specific book or generic domain expertise. Without this, any improvement could be the instruction style, not the book reference.

### Ablation Controls

For publication-grade rigor, the framework supports additional ablation conditions that rule out alternative explanations:

| Condition | Role | What It Tests |
|-----------|------|---------------|
| **Irrelevant Book** | treatment | Same prompt structure, wrong-domain book. Rules out prompt specificity effects. |
| **Fake Book** | treatment | Same prompt structure, made-up book/author. Rules out generic "book-priming" effects. |

If the primed condition improves biology scores but irrelevant_book and fake_book don't, the effect is specific to the book's content — not just the instruction structure.

### Domain Specificity as the Key Metric

The experiment doesn't just ask *"did priming help?"* — it asks *"did priming help **more** in the book's domain than in unrelated domains?"*

Priming with a biology textbook should improve biology scores. It should **not** improve law or history scores. If it does improve everything equally, the book reference isn't doing anything special — it's just a fancier prompt.

The 12 non-matched domains serve as built-in negative controls within every single experiment. Domain specificity is tested per treatment condition via mixed-effects model (with Welch's t-test fallback).

## Theoretical Framework

This project tests a *behavioral* question: does naming a specific book in a prompt change domain-specific performance? Several candidate mechanisms could explain such an effect:

1. **Attention focusing**: The book reference may function as a domain-specific attention cue, causing the model to preferentially weight domain-relevant knowledge during generation.
2. **Training distribution activation**: If the model encountered the referenced book during training, the reference may activate representations from that training context — similar to how context cues in human memory research facilitate recall of associated information.
3. **Instruction compliance**: The model may interpret the book reference as a stronger instruction to apply domain knowledge, independent of any specific book content.
4. **Prompt specificity**: More specific prompts (naming a particular source vs. generic framing) may produce better-structured reasoning.

The experimental design disentangles these mechanisms:
- The **control condition** uses equally specific prompting (naming the task category) without a book reference, isolating mechanisms 3 and 4.
- The **irrelevant book** condition uses a real but wrong-domain book, isolating mechanism 2 from 1.
- The **fake book** condition uses a made-up book, testing whether mere book-reference structure matters regardless of training exposure.

If primed > control *only in matched domains*, and irrelevant_book and fake_book do not show this pattern, the most parsimonious explanation involves mechanism 1 or 2 — the book reference activates domain-specific processing that generic prompting does not.

This framework does not claim to resolve the mechanism definitively. It establishes *whether* the effect exists and *whether* it is domain-specific, providing the empirical basis for mechanistic follow-up work.

## Supported Benchmarks

| Benchmark | Source | Tasks | Format | Frontier Model Range |
|-----------|--------|-------|--------|---------------------|
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | TIGER-Lab | 12,032 | 10-choice MCQ | 80-90% |
| [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) | Idavidrein | 198 | 4-choice MCQ | 75-94% |
| [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle) | CAIS | 2,500 | MCQ + short answer | 24-53% |

All loaded automatically from HuggingFace with per-category sampling. Local YAML eval sets also supported.

## Pre-Built Experiments

| Config | Book | Matched Domains | Mismatched (Negative Control) |
|--------|------|-----------------|-------------------------------|
| `biology_campbell_mmlu_pro` | Campbell Biology — Urry et al. | biology, health | 12 other domains |
| `physics_feynman_mmlu_pro` | The Feynman Lectures — Feynman | physics, engineering, math | 11 other domains |
| `psychology_kahneman_mmlu_pro` | Thinking, Fast and Slow — Kahneman | psychology, philosophy | 12 other domains |
| `cs_clrs_mmlu_pro` | Introduction to Algorithms — Cormen et al. | computer science, math | 12 other domains |
| `biology_campbell_mmlu_pro_ablation` | Campbell Biology + ablation controls | biology, health | 12 other domains |

## Quick Start

```bash
git clone https://github.com/gregtuc/ai-evals.git
cd ai-evals
pip install -e ".[dev]"
```

Set your API key:
```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

Run an experiment:
```bash
# Preview what will run with cost estimate (no API calls)
ai-evals run configs/biology_campbell_mmlu_pro.yaml --dry-run

# Run it (async with configurable concurrency)
ai-evals run configs/biology_campbell_mmlu_pro.yaml

# Run with higher concurrency
ai-evals run configs/biology_campbell_mmlu_pro.yaml --concurrency 20

# Analyze results (includes power analysis)
ai-evals analyze results/<experiment-dir>/

# Focus domain analysis on a specific treatment
ai-evals analyze results/<experiment-dir>/ --treatment primed
```

## Statistical Methods

- **Role-based pair discovery** — automatically compares all (baseline, treatment), (baseline, control), and (control, treatment) pairs
- **McNemar's test** for binary pass/fail outcomes (using `scipy.stats.binomtest` for exact tests)
- **Wilcoxon signed-rank / paired t-test** for continuous scores
- **Holm-Bonferroni correction** across all comparisons to control false discovery
- **Bootstrap 95% confidence intervals** on mean differences
- **Mixed-effects model** for domain specificity: `effect ~ is_matched + (1|category)`, accounting for within-category clustering (falls back to Welch's t-test if statsmodels not installed)
- **Measurement bias audit** — extraction method chi-square independence test, response length distributions by condition
- **Cohen's d** effect sizes reported alongside p-values
- **Power analysis** — a priori (in dry-run/validate) and post-hoc, with minimum detectable effect sizes and sample size recommendations
- **Multi-seed replication** — run with `--multi-seed N` to repeat the experiment with different task samples, with cross-seed consistency analysis
- **Pre-registration** — `ai-evals pre-register` generates a hash-verified analysis plan before data collection
- **Temperature 0.3** with multiple runs per task to produce genuine variance for statistical testing
- **Randomized condition ordering** per task to prevent sequence effects

## Example Configs

### Simple (3 conditions, backward-compatible)

```yaml
name: "biology-campbell-mmlu-pro"

models:
  - provider: anthropic
    model: claude-sonnet-4-20250514

book:
  title: "Campbell Biology"
  author: "Lisa A. Urry, Michael L. Cain, Steven A. Wasserman"

benchmark:
  name: mmlu_pro
  domain_matched:
    - biology
    - health
  sample_per_category: 50   # 700 tasks total, cost-controlled

runs_per_task: 3
temperature: 0.3
```

### Ablation (5 conditions with controls for alternative explanations)

```yaml
name: "biology-campbell-ablation"

models:
  - provider: anthropic
    model: claude-sonnet-4-20250514

book:
  title: "Campbell Biology"
  author: "Lisa A. Urry, Michael L. Cain, Steven A. Wasserman"

conditions:
  baseline:
    role: baseline
    template: "{task_prompt}"
  control:
    role: control
    template: "Draw on your deep expertise and knowledge in this domain...\n\n{task_prompt}"
  primed:
    role: treatment
    book_vars: true
    template: 'Draw on your deep expertise and knowledge from "{book_title}" by {book_author}...\n\n{task_prompt}'
  irrelevant_book:
    role: treatment
    book_vars: true
    book_override:
      title: "The Feynman Lectures on Physics"
      author: "Richard P. Feynman"
    template: 'Draw on your deep expertise and knowledge from "{book_title}" by {book_author}...\n\n{task_prompt}'
  fake_book:
    role: treatment
    book_vars: true
    book_override:
      title: "Advanced Principles of Cellular Dynamics"
      author: "Dr. Robert J. Thornfield"
    template: 'Draw on your deep expertise and knowledge from "{book_title}" by {book_author}...\n\n{task_prompt}'

benchmark:
  name: mmlu_pro
  domain_matched: [biology, health]
  sample_per_category: 50

runs_per_task: 3
temperature: 0.3
concurrency: 10
```

## Adding Your Own Experiment

Pick any well-known book. Pick the MMLU-Pro categories it covers. Create a config:

```yaml
name: "your-experiment"
models:
  - provider: anthropic
    model: claude-sonnet-4-20250514
book:
  title: "Your Book Title"
  author: "Author Name"
benchmark:
  name: mmlu_pro
  domain_matched:
    - relevant_category_1
    - relevant_category_2
  sample_per_category: 50
runs_per_task: 3
temperature: 0.3
```

For rigorous testing, add ablation conditions using the `conditions:` format shown above.

Available MMLU-Pro categories: `biology`, `business`, `chemistry`, `computer science`, `economics`, `engineering`, `health`, `history`, `law`, `math`, `other`, `philosophy`, `physics`, `psychology`

## Project Structure

```
ai-evals/
├── src/ai_evals/
│   ├── config.py              # Flexible condition system with roles and book overrides
│   ├── benchmark_loader.py    # HuggingFace dataset → EvalTask (per-category sampling)
│   ├── runner.py              # Async experiment runner with concurrency + cost estimation
│   ├── analysis.py            # Role-based pair discovery, power analysis, domain specificity
│   ├── results.py             # JSONL storage with resume, condition roles, task metadata
│   ├── scorers/               # MCQ (last-line priority), exact match, code execution, LLM judge
│   ├── models/                # Anthropic, OpenAI with sync/async and SDK retry support
│   └── cli.py                 # Command-line interface
├── configs/                   # Pre-built experiment configurations (incl. ablation)
├── evals/                     # Local YAML eval tasks (supplementary)
├── tests/                     # Test suite (config, MCQ, analysis, benchmark loader)
└── results/                   # Experiment outputs (gitignored)
```

## CLI Reference

```bash
ai-evals run <config.yaml>                    # Run experiment
ai-evals run <config.yaml> --dry-run          # Preview with cost estimate
ai-evals run <config.yaml> --concurrency 20   # Override concurrency
ai-evals run <config.yaml> --resume <dir>     # Resume interrupted experiment
ai-evals run <config.yaml> --multi-seed 5     # Run 5 replications with different seeds
ai-evals analyze <results-dir>                # Full analysis with power report
ai-evals analyze <results-dir> --treatment X  # Domain analysis for specific treatment
ai-evals pre-register <config.yaml>           # Generate pre-registration document
ai-evals validate <config.yaml>               # Validate config with power analysis
ai-evals list-evals --benchmark mmlu_pro      # Browse benchmark tasks
```

## Limitations

- **Behavioral, not mechanistic**: This framework tests *whether* mentioning a book changes domain-specific performance, not *why*. The theoretical framework section describes candidate mechanisms and how ablation conditions help distinguish them, but definitive mechanistic claims require follow-up work.
- **Single-seed mitigation**: Each experiment uses one random seed for task sampling by default. Use `--multi-seed N` to run N replications with different seeds and the cross-seed analyzer to verify that conclusions are stable across samples.
- **Temperature-based variance**: Multiple runs at temperature > 0 provide within-task variance but are not independent observations. The task remains the unit of analysis; multiple runs reduce noise but do not increase effective sample size.
- **Prompt sensitivity**: Even with structurally parallel templates, LLM behavior is sensitive to exact wording. Small template changes can affect results.
- **Category clustering**: The mixed-effects model accounts for within-category correlation in domain specificity tests. If statsmodels is not installed, the fallback Welch's t-test treats tasks as independent, which may inflate significance when few categories contain many tasks.

## License

MIT
