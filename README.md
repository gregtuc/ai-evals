# ai-evals

**Do LLMs perform better when you remind them of a book they've already read?**

Large language models ingest millions of books during training. That knowledge doesn't disappear — it becomes latent, distributed across billions of parameters. This project tests a simple idea: if you tell a model to recall a specific book before answering a question, does performance improve in that book's domain?

Not by feeding it the book. Just by saying: *"Recall everything you know from Campbell Biology by Urry et al."*

## The Experiment

Each experiment runs a full academic benchmark (e.g., [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) with 12,000+ questions across 14 domains) under three conditions:

| Condition | Prompt | Purpose |
|-----------|--------|---------|
| **Baseline** | Raw question, no framing | Ground truth performance |
| **Control** | "Think carefully and draw on your expertise..." | Isolate the effect of elaborate prompting |
| **Primed** | "Recall everything you know from [Book] by [Author]..." | Test book-specific activation |

The control condition is critical. Without it, any improvement from priming could just be the model responding to a more detailed instruction — not the book reference itself.

### Domain Specificity as the Key Metric

The experiment doesn't just ask *"did priming help?"* — it asks *"did priming help **more** in the book's domain than in unrelated domains?"*

Priming with a biology textbook should improve biology scores. It should **not** improve law or history scores. If it does improve everything equally, the book reference isn't doing anything special — it's just a fancier prompt.

The 12 non-matched domains serve as built-in negative controls within every single experiment.

## Supported Benchmarks

| Benchmark | Source | Tasks | Format | Frontier Model Range |
|-----------|--------|-------|--------|---------------------|
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | TIGER-Lab | 12,032 | 10-choice MCQ | 80-90% |
| [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) | Idavidrein | 198 | 4-choice MCQ | 75-94% |
| [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle) | CAIS | 2,500 | MCQ + short answer | 24-53% |

All loaded automatically from HuggingFace. Local YAML eval sets also supported.

## Pre-Built Experiments

| Config | Book | Matched Domains | Mismatched (Negative Control) |
|--------|------|-----------------|-------------------------------|
| `biology_campbell_mmlu_pro` | Campbell Biology — Urry et al. | biology, health | 12 other domains |
| `physics_feynman_mmlu_pro` | The Feynman Lectures — Feynman | physics, engineering, math | 11 other domains |
| `psychology_kahneman_mmlu_pro` | Thinking, Fast and Slow — Kahneman | psychology, philosophy | 12 other domains |
| `cs_clrs_mmlu_pro` | Introduction to Algorithms — Cormen et al. | computer science, math | 12 other domains |

## Quick Start

```bash
git clone https://github.com/gregtuc/ai-evals.git
cd ai-evals
pip install -e .
```

Set your API key:
```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

Run an experiment:
```bash
# Preview what will run (no API calls)
ai-evals run configs/biology_campbell_mmlu_pro.yaml --dry-run

# Run it
ai-evals run configs/biology_campbell_mmlu_pro.yaml

# Analyze results
ai-evals analyze results/<experiment-dir>/
```

## Statistical Methods

- **Paired comparisons** across all three conditions per task
- **McNemar's test** for binary pass/fail outcomes
- **Wilcoxon signed-rank / paired t-test** for continuous scores
- **Holm-Bonferroni correction** across all comparisons to control false discovery
- **Bootstrap 95% confidence intervals** on mean differences
- **Permutation test** for domain specificity (are matched-domain effect sizes significantly larger than mismatched?)
- **Cohen's d** effect sizes reported alongside p-values
- **Temperature 0.3** with multiple runs per task to produce genuine variance for statistical testing
- **Randomized condition ordering** per task to prevent sequence effects

## Example Config

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

Available MMLU-Pro categories: `biology`, `business`, `chemistry`, `computer science`, `economics`, `engineering`, `health`, `history`, `law`, `math`, `other`, `philosophy`, `physics`, `psychology`

## Project Structure

```
ai-evals/
├── src/ai_evals/
│   ├── benchmark_loader.py   # HuggingFace dataset → EvalTask conversion
│   ├── runner.py              # 3-condition experiment loop
│   ├── analysis.py            # Statistical analysis + domain specificity
│   ├── scorers/               # MCQ, exact match, code execution, LLM judge
│   ├── models/                # Anthropic, OpenAI provider abstraction
│   └── cli.py                 # Command-line interface
├── configs/                   # Pre-built experiment configurations
├── evals/                     # Local YAML eval tasks (supplementary)
└── results/                   # Experiment outputs (gitignored)
```

## CLI Reference

```bash
ai-evals run <config.yaml>              # Run experiment
ai-evals run <config.yaml> --dry-run    # Preview without API calls
ai-evals run <config.yaml> --resume <dir>  # Resume interrupted experiment
ai-evals analyze <results-dir>          # Statistical analysis
ai-evals validate <config.yaml>         # Validate config
ai-evals list-evals --benchmark mmlu_pro  # Browse benchmark tasks
```

## License

MIT
