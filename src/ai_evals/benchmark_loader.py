"""Load established benchmarks from HuggingFace and convert to EvalTask format.

Supported benchmarks:
- mmlu_pro: MMLU-Pro (12K tasks, 10-choice MCQ, 14 categories)
- gpqa_diamond: GPQA Diamond (198 tasks, 4-choice MCQ, graduate-level science)
- hle: Humanity's Last Exam (2,500 tasks, MCQ + short-answer, expert-level)
"""

from __future__ import annotations

import random
import string

from ai_evals.config import EvalTask, ScorerConfig

LETTERS = list(string.ascii_uppercase[:10])  # A-J


def load_benchmark(
    name: str,
    sample_per_category: int | None = None,
    seed: int = 42,
) -> list[EvalTask]:
    """Load a benchmark dataset and convert to EvalTask list.

    Args:
        name: Benchmark name (mmlu_pro, gpqa_diamond, hle)
        sample_per_category: Max tasks per category (None = all)
        seed: Random seed for sampling and answer shuffling
    """
    if name == "mmlu_pro":
        return _load_mmlu_pro(sample_per_category=sample_per_category, seed=seed)
    elif name == "gpqa_diamond":
        return _load_gpqa_diamond(sample_per_category=sample_per_category, seed=seed)
    elif name == "hle":
        return _load_hle(sample_per_category=sample_per_category, seed=seed)
    else:
        raise ValueError(f"Unknown benchmark: {name}. Supported: mmlu_pro, gpqa_diamond, hle")


def list_benchmark_categories(name: str) -> list[str]:
    """List available categories for a benchmark (loads the dataset)."""
    tasks = load_benchmark(name, sample_per_category=1)
    return sorted(set(t.category for t in tasks))


def _load_mmlu_pro(
    sample_per_category: int | None = None,
    seed: int = 42,
) -> list[EvalTask]:
    """Load MMLU-Pro from HuggingFace.

    Dataset: TIGER-Lab/MMLU-Pro
    Format: 10-choice MCQ, 14 categories
    Fields: question, options (list), answer (letter), answer_index, category
    """
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    # Group by category for sampling
    by_category: dict[str, list] = {}
    for row in ds:
        cat = row["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(row)

    rng = random.Random(seed)
    tasks = []

    for cat in sorted(by_category.keys()):
        rows = by_category[cat]
        if sample_per_category and len(rows) > sample_per_category:
            rows = rng.sample(rows, sample_per_category)

        for row in rows:
            question = row["question"]
            options = row["options"]
            answer_letter = row["answer"]

            # Format as lettered MCQ
            options_text = "\n".join(
                f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options) if opt
            )
            prompt = (
                f"{question}\n\n{options_text}\n\n"
                f"Answer with just the letter (A, B, C, etc.)."
            )

            task_id = f"mmlu_pro_{row.get('question_id', rng.randint(0, 999999))}"

            tasks.append(EvalTask(
                id=task_id,
                category=cat,
                name=question[:80],
                prompt=prompt,
                scorer=ScorerConfig(type="mcq", expected=answer_letter),
                metadata={"benchmark": "mmlu_pro", "difficulty": "benchmark"},
            ))

    return tasks


def _load_gpqa_diamond(
    sample_per_category: int | None = None,
    seed: int = 42,
) -> list[EvalTask]:
    """Load GPQA Diamond from HuggingFace.

    Dataset: Idavidrein/gpqa (gpqa_diamond config)
    Format: 4-choice MCQ, graduate-level science
    Fields: Question, Correct Answer, Incorrect Answer 1/2/3, Subdomain
    Note: Answers are NOT pre-shuffled — we shuffle deterministically.
    """
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    rng = random.Random(seed)

    # Group by subdomain for per-category sampling (matching MMLU-Pro/HLE pattern)
    by_category: dict[str, list] = {}
    for row in ds:
        category = row.get("Subdomain", row.get("subdomain", "gpqa_diamond"))
        if category:
            category = category.lower().replace(" ", "_")
        else:
            category = "gpqa_diamond"
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(row)

    tasks = []
    global_idx = 0

    for cat in sorted(by_category.keys()):
        rows = by_category[cat]
        if sample_per_category and len(rows) > sample_per_category:
            rows = rng.sample(rows, sample_per_category)

        for row in rows:
            question = row["Question"]
            correct = row["Correct Answer"]
            incorrect = [
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]

            # Shuffle answers deterministically
            all_answers = [correct] + incorrect
            rng_shuffle = random.Random(seed + global_idx)  # unique seed per question
            rng_shuffle.shuffle(all_answers)

            # Find which letter is the correct answer after shuffling
            correct_idx = all_answers.index(correct)
            correct_letter = LETTERS[correct_idx]

            # Format as MCQ
            options_text = "\n".join(
                f"{LETTERS[j]}) {ans}" for j, ans in enumerate(all_answers)
            )
            prompt = (
                f"{question}\n\n{options_text}\n\n"
                f"Answer with just the letter (A, B, C, or D)."
            )

            tasks.append(EvalTask(
                id=f"gpqa_{cat}_{global_idx:04d}",
                category=cat,
                name=question[:80],
                prompt=prompt,
                scorer=ScorerConfig(type="mcq", expected=correct_letter),
                metadata={"benchmark": "gpqa_diamond", "difficulty": "graduate"},
            ))
            global_idx += 1

    return tasks


def _load_hle(
    sample_per_category: int | None = None,
    seed: int = 42,
) -> list[EvalTask]:
    """Load Humanity's Last Exam from HuggingFace.

    Dataset: cais/hle
    Format: Mix of MCQ and short-answer, expert-level
    Fields: question, answer, image, category, question_type
    Note: Skip image-based questions (text-only model support).
    """
    from datasets import load_dataset

    ds = load_dataset("cais/hle", split="test")

    rng = random.Random(seed)
    skipped_image = 0

    # Group by category
    by_category: dict[str, list] = {}
    for row in ds:
        # Skip image-based questions
        if row.get("image") is not None:
            skipped_image += 1
            continue

        cat = row.get("category", "general")
        if cat:
            cat = cat.lower().replace(" ", "_")
        else:
            cat = "general"

        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(row)

    if skipped_image > 0:
        import sys
        print(f"[benchmark_loader] Skipped {skipped_image} image-based HLE questions", file=sys.stderr)

    tasks = []
    for cat in sorted(by_category.keys()):
        rows = by_category[cat]
        if sample_per_category and len(rows) > sample_per_category:
            rows = rng.sample(rows, sample_per_category)

        for i, row in enumerate(rows):
            question = row["question"]
            answer = row["answer"]

            # Determine scorer based on answer format
            # Short answers get exact_match; if answer looks like a single letter, use mcq
            if len(answer.strip()) == 1 and answer.strip().upper() in "ABCDEFGHIJ":
                scorer = ScorerConfig(type="mcq", expected=answer.strip().upper())
            else:
                scorer = ScorerConfig(type="exact_match", expected=answer)

            prompt = f"{question}\n\nAnswer as concisely as possible."

            tasks.append(EvalTask(
                id=f"hle_{cat}_{i:04d}",
                category=cat,
                name=question[:80],
                prompt=prompt,
                scorer=scorer,
                metadata={"benchmark": "hle", "difficulty": "expert"},
            ))

    return tasks
