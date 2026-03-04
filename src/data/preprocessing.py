"""Data preprocessing and split creation.

Handles image resizing, text cleaning, and train/val/test splitting
with support for holding out specific categories for generalization testing.
"""

import random
from collections import defaultdict


def create_splits(
    samples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    holdout_categories: list[str] | None = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split samples into train/val/test sets.

    Optionally holds out specific categories entirely for generalization testing.

    Args:
        samples: List of sample dicts (must have 'label' key, optionally 'category').
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation. Test gets the remainder.
        holdout_categories: Categories to exclude from train/val for generalization eval.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'train', 'val', 'test', 'generalization'.
    """
    rng = random.Random(seed)

    generalization = []
    remaining = []

    if holdout_categories:
        for s in samples:
            if s.get("category") in holdout_categories:
                generalization.append(s)
            else:
                remaining.append(s)
    else:
        remaining = list(samples)

    rng.shuffle(remaining)

    n = len(remaining)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = remaining[:n_train]
    val = remaining[n_train : n_train + n_val]
    test = remaining[n_train + n_val :]

    return {
        "train": train,
        "val": val,
        "test": test,
        "generalization": generalization,
    }


def print_split_stats(splits: dict[str, list[dict]]) -> None:
    """Print label distribution for each split."""
    for name, samples in splits.items():
        label_counts = defaultdict(int)
        for s in samples:
            label_counts[s.get("label", "unknown")] += 1
        total = len(samples)
        print(f"{name}: {total} samples | {dict(label_counts)}")
