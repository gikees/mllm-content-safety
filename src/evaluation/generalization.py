"""Generalization evaluation on held-out categories.

Tests whether the model can detect violations in categories
it has never seen during training.
"""

from .metrics import compute_metrics, compute_per_category_metrics, print_results


def evaluate_generalization(
    classifier,
    generalization_samples: list[dict],
    verbose: bool = False,
) -> dict:
    """Evaluate on held-out violation categories.

    Args:
        classifier: SafetyClassifier instance (fine-tuned).
        generalization_samples: Samples from categories not seen during training.
        verbose: If True, print per-category breakdowns.

    Returns:
        Overall metrics + per-category breakdown.
    """
    if not generalization_samples:
        print("No generalization samples provided.")
        return {}

    y_true = []
    y_pred = []
    categories = []

    for sample in generalization_samples:
        pred = classifier.predict(sample["image"], sample["text"])
        is_unsafe = "unsafe" in pred["classification"].lower()

        y_true.append(sample["label"])
        y_pred.append(1 if is_unsafe else 0)
        categories.append(sample.get("category", "unknown"))

    overall = compute_metrics(y_true, y_pred)
    print_results(overall, "Generalization Evaluation (Overall)")

    per_category = {}
    if verbose:
        per_category = compute_per_category_metrics(y_true, y_pred, categories)
        for cat, metrics in per_category.items():
            print_results(metrics, f"Category: {cat}")

    return {"overall": overall, "per_category": per_category}
