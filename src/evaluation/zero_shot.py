"""Zero-shot evaluation for content safety classification.

Evaluates the base model (without fine-tuning) to establish baselines.
"""

from src.data.cot_templates import parse_cot_response
from .metrics import compute_metrics, print_results


def evaluate_zero_shot(
    classifier,
    test_samples: list[dict],
    verbose: bool = False,
) -> dict:
    """Evaluate zero-shot classification performance.

    Args:
        classifier: SafetyClassifier instance (base model, no adapter).
        test_samples: Labeled test samples.
        verbose: If True, print per-sample predictions.

    Returns:
        Metrics dict.
    """
    y_true = []
    y_pred = []
    y_scores = []

    for i, sample in enumerate(test_samples):
        pred = classifier.predict(sample["image"], sample["text"])
        is_unsafe = "unsafe" in pred["classification"].lower()

        y_true.append(sample["label"])
        y_pred.append(1 if is_unsafe else 0)
        y_scores.append(pred["severity"] / 5.0)

        if verbose:
            label_str = "unsafe" if sample["label"] == 1 else "safe"
            pred_str = "unsafe" if is_unsafe else "safe"
            match = "OK" if (sample["label"] == 1) == is_unsafe else "MISS"
            print(f"  [{match}] #{i}: true={label_str}, pred={pred_str}, severity={pred['severity']}")

    metrics = compute_metrics(y_true, y_pred, y_scores)
    print_results(metrics, "Zero-Shot Evaluation")
    return metrics
