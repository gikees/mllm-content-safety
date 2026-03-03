"""Evaluation metrics for content safety classification.

Computes standard classification metrics plus per-category breakdowns
and risk severity calibration metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float] | None = None,
) -> dict:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred: Predicted labels (0/1).
        y_scores: Optional continuous risk scores for AUROC.

    Returns:
        Dict of metric names to values.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "support": len(y_true),
    }

    if y_scores is not None and len(set(y_true)) > 1:
        results["auroc"] = roc_auc_score(y_true, y_scores)

    return results


def compute_per_category_metrics(
    y_true: list[int],
    y_pred: list[int],
    categories: list[str],
) -> dict[str, dict]:
    """Compute metrics broken down by category.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        categories: Category label for each sample.

    Returns:
        Dict mapping category name to metrics dict.
    """
    cat_metrics = {}
    unique_cats = sorted(set(categories))

    for cat in unique_cats:
        mask = [c == cat for c in categories]
        cat_true = [t for t, m in zip(y_true, mask) if m]
        cat_pred = [p for p, m in zip(y_pred, mask) if m]

        if cat_true:
            cat_metrics[cat] = compute_metrics(cat_true, cat_pred)

    return cat_metrics


def compute_severity_calibration(
    true_severities: list[int],
    pred_severities: list[int],
) -> dict:
    """Measure how well predicted severity matches ground truth.

    Args:
        true_severities: Ground truth severity scores (1-5).
        pred_severities: Predicted severity scores (1-5).

    Returns:
        Calibration metrics.
    """
    true_arr = np.array(true_severities)
    pred_arr = np.array(pred_severities)

    return {
        "mae": float(np.mean(np.abs(true_arr - pred_arr))),
        "correlation": float(np.corrcoef(true_arr, pred_arr)[0, 1])
        if len(set(true_severities)) > 1
        else 0.0,
        "exact_match": float(np.mean(true_arr == pred_arr)),
    }


def print_results(metrics: dict, title: str = "Results") -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:>15}: {value:.4f}")
        else:
            print(f"  {key:>15}: {value}")
    print()
