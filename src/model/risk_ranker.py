"""Risk ranking module.

Converts CoT classification outputs into calibrated risk scores
and applies threshold-based filtering for different precision/recall tradeoffs.
"""

from dataclasses import dataclass


@dataclass
class RiskResult:
    classification: str  # "safe" or "unsafe"
    severity: int  # 1-5
    confidence: float  # 0.0-1.0
    reasoning: str
    action: str  # "allow", "review", "remove"


class RiskRanker:
    """Converts classifier outputs into actionable risk decisions.

    Uses severity scores and optional confidence calibration to assign
    content to action tiers: allow, review, or remove.
    """

    def __init__(
        self,
        review_threshold: int = 2,
        remove_threshold: int = 4,
    ):
        """
        Args:
            review_threshold: Minimum severity to flag for human review.
            remove_threshold: Minimum severity for automatic removal.
        """
        self.review_threshold = review_threshold
        self.remove_threshold = remove_threshold

    def rank(self, prediction: dict) -> RiskResult:
        """Convert a classifier prediction into a risk decision.

        Args:
            prediction: Output from SafetyClassifier.predict().

        Returns:
            RiskResult with action recommendation.
        """
        classification = prediction.get("classification", "").lower().strip()
        severity = prediction.get("severity", 0)
        reasoning = prediction.get("reasoning", "")

        is_unsafe = "unsafe" in classification

        if not is_unsafe and severity < self.review_threshold:
            action = "allow"
        elif severity >= self.remove_threshold:
            action = "remove"
        else:
            action = "review"

        confidence = min(severity / 5.0, 1.0) if is_unsafe else max(1.0 - severity / 5.0, 0.0)

        return RiskResult(
            classification="unsafe" if is_unsafe else "safe",
            severity=severity,
            confidence=confidence,
            reasoning=reasoning,
            action=action,
        )

    def rank_batch(self, predictions: list[dict]) -> list[RiskResult]:
        """Rank a batch of predictions."""
        return [self.rank(p) for p in predictions]

    def summary(self, results: list[RiskResult]) -> dict:
        """Compute summary statistics for a batch of risk results."""
        total = len(results)
        if total == 0:
            return {}

        action_counts = {"allow": 0, "review": 0, "remove": 0}
        severity_sum = 0

        for r in results:
            action_counts[r.action] += 1
            severity_sum += r.severity

        return {
            "total": total,
            "action_distribution": {k: v / total for k, v in action_counts.items()},
            "action_counts": action_counts,
            "mean_severity": severity_sum / total,
        }
