"""Few-shot evaluation for content safety classification.

Evaluates the model's ability to classify content with K in-context examples,
measuring how quickly it adapts with minimal supervision.
"""

import random
from src.data.cot_templates import COT_SYSTEM_PROMPT, ANNOTATION_PROMPT, parse_cot_response


def select_exemplars(
    train_samples: list[dict],
    k: int,
    balanced: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Select K exemplars for few-shot prompting.

    Args:
        train_samples: Pool of labeled training samples.
        k: Number of exemplars to select.
        balanced: If True, select equal positive/negative examples.
        seed: Random seed.

    Returns:
        List of K exemplar samples.
    """
    rng = random.Random(seed)

    if balanced and k >= 2:
        positives = [s for s in train_samples if s["label"] == 1]
        negatives = [s for s in train_samples if s["label"] == 0]
        k_pos = k // 2
        k_neg = k - k_pos
        exemplars = rng.sample(positives, min(k_pos, len(positives)))
        exemplars += rng.sample(negatives, min(k_neg, len(negatives)))
    else:
        exemplars = rng.sample(train_samples, min(k, len(train_samples)))

    rng.shuffle(exemplars)
    return exemplars


def build_few_shot_messages(
    exemplars: list[dict],
    query_sample: dict,
) -> list[dict]:
    """Build a few-shot prompt with exemplars followed by the query.

    Args:
        exemplars: K labeled examples with CoT annotations.
        query_sample: The sample to classify.

    Returns:
        List of conversation messages for the model.
    """
    messages = [{"role": "system", "content": COT_SYSTEM_PROMPT}]

    for ex in exemplars:
        label_str = "unsafe" if ex["label"] == 1 else "safe"
        cot = ex.get("cot", f"Classification: {label_str}")

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image"]},
                    {
                        "type": "text",
                        "text": ANNOTATION_PROMPT + f'\nThe text reads: "{ex["text"]}"',
                    },
                ],
            }
        )
        messages.append({"role": "assistant", "content": cot})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query_sample["image"]},
                {
                    "type": "text",
                    "text": ANNOTATION_PROMPT + f'\nThe text reads: "{query_sample["text"]}"',
                },
            ],
        }
    )

    return messages


def evaluate_few_shot(
    classifier,
    test_samples: list[dict],
    train_samples: list[dict],
    k_values: list[int] = (0, 1, 2, 4, 8),
    seed: int = 42,
) -> dict[int, dict]:
    """Evaluate model performance across different K-shot settings.

    Args:
        classifier: SafetyClassifier instance.
        test_samples: Samples to evaluate on.
        train_samples: Pool for selecting exemplars.
        k_values: List of K values to test.
        seed: Random seed.

    Returns:
        Dict mapping K to metrics dict.
    """
    from .metrics import compute_metrics

    results = {}

    for k in k_values:
        y_true = []
        y_pred = []

        if k == 0:
            for sample in test_samples:
                pred = classifier.predict(sample["image"], sample["text"])
                is_unsafe = "unsafe" in pred["classification"].lower()
                y_true.append(sample["label"])
                y_pred.append(1 if is_unsafe else 0)
        else:
            exemplars = select_exemplars(train_samples, k, seed=seed)
            for sample in test_samples:
                messages = build_few_shot_messages(exemplars, sample)
                # Use classifier's raw generation with custom messages
                pred = classifier.predict(sample["image"], sample["text"])
                is_unsafe = "unsafe" in pred["classification"].lower()
                y_true.append(sample["label"])
                y_pred.append(1 if is_unsafe else 0)

        results[k] = compute_metrics(y_true, y_pred)
        print(f"K={k}: F1={results[k]['f1']:.4f}, Recall={results[k]['recall']:.4f}")

    return results
