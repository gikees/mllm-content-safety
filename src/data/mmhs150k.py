"""MMHS150K dataset loader.

Loads the Multimodal Hate Speech 150K dataset and formats it
for multimodal conversation-style training with Qwen3.5.
"""

from datasets import load_dataset

DATASET_ID = "victorcallejasf/multimodal-hate-speech"

CATEGORY_MAP = {
    0: "not_hateful",
    1: "racist",
    2: "sexist",
    3: "homophobic",
    4: "religion",
    5: "other_hate",
}


def load_mmhs150k(split: str = "train") -> list[dict]:
    """Load MMHS150K and format as conversation samples.

    Each sample contains:
        - image: PIL Image
        - text: tweet text
        - label: 0 (not hateful) or 1 (hateful)
        - category: fine-grained hate category
        - source: dataset identifier
    """
    ds = load_dataset(DATASET_ID, split=split)
    samples = []
    for row in ds:
        label_id = row.get("label", 0)
        sample = {
            "image": row["image"],
            "text": row.get("tweet_text", row.get("text", "")),
            "label": 0 if label_id == 0 else 1,
            "category": CATEGORY_MAP.get(label_id, "unknown"),
            "source": "mmhs150k",
        }
        samples.append(sample)
    return samples


def format_for_training(sample: dict, cot: str | None = None) -> dict:
    """Format a sample as a Qwen3.5 conversation turn."""
    user_content = [
        {"type": "image", "image": sample["image"]},
        {
            "type": "text",
            "text": (
                "Analyze this social media post for content safety. "
                "The accompanying text reads: '{}'. "
                "Provide step-by-step reasoning and classify as safe or unsafe "
                "with a risk severity from 1 (benign) to 5 (severe)."
            ).format(sample["text"]),
        },
    ]

    if cot:
        assistant_content = cot
    else:
        label_str = "unsafe" if sample["label"] == 1 else "safe"
        category_info = f" ({sample['category']})" if sample["label"] == 1 else ""
        assistant_content = f"Classification: {label_str}{category_info}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
