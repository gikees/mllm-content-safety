"""HatefulMemes dataset loader.

Loads the HatefulMemes dataset from HuggingFace (Multimodal-Fatima version
with embedded PIL images) and formats it for multimodal conversation-style
training with Qwen3.5.
"""

from datasets import load_dataset

DATASET_ID = "Multimodal-Fatima/Hatefulmemes_train"


def load_hateful_memes(split: str = "train") -> list[dict]:
    """Load HatefulMemes and format as conversation samples.

    Each sample contains:
        - image: PIL Image
        - text: overlay text from the meme
        - label: 0 (not hateful) or 1 (hateful)
        - source: dataset identifier
    """
    ds = load_dataset(DATASET_ID, split=split)
    samples = []
    for row in ds:
        sample = {
            "image": row["image"],
            "text": row.get("text", ""),
            "label": row["label"],
            "source": "hateful_memes",
        }
        samples.append(sample)
    return samples


def format_for_training(sample: dict, cot: str | None = None) -> dict:
    """Format a sample as a Qwen3.5 conversation turn.

    Args:
        sample: Raw sample from load_hateful_memes.
        cot: Optional pre-generated Chain-of-Thought annotation.

    Returns:
        Conversation dict with 'messages' key.
    """
    user_content = [
        {"type": "image", "image": sample["image"]},
        {
            "type": "text",
            "text": (
                "Analyze this meme for content safety. "
                "The text overlay reads: '{}'. "
                "Provide step-by-step reasoning and classify as safe or unsafe "
                "with a risk severity from 1 (benign) to 5 (severe)."
            ).format(sample["text"]),
        },
    ]

    if cot:
        assistant_content = cot
    else:
        label_str = "unsafe" if sample["label"] == 1 else "safe"
        assistant_content = f"Classification: {label_str}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
