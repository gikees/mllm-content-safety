"""Generate Chain-of-Thought annotations for training data.

Uses a larger model (or the base Qwen3.5) to generate structured CoT
annotations for each training sample, which are then used as training targets.
"""

import json
import argparse
from pathlib import Path

from src.model.classifier import SafetyClassifier
from src.data.hateful_memes import load_hateful_memes, format_for_training
from src.data.mmhs150k import load_mmhs150k, format_for_training as format_mmhs


def generate_cot_annotations(
    classifier: SafetyClassifier,
    samples: list[dict],
    format_fn,
    output_path: str,
    max_samples: int | None = None,
):
    """Generate CoT annotations and save formatted training data.

    Args:
        classifier: Model to generate CoT annotations.
        samples: Raw samples to annotate.
        format_fn: Dataset-specific formatting function.
        output_path: Path to save JSONL output.
        max_samples: Optional limit on number of samples to annotate.
    """
    if max_samples:
        samples = samples[:max_samples]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = 0
    with open(output_path, "w") as f:
        for i, sample in enumerate(samples):
            try:
                pred = classifier.predict(sample["image"], sample["text"])
                cot = pred["reasoning"]
                formatted = format_fn(sample, cot=cot)
                f.write(json.dumps(formatted) + "\n")
                annotated += 1

                if (i + 1) % 100 == 0:
                    print(f"  Annotated {i + 1}/{len(samples)} samples")

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                continue

    print(f"Saved {annotated} annotated samples to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--output", default="data/train_cot.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("Loading model for CoT generation...")
    classifier = SafetyClassifier.from_config(args.model_config, device=args.device)

    print("Loading datasets...")
    hm_samples = load_hateful_memes("train")
    mmhs_samples = load_mmhs150k("train")

    all_samples = hm_samples + mmhs_samples
    print(f"Total samples to annotate: {len(all_samples)}")

    def format_fn(sample, cot=None):
        if sample["source"] == "hateful_memes":
            return format_for_training(sample, cot=cot)
        return format_mmhs(sample, cot=cot)

    generate_cot_annotations(
        classifier,
        all_samples,
        format_fn,
        args.output,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
