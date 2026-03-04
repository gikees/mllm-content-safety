"""Generate Chain-of-Thought annotations for training data.

Uses a larger model (or the base Qwen3.5) to generate structured CoT
annotations for each training sample, which are then used as training targets.
"""

import json
import argparse
from pathlib import Path

from src.model.classifier import SafetyClassifier
from src.data.hateful_memes import load_hateful_memes, format_for_training
from src.data.preprocessing import create_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--output", default="data/train_cot.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print("Loading model for CoT generation...")
    classifier = SafetyClassifier.from_config(args.model_config, device=args.device)

    print("Loading dataset...")
    samples = load_hateful_memes("train")
    splits = create_splits(samples)
    train_samples = splits["train"]

    if args.max_samples:
        train_samples = train_samples[: args.max_samples]

    print(f"Generating CoT for {len(train_samples)} samples...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = 0
    with open(output_path, "w") as f:
        for i, sample in enumerate(train_samples):
            try:
                pred = classifier.predict(sample["image"], sample["text"])
                cot = pred["reasoning"]
                formatted = format_for_training(sample, cot=cot)
                f.write(json.dumps(formatted) + "\n")
                annotated += 1

                if (i + 1) % 10 == 0:
                    print(f"  Annotated {i + 1}/{len(train_samples)} samples")

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                continue

    print(f"Saved {annotated} annotated samples to {output_path}")


if __name__ == "__main__":
    main()
