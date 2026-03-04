"""Download and prepare datasets for training.

Downloads HatefulMemes from HuggingFace, creates train/val/test splits,
and saves metadata to the data/ directory.
"""

import json
from pathlib import Path

from src.data.hateful_memes import load_hateful_memes
from src.data.preprocessing import create_splits, print_split_stats


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Downloading HatefulMemes...")
    samples = load_hateful_memes("train")
    print(f"  Loaded {len(samples)} samples")

    print("\nCreating splits...")
    splits = create_splits(samples)
    print_split_stats(splits)

    for split_name, split_samples in splits.items():
        out_path = data_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for s in split_samples:
                record = {
                    "text": s["text"],
                    "label": s["label"],
                    "source": s["source"],
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Saved {len(split_samples)} samples to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
