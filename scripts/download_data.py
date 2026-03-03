"""Download and prepare datasets for training.

Downloads HatefulMemes and MMHS150K from HuggingFace,
formats them, and saves to the data/ directory.
"""

import json
from pathlib import Path

from src.data.hateful_memes import load_hateful_memes
from src.data.mmhs150k import load_mmhs150k
from src.data.preprocessing import create_splits, print_split_stats


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Downloading HatefulMemes...")
    hm_samples = load_hateful_memes("train")
    print(f"  Loaded {len(hm_samples)} samples")

    print("Downloading MMHS150K...")
    mmhs_samples = load_mmhs150k("train")
    print(f"  Loaded {len(mmhs_samples)} samples")

    all_samples = hm_samples + mmhs_samples
    print(f"\nTotal samples: {len(all_samples)}")

    print("\nCreating splits...")
    splits = create_splits(
        all_samples,
        holdout_categories=["disability", "nationality"],
    )
    print_split_stats(splits)

    for split_name, samples in splits.items():
        out_path = data_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for s in samples:
                record = {
                    "text": s["text"],
                    "label": s["label"],
                    "category": s.get("category", ""),
                    "source": s["source"],
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Saved {len(samples)} samples to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
