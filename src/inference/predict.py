"""Single-image inference script.

Usage:
    python -m src.inference.predict --image path/to/image.jpg --text "meme text"
    python -m src.inference.predict --image path/to/image.jpg --text "meme text" --adapter checkpoints/final
"""

import argparse
from pathlib import Path

from PIL import Image

from src.model.classifier import SafetyClassifier
from src.model.risk_ranker import RiskRanker


def main():
    parser = argparse.ArgumentParser(description="Run content safety inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--text", type=str, default="", help="Text overlay or caption")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    print("Loading model...")
    classifier = SafetyClassifier.from_config(
        args.model_config,
        adapter_path=args.adapter,
        device=args.device,
    )

    print("Running inference...")
    prediction = classifier.predict(image, args.text)

    ranker = RiskRanker()
    risk = ranker.rank(prediction)

    print(f"\n{'=' * 50}")
    print(f" Content Safety Analysis")
    print(f"{'=' * 50}")
    print(f"  Classification: {risk.classification}")
    print(f"  Risk Severity:  {risk.severity}/5")
    print(f"  Confidence:     {risk.confidence:.2f}")
    print(f"  Action:         {risk.action}")
    print(f"\n  Reasoning:")
    print(f"  {prediction['reasoning']}")
    print()


if __name__ == "__main__":
    main()
