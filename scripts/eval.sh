#!/bin/bash
set -e

ADAPTER_PATH="${1:-checkpoints/final}"

echo "Running evaluation suite..."
echo "Adapter: $ADAPTER_PATH"
echo ""

echo "=== Zero-shot (base model) ==="
python -c "
from src.model.classifier import SafetyClassifier
from src.evaluation.zero_shot import evaluate_zero_shot
from src.data.hateful_memes import load_hateful_memes

classifier = SafetyClassifier.from_config()
test = load_hateful_memes('test')[:200]
evaluate_zero_shot(classifier, test, verbose=False)
"

echo ""
echo "=== Few-shot evaluation ==="
python -c "
from src.model.classifier import SafetyClassifier
from src.evaluation.few_shot import evaluate_few_shot
from src.data.hateful_memes import load_hateful_memes

classifier = SafetyClassifier.from_config(adapter_path='$ADAPTER_PATH')
train = load_hateful_memes('train')
test = load_hateful_memes('test')[:200]
evaluate_few_shot(classifier, test, train, k_values=[0, 1, 2, 4, 8])
"

echo ""
echo "Evaluation complete!"
