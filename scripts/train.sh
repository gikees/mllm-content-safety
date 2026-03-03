#!/bin/bash
set -e

echo "Starting LoRA fine-tuning..."
python -m src.training.finetune \
    --model-config configs/model.yaml \
    --train-config configs/train.yaml \
    --data-path data/train_cot.jsonl

echo "Training complete!"
