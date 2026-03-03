#!/bin/bash
# Quick dry run to catch bugs before the full pipeline
# Runs the entire pipeline on ~10 samples
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

echo "========================================="
echo " DRY RUN - Smoke test (10 samples)"
echo "========================================="

cd ~/mllm-content-safety

eval "$(conda shell.bash hook)"
conda activate mllm-content-safety

# 1. Download data
echo "[1/4] Downloading datasets..."
python scripts/download_data.py

# 2. Generate CoT on 10 samples
echo "[2/4] Generating CoT (10 samples)..."
python scripts/generate_cot.py --max-samples 10

# 3. Train for 1 step
echo "[3/4] Training (1 step sanity check)..."
python -c "
from src.training.finetune import load_configs, setup_model, load_training_data
from trl import SFTTrainer, SFTConfig

model_cfg, train_cfg = load_configs()
model, processor = setup_model(model_cfg)
dataset = load_training_data('data/train_cot.jsonl')

args = SFTConfig(
    output_dir='checkpoints/dry_run',
    max_steps=1,
    per_device_train_batch_size=1,
    max_seq_length=512,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=1,
    report_to='none',
)

trainer = SFTTrainer(model=model, processing_class=processor, train_dataset=dataset, args=args)
trainer.train()
print('Training step OK')
trainer.save_model('checkpoints/dry_run')
print('Save OK')
"

# 4. Inference on one sample
echo "[4/4] Testing inference..."
python -c "
from src.model.classifier import SafetyClassifier
from src.model.risk_ranker import RiskRanker
from PIL import Image
import json

# Load first sample to get an image
with open('data/train.jsonl') as f:
    sample = json.loads(f.readline())

classifier = SafetyClassifier.from_config(adapter_path='checkpoints/dry_run')
# Test with a dummy image since jsonl doesn't store images directly
from datasets import load_dataset
ds = load_dataset('neuralcatcher/hateful_memes', split='train[:1]')
pred = classifier.predict(ds[0]['image'], ds[0].get('text', 'test'))

ranker = RiskRanker()
result = ranker.rank(pred)

print(f'Classification: {result.classification}')
print(f'Severity: {result.severity}/5')
print(f'Action: {result.action}')
print(f'Reasoning: {pred[\"reasoning\"][:200]}...')
"

echo ""
echo "========================================="
echo " DRY RUN PASSED"
echo "========================================="
