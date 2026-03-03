#!/bin/bash
# Full pipeline for remote GPU server
# Usage: ssh egor@69.30.0.75 'bash -s' < scripts/remote_setup.sh
#    OR: scp to remote, then run: bash remote_setup.sh
set -e

# Use GPU 1 (GPU 0 is occupied)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

echo "========================================="
echo " MLLM Content Safety - Remote Setup"
echo "========================================="

# --- 1. Clone repo ---
if [ ! -d "mllm-content-safety" ]; then
    echo "[1/6] Cloning repo..."
    git clone https://github.com/gikees/mllm-content-safety.git
else
    echo "[1/6] Repo already exists, pulling latest..."
    cd mllm-content-safety && git pull && cd ..
fi
cd mllm-content-safety

# --- 2. Setup conda environment ---
echo "[2/6] Setting up environment..."
if ! conda env list | grep -q "mllm-content-safety"; then
    conda create -n mllm-content-safety python=3.11 -y
fi

export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate mllm-content-safety

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets peft accelerate trl \
    pillow pyyaml scikit-learn wandb ruff

# Try installing unsloth (optional, may fail on some setups)
pip install unsloth 2>/dev/null || echo "Warning: unsloth install failed, using standard PEFT"

# --- 3. Download data ---
echo "[3/6] Downloading datasets..."
mkdir -p data
python scripts/download_data.py

# --- 4. Generate CoT annotations ---
echo "[4/6] Generating CoT annotations..."
python scripts/generate_cot.py --max-samples 5000

# --- 5. Train ---
echo "[5/6] Starting LoRA fine-tuning..."
python -m src.training.finetune \
    --model-config configs/model.yaml \
    --train-config configs/train.yaml \
    --data-path data/train_cot.jsonl

# --- 6. Evaluate ---
echo "[6/6] Running evaluation..."
bash scripts/eval.sh checkpoints/final

echo ""
echo "========================================="
echo " Pipeline complete!"
echo "========================================="
echo "Checkpoints saved to: checkpoints/final"
echo "Run inference: python -m src.inference.predict --image <path> --text <text> --adapter checkpoints/final"
