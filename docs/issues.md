# Issues Log

## Issue 1: `.gitignore` excluded `src/data/`
- **Status**: Fixed
- **Cause**: `.gitignore` had `data/` which matched `src/data/` too
- **Fix**: Changed to `/data/` to only match top-level data directory

## Issue 2: `ModuleNotFoundError: No module named 'src'`
- **Status**: Fixed
- **Cause**: Project not installed as package, scripts couldn't find `src` module
- **Fix**: Added `export PYTHONPATH="$PWD:$PYTHONPATH"` to shell scripts

## Issue 3: HatefulMemes dataset `KeyError: 'image'`
- **Status**: Fixed
- **Cause**: `neuralcatcher/hateful_memes` has `img` column (path string), not `image` (PIL Image)
- **Fix**: Switched to `Multimodal-Fatima/Hatefulmemes_train` which has embedded PIL images with columns: `image`, `text`, `label`

## Issue 4: MMHS150K dataset not found
- **Status**: Dropped
- **Cause**: `victorcallejasf/multimodal-hate-speech` doesn't exist on HuggingFace Hub
- **Decision**: Dropped MMHS150K for now. Using HatefulMemes (8,500 samples) with our own train/val/test splits. Can add more datasets later.

## Issue 5: conda not found on remote SSH
- **Status**: Fixed
- **Cause**: Non-interactive SSH sessions don't source shell profile
- **Fix**: Added `export PATH="$HOME/miniconda3/bin:$PATH"` to scripts

## Issue 6: torch/torchvision version mismatch
- **Status**: Fixed
- **Cause**: Conda installed torch 2.10+cu128 but torchvision 0.20.1 (for torch 2.5.x)
- **Fix**: `pip install "torchvision>=0.25" --index-url https://download.pytorch.org/whl/cu128` → installed 0.25.0+cu128

## Issue 7: Qwen3.5 model rejects vision inputs (`pixel_values`, `image_grid_thw`)
- **Status**: Fixed
- **Cause**: Used `AutoModelForCausalLM` (text-only). Qwen3.5 architecture is `Qwen3_5ForConditionalGeneration`, needs `AutoModelForImageTextToText`.
- **Fix**: Replaced `AutoModelForCausalLM` → `AutoModelForImageTextToText` in classifier.py and finetune.py. Also `torch_dtype` → `dtype` (deprecated param).

## Issue 8: transformers too old for Qwen3.5
- **Status**: Fixed
- **Cause**: transformers 4.57.2 didn't recognize `qwen3_5` model type
- **Fix**: `pip install --upgrade transformers` → 5.3.0
