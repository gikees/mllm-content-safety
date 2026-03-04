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
