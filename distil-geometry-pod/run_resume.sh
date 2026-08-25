#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export ONLY=Dirigibles,Caverns N_KEEP=10000 CKPT_EVERY=250
python src/train.py     2>&1 | tee logs/03_train.log
python src/wspace.py    2>&1 | tee logs/04_wspace.log
python src/induced.py   2>&1 | tee logs/05_induced.log
python src/s2e.py       2>&1 | tee logs/06_s2e.log
python src/compare.py   2>&1 | tee logs/07_compare.log
echo PILOT_DONE
