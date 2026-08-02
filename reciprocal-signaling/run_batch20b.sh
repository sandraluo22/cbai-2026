#!/bin/bash
# batch20b (.140): PUBLIC one-sentence remarks in the elephant game
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
MODEL=Qwen72 NOTES=public QUIZ=elephant,penguin,shadow,n30,n17 ROUNDS=20 SEEDS=2 OUT=runs/elephant_public/Qwen72 python src/elephant.py 2>&1 | tee -a batch20.log
MODEL=Qwen32 NOTES=public QUIZ=elephant,penguin,shadow,n30,n17 ROUNDS=20 SEEDS=2 OUT=runs/elephant_public/Qwen32 python src/elephant.py 2>&1 | tee -a batch20.log
echo BATCH20B_DONE
