#!/bin/bash
# New elephant-game conditions: number quizzes + poisoned-clue variants.
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
NEW=n30,n17,n64,n36,elephant_wrong,piano_wrong,n30_wrong
for MODEL in Qwen32 Llama70 Qwen72; do
  MODEL=$MODEL QUIZ=$NEW ROUNDS=5 SEEDS=2 python src/elephant.py 2>&1 | tee -a batch4.log
done
echo BATCH4_DONE
