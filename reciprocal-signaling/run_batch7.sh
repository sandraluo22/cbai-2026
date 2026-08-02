#!/bin/bash
# 20-round elephant runs, all quizzes, both seeds; waits for batch6 (gossip) first.
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch6.sh' >/dev/null; do sleep 60; done
for MODEL in Qwen32 Llama70 Qwen72; do
  MODEL=$MODEL ROUNDS=20 SEEDS=2 OUT=runs/elephant_r20/$MODEL python src/elephant.py 2>&1 | tee -a batch7.log
done
echo BATCH7_DONE
