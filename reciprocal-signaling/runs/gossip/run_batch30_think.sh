#!/bin/bash
# batch30 (pod port 12272): Qwen3-32B native thinking, 80/20 frame, no scratchpad, 2 seeds.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
A=11110111101111011110; B=01000010000100001000
for SEED in 0 1; do
  MODEL=Qwen32 VAR=curve SCHED="$A;$B" ROUNDS=20 STEPS=40 SEED=$SEED \
    OUT=xmodel_think/Qwen32_think_s$SEED python qsg_local_think.py 2>&1 | tee -a batch30.log
done
echo BATCH30_DONE | tee -a batch30.log
