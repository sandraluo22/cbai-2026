#!/bin/bash
# batch27: fixed speaker order (deterministic round-robin pairs), Qwen32, 80/20 pair, 3 seeds.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1 2; do
  MODEL=Qwen32 VAR=curve ORDER=fixed FRESH=1 ROUNDS=20 STEPS=40 \
    SCHED="11110111101111011110;01000010000100001000" SEED=$SEED \
    OUT=order/Qwen32_fixed_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch27.log
done
echo BATCH27_DONE | tee -a batch27.log
