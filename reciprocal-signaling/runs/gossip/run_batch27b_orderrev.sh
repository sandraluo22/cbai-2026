#!/bin/bash
# batch27b (pod .140): after batch28, fixed order with the WRONG tower (P2) speaking first.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH28_DONE batch28.log 2>/dev/null; do sleep 120; done
for SEED in 0 1 2; do
  MODEL=Qwen32 VAR=curve ORDER=fixedrev FRESH=1 ROUNDS=20 STEPS=40 \
    SCHED="11110111101111011110;01000010000100001000" SEED=$SEED \
    OUT=order/Qwen32_fixedrev_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch27b.log
done
echo BATCH27B_DONE | tee -a batch27b.log
