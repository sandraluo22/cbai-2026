#!/bin/bash
# batch34 (pod .140): dense-frame duel control — historic record density (75 steps, 30 rounds).
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH32_DONE batch32.log 2>/dev/null; do sleep 180; done
T=111111111111111111111111111111; L=000000000000000000000000000000
for SEED in 0 1; do
  MODEL=Qwen32 VAR=curve FRESH=1 ROUNDS=30 STEPS=75 SCHED="$T;$L" SEED=$SEED \
    OUT=dense/Qwen32_duel100dense_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch34.log
done
echo BATCH34_DONE | tee -a batch34.log
