#!/bin/bash
# batch33 (pod .149): after Llama duels — Qwen32 minimal N=3 isolation:
# 2 neutrals + 1 perfect truthteller (SCHED all-1) or 1 perfect liar (all-0).
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH29_DONE batch29.log 2>/dev/null; do sleep 120; done
T=11111111111111111111; L=00000000000000000000
for SEED in 0 1 2; do
  MODEL=Qwen32 VAR=curve NAGENTS=3 FRESH=1 ROUNDS=20 STEPS=24 SCHED="$T" SEED=$SEED \
    OUT=n3/Qwen32_tower100_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch33.log
  MODEL=Qwen32 VAR=curve NAGENTS=3 FRESH=1 ROUNDS=20 STEPS=24 SCHED="$L" SEED=$SEED \
    OUT=n3/Qwen32_tower0_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch33.log
done
echo BATCH33_DONE | tee -a batch33.log
