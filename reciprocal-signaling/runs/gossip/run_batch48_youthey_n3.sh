#!/bin/bash
# batch48 (pod D): you/they format on the N=3 ISOLATION cells (1 tower + 2 neutrals),
# matched to batch33 (STEPS=24). Old-format baselines: tower100 0.88, tower0 0.98.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
T=11111111111111111111; L=00000000000000000000
for SEED in 0 1 2; do
  PRESENT=youthey MODEL=Qwen32 VAR=curve NAGENTS=3 FRESH=1 ROUNDS=20 STEPS=24 SCHED="$T" \
    SEED=$SEED OUT=youthey/Qwen32_tower100_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch48.log
  PRESENT=youthey MODEL=Qwen32 VAR=curve NAGENTS=3 FRESH=1 ROUNDS=20 STEPS=24 SCHED="$L" \
    SEED=$SEED OUT=youthey/Qwen32_tower0_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch48.log
done
echo BATCH48_DONE | tee -a batch48.log
