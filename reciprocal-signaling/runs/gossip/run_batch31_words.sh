#!/bin/bash
# batch31 (pod .140): after batch27b — word labels, Qwen32: 100/0 duel + 80/20, 2 seeds each.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH27B_DONE batch27b.log 2>/dev/null; do sleep 120; done
D=11111111111111111111; Z=00000000000000000000
A=11110111101111011110; B=01000010000100001000
for SEED in 0 1; do
  WORDS=1 MODEL=Qwen32 VAR=curve FRESH=1 ROUNDS=20 STEPS=40 SCHED="$D;$Z" SEED=$SEED \
    OUT=words/Qwen32_duel100w_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch31.log
  WORDS=1 MODEL=Qwen32 VAR=curve FRESH=1 ROUNDS=20 STEPS=40 SCHED="$A;$B" SEED=$SEED \
    OUT=words/Qwen32_8020w_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch31.log
done
echo BATCH31_DONE | tee -a batch31.log
