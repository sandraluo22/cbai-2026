#!/bin/bash
# batch28 (pod .140): waits for batch27's DONE marker, then 100/0 duel — Qwen72 + Qwen32 anchor.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH27_DONE batch27.log 2>/dev/null; do sleep 120; done
A=11111111111111111111; B=00000000000000000000
for M in Qwen72 Qwen32; do
for SEED in 0 1; do
  MODEL=$M VAR=curve FRESH=1 ROUNDS=20 STEPS=40 SCHED="$A;$B" SEED=$SEED \
    OUT=duel100/${M}_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch28.log
done
done
echo BATCH28_DONE | tee -a batch28.log
