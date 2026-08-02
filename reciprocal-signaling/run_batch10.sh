#!/bin/bash
# batch10: fresh labels + EARLY-STOP rounds (end after 3 unanimous conversations).
# 5 players, 50 rounds: 1 truth-teller / 1 liar / 1 of each. 2 seeds.
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
for V in informed_all misinformed_all duel; do
  FRESH=1 EARLYSTOP=3 ROUNDS=50 VAR=$V SEED=$SEED OUT=earlystop/Qwen32_${V}_s$SEED python qsg_gossip.py 2>&1 | tee -a batch10.log
done
done
echo BATCH10_DONE
