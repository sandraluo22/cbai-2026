#!/bin/bash
# batch26b (.140): cross-model — Qwen72 + harmonized Qwen32 reference
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch25[b].sh' >/dev/null; do sleep 60; done
for M in Qwen72 Qwen32; do
for SEED in 0 1; do
  FRESH=1 ROUNDS=20 STEPS=40 VAR=curve SCHED='11110111101111011110;01000010000100001000' MODEL=$M SEED=$SEED OUT=xmodel/${M}_none_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch26.log
  FRESH=1 ROUNDS=20 STEPS=40 VAR=curve SCHED='11110111101111011110;01000010000100001000' MODEL=$M NOTES=append SEED=$SEED OUT=xmodel/${M}_notes_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch26.log
done
done
echo BATCH26B_DONE
