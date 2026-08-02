#!/bin/bash
# batch17a (.149): notes x graded 80/20 — can an explicit state enable noisy-evidence integration?
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
for NM in update append; do
  FRESH=1 EARLYSTOP=3 ROUNDS=30 VAR=graded P1REL=0.8 P2REL=0.2 NOTES=$NM SEED=$SEED OUT=notes/Qwen32_g8020_${NM}_s$SEED python qsg_gossip.py 2>&1 | tee -a batch17.log
done
done
echo BATCH17A_DONE
