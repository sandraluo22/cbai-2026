#!/bin/bash
# batch17b (.140): notes x betrayal+window3 — can a note carry trust/distrust past the memory boundary?
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
for NM in update append; do
  FRESH=1 EARLYSTOP=3 WINDOW=3 SWITCH_AT=10 ROUNDS=20 VAR=betrayal NOTES=$NM SEED=$SEED OUT=notes/Qwen32_betrayal_w3_${NM}_s$SEED python qsg_gossip.py 2>&1 | tee -a batch17.log
done
done
echo BATCH17B_DONE
