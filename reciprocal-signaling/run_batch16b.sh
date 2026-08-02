#!/bin/bash
# batch16b (.140): 3-round memory window — alternator (5-round blocks)
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 WINDOW=3 ALT_PERIOD=5 ROUNDS=30 VAR=alternator SEED=$SEED OUT=window3/Qwen32_alternator_s$SEED python qsg_gossip.py 2>&1 | tee -a batch16.log
done
echo BATCH16B_DONE
