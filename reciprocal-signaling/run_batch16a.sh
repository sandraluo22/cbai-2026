#!/bin/bash
# batch16a (.149): 3-round memory window — duel + betrayal
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 WINDOW=3 ROUNDS=20 VAR=duel SEED=$SEED OUT=window3/Qwen32_duel_s$SEED python qsg_gossip.py 2>&1 | tee -a batch16.log
  FRESH=1 EARLYSTOP=3 WINDOW=3 SWITCH_AT=10 ROUNDS=20 VAR=betrayal SEED=$SEED OUT=window3/Qwen32_betrayal_s$SEED python qsg_gossip.py 2>&1 | tee -a batch16.log
done
echo BATCH16A_DONE
