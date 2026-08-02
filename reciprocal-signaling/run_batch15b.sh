#!/bin/bash
# batch15b (.140): graded 60/40 + delayed + reversal + natural-domain task switch
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 ROUNDS=30 VAR=graded P1REL=0.6 P2REL=0.4 SEED=$SEED OUT=polar/Qwen32_g6040_s$SEED python qsg_gossip.py 2>&1 | tee -a batch15.log
  FRESH=1 EARLYSTOP=3 ROUNDS=20 SWITCH_AT=10 VAR=delayed SEED=$SEED OUT=polar/Qwen32_delayed_s$SEED python qsg_gossip.py 2>&1 | tee -a batch15.log
  FRESH=1 EARLYSTOP=3 ROUNDS=30 SWITCH_AT=10 VAR=reversal SEED=$SEED OUT=polar/Qwen32_reversal_s$SEED python qsg_gossip.py 2>&1 | tee -a batch15.log
  FRESH=1 EARLYSTOP=3 ROUNDS=25 SWITCH_AT=10 VAR=switch_natural SEED=$SEED OUT=polar/Qwen32_natural_s$SEED python qsg_gossip.py 2>&1 | tee -a batch15.log
done
echo BATCH15B_DONE
