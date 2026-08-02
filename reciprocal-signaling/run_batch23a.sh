#!/bin/bash
# batch23a (.149): duel cells — justification channel, then influence self-report
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=duel JUSTIFY=1 SEED=$SEED OUT=justify/Qwen32_duel_justify_s$SEED python qsg_gossip.py 2>&1 | tee -a batch23.log
done
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=duel WHOINF=1 SEED=$SEED OUT=whoinf/Qwen32_duel_whoinf_s$SEED python qsg_gossip.py 2>&1 | tee -a batch23.log
done
echo BATCH23A_DONE
