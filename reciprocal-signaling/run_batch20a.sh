#!/bin/bash
# batch20a (.149): decision-time reliability assessment spliced into probe reads
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 ROUNDS=30 VAR=graded P1REL=0.8 P2REL=0.2 DECIDE=1 SEED=$SEED OUT=decide/Qwen32_g8020_decide_s$SEED python qsg_gossip.py 2>&1 | tee -a batch20.log
  FRESH=1 EARLYSTOP=3 WINDOW=3 SWITCH_AT=10 ROUNDS=20 VAR=betrayal DECIDE=1 SEED=$SEED OUT=decide/Qwen32_betrayal_w3_decide_s$SEED python qsg_gossip.py 2>&1 | tee -a batch20.log
done
echo BATCH20A_DONE
