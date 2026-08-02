#!/bin/bash
# batch22a (.149): per-CONVERSATION freeform note (single running note, replaced each conversation)
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=graded P1REL=0.8 P2REL=0.2 NOTES=convfree SEED=$SEED OUT=convnotes/Qwen32_g8020_convfree_s$SEED python qsg_gossip.py 2>&1 | tee -a batch22.log
  FRESH=1 EARLYSTOP=3 WINDOW=3 SWITCH_AT=10 ROUNDS=20 VAR=betrayal NOTES=convfree SEED=$SEED OUT=convnotes/Qwen32_betrayal_w3_convfree_s$SEED python qsg_gossip.py 2>&1 | tee -a batch22.log
done
echo BATCH22A_DONE
