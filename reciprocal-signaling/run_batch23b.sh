#!/bin/bash
# batch23b (.140): misinformed cells — justification channel, then influence self-report
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=misinformed_all JUSTIFY=1 SEED=$SEED OUT=justify/Qwen32_misinf_justify_s$SEED python qsg_gossip.py 2>&1 | tee -a batch23.log
done
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 ROUNDS=20 VAR=misinformed_all WHOINF=1 SEED=$SEED OUT=whoinf/Qwen32_misinf_whoinf_s$SEED python qsg_gossip.py 2>&1 | tee -a batch23.log
done
echo BATCH23B_DONE
