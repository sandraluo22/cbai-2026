#!/bin/bash
# batch15a (.149): polarization pairs + graded 80/20, 80/60; then counterfactual removal replay
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
  FRESH=1 EARLYSTOP=3 ROUNDS=30 VAR=pair_tt SEED=$SEED OUT=polar/Qwen32_pair_tt_s$SEED python qsg_gossip.py 2>&1 | tee -a batch15.log
  FRESH=1 EARLYSTOP=3 ROUNDS=30 VAR=pair_ll SEED=$SEED OUT=polar/Qwen32_pair_ll_s$SEED python qsg_gossip.py 2>&1 | tee -a batch15.log
  FRESH=1 EARLYSTOP=3 ROUNDS=30 VAR=graded P1REL=0.8 P2REL=0.2 SEED=$SEED OUT=polar/Qwen32_g8020_s$SEED python qsg_gossip.py 2>&1 | tee -a batch15.log
  FRESH=1 EARLYSTOP=3 ROUNDS=30 VAR=graded P1REL=0.8 P2REL=0.6 SEED=$SEED OUT=polar/Qwen32_g8060_s$SEED python qsg_gossip.py 2>&1 | tee -a batch15.log
done
python cf_removal.py earlystop/Qwen32_duel_s0/gossip_s0_transcript.jsonl earlystop/Qwen32_duel_s1/gossip_s1_transcript.jsonl 2>&1 | tee -a batch15.log
echo BATCH15A_DONE
