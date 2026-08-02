#!/bin/bash
# batch6: (A) no-sentence arm 5r, (B) 10-round P-label runs (all vars + duel),
# (C) 10-round names runs (informed_all, misinformed_all)
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for V in none informed_all misinformed_all; do
  SENT=0 VAR=$V OUT=nosent/Qwen32_$V python qsg_gossip.py 2>&1 | tee -a batch6.log
done
for V in none informed_r1 informed_all misinformed_r1 misinformed_all duel; do
  ROUNDS=10 VAR=$V OUT=r10/Qwen32_$V python qsg_gossip.py 2>&1 | tee -a batch6.log
done
for V in informed_all misinformed_all; do
  ROUNDS=10 NAMES=1 VAR=$V OUT=r10names/Qwen32names_$V python qsg_gossip.py 2>&1 | tee -a batch6.log
done
python gossip_analyze.py nosent 2>&1 | tee -a batch6.log
python gossip_analyze.py r10 2>&1 | tee -a batch6.log
python gossip_analyze.py r10names 2>&1 | tee -a batch6.log
echo BATCH6_DONE
