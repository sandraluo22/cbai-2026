#!/bin/bash
# batch8: FRESH labels every round (no independence sentence needed), 50 rounds,
# the four clue conditions. Waits for batch7 (elephant 20r).
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch7.sh' >/dev/null; do sleep 60; done
cd runs/gossip
for V in informed_r1 informed_all misinformed_r1 misinformed_all; do
  FRESH=1 ROUNDS=50 VAR=$V OUT=fresh50/Qwen32_$V python qsg_gossip.py 2>&1 | tee -a batch8.log
done
python gossip_analyze.py fresh50 2>&1 | tee -a batch8.log
echo BATCH8_DONE
