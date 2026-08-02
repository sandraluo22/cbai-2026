#!/bin/bash
# NAMES variation of the gossip game: waits for batch4 (number quizzes) to finish.
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch4.sh' >/dev/null; do sleep 60; done
cd runs/gossip
for V in none informed_r1 informed_all misinformed_r1 misinformed_all; do
  NAMES=1 VAR=$V python qsg_gossip.py 2>&1 | tee -a gossip_names.log
done
python gossip_analyze.py . 2>&1 | tee -a gossip_names.log
echo BATCH5_DONE
