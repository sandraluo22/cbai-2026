#!/bin/bash
# batch9: 3-agent duel microscope — P1 truth clue, P2 wrong clue, P3 neutral.
# Fresh labels, 30 rounds. Waits for batch8.
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch8.sh' >/dev/null; do sleep 60; done
for SEED in 0 1 2; do
  FRESH=1 NAGENTS=3 STEPS=60 ROUNDS=30 VAR=duel SEED=$SEED OUT=duel3/Qwen32_s$SEED python qsg_gossip.py 2>&1 | tee -a batch9.log
done
python gossip_analyze.py duel3 2>&1 | tee -a batch9.log
echo BATCH9_DONE
