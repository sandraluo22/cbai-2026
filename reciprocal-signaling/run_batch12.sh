#!/bin/bash
# batch12: task-switch arms — 10 naming rounds then 15 A/B category rounds.
# switch_duel (both towers), switch_informed / switch_misinformed (tower + seeded
# dissenting neutral in each phase-2 round's memory). Waits for batch11.
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch11.sh' >/dev/null; do sleep 60; done
for SEED in 0 1; do
for V in switch_duel switch_informed switch_misinformed; do
  FRESH=1 EARLYSTOP=3 SWITCH_AT=10 ROUNDS=25 VAR=$V SEED=$SEED OUT=switchtask/Qwen32_${V}_s$SEED python qsg_gossip.py 2>&1 | tee -a batch12.log
done
done
echo BATCH12_DONE
