#!/bin/bash
# batch11: duel_remove (liar vanishes after r10) + betrayal (truth-teller lies from r11).
# Fresh labels, early-stop (neutrals-only unanimity), 30 rounds, 2 seeds.
set -x
cd "$(dirname "$0")/runs/gossip"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for SEED in 0 1; do
for V in duel_remove betrayal; do
  FRESH=1 EARLYSTOP=3 SWITCH_AT=10 ROUNDS=30 VAR=$V SEED=$SEED OUT=switch/Qwen32_${V}_s$SEED python qsg_gossip.py 2>&1 | tee -a batch11.log
done
done
echo BATCH11_DONE
