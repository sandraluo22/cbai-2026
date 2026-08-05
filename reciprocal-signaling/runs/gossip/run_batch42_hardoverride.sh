#!/bin/bash
# batch42 (pod A): hard-item override — trusted tower pushes WRONG answers on the hard bank,
# discredited liar pushes correct. Isolates reputation from plausibility on uncertain items.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q SOLO_BASELINE_DONE solo_baseline.log 2>/dev/null; do sleep 180; done
for SEED in 0 1; do
  HARD=1 MODEL=Qwen32 VAR=trivia_override FRESH=1 ROUNDS=20 STEPS=40 SWITCH_AT=10 SEED=$SEED \
    OUT=transfer/Qwen32_hardoverride_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch42.log
done
echo BATCH42_DONE | tee -a batch42.log
