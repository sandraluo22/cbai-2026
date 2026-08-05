#!/bin/bash
# batch35 (pod .149): group-size sweep — 1 perfect truthteller (or liar) + n neutrals,
# n = 1,3,4,5 (n=2 covered by batch33). STEPS = 4 sweeps of ordered pairs.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH33_DONE batch33.log 2>/dev/null; do sleep 120; done
T=11111111111111111111; L=00000000000000000000
for NA in 2 4 5 6; do
  STEPS=$((4 * NA * (NA - 1)))
  for SEED in 0 1; do
    MODEL=Qwen32 VAR=curve NAGENTS=$NA FRESH=1 ROUNDS=20 STEPS=$STEPS SCHED="$T" SEED=$SEED \
      OUT=nsweep/Qwen32_T_n$((NA-1))_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch35.log
    MODEL=Qwen32 VAR=curve NAGENTS=$NA FRESH=1 ROUNDS=20 STEPS=$STEPS SCHED="$L" SEED=$SEED \
      OUT=nsweep/Qwen32_L_n$((NA-1))_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch35.log
  done
done
echo BATCH35_DONE | tee -a batch35.log
