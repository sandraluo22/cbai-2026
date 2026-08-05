#!/bin/bash
# batch36 (pod C): dyad with NATIVE THINKING — 1 perfect liar (or truthteller) + 1 neutral,
# Qwen3-32B enable_thinking, generative emissions+probes. Chained behind mech head sweep.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q MECH_HEADS_DONE mech_heads.log 2>/dev/null; do sleep 120; done
T=11111111111111111111; L=00000000000000000000
for SEED in 0 1; do
  NAGENTS=2 MODEL=Qwen32 VAR=curve ROUNDS=20 STEPS=8 SCHED="$L" SEED=$SEED \
    OUT=nsweep_think/Qwen32think_L_n1_s$SEED python qsg_local_think.py 2>&1 | tee -a batch36.log
  NAGENTS=2 MODEL=Qwen32 VAR=curve ROUNDS=20 STEPS=8 SCHED="$T" SEED=$SEED \
    OUT=nsweep_think/Qwen32think_T_n1_s$SEED python qsg_local_think.py 2>&1 | tee -a batch36.log
done
echo BATCH36_DONE | tee -a batch36.log
