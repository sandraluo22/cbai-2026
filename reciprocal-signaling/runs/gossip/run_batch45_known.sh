#!/bin/bash
# batch45 (pod D, 213.181.104.59): Qwen32 THINKING + "someone knows the answer" hint.
# (a) 1 truthteller + 2 neutrals; (b) 1 truthteller + 1 liar + 2 neutrals.
# Matched to batch41 protocol (no FRESH env -> independence sentence present, as in baselines).
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
T=11111111111111111111; L=00000000000000000000
for SEED in 0 1; do
  KNOWN=1 NAGENTS=3 MODEL=Qwen32 VAR=curve ROUNDS=20 STEPS=12 SCHED="$T" SEED=$SEED \
    OUT=known_think/Qwen32think_known_T_s$SEED python qsg_local_think.py 2>&1 | tee -a batch45.log
  KNOWN=1 NAGENTS=4 MODEL=Qwen32 VAR=curve ROUNDS=20 STEPS=24 SCHED="$T;$L" SEED=$SEED \
    OUT=known_think/Qwen32think_known_TL_s$SEED python qsg_local_think.py 2>&1 | tee -a batch45.log
done
echo BATCH45_DONE | tee -a batch45.log
