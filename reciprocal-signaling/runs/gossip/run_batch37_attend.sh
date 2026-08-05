#!/bin/bash
# batch37 (pod C): dyad + explicit source-attention instruction (ATTEND=1).
# 1 perfect liar (or truthteller) + 1 neutral; does the instruction flip 20/20 capture?
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH36_DONE batch36.log 2>/dev/null; do sleep 120; done
T=11111111111111111111; L=00000000000000000000
for SEED in 0 1; do
  ATTEND=1 NAGENTS=2 MODEL=Qwen32 VAR=curve FRESH=1 ROUNDS=20 STEPS=8 SCHED="$L" SEED=$SEED \
    OUT=attend/Qwen32_L_n1_attend_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch37.log
  ATTEND=1 NAGENTS=2 MODEL=Qwen32 VAR=curve FRESH=1 ROUNDS=20 STEPS=8 SCHED="$T" SEED=$SEED \
    OUT=attend/Qwen32_T_n1_attend_s$SEED python qsg_gossip_fast.py 2>&1 | tee -a batch37.log
done
echo BATCH37_DONE | tee -a batch37.log
