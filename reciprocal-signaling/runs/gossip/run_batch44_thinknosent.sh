#!/bin/bash
# batch44 (pod A): Qwen thinking duel WITHOUT the independence sentence (FRESH=1 kills INDEP
# in the imported prompt builder). Tests whether the sentence caused the think-duel null.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
T=11111111111111111111; L=00000000000000000000
FRESH=1 NAGENTS=3 MODEL=Qwen32 VAR=curve ROUNDS=20 STEPS=12 SCHED="$T;$L" SEED=0 \
  OUT=duel_think/Qwen32think_nosent_n1_s0 python qsg_local_think.py 2>&1 | tee -a batch44.log
FRESH=1 NAGENTS=5 MODEL=Qwen32 VAR=curve ROUNDS=20 STEPS=40 SCHED="$T;$L" SEED=0 \
  OUT=duel_think/Qwen32think_nosent_n3_s0 python qsg_local_think.py 2>&1 | tee -a batch44.log
echo BATCH44_DONE | tee -a batch44.log
