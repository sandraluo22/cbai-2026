#!/bin/bash
# batch46 (pod D): (1) attention analysis at the answer position; (2) THINKING duel with the
# explicit per-player bookkeeping order (TRACK=1) — strongest prompt intervention yet.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python mech_attention.py 2>&1 | tee -a attn.log
T=11111111111111111111; L=00000000000000000000
for SEED in 0 1; do
  TRACK=1 NAGENTS=4 MODEL=Qwen32 VAR=curve ROUNDS=20 STEPS=24 SCHED="$T;$L" SEED=$SEED \
    OUT=track_think/Qwen32think_track_TL_s$SEED python qsg_local_think.py 2>&1 | tee -a batch46.log
done
echo BATCH46_DONE | tee -a batch46.log
