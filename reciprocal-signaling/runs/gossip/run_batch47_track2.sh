#!/bin/bash
# batch47 (pod D): TRACK rerun with the bookkeeping order given to NEUTRALS ONLY
# (batch46 leaked it to the towers, whose clue fidelity fell to ~0.45 — result void).
# Chained behind the subliminal run so the two don't contend for the GPU.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q HIDDEN_DONE ../subliminal/sub.log 2>/dev/null; do sleep 120; done
T=11111111111111111111; L=00000000000000000000
for SEED in 0 1; do
  TRACK=1 NAGENTS=4 MODEL=Qwen32 VAR=curve ROUNDS=20 STEPS=24 SCHED="$T;$L" SEED=$SEED \
    OUT=track_think2/Qwen32think_track2_TL_s$SEED python qsg_local_think.py 2>&1 | tee -a batch47.log
done
echo BATCH47_DONE | tee -a batch47.log
