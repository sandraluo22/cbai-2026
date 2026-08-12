#!/bin/bash
# Decisive: no-record context with the speaker order SWAPPED. Does the credulity push
# follow the designated source, or simply whoever is mentioned first?
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 LOAD8=1
SWAPORDER=1 LAYERS=50 ALPHA=2 NEX=40 MODES=norecord,duel100 python cross_steer2.py 2>&1 | tee cross_steer5.log
cp mech_out/cross_steer2.json mech_out/cross_steer5.json
echo CROSS_STEER5_DONE | tee -a cross_steer5.log
