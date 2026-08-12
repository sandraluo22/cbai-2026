#!/bin/bash
# what does the credibility push act ON? record-supported label, asserted reliability,
# a lone source, or nothing in particular?
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 LOAD8=1
until grep -qE 'EMOTION_FIXED_DONE|Traceback' emotion4.log 2>/dev/null; do sleep 60; done
LAYERS=50 ALPHA=1,2 NEX=40 MODES=norecord,tally_true,single python cross_steer2.py 2>&1 | tee cross_steer4.log
cp mech_out/cross_steer2.json mech_out/cross_steer4.json
echo CROSS_STEER4_DONE | tee -a cross_steer4.log
