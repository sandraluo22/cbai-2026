#!/bin/bash
# cross-domain credibility steering, chained behind the emotion run
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 LOAD8=1
until grep -qE 'EMOTION_DONE|Traceback' emotion.log 2>/dev/null; do sleep 60; done
LAYERS=20,30,40,50,56 ALPHA=1,2 NDOC=12 NEX=8 python cross_steer.py 2>&1 | tee cross_steer.log
