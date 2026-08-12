#!/bin/bash
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 LOAD8=1
until grep -qE 'EMOTION_DONE|Traceback' emotion2.log 2>/dev/null; do sleep 60; done
LAYERS=50,56 ALPHA=1,2 NEX=16 MODES=duel80,duel100 python cross_steer2.py 2>&1 | tee cross_steer2.log
