#!/bin/bash
# emotion contagion, chained behind the news patching stage
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 LOAD8=1
until grep -qE 'NEWS_DONE|Traceback' news_c.log 2>/dev/null; do sleep 60; done
LAYER=40 ALPHA=1.2 NGEN=6 python emotion_contagion.py 2>&1 | tee emotion.log
