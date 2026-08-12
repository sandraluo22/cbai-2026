#!/bin/bash
# (1) emotion positive control: identical pipeline, emotion words ALLOWED
# (2) larger paired credibility-injection at L50
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 LOAD8=1
NOBAN=1 LAYER=40 ALPHA=2.4 NGEN=20 python emotion_contagion.py 2>&1 | tee emotion_noban.log
LAYERS=50 ALPHA=1,2,3 NEX=40 MODES=duel80,duel100 python cross_steer2.py 2>&1 | tee cross_steer3.log
