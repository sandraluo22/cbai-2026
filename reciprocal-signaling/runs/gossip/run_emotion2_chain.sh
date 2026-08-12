#!/bin/bash
# emotion replication: more samples, stronger steering, independent readout direction
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 LOAD8=1
until grep -qE 'CROSS_STEER_DONE|Traceback' cross_steer.log 2>/dev/null; do sleep 60; done
LAYER=40 ALPHA=2.4 NGEN=20 python emotion_contagion.py 2>&1 | tee emotion2.log
