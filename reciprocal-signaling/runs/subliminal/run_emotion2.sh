#!/bin/bash
set -x
cd /workspace/mm/reciprocal-signaling/runs/subliminal
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH47_DONE ../gossip/batch47.log 2>/dev/null; do sleep 120; done
LOAD8=0 NSEED=8 MODE=emotion python subliminal2.py 2>&1 | tee -a sub2.log
