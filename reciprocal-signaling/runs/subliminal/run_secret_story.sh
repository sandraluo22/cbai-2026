#!/bin/bash
set -x
cd /workspace/mm/reciprocal-signaling/runs/subliminal
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q TOPICS_DONE sub3.log 2>/dev/null; do sleep 120; done
NSEED=4 python secret_story.py 2>&1 | tee -a story.log
