#!/bin/bash
set -x
cd /workspace/mm/reciprocal-signaling/runs/subliminal
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q EMOTION_DONE sub2.log 2>/dev/null; do sleep 120; done
for T in dinner travel music; do
  LOAD8=0 NSEED=4 MODE=hidden TOPIC=$T python subliminal3.py 2>&1 | tee -a sub3.log
done
echo TOPICS_DONE | tee -a sub3.log
