#!/bin/bash
# emotion, fixed: induce A's state by PROMPT (verified stronger than steering), keep the
# emotion-word ban ON. B_detect now doubles as the manipulation check on A's text.
set -x
cd /workspace/mm/reciprocal-signaling/runs/gossip
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 LOAD8=1
MODE=prompt LAYER=40 ALPHA=0 NGEN=20 python emotion_contagion.py 2>&1 | tee emotion3.log
mkdir -p emotion_out_prompt && cp emotion_out/runs.json emotion_out_prompt/runs.json
MODE=both LAYER=40 ALPHA=2.4 NGEN=20 python emotion_contagion.py 2>&1 | tee emotion4.log
mkdir -p emotion_out_both && cp emotion_out/runs.json emotion_out_both/runs.json
echo EMOTION_FIXED_DONE | tee -a emotion4.log
