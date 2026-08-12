#!/bin/bash
# Phase 11 — three things:
#   1. story families reframed so the model is the narrator of its own account
#      (they previously said "you are reading an account of someone you work with",
#      which makes the model a spectator and engages nothing about its own trust)
#   2. the conversation tracking run with the pos-minus-mixed-neutral directions too,
#      not only pos-minus-negative
#   3. fit a direction to PREDICT the model's stated trust, and ask whether it
#      generalises across families and whether it tracks the conversation
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

LAYERS=all NITEM=16 MODEL=${MODEL:-Qwen32} python src/build_vectors2.py 2>&1 | tee build11.log
for T in full addm; do
  TAG=$T python src/compare2.py 2>&1 | tee compare11_$T.log
  TAG=$T python src/plots2.py  2>&1 | tee plots11_$T.log
  cp out/v2_heatmap_$T.png out/v11_heatmap_$T.png
done
NNAME=12 MODEL=${MODEL:-Qwen32} \
  VECS=direct_b.full,direct_b.addm,story_trust.full,story_trust.addm,game_b.full,game_b.addm,warmth_b.full,warmth_b.addm \
  python src/project.py 2>&1 | tee project11.log
cp out/project.json out/project11.json

LAYERS=27,35,45,52 NITEM=12 MODEL=${MODEL:-Qwen32} \
  python src/fit_direction.py 2>&1 | tee fit11.log
echo TRUST_PHASE11_DONE | tee -a fit11.log
