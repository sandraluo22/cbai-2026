#!/bin/bash
# Phase 5 — where relative to the partner's name does the injection land, and are any
# of these differences bigger than their own error bars? Run 1 reported arm means with
# no spread; steer_qsg now stores raw per-prompt margins so the arm-vs-base difference
# can be taken paired, per prompt.
#   pre  = token immediately before each mention of the partner's name
#   at   = the name tokens themselves
#   post = token immediately after   (in a decoder this is the first position that has
#          attended to the name, so it may carry more about that player than the name)
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VEC=trait,record,recordR,rationale,valence,arbitrary
for A in name2 last; do
  VECS=$VEC ANCHOR=$A ALPHA=${ALPHA:-0.25} STAGES=grid MODEL=${MODEL:-Qwen32} \
    python src/steer_qsg.py 2>&1 | tee steer5_$A.log
  cp out/steer_qsg.json out/steer5_$A.json
  FILE=out/steer5_$A.json LAYER=46 python src/analyze_steer.py 2>&1 | tee analyze5_$A.log
done
echo TRUST_PHASE5_DONE | tee -a analyze5_last.log
