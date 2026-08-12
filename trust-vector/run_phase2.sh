#!/bin/bash
# Phase 2, after compare.py returned separation ~= 0 (the candidate directions are
# no more alike than they are like valence/competence/tall-short):
#   a) build the control-residualised directions (*R) and append them to vectors.npz
#   b) re-validate everything against BOTH probe families, so a direction has to
#      raise the no-evidence probe AND lower the good-record probe
#   c) run the game with the controls promoted to full arms — if valence moves the
#      game as much as trait does, the steering result is a valence result
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ANCHOR=last  WRITE=1 python src/residualize.py 2>&1 | tee resid_last.log
ANCHOR=name2 WRITE=1 python src/residualize.py 2>&1 | tee resid_name2.log

SKIP_VEC=1 VALIDATE=1 ALPHA=0.5 MODEL=${MODEL:-Qwen32} \
  python src/build_vectors.py 2>&1 | tee validate2.log

VECS=trait,record,news,traitR,recordR,valence,arbitrary ALPHA=${ALPHA:-0.25} \
  STAGES=grid,curve MODEL=${MODEL:-Qwen32} \
  python src/steer_qsg.py 2>&1 | tee steer.log
cp out/steer_qsg.json out/steer_qsg_a${ALPHA:-0.25}.json
echo TRUST_PHASE2_DONE | tee -a steer.log
