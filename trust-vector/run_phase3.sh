#!/bin/bash
# Phase 3 — the two behavioural dissociations, because cosine is not function.
# cos(trait, competence) = +0.59, so geometry cannot tell them apart. These
# scenarios can: scenario A asks whether the direction makes the model believe an
# UNVERIFIABLE promise specifically (trust) rather than merely feel positive
# (valence); scenario B is built so trust and competence predict OPPOSITE signs.
# Also re-runs the game grid with `competence` as a full arm, since the same logic
# applies there: a capable partner is a more dangerous defector.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until grep -qaE 'TRUST_PHASE2_DONE|Traceback' steer.log 2>/dev/null; do sleep 60; done

VECS=trait,record,traitR,recordR,news,valence,competence,arbitrary \
  ALPHA=${ALPHA:-0.25} NITEM=12 MODEL=${MODEL:-Qwen32} \
  python src/dissociate.py 2>&1 | tee dissociate.log

# dose-response on the dissociation: a real effect scales, a fluke does not
for A in 0.5 1.0; do
  VECS=trait,record,traitR,recordR,valence,competence ALPHA=$A NITEM=12 \
    MODEL=${MODEL:-Qwen32} python src/dissociate.py 2>&1 | tee dissociate_a$A.log
  cp out/dissociate.json out/dissociate_a$A.json
done

# the iterated game with competence promoted to a full arm
VECS=trait,record,traitR,recordR,valence,competence STAGES=grid ALPHA=${ALPHA:-0.25} \
  MODEL=${MODEL:-Qwen32} python src/steer_qsg.py 2>&1 | tee steer_comp.log
cp out/steer_qsg.json out/steer_qsg_competence.json
echo TRUST_PHASE3_DONE | tee -a dissociate.log
