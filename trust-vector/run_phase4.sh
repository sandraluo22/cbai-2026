#!/bin/bash
# Phase 4 — two things run 1 got wrong or never did.
#
# (a) ANCHOR MISMATCH. Every steering run so far used vectors read at the LAST token
#     of the derivation passage, then injected them at the PARTNER'S NAME tokens.
#     A last-token vector is an "about to emit output" direction, so it is no
#     surprise it worked ~20x better at the answer slot than at the name. This runs
#     the matched design: read at the second mention of the name, inject at name
#     positions. Both anchors are run so the comparison is direct.
#
# (b) RATIONALE DERIVATION. All previous methods describe a third party in the user
#     turn. These two put the contrast in the model's own begun reply -- trusting vs
#     refusing rationale, rejoining a shared tail -- which is in-domain for the
#     decision being steered and is first-person stance rather than attribution.
#       rationale : offer that pays off only if the counterparty keeps their word
#       gamerat   : the round-7 iterated-game decision
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cp out/vectors.npz out/vectors_run1.npz 2>/dev/null

MODEL=${MODEL:-Qwen32} LAYERS=all NPAIR=24 ALPHA=0.5 VALIDATE=1 \
  python src/build_vectors.py 2>&1 | tee build4.log

for A in last name2; do
  ANCHOR=$A WRITE=1 python src/residualize.py 2>&1 | tee resid4_$A.log
  ANCHOR=$A PLOT=1 python src/compare.py 2>&1 | tee compare4_$A.log
  cp out/compare.json out/compare4_$A.json
  cp out/compare.png  out/compare4_$A.png
done

# the matched-anchor test: name-derived vector -> name positions, vs last-derived
VEC=trait,record,rationale,gamerat,recordR,rationaleR,valence,arbitrary
for A in name2 last; do
  VECS=$VEC ANCHOR=$A ALPHA=${ALPHA:-0.25} STAGES=grid MODEL=${MODEL:-Qwen32} \
    python src/steer_qsg.py 2>&1 | tee steer4_$A.log
  cp out/steer_qsg.json out/steer4_$A.json
done

VECS=$VEC ANCHOR=name2 ALPHA=${ALPHA:-0.25} NITEM=12 MODEL=${MODEL:-Qwen32} \
  python src/dissociate.py 2>&1 | tee dissociate4.log
cp out/dissociate.json out/dissociate4_name2.json
echo TRUST_PHASE4_DONE | tee -a steer4_last.log
