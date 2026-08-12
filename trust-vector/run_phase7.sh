#!/bin/bash
# Phase 7 — rebuild v2 with a CONTENT-MATCHED neutral.
# Run 1 of v2 gave cos(pos-neu, neu-neg) of -0.5 to -0.99: `neu` ("has yet to prove
# anything") differs from BOTH pos and neg by having no described history at all, so
# it is not a midpoint on any trust axis. `mix` supplies the same amount and
# specificity of evidence pointing both ways. If cos(addm, subm) comes out positive
# where cos(add, sub) was negative, the earlier number was a content artifact.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until grep -qaE 'TRUST_PHASE6_DONE|Traceback' analyze6.log 2>/dev/null; do sleep 60; done

LAYERS=all NITEM=${NITEM:-16} MODEL=${MODEL:-Qwen32} \
  python src/build_vectors2.py 2>&1 | tee build7.log
for T in full addm subm; do
  TAG=$T python src/compare2.py 2>&1 | tee compare7_$T.log
done
VECFILE=vectors2.npz ANCHOR=last ALPHA=${ALPHA:-0.25} STAGES=grid MODEL=${MODEL:-Qwen32} \
  VECS=direct_b.full,game_b.full,relational.full,story_trust.full,direct_b.addm,game_b.addm,comp_b.full,warmth_b.full \
  python src/steer_qsg.py 2>&1 | tee steer7.log
cp out/steer_qsg.json out/steer7.json
FILE=out/steer7.json python src/analyze_steer.py 2>&1 | tee analyze7.log
echo TRUST_PHASE7_DONE | tee -a analyze7.log
