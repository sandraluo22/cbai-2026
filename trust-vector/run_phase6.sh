#!/bin/bash
# Phase 6 — protocol v2 end to end. New stimuli (Sandra's spec, generalised across
# 8 settings and 12 names), read at an appended bare name token, three conditions
# (positive / neutral / negative), components (competence, honesty, reliability)
# tested separately rather than as throwaway decoys, model-generated stories, and a
# conversation-trajectory measurement instead of only steering.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

NSTORY=${NSTORY:-16} MODEL=${MODEL:-Qwen32} python src/gen_stories.py 2>&1 | tee gen6.log
LAYERS=all NITEM=${NITEM:-16} MODEL=${MODEL:-Qwen32} \
  python src/build_vectors2.py 2>&1 | tee build6.log
for T in full add sub; do
  TAG=$T python src/compare2.py 2>&1 | tee compare6_$T.log
done
MODEL=${MODEL:-Qwen32} python src/project.py 2>&1 | tee project6.log

# steer the game with the v2 directions, paired error bars on every arm
VECFILE=vectors2.npz ANCHOR=last ALPHA=${ALPHA:-0.25} STAGES=grid MODEL=${MODEL:-Qwen32} \
  VECS=direct_b.full,game_b.full,relational.full,story_trust.full,comp_b.full,hon_b.full,rel_b.full,warmth_b.full \
  python src/steer_qsg.py 2>&1 | tee steer6.log
cp out/steer_qsg.json out/steer6.json
FILE=out/steer6.json python src/analyze_steer.py 2>&1 | tee analyze6.log
echo TRUST_PHASE6_DONE | tee -a analyze6.log
