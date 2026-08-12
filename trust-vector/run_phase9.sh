#!/bin/bash
# Phase 9 — v2 with the confounds found by inspection removed:
#   * relation crossed (subordinate / peer / superior / counterparty / service /
#     friend / stranger) instead of every scenario being "you are the boss"
#   * elaborated descriptions reworded so the same sentence works in any relation
#   * generated stories told what NOT to write about, so the dimensions stop bleeding
#   * generated neutral is mixed evidence in the same genre, not a stranger vignette
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until grep -qaE 'TRUST_PLOTS_DONE|Traceback' plots2_full.log 2>/dev/null; do sleep 60; done
cp out/stories.json out/stories_run1.json 2>/dev/null
cp out/vectors2.npz out/vectors2_run1.npz 2>/dev/null

NSTORY=16 MODEL=${MODEL:-Qwen32} python src/gen_stories.py 2>&1 | tee gen9.log
LAYERS=all NITEM=16 MODEL=${MODEL:-Qwen32} python src/build_vectors2.py 2>&1 | tee build9.log
for T in full addm subm; do TAG=$T python src/compare2.py 2>&1 | tee compare9_$T.log; done
MODEL=${MODEL:-Qwen32} NNAME=12 \
  VECS=direct_b.full,game_b.full,story_trust.full,relational.full,comp_b.full,warmth_b.full,story_trust@acct.full,story_trust@storynb.full \
  python src/project.py 2>&1 | tee project9.log
cp out/project.json out/project9.json
for T in full addm; do TAG=$T python src/plots2.py 2>&1 | tee plots9_$T.log; done
echo TRUST_PHASE9_DONE | tee -a plots9_full.log
