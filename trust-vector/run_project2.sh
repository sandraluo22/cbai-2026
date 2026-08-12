#!/bin/bash
# Re-run the conversation projection with per-name error bars, all 12 names, and the
# content-matched-neutral directions included. Run 1 of it showed every direction
# separating the two conversations -- including the warmth decoy -- so the question
# is whether any of those differences survive their own spread.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until grep -qaE 'TRUST_PHASE7_DONE|Traceback' analyze7.log 2>/dev/null; do sleep 60; done
NNAME=12 MODEL=${MODEL:-Qwen32} \
  VECS=direct_b.full,game_b.full,story_trust.full,relational.full,direct_b.addm,comp_b.full,hon_b.full,rel_b.full,warmth_b.full \
  python src/project.py 2>&1 | tee project8.log
cp out/project.json out/project8.json
echo TRUST_PROJECT2_DONE | tee -a project8.log
