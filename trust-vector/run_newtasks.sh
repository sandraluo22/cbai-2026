#!/bin/bash
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# slide 3 needs every direction, not the 8 it currently has
NNAME=6 MODEL=${MODEL:-Qwen32} LAYERS=45,52 \
  VECS=direct_a.full,direct_b.full,relational.full,game_a.full,game_b.full,comp_b.full,hon_b.full,rel_b.full,warmth_b.full,story_trust.full,story_trust@acct.full,story_trust@story.full,story_trust@acctnb.full,story_trust@storynb.full \
  python src/project.py 2>&1 | tee project_all.log
cp out/project.json out/project_all.json
LAYERS=45 ALPHAS=0.2,0.35,0.5 MODEL=${MODEL:-Qwen32} python src/newtasks.py 2>&1 | tee newtasks.log
echo TRUST_NEWTASKS_DONE | tee -a newtasks.log
