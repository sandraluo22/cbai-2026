#!/bin/bash
# Does the model actually get suspicious? Remove the scripted assistant turns and let
# it answer. Also projects the read-out along the model's OWN conversation, and runs
# the projection for the four generation-prompt variants that the phase-9 run missed.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MAXNEW=110 TEMP=0 NNAME=4 MODEL=${MODEL:-Qwen32} \
  VECS=story_trust.full,direct_b.full,warmth_b.full LAYERS=45,52 \
  python src/elicit.py 2>&1 | tee elicit.log
NNAME=12 MODEL=${MODEL:-Qwen32} \
  VECS=story_trust@acct.full,story_trust@story.full,story_trust@acctnb.full,story_trust@storynb.full,story_trust.full \
  python src/project.py 2>&1 | tee project_variants.log
cp out/project.json out/project_variants.json
echo TRUST_ELICIT_DONE | tee -a elicit.log
