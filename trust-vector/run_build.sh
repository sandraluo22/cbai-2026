#!/bin/bash
# Stage 1-3: derive every candidate trust direction, validate each in its own
# domain, compare them against each other and against the controls.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MODEL=${MODEL:-Qwen32} LAYERS=${LAYERS:-all} NPAIR=${NPAIR:-24} ALPHA=0.5 VALIDATE=1 \
  python src/build_vectors.py 2>&1 | tee build.log

for ANCHOR in last name2; do
  ANCHOR=$ANCHOR python src/compare.py 2>&1 | tee compare_$ANCHOR.log
  cp out/compare.json out/compare_$ANCHOR.json
  cp out/compare.png out/compare_$ANCHOR.png 2>/dev/null
done
echo TRUST_BUILD_DONE | tee -a build.log
