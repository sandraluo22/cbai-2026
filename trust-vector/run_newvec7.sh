#!/bin/bash
# Sandra's 2026-08-14 evening batch:
#   story_all       read at EVERY in-story mention, mean-pooled (1a)
#   advisor nulls   battery with 4 extra random seeds + a zero-vector harness
#                   check, so the first-slot effect has a proper null band (1c)
#   pad_probe       recency test of the second-slot gain: filler between the
#                   adviser lines and the question (1b, in-house test)
#   opengen         open-ended "What would you trust X with?", steered vs
#                   control, infamous + in-context-distrusted people (2)
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

FAMS=story_all NITEM=96 python src/newvec_build.py 2>&1 | tee nv12_build.log

DIRS_FILTER="story_all,story_trust,optim" ADD_NULLS=1 ARM_LAYERS=45 \
  ADV_PAIR=Ana,Bruno ALPHA=0.5 python src/advisor_run.py 2>&1 | tee nv12_advisor.log
cp out/advisor_battery.json out/newvec_advisor_nulls.json

python src/pad_probe.py 2>&1 | tee nv12_pad.log

python src/opengen.py 2>&1 | tee nv12_opengen.log

echo NEWVEC7_ALL_DONE | tee -a nv12_opengen.log
