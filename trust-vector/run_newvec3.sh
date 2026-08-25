#!/bin/bash
# storyend_x (Sandra 2026-08-14): the existing story bank deterministically
# restructured to the end-name form -- no new generation, content held fixed --
# so structure and story-sample are unconfounded. Waits for run_newvec2.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

until grep -q NEWVEC2_ALL_DONE nv2_pushpull.log 2>/dev/null; do sleep 60; done

FAMS=storyend_x NITEM=96 python src/newvec_build.py 2>&1 | tee nv3_build.log

DIRS_FILTER="storyend_x,storyend,story_trust,warmth_b" LAYERS=45 NPROBE=6 \
  python src/sweep_all.py 2>&1 | tee nv3_sweep.log
cp out/sweep_all.json out/newvec_sweep3.json

DIRS_FILTER="storyend_x,storyend,story_trust,warmth_b,random" \
  python src/pushpull.py 2>&1 | tee nv3_pushpull.log
cp out/pushpull.json out/pushpull_storyendx.json

DIRS_FILTER="storyend_x" ARM_LAYERS=45 ADV_PAIR=Ana,Bruno ALPHA=0.5 \
  python src/advisor_run.py 2>&1 | tee nv3_advisor.log
cp out/advisor_battery.json out/newvec_advisor3.json

echo NEWVEC3_ALL_DONE | tee -a nv3_advisor.log
