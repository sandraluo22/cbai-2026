#!/bin/bash
# story_posavg (Sandra 2026-08-15): ONE vector averaged across four reads of the
# same stories with the name in different places (appended / end / all mentions
# / NEW mid-story), so read-position components cancel in the DERIVATION.
# The advisor scenario is strictly the untouched testbed: does the averaged
# vector still steer, and does the second-slot asymmetry shrink vs story_trust?
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

FAMS=storymid_x NITEM=96 python src/newvec_build.py 2>&1 | tee nv16_build.log
SKIP_OPT=1 python src/newvec_opt.py 2>&1 | tee nv16_avg.log

DIRS_FILTER="story_posavg,story_trust,storymid_x,warmth_b" LAYERS=45 NPROBE=6 \
  python src/sweep_all.py 2>&1 | tee nv16_sweep.log
cp out/sweep_all.json out/newvec_sweep8.json

DIRS_FILTER="story_posavg,story_trust,warmth_b" ARM_LAYERS=45 \
  ADV_PAIR=Ana,Bruno ALPHA=0.5 python src/advisor_run.py 2>&1 | tee nv16_advisor.log
cp out/advisor_battery.json out/newvec_advisor8.json

DIRS_FILTER="story_posavg,story_trust,warmth_b,random" \
  python src/pushpull.py 2>&1 | tee nv16_pushpull.log
cp out/pushpull.json out/pushpull_posavg.json

echo NEWVEC8_ALL_DONE | tee -a nv16_pushpull.log
