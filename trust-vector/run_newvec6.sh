#!/bin/bash
# nomfame (Sandra's design): famous people the model names as genuinely
# trusted / distrusted, vector read at the FIRST token of each generated name.
# Trust context precedes the read (unlike story_first), the mention is fresh
# (unlike the end-of-passage reads), and famous names dodge the personal-
# relationship refusals that starved `nominate`. Validate on all three beds.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

FAMS=nomfame NGEN=12 python src/newvec_build.py 2>&1 | tee nv7_build.log

DIRS_FILTER="nomfame,nominate,story_trust,warmth_b" LAYERS=45 NPROBE=6 \
  python src/sweep_all.py 2>&1 | tee nv7_sweep.log
cp out/sweep_all.json out/newvec_sweep7.json

DIRS_FILTER="nomfame,story_trust" ARM_LAYERS=45 ADV_PAIR=Ana,Bruno ALPHA=0.5 \
  python src/advisor_run.py 2>&1 | tee nv7_advisor.log
cp out/advisor_battery.json out/newvec_advisor7.json

DIRS_FILTER="nomfame,story_trust,warmth_b,random" \
  python src/pushpull.py 2>&1 | tee nv7_pushpull.log
cp out/pushpull.json out/pushpull_nomfame.json

echo NEWVEC6_ALL_DONE | tee -a nv7_pushpull.log
