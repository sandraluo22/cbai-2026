#!/bin/bash
# 2026-08-13 evening batch — four new derivations + validation (Sandra's spec):
#   (1) storyend: stories that withhold the name until "This person is {n}.",
#       read directly at that single name token (+ story_comb = avg with the
#       existing story batch)
#   (2) nominate: model names people it trusts / distrusts, diff at those names
#   (3) avg_all: unit mean of every existing trust method
#   (4) optim: magnitude-bounded vector optimized at the name tokens to steer
#       "{n} is someone I [trust/don't trust]" over frames x names, averaged
# Validation: stated-trust sweep (do-you-trust-X) at L45/52, advisor battery
# plain + conditional at L45, both against story_trust/direct_b/FITTED/warmth/random.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python src/gen_storyend.py 2>&1 | tee nv_genstory.log
python src/newvec_build.py 2>&1 | tee nv_build.log
python src/newvec_opt.py 2>&1 | tee nv_opt.log

NEWDIRS="storyend,nominate,story_comb,avg_all,optim"
REFDIRS="story_trust,direct_b,FITTED trust,warmth_b"

DIRS_FILTER="$NEWDIRS,$REFDIRS" LAYERS=45,52 NPROBE=6 \
  python src/sweep_all.py 2>&1 | tee nv_sweep.log
cp out/sweep_all.json out/newvec_sweep.json

DIRS_FILTER="$NEWDIRS,$REFDIRS" ARM_LAYERS=45 ADV_PAIR=Ana,Bruno ALPHA=0.5 \
  python src/advisor_run.py 2>&1 | tee nv_advisor.log
cp out/advisor_battery.json out/newvec_advisor.json

echo NEWVEC_ALL_DONE | tee -a nv_advisor.log
