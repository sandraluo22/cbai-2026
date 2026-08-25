#!/bin/bash
# Follow-up to run_newvec.sh: the two checks the optim result needs before it
# means anything.
#   (a) optim_like — the SAME optimization with " like"/" dis" margin words:
#       the optimized-the-same-way decoy. If it steers the stated-TRUST probe
#       as hard as optim does, the optimizer found generic steering power.
#   (b) push-pull with every new direction — the position-cancelled entity
#       differential, the project's standing estimand for entity steering.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OPT_KEY=optim_like OPT_POSW=" like" OPT_NEGW=" dis" OPT_LAYERS=45 \
  python src/newvec_opt.py 2>&1 | tee nv2_optlike.log

DIRS_FILTER="optim,optim_like,FITTED trust,warmth_b" LAYERS=45 NPROBE=6 \
  python src/sweep_all.py 2>&1 | tee nv2_sweep.log
cp out/sweep_all.json out/newvec_sweep2.json

DIRS_FILTER="optim,optim_like,avg_all,story_comb,storyend,nominate,story_trust,FITTED trust,direct_b,warmth_b,random" \
  python src/pushpull.py 2>&1 | tee nv2_pushpull.log
cp out/pushpull.json out/pushpull_newvec.json

echo NEWVEC2_ALL_DONE | tee -a nv2_pushpull.log
