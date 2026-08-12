#!/bin/bash
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python src/prior_src.py 2>&1 | tee prior.log
ALPHA=0.5 python src/advisor_run.py 2>&1 | tee battery.log
NNAME=6 LAYERS=45,52 VECS=prior_wiki.full,prior_src.full,prior_expert.full,direct_b.full,FITTED\ trust \
  python src/project.py 2>&1 | tee project_prior.log
cp out/project.json out/project_prior.json
LAYERS=45,52 NPROBE=6 python src/sweep_all.py 2>&1 | tee sweep_all2.log
echo BATTERY_ALL_DONE | tee -a sweep_all2.log
