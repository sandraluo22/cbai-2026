#!/bin/bash
# Redo everything at the NAME-TOKEN injection site, with ALL directions everywhere.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LAYERS=45,52 NPROBE=6 MODEL=${MODEL:-Qwen32} python src/sweep_all.py 2>&1 | tee sweep_all.log
NNAME=6 LAYERS=45,52 VECS=all MODEL=${MODEL:-Qwen32} python src/project.py 2>&1 | tee project_all.log
cp out/project.json out/project_all.json
LAYERS=45 ALPHAS=0.2,0.35,0.5 NSCAM_NAME=2 VECS=all MODEL=${MODEL:-Qwen32} \
  python src/newtasks.py 2>&1 | tee newtasks.log
echo TRUST_REDO_DONE | tee -a newtasks.log
