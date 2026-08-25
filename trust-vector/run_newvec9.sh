#!/bin/bash
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/pylibs
python src/avg_variants.py 2>&1 | tee nv20_avgvar.log
python src/promptforms.py 2>&1 | tee nv20_forms.log
python src/dissoc.py 2>&1 | tee nv20_dissoc.log
DIRS_FILTER="avg_all,avg_nofit,avg_core,story_trust,warmth_b" LAYERS=45 NPROBE=6 \
  python src/sweep_all.py 2>&1 | tee nv20_sweep.log
cp out/sweep_all.json out/newvec_sweep9.json
DIRS_FILTER="avg_all,avg_nofit,avg_core,story_trust,warmth_b,random" \
  python src/pushpull.py 2>&1 | tee nv20_pushpull.log
cp out/pushpull.json out/pushpull_avgvar.json
echo NEWVEC9_ALL_DONE | tee -a nv20_pushpull.log
