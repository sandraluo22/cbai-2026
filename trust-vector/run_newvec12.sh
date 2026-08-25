#!/bin/bash
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/pylibs
python src/syco_vec2.py 2>&1 | tee nv24_vec2.log
DIRS_FILTER="syco_name2,syco_endname,syco_caa,story_trust,warmth_b" LAYERS=45 NPROBE=6 \
  python src/sweep_all.py 2>&1 | tee nv24_sweep.log
cp out/sweep_all.json out/newvec_sweep12.json
DIRS_FILTER="syco_name2,syco_endname,syco_caa,story_trust,warmth_b,random" \
  python src/pushpull.py 2>&1 | tee nv24_pushpull.log
cp out/pushpull.json out/pushpull_syco2.json
echo NEWVEC12_ALL_DONE | tee -a nv24_pushpull.log
