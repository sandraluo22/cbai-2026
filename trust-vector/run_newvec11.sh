#!/bin/bash
# CAA sycophancy vector (Rimsky et al. 2024 recipe on Anthropic syco evals):
# derive + home-bed validation, then test on every standing bed.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/pylibs
python src/syco_vec.py 2>&1 | tee nv23_vec.log
DIRS_FILTER="syco_caa,optim_like,warmth_b,story_trust" LAYERS=45 NPROBE=6 \
  python src/sweep_all.py 2>&1 | tee nv23_sweep.log
cp out/sweep_all.json out/newvec_sweep11.json
DIRS_FILTER="syco_caa,story_trust,optim,warmth_b,random" \
  python src/pushpull.py 2>&1 | tee nv23_pushpull.log
cp out/pushpull.json out/pushpull_syco.json
DIRS="syco_caa,optim,optim_like,random" OUTNAME=testimony_syco.json \
  python src/testimony.py 2>&1 | tee nv23_testimony.log
DIRS="random,syco_caa,optim_like,warmth_b" python src/dissoc.py 2>&1 | tee nv23_dissoc.log
cp out/dissoc.json out/dissoc_syco.json
STEER_ONLY=1 STEER_N=32 STEER_DIRS="syco_caa,optim_like,random" \
  python src/moneyspec.py 2>&1 | tee nv23_money.log
cp out/moneyspec_steeronly.json out/moneyspec_steeronly_syco.json
SPEC=objects STEER_ONLY=1 STEER_N=32 STEER_DIRS="syco_caa,optim_like,random" \
  python src/moneyspec.py 2>&1 | tee nv23_objects.log
cp out/moneyspec_objects_steeronly.json out/objects_steeronly_syco.json
echo NEWVEC11_ALL_DONE | tee -a nv23_objects.log
