#!/bin/bash
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/pylibs
python src/cos_matrix.py 2>&1 | tee nv22_cos.log
STEER_ONLY=1 STEER_N=32 STEER_DIRS="FITTED trust,optim,optim_like,story_comb,story_trust,story_warmth,warmth_b,nomfame,random" python src/moneyspec.py 2>&1 | tee nv22_money.log
SPEC=objects STEER_ONLY=1 STEER_N=32 STEER_DIRS="FITTED trust,optim,optim_like,story_comb,story_trust,story_warmth,warmth_b,nomfame,random" python src/moneyspec.py 2>&1 | tee nv22_objects.log
DIRS="FITTED trust,optim,story_comb,story_trust,story_warmth,optim_like,warmth_b,random" \
  OUTNAME=syco_final.json python src/syco.py 2>&1 | tee nv22_syco.log
DIRS="FITTED trust,optim,optim_like,story_trust,story_warmth,story_comb,warmth_b,random" \
  OUTNAME=testimony_final.json python src/testimony.py 2>&1 | tee nv22_testimony.log
echo RERUN_B_DONE | tee -a nv22_testimony.log
