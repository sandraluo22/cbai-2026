#!/bin/bash
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LAYERS=45,52 NNAME=6 ALPHA=0.25,0.5,1.0 MODEL=${MODEL:-Qwen32} \
  python src/fit3.py 2>&1 | tee fit3.log
echo TRUST_FIT3_DONE | tee -a fit3.log
