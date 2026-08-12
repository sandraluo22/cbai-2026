#!/bin/bash
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until grep -qaE 'TRUST_PHASE11_DONE|Traceback' fit11.log 2>/dev/null; do sleep 45; done
LAYERS=35,45,52 NITEM=10 MODEL=${MODEL:-Qwen32} python src/fit2.py 2>&1 | tee fit2.log
echo TRUST_FIT2_DONE | tee -a fit2.log
