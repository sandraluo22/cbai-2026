#!/bin/bash
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LAYERS=45,52 NPROBE=6 MODEL=${MODEL:-Qwen32} python src/sweep_all.py 2>&1 | tee sweep_all.log
echo TRUST_SWEEPALL_DONE | tee -a sweep_all.log
