#!/bin/bash
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
until grep -qaE 'TRUST_PROJECT2_DONE|Traceback' project8.log 2>/dev/null; do sleep 60; done
for T in full addm; do TAG=$T python src/plots2.py 2>&1 | tee plots2_$T.log; done
echo TRUST_PLOTS_DONE | tee -a plots2_full.log
