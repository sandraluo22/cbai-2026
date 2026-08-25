#!/bin/bash
# FULL RUN. 28 concepts x 2 init blocks x 2 data seeds = 112 adapters. Only
# after run_pilot.sh has cleared the weight-space gate.
#
# train_lora skips any adapter that already exists, and orders by init block, so
# a partial run still leaves a COMPLETE, analysable block.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export INIT_SEEDS="${INIT_SEEDS:-0,1}"
export DATA_SEEDS="${DATA_SEEDS:-0,1}"
unset ONLY

mkdir -p out logs
python src/gen_data.py             2>&1 | tee logs/11_gen.log
python src/build_vecs.py           2>&1 | tee logs/12_vec.log
python src/train_lora.py           2>&1 | tee logs/13_lora.log
python src/wspace.py               2>&1 | tee logs/14_wspace.log
python src/induced.py              2>&1 | tee logs/15_induced.log

# Read position is a FACTOR, not a setting (trust-vector lost a week to a result
# that existed at one read position only). Init BLOCK is also a factor, for a
# different reason: dW coordinates live in a per-block random basis, so tier and
# mapping tables are within-block and every claim must survive both blocks.
for POS in response last; do
  for BLOCK in 0 1; do
    POS=$POS BLOCK=$BLOCK python src/compare.py 2>&1 \
      | tee "logs/16_compare_${POS}_b${BLOCK}.log"
  done
done
echo FULL_DONE
