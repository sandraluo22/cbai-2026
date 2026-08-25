#!/bin/bash
# PILOT. Small concept set, full pipeline, two hard gates before any geometry:
#   gen.py     the teacher must express the trait, and the numeric filter must
#              leave the data free of the concept word (else the student can learn
#              it by surface imitation and the design is void)
#   induced.py the trait must actually transmit to the student; adapters where it
#              did not are dropped by compare.py, because their dW is noise
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export N_CONCEPTS="${N_CONCEPTS:-2}" N_KEEP="${N_KEEP:-10000}" CKPT_EVERY="${CKPT_EVERY:-250}"
export INIT_SEEDS="${INIT_SEEDS:-0}" DATA_SEEDS="${DATA_SEEDS:-0}"
mkdir -p out logs
python src/gen.py       2>&1 | tee logs/01_gen.log
python src/vectors.py   2>&1 | tee logs/02_vec.log
python src/train.py     2>&1 | tee logs/03_train.log
python src/wspace.py    2>&1 | tee logs/04_wspace.log
python src/induced.py   2>&1 | tee logs/05_induced.log
python src/s2e.py       2>&1 | tee logs/06_s2e.log
python src/compare.py   2>&1 | tee logs/07_compare.log
echo PILOT_DONE
