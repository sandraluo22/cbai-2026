#!/bin/bash
# GATE RUN. 4 concepts x 3 seeds, ~1-2 h. Do not run run_full.sh until this has
# been read.
#
# The pilot exists to answer one question before 28 concepts get trained:
#   does a LoRA for a concept look like ITSELF across training seeds?
# If cos(dW_c^s0, dW_c^s1) sits near the cross-concept floor, weight-space
# cosine is not a representation of the concept and every geometry number in the
# full study is capped at noise. That is a result worth having in two hours
# rather than two days.
#
# The pilot set (concepts.PILOT) carries every tier the full study needs:
#   verbose/terse    antonym pair -> signed-vs-magnitude dissociation
#   terse/terse_b    twin pair    -> the data ceiling
#   french           unrelated    -> the floor
#
# What to read in the output, in order:
#   1. gen  manipulation-check d per concept          (must clear 0.8)
#   2. vec  split-half reliability, steering gain vs random, integrity
#   3. lora per-adapter behavioural gain over base    (did it learn it)
#   4. w    seed-ceiling per representation           <- THE GATE
#   5. cmp  tier table + antonym test
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export ONLY="verbose,terse,terse_b,french"
export INIT_SEEDS="${INIT_SEEDS:-0,1}"
export DATA_SEEDS="${DATA_SEEDS:-0,1}"

mkdir -p out logs
python src/mock_test.py            2>&1 | tee logs/00_mock.log
python src/gen_data.py             2>&1 | tee logs/01_gen.log
python src/build_vecs.py           2>&1 | tee logs/02_vec.log
python src/train_lora.py           2>&1 | tee logs/03_lora.log
python src/wspace.py               2>&1 | tee logs/04_wspace.log
python src/induced.py              2>&1 | tee logs/05_induced.log
python src/compare.py              2>&1 | tee logs/06_compare.log
echo PILOT_DONE
