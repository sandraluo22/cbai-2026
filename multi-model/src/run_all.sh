#!/usr/bin/env bash
# Drive all multi-model stages in order, writing to $RUN_DIR.
# Run from the multi-model/ project root (so `python src/...` resolves and
# common.py finds ../cross-model/src and ../emotion).
#
#   PRESET=smoke      RUN_DIR=runs/smoke bash src/run_all.sh   # CPU end-to-end
#   PRESET=gemma_qwen RUN_DIR=runs/main  bash src/run_all.sh   # real 8B run
set -euo pipefail

PRESET="${PRESET:-gemma_qwen}"
RUN_DIR="${RUN_DIR:-runs/main}"
export PRESET RUN_DIR
mkdir -p "$RUN_DIR"

echo "==== [1/4] emotion vectors: Llama ===="
MODEL=Llama python src/build_emotion_vectors.py
echo "==== [2/4] emotion vectors: Qwen ===="
MODEL=Qwen  python src/build_emotion_vectors.py
echo "==== [3/4] Exp1 grid transfer ===="
python src/exp1_grid_transfer.py
echo "==== [4/4] Exp2 sadness transfer ===="
python src/exp2_sadness_transfer.py
echo "==== ALL DONE -> $RUN_DIR ===="
ls -la "$RUN_DIR"
