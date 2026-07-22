#!/usr/bin/env bash
# Run both games (reference pragmatic agents, CPU) into $RUN_DIR.
#   RUN_DIR=runs/main bash src/run_all.sh
set -euo pipefail
RUN_DIR="${RUN_DIR:-runs/main}"
export RUN_DIR PYTHONPATH=src
mkdir -p "$RUN_DIR"
echo "==== GAME 1: perturbation / KL-coupling ===="
python3 src/game1_coupling.py
echo "==== GAME 2: sequential-reveal Codenames (fused with the coupling instrument) ===="
python3 src/game2_codenames.py
echo "==== DONE -> $RUN_DIR ===="
ls -la "$RUN_DIR"
