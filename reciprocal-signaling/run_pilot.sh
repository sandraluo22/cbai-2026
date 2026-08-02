#!/bin/bash
# Reciprocal-signaling pilot v2 (free-text belief probe, 'is' format, no FC).
# Qwen32 grid + controls, then Llama70 and Qwen72 mains.
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

for M in 1 2 3; do
  MODEL=Qwen32 M=$M COND=main python src/run_games.py 2>&1 | tee -a pilot.log
done
for C in static oneway diffmis shuffled; do
  MODEL=Qwen32 M=2 COND=$C python src/run_games.py 2>&1 | tee -a pilot.log
done
for M in 1 2 3; do
  MODEL=Llama70 M=$M COND=main python src/run_games.py 2>&1 | tee -a pilot.log
done
for M in 1 2 3; do
  MODEL=Qwen72 M=$M COND=main python src/run_games.py 2>&1 | tee -a pilot.log
done
python src/analyze.py runs 2>&1 | tee -a pilot.log
echo PILOT_DONE
