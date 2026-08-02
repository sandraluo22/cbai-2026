#!/bin/bash
# Pilot part 2: the two big models (Qwen32 grid + controls already done in part 1).
# Waits for the Qwen2.5-72B shard download (download_72b.log ends with DL_DONE).
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

for M in 1 2 3; do
  MODEL=Llama70 M=$M COND=main python src/run_games.py 2>&1 | tee -a pilot.log
done

for i in $(seq 1 240); do
  grep -q DL_DONE download_72b.log 2>/dev/null && break
  sleep 60
done
if ! grep -q DL_DONE download_72b.log 2>/dev/null; then
  echo "72B_DOWNLOAD_NOT_READY — skipping Qwen72" | tee -a pilot.log
else
  for M in 1 2 3; do
    MODEL=Qwen72 M=$M COND=main python src/run_games.py 2>&1 | tee -a pilot.log
  done
fi
python src/analyze.py runs 2>&1 | tee -a pilot.log
echo PILOT_DONE
