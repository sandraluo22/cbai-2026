#!/bin/bash
# Stage 4: push each direction into the iterated games. Layers default to the three
# that best moved the held-out trust question in build_vectors' validation stage.
set -x
cd /workspace/mm/trust-vector
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until grep -qE 'TRUST_BUILD_DONE|Traceback' build.log 2>/dev/null; do sleep 60; done

# main grid: all six directions (four candidates + two controls), both schedules
MODEL=${MODEL:-Qwen32} ALPHA=${ALPHA:-0.25} STAGES=grid,curve \
  python src/steer_qsg.py 2>&1 | tee steer.log
cp out/steer_qsg.json out/steer_qsg_a${ALPHA:-0.25}.json

# dose-response: an effect that does not scale with alpha is not the vector's
for A in 0.5 1.0; do
  ALPHA=$A STAGES=grid python src/steer_qsg.py 2>&1 | tee steer_a$A.log
  cp out/steer_qsg.json out/steer_qsg_a$A.json
done
echo TRUST_STEER_DONE | tee -a steer.log
