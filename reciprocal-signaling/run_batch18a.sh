#!/bin/bash
# batch18a (.149): elephant + scratchpad notes, Qwen32. Waits for batch17a.
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch17[a].sh' >/dev/null; do sleep 60; done
MODEL=Qwen32 NOTES=append QUIZ=elephant,penguin,shadow,n30,n17 ROUNDS=20 SEEDS=2 OUT=runs/elephant_notes/Qwen32 python src/elephant.py 2>&1 | tee -a batch18.log
echo BATCH18A_DONE
