#!/bin/bash
# batch18b (.140): elephant + scratchpad notes, Qwen72. Waits for batch17b.
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f 'run_batch17[b].sh' >/dev/null; do sleep 60; done
MODEL=Qwen72 NOTES=append QUIZ=elephant,penguin,shadow,n30,n17 ROUNDS=20 SEEDS=2 OUT=runs/elephant_notes/Qwen72 python src/elephant.py 2>&1 | tee -a batch18.log
echo BATCH18B_DONE
