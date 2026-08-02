#!/bin/bash
# gossip v2.1 (independence sentence) + elephant quiz battery
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd runs/gossip
for V in none informed_r1 informed_all misinformed_r1 misinformed_all; do
  VAR=$V python qsg_gossip.py 2>&1 | tee -a gossip21.log
done
python gossip_analyze.py . 2>&1 | tee -a gossip21.log
cd ../..
for MODEL in Qwen32 Llama70 Qwen72; do
  MODEL=$MODEL ROUNDS=5 SEEDS=2 python src/elephant.py 2>&1 | tee -a batch3.log
done
echo BATCH3_DONE
