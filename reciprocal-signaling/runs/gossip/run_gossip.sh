#!/bin/bash
# All five gossip variations, one seed, Qwen32.
set -x
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for V in none informed_r1 informed_all misinformed_r1 misinformed_all; do
  VAR=$V python qsg_gossip.py 2>&1 | tee -a gossip.log
done
python gossip_analyze.py . 2>&1 | tee -a gossip.log
echo GOSSIP_DONE
