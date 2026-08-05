#!/bin/bash
# batch39 (pod B): scripted tower in the non-converging hidden-profile game.
# penguin @ Qwen72 (historically 0/6 convergence). Arms: truth tower ("penguin"),
# liar tower ("puffin"), both. 2 seeds, 20 rounds, base mode (no notes).
set -x
cd /workspace/mm/reciprocal-signaling
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q BATCH35_DONE runs/gossip/batch35.log 2>/dev/null; do sleep 180; done
for TW in "penguin" "puffin" "penguin;puffin"; do
  TAG=$(echo "$TW" | tr ';' '-')
  TOWER="$TW" MODEL=Qwen72 QUIZ=penguin ROUNDS=20 SEEDS=2 \
    OUT=runs/elephant_tower/Qwen72_$TAG python src/elephant.py 2>&1 | tee -a runs/gossip/batch39.log
done
echo BATCH39_DONE | tee -a runs/gossip/batch39.log
