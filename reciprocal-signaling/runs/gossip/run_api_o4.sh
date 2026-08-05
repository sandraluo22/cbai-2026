#!/bin/bash
set -x
cd /Users/sandraluo/cbai-2026/reciprocal-signaling/runs/gossip
eval "$(grep -E '^export OPENAI_API_KEY' ~/.zshrc)"
A=11110111101111011110; B=01000010000100001000
for SEED in 0 1; do
  THINK=2048 MODEL=o4-mini VAR=curve SCHED="$A;$B" ROUNDS=20 STEPS=40 SEED=$SEED \
    OUT=xmodel_think/o4-mini_think_s$SEED python3 qsg_api.py 2>&1 | tee -a api_o4.log
done
echo API_O4_DONE | tee -a api_o4.log
