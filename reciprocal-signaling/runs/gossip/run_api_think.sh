#!/bin/bash
set -x
cd /Users/sandraluo/cbai-2026/reciprocal-signaling/runs/gossip
eval "$(grep -E '^export (ANTHROPIC|OPENAI)_API_KEY' ~/.zshrc)"
unset ANTHROPIC_BASE_URL
A=11110111101111011110; B=01000010000100001000
for M in claude-sonnet-5 o4-mini; do
for SEED in 0 1; do
  THINK=1024 MODEL=$M VAR=curve SCHED="$A;$B" ROUNDS=20 STEPS=40 SEED=$SEED \
    OUT=xmodel_think/${M}_think_s$SEED python3 qsg_api.py 2>&1 | tee -a api_think.log
done
done
echo API_THINK_DONE | tee -a api_think.log
