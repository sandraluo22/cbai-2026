#!/bin/bash
set -x
cd /Users/sandraluo/cbai-2026/reciprocal-signaling/runs/gossip
eval "$(grep -E '^export (ANTHROPIC|OPENAI)_API_KEY' ~/.zshrc)"
unset ANTHROPIC_BASE_URL
T=11111111111111111111; L=00000000000000000000
for M in o4-mini; do
for SEED in 2 3; do
  THINK=2048 MODEL=$M VAR=curve SCHED="$T;$L" ROUNDS=20 STEPS=40 SEED=$SEED \
    OUT=duel_think/${M}_think_duel_s$SEED python3 qsg_api.py 2>&1 | tee -a api_o4duel2.log
done
done
echo API_O4DUEL2_DONE | tee -a api_o4duel2.log
