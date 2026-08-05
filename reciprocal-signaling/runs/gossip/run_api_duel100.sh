#!/bin/bash
set -x
cd /Users/sandraluo/cbai-2026/reciprocal-signaling/runs/gossip
eval "$(grep -E '^export (ANTHROPIC|OPENAI)_API_KEY' ~/.zshrc)"
unset ANTHROPIC_BASE_URL
A=11111111111111111111; B=00000000000000000000
for M in claude-sonnet-5 claude-haiku-4-5-20251001 gpt-4o; do
for SEED in 0 1; do
  MODEL=$M VAR=curve SCHED="$A;$B" ROUNDS=20 STEPS=40 SEED=$SEED \
    OUT=duel100/${M}_s$SEED python3 qsg_api.py 2>&1 | tee -a api_duel100.log
done
done
echo API_DUEL100_DONE | tee -a api_duel100.log
