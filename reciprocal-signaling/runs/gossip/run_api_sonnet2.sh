#!/bin/bash
set -x
cd "$(dirname "$0")"
eval "$(grep -E '^export (ANTHROPIC|OPENAI)_API_KEY' ~/.zshrc)"
unset ANTHROPIC_BASE_URL
A="11110111101111011110"; B="01000010000100001000"
for M in claude-sonnet-5; do
for SEED in 2 3; do
  MODEL=$M VAR=curve SCHED="$A;$B" ROUNDS=20 STEPS=40 SEED=$SEED OUT=xmodel/${M}_none_s$SEED python3 qsg_api.py 2>&1 | tee -a api_sonnet2.log
  MODEL=$M VAR=curve SCHED="$A;$B" ROUNDS=20 STEPS=40 NOTES=append SEED=$SEED OUT=xmodel/${M}_notes_s$SEED python3 qsg_api.py 2>&1 | tee -a api_sonnet2.log
done
done
echo API_SONNET2_DONE
