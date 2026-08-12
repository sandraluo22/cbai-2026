#!/bin/bash
# Seed replications of the two recursive conditions (the oscillation/sharpening claims).
cd /workspace/mm/reciprocal-signaling/recursive-ft
export HF_HOME=/workspace/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for SEED in 1 2; do
  for COND in reciprocal self; do
    echo "=== $COND s$SEED $(date) ==="
    COND=$COND GENS=10 NPOOL=400 STEPS=60 SEED=$SEED \
      python run_recursion.py > recursion_${COND}_s${SEED}.log 2>&1
    tail -2 recursion_${COND}_s${SEED}.log
  done
done
echo "ALL_SEEDS_DONE $(date)"
