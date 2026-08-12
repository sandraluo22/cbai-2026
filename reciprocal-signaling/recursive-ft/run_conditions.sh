#!/bin/bash
# Sequential recursion runs: all four conditions from the same A_0/B_0 checkpoint.
# Gate downstream waiters on the RECURSION_DONE_<cond> markers in recursion_<cond>.log,
# never on pgrep (see project gotchas).
cd /workspace/mm/reciprocal-signaling/recursive-ft
export HF_HOME=/workspace/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for COND in reciprocal self frozen static; do
  echo "=== $COND $(date) ==="
  COND=$COND GENS=10 NPOOL=400 STEPS=60 SEED=0 \
    python run_recursion.py > recursion_${COND}.log 2>&1
  tail -2 recursion_${COND}.log
done
echo "ALL_CONDITIONS_DONE $(date)"
