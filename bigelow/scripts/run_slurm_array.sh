#!/usr/bin/env bash
#SBATCH --job-name=belief-feedback
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-17
# Slurm array wrapper: each array index runs one pipeline stage of the full
# configuration in dependency order. Because every stage is idempotent, the
# array can also be submitted with dependencies:
#   sbatch --dependency=afterok:<prev> scripts/run_slurm_array.sh configs/full.yaml
set -euo pipefail
CONFIG="${1:-configs/full.yaml}"
cd "$(dirname "$0")/.."

STAGES=(
  generate-worlds validate-data train-steering calibrate-steering
  run-exogenous-emission run-exogenous-receiver fit-models run-network
  run-recycling run-hysteresis run-phase-boundary run-jacobian
  run-mechanistic run-robustness analyze make-tables make-figures make-report
)
STAGE="${STAGES[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "Slurm task ${SLURM_ARRAY_TASK_ID:-0}: ${STAGE} (${CONFIG})"
python3 -m belief_feedback.cli "${STAGE}" --config "${CONFIG}"
