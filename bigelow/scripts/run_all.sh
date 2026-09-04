#!/usr/bin/env bash
# Complete dependency-ordered pipeline. Idempotent: each stage skips already
# completed, valid artifacts; the run stops on schema or integrity failure.
set -euo pipefail
CONFIG="${1:-configs/smoke.yaml}"
cd "$(dirname "$0")/.."

STAGES=(
  generate-worlds
  validate-data
  train-steering
  calibrate-steering
  run-exogenous-emission
  run-exogenous-receiver
  fit-models
  run-network
  run-recycling
  run-hysteresis
  run-phase-boundary
  run-jacobian
  run-mechanistic
  run-robustness
  analyze
  make-tables
  make-figures
  make-report
)

for stage in "${STAGES[@]}"; do
  echo "==================================================================="
  echo ">>> ${stage} (${CONFIG})"
  python3 -m belief_feedback.cli "${stage}" --config "${CONFIG}"
done
echo "Pipeline complete for ${CONFIG}."
