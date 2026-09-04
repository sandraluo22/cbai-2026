#!/usr/bin/env bash
# Collect Slurm run artifacts into a timestamped bundle for transfer.
set -euo pipefail
CONFIG_NAME="${1:-full}"
cd "$(dirname "$0")/.."
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="collected_${CONFIG_NAME}_${STAMP}.tar.gz"
tar czf "${OUT}" \
  "artifacts/runs/${CONFIG_NAME}" \
  "artifacts/models/${CONFIG_NAME}" \
  "artifacts/figures/${CONFIG_NAME}" \
  "artifacts/figure_data/${CONFIG_NAME}" \
  "artifacts/tables/${CONFIG_NAME}" \
  "artifacts/reports/${CONFIG_NAME}" \
  "artifacts/manifests/${CONFIG_NAME}" \
  "artifacts/vectors/${CONFIG_NAME}" 2>/dev/null || true
echo "wrote ${OUT}"
