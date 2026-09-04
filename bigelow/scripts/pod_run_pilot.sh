#!/usr/bin/env bash
# Pilot pipeline launcher for the GPU pod.
set -euo pipefail
cd /workspace/bigelow
export HF_HOME=/workspace/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
bash scripts/run_all.sh configs/pilot.yaml
