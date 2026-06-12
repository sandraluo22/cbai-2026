#!/usr/bin/env bash
# Install deps on a fresh runpod box. Run ON the pod (run.sh does this for you).
# The base image is PEP-668 "externally managed" and ephemeral, so installing
# into the system env with --break-system-packages is the right call here.
# torch/numpy/Pillow ship preinstalled; the pins below are already satisfied so
# pip only fetches transformers/datasets/scikit-learn/matplotlib/accelerate.
set -euo pipefail
PIP="pip install --break-system-packages"
$PIP -r requirements.txt
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
