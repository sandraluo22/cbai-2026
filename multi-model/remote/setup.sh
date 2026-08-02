#!/usr/bin/env bash
# Install deps on the GPU pod. torch ships preinstalled; don't touch it.
set -euo pipefail
PIP="pip install --break-system-packages -q"
$PIP 'transformers>=4.56' 'datasets>=2.19' 'scikit-learn>=1.3' 'huggingface_hub>=0.24' numpy matplotlib || \
  pip install -q 'transformers>=4.56' 'datasets>=2.19' 'scikit-learn>=1.3' 'huggingface_hub>=0.24' numpy matplotlib
python - <<'PY'
import torch, transformers, datasets, sklearn
print("cuda:", torch.cuda.is_available(),
      "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| datasets", datasets.__version__, "| sklearn", sklearn.__version__)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
