"""10-line example: pull all agents' layer-L activations for a given round.

This is exactly the input the later linear-transformation project needs:
an [N, hidden_dim] matrix of agents' internal belief representations.

    python scripts/load_activations_example.py results/<run_id> --round 5 --layer 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qsg.activations import layer_matrix_for_round, load_activations

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--round", type=int, default=0)
ap.add_argument("--layer", type=int, default=0)
args = ap.parse_args()

arr, meta = load_activations(args.run_dir)               # (rounds+1, N, n_layers, hidden_dim)
print("store shape:", meta["shape"], "dtype:", meta["dtype"])
print("model:", meta["model_name"], "| anchor_token_id:", meta["anchor_token_id"])

X = layer_matrix_for_round(arr, meta, round_idx=args.round, layer=args.layer)
print(f"layer {args.layer}, round {args.round}: X.shape = {X.shape}  # [N, hidden_dim]")
print("per-agent activation norms:", [round(float((X[i] ** 2).sum() ** 0.5), 2) for i in range(X.shape[0])])
