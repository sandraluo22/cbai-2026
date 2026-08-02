"""Tier 2: extract the greed/selfishness direction v via contrastive prompts.

v[k] = unit-normed( mean(last-token residual | greedy phrasing)
                  - mean(last-token residual | generous phrasing) )
for every hidden_states index k (0 = embeddings, k = output of block k-1).

During tier-2 episode generation, B's forward passes run under
Steering(model, v, cfg.steer_layers, coef = cfg.steer_scale * alpha):
the latent now lives in B's activations, not its prompt, which is what makes
the later same-space test cos(v, w) clean.

Run:
  python src/steering.py --preset tier2              # extract + save v
  python src/steering.py --preset tier2 --calibrate  # eyeball coherence vs scale
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config                      # noqa: E402
from modeling import (Steering, capture_last_token_text,   # noqa: E402
                      generate, load_model)
from prompts import contrast_pairs                         # noqa: E402


DIRECTION_FILE = "greed_direction.npz"


def extract_direction(model, tok, cfg: Config) -> np.ndarray:
    """Return v as float32 [n_blocks+1, d], unit norm per layer."""
    greedy, generous = [], []
    for g_text, n_text in contrast_pairs(cfg.pie, cfg.n_contrast_pairs):
        greedy.append(capture_last_token_text(model, tok, g_text))
        generous.append(capture_last_token_text(model, tok, n_text))
    v = (np.stack(greedy).astype(np.float32).mean(0)
         - np.stack(generous).astype(np.float32).mean(0))       # [L+1, d]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def save_direction(v: np.ndarray, cfg: Config) -> str:
    path = os.path.join(cfg.run_dir(), DIRECTION_FILE)
    np.savez(path, v=v, steer_layers=np.array(cfg.steer_layers))
    return path


def load_direction(cfg: Config) -> np.ndarray:
    path = os.path.join(cfg.run_dir(), DIRECTION_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python src/steering.py --preset "
            f"{cfg.name}` first (tier 2 needs the greed direction)."
        )
    return np.load(path)["v"].astype(np.float32)


def calibrate(model, tok, cfg: Config, v: np.ndarray) -> None:
    """Print sample generations at several steering scales so the operator can
    pick the largest cfg.steer_scale at which B stays coherent (the usual
    failure mode of too-large scales is degenerate repetition)."""
    from prompts import FORMAT_B_OFFER, system_b_tier2
    messages = [
        {"role": "system", "content": system_b_tier2(cfg.pie, cfg.n_rounds)},
        {"role": "user", "content":
            f"[Round 1/{cfg.n_rounds}. Totals so far -- you: 0, opponent: 0.] "
            f"Make your opening offer. {FORMAT_B_OFFER.format(pie=cfg.pie)}"},
    ]
    s = cfg.steer_scale
    for coef in [-2 * s, -s, -s / 2, 0.0, s / 2, s, 2 * s]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            steer = Steering(model, v, cfg.steer_layers, coef)
            text = generate(model, tok, messages, cfg, steer=steer)
        print(f"--- coef={coef:+5.2f} ---\n{text}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="tier2")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    cfg = get_config(args.preset)

    torch.manual_seed(cfg.seed)
    model, tok = load_model(cfg)
    v = extract_direction(model, tok, cfg)
    path = save_direction(v, cfg)
    print(f"saved greed direction {v.shape} -> {path}")
    if args.calibrate:
        calibrate(model, tok, cfg, v)


if __name__ == "__main__":
    main()
