"""The same-space test (tier 2 + same weights): one dot product.

We hold two directions in the SAME vector space:
  v : the greed direction we steer B with        (steering.py)
  w : the direction in A that encodes A's-estimate-of-B's-greed (probes.py)

If cos(v, w) is high, A represents "B is greedy" using the machinery it would
use to BE greedy -- geometric evidence for simulation-theory opponent
modeling. If they're orthogonal, A maintains a dissociated "other-agent"
encoding. Either answer is a finding.

Both v and w are stored per hidden_states index, so the comparison is
layerwise; the null is the expected |cos| of random directions in d
dimensions, ~ 1/sqrt(d) (a permutation band is drawn too).

Output: samespace.npz + samespace.pdf/png; headline number printed.

Run:  python src/samespace.py --preset tier2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config          # noqa: E402
from steering import load_direction    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="tier2")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    run_dir = cfg.run_dir()
    if cfg.tier != 2:
        print("note: preset is tier 1 -- cos(v, w) is still computable but "
              "the clean interpretation needs tier 2 (alpha delivered via v).")

    v = load_direction(cfg)                                     # [L+1, d]
    w = np.load(os.path.join(run_dir, "probe_directions.npz"))["w"]
    assert v.shape == w.shape, (v.shape, w.shape)
    n_layers, d = v.shape

    def unit(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    cos = np.sum(unit(v) * unit(w), axis=1)                     # [L+1]

    # permutation null: random rotations of w's entries within each layer
    rng = np.random.default_rng(cfg.seed + 42)
    null = np.array([[float(np.dot(unit(v[l]), unit(rng.permutation(w[l]))))
                      for _ in range(200)] for l in range(n_layers)])
    lo, hi = np.percentile(null, [2.5, 97.5], axis=1)

    r2 = np.load(os.path.join(run_dir, "probe_r2.npz"))["r2"]
    best_l = int(np.argmax(r2[:, -1]))
    print(f"expected |cos| for random directions: {1 / np.sqrt(d):.4f}")
    print(f"cos(v, w) at best probe layer {best_l}: {cos[best_l]:+.4f}")
    print(f"max |cos(v, w)| over layers: {np.abs(cos).max():.4f} "
          f"at layer {int(np.abs(cos).argmax())}")

    np.savez(os.path.join(run_dir, "samespace.npz"), cos=cos,
             null_lo=lo, null_hi=hi, best_probe_layer=best_l, dim=d)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(layers, lo, hi, color="gray", alpha=0.3,
                    label="permutation null (95%)")
    ax.plot(layers, cos, marker="o", label="cos(v, w)")
    ax.axvline(best_l, color="tab:red", ls=":", lw=1,
               label=f"best probe layer ({best_l})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("layer (hidden_states index)")
    ax.set_ylabel("cosine similarity")
    ax.set_title("does A represent B's greed with B's greed direction?\n"
                 f"v = steering dir, w = probe dir ({cfg.name})")
    ax.legend(fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(run_dir, f"samespace.{ext}"),
                    bbox_inches="tight", dpi=150)
    print(f"wrote samespace.npz and samespace plot -> {run_dir}")


if __name__ == "__main__":
    main()
