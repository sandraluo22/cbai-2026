"""Ridge probes: predict B's alpha from A's residual stream, per layer per
turn. The headline plot is probe R^2 as a function of turn number and layer --
A's internal estimate of B sharpening over the episode, watched from inside.

Also fits the same probes on the shadow observer's activations (if captured)
so the participation-vs-spectation gap is one overlay, and saves per-layer
probe DIRECTIONS w (final turn, normalized ridge coefficients) for the causal
step and the same-space test.

Outputs under <run_dir>/
  probe_r2.npz            r2 [n_layers+1, n_rounds] (+ shadow_r2 if present)
  probe_directions.npz    w  [n_layers+1, d] (final-turn probes, unit norm)
  probe_heatmap.pdf/png   layers x turns heatmap
  probe_curves.pdf/png    R^2 vs turn at selected layers (+ shadow at best)

Run:  python src/probes.py --preset default
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config      # noqa: E402
from episodes import load_all_shards       # noqa: E402


def split_by_episode(n: int, test_frac: float, seed: int):
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    return perm[n_test:], perm[:n_test]     # train_idx, test_idx


def fit_r2_grid(acts: np.ndarray, alpha: np.ndarray, cfg: Config):
    """acts [N, T, L+1, d] fp16, alpha [N]. Returns (r2 [L+1, T],
    w [L+1, d] final-turn unit-norm probe directions). Held out BY EPISODE;
    ridge penalty chosen per cell by efficient LOO-CV on the train split."""
    from sklearn.linear_model import RidgeCV

    n, n_turns, n_layers, _ = acts.shape
    tr, te = split_by_episode(n, cfg.test_frac, cfg.seed)
    y_tr, y_te = alpha[tr], alpha[te]
    r2 = np.zeros((n_layers, n_turns))
    w = np.zeros((n_layers, acts.shape[-1]), dtype=np.float32)

    for l in range(n_layers):
        for t in range(n_turns):
            X = acts[:, t, l, :].astype(np.float32)
            reg = RidgeCV(alphas=cfg.ridge_alphas).fit(X[tr], y_tr)
            r2[l, t] = reg.score(X[te], y_te)
            if t == n_turns - 1:
                coef = reg.coef_.astype(np.float32)
                nrm = np.linalg.norm(coef)
                w[l] = coef / nrm if nrm > 0 else coef
        print(f"  layer {l:2d}: R2 turn1={r2[l, 0]:+.3f} "
              f"turnT={r2[l, -1]:+.3f}")
    return r2, w


def plot_all(r2: np.ndarray, shadow_r2, cfg: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers, n_turns = r2.shape
    run_dir = cfg.run_dir()

    # heatmap: layers x turns
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.clip(r2, 0, None), aspect="auto", origin="lower",
                   cmap="viridis", extent=[0.5, n_turns + 0.5, -0.5,
                                           n_layers - 0.5])
    ax.set_xlabel("turn (round)")
    ax.set_ylabel("layer (hidden_states index)")
    ax.set_title(f"probe R$^2$ for B's $\\alpha$ from A's residual stream "
                 f"({cfg.name})")
    fig.colorbar(im, label="held-out R$^2$ (clipped at 0)")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(run_dir, f"probe_heatmap.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)

    # curves: R^2 vs turn at a few layers + best layer, with shadow overlay
    best_l = int(np.argmax(r2[:, -1]))
    picks = sorted({1, n_layers // 4, n_layers // 2, 3 * n_layers // 4,
                    n_layers - 1, best_l})
    fig, ax = plt.subplots(figsize=(6, 4))
    turns = np.arange(1, n_turns + 1)
    for l in picks:
        lw, z = (2.5, 3) if l == best_l else (1.2, 2)
        ax.plot(turns, r2[l], marker="o", lw=lw, zorder=z,
                label=f"layer {l}" + (" (best)" if l == best_l else ""))
    if shadow_r2 is not None:
        ax.plot(turns, shadow_r2[int(np.argmax(shadow_r2[:, -1]))],
                marker="s", ls="--", color="k", lw=2,
                label="shadow observer (best layer)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("turn (round)")
    ax.set_ylabel("held-out R$^2$")
    ax.set_title(f"A's internal estimate of B's $\\alpha$ sharpens over the "
                 f"episode ({cfg.name})")
    ax.legend(fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(run_dir, f"probe_curves.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="default")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    run_dir = cfg.run_dir()

    acts, alpha, episode, fallback = load_all_shards(
        os.path.join(run_dir, "acts"))
    print(f"A acts: {acts.shape}, alpha in [{alpha.min():.2f}, "
          f"{alpha.max():.2f}], median fallbacks/ep {np.median(fallback):.0f}")
    r2, w = fit_r2_grid(acts, alpha, cfg)

    shadow_r2 = None
    sdir = os.path.join(run_dir, "shadow_acts")
    if os.path.isdir(sdir) and any(f.startswith("acts_")
                                   for f in os.listdir(sdir)):
        s_acts, s_alpha, s_ep, _ = load_all_shards(sdir)
        # align on common episodes (shadow may lag behind the corpus)
        common = np.intersect1d(episode, s_ep)
        a_sel = np.isin(s_ep, common)
        print(f"shadow acts: {s_acts.shape} ({common.size} common episodes)")
        shadow_r2, _ = fit_r2_grid(s_acts[a_sel], s_alpha[a_sel], cfg)

    out = {"r2": r2, "layers": np.arange(r2.shape[0]),
           "turns": np.arange(1, r2.shape[1] + 1)}
    if shadow_r2 is not None:
        out["shadow_r2"] = shadow_r2
    np.savez(os.path.join(run_dir, "probe_r2.npz"), **out)
    np.savez(os.path.join(run_dir, "probe_directions.npz"), w=w)

    best_l = int(np.argmax(r2[:, -1]))
    print(f"best layer {best_l}: R2 by turn "
          + " ".join(f"{v:+.2f}" for v in r2[best_l]))
    if shadow_r2 is not None:
        gap = r2[best_l, -1] - shadow_r2.max(0)[-1]
        print(f"participation gap at final turn (A best - shadow best): "
              f"{gap:+.3f}")
    plot_all(r2, shadow_r2, cfg)
    print(f"wrote probe_r2.npz, probe_directions.npz and plots -> {run_dir}")


if __name__ == "__main__":
    main()
