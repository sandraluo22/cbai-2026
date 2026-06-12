"""Per-layer activation similarity heatmaps + a layer-by-layer slideshow (GIF).

Unlike the belief heatmaps (which use the K-way simplex belief), these are built
from the SAVED RESIDUAL-STREAM ACTIVATIONS at the anchor. For each layer L:

    population activation state per round  =  mean over agents of activations[round, :, L, :]
    cell S[i,j]                            =  cosine similarity( state@round_i , state@round_j )

One round×round heatmap per layer; assembled into activation_slideshow.gif stepping
through all layers. Saved per representative run, alongside per-layer PNGs and the
raw matrices (.npy).

    python generate_activation_slideshows.py            # representative run per experiment
    python generate_activation_slideshows.py <run_dir>  # a specific run
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from qsg.activations import load_activations


def pop_cosine(arr, layer: int) -> np.ndarray:
    """COSINE OF THE MEANS: average agents first, then cosine between rounds.
    states[round] = mean_agents act[round,:,layer]; S[i,j] = cos(states[i], states[j])."""
    states = np.asarray(arr[:, :, layer, :].mean(axis=1), dtype=np.float64)   # (rounds, hidden)
    u = states / np.clip(np.linalg.norm(states, axis=1, keepdims=True), 1e-12, None)
    return u @ u.T


def mean_cosine(arr, layer: int) -> np.ndarray:
    """MEAN OF THE COSINES: cosine per agent between rounds, then average over agents.
    S[i,j] = (1/N) Σ_n cos(act[i,n,layer], act[j,n,layer])  (same agent n matched
    across rounds). Keeps each agent's vector instead of collapsing to the mean."""
    a = np.asarray(arr[:, :, layer, :], dtype=np.float64)                     # (rounds, N, hidden)
    u = a / np.clip(np.linalg.norm(a, axis=2, keepdims=True), 1e-12, None)
    return np.einsum("inh,jnh->ij", u, u) / u.shape[1]


METRICS = {
    "cos_of_means": (pop_cosine, "", "cos-of-means"),
    "mean_cos": (mean_cosine, "_meancos", "mean-of-cosines (per-agent)"),
}


def make_slideshow(run_dir: str | Path, title: str, metric: str = "cos_of_means") -> tuple[Path, int]:
    run_dir = Path(run_dir)
    metric_fn, suffix, metric_label = METRICS[metric]
    arr, meta = load_activations(run_dir)
    n_layers = meta["n_layers"]
    layer_ids = meta["layer_indices"]

    mats = [metric_fn(arr, li) for li in range(n_layers)]
    vmin = min(float(m.min()) for m in mats)          # shared color scale across layers

    out = run_dir / "plots" / f"activation_layers{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = run_dir / "plots" / f"activation_slideshow{suffix}.pdf"
    with PdfPages(pdf_path) as pdf:
        for li in range(n_layers):
            fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=130)
            im = ax.imshow(mats[li], origin="lower", cmap="magma", vmin=vmin, vmax=1.0)
            ax.set_title(f"{title}\n{metric_label} — layer {layer_ids[li]}/{layer_ids[-1]}")
            ax.set_xlabel("round j"); ax.set_ylabel("round i")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out / f"layer_{layer_ids[li]:02d}.png")
            pdf.savefig(fig)
            plt.close(fig)

    np.save(run_dir / "plots" / f"activation_simmats{suffix}.npy", np.stack(mats))
    return pdf_path, n_layers


# representative run (seed0, full/non-ablate) per experiment
REPRESENTATIVE = {
    "results_exp":   ("results_exp/text_two_layer_N6_a0p3_m1p0_seed0", "exp1: detailed clues (two_layer)"),
    "results_exp2":  ("results_exp2/text_two_layer_N6_a0p3_m1p0_seed0", "exp2: generic attributes (two_layer)"),
    "results_hard":  ("results_hard/text_reasoning_exchange_N8_a0p3_m1p0_seed0", "hard: 1 ambiguous clue (reasoning)"),
    "results_talk":  ("results_talk/text_reasoning_exchange_N8_a0p3_m1p0_seed0", "talk: QSG reasoning exchange"),
    "results_image": ("results_image/image_reasoning_exchange_N6_a0p3_m1p0_seed0", "image: elephant patches (Qwen2-VL)"),
}


def main():
    # metric: "mean_cos" (default, the new one) | "cos_of_means" | "both"
    metric = sys.argv[1] if len(sys.argv) > 1 else "mean_cos"
    metrics = ["cos_of_means", "mean_cos"] if metric == "both" else [metric]
    for exp, (rd, title) in REPRESENTATIVE.items():
        if not (Path(rd) / "activations").exists():
            print(f"{exp:14s} -> no activations at {rd}")
            continue
        for m in metrics:
            pdf, L = make_slideshow(rd, title, metric=m)
            print(f"{exp:14s} [{m:12s}] -> {pdf}  ({L} layers)")


if __name__ == "__main__":
    main()
