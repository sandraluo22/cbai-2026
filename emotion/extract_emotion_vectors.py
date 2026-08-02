"""Build per-emotion 'emotion vectors' from the collected Q activations,
following the Anthropic emotion-vectors recipe, adapted to GoEmotions:

  1. Labeled data .... GoEmotions examples grouped by their primary emotion.
  2. Per-example act . the Q activation at the COLLECTED ANCHOR TOKEN (the ':'
                       after Assistant) — one vector per example per layer.
                       (No 50th-token averaging; we use the single token that
                       was already hooked.)
  3. Mean + center ... average per emotion, then subtract the mean across the
                       (non-neutral) emotions -> strips generic "story-ness".
  4. Remove confounds  PCA (enough comps for 50% variance) of the NEUTRAL
                       examples' activations, projected out of each vector.

Done per layer. Outputs (under <run>/emotion_vectors/):
  emotion_vectors_raw.npy     (E, L, H)  centered, before confound removal
  emotion_vectors_clean.npy   (E, L, H)  after projecting out neutral PCs
  meta.json                   emotion order, per-emotion counts, n_pcs/layer
  emotion_cos_heatmaps.pdf    per-layer emotion-emotion cosine sim (validation)
  npcs_for_50pct_var.png      # neutral PCs needed for 50% var, by layer

Usage (on a box with the activations on disk):
  python extract_emotion_vectors.py results/all_full           # uses q_acts
  python extract_emotion_vectors.py results/all_full --act a1   # or a1/a2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from goemotions_utils import GOEMOTIONS_LABELS

NEUTRAL = GOEMOTIONS_LABELS.index("neutral")
EMO_IDS = [i for i in range(len(GOEMOTIONS_LABELS)) if i != NEUTRAL]  # 27
EMO_NAMES = [GOEMOTIONS_LABELS[i] for i in EMO_IDS]
VAR_TARGET = 0.5


def stream_accumulate(dat_path: Path, primary: np.ndarray, L: int, H: int,
                      chunk: int = 1024):
    """One sequential pass over the (N,L,H) float16 file. Returns:
      sums  (n_labels, L, H)  per-label activation sums
      cnts  (n_labels,)       per-label counts
      neutral_acts (n_neutral, L, H) float16  raw neutral activations
    """
    N = len(primary)
    rec = L * H * 2
    n_lab = len(GOEMOTIONS_LABELS)
    sums = np.zeros((n_lab, L, H), dtype=np.float64)
    cnts = np.zeros(n_lab, dtype=np.int64)
    n_neutral = int((primary == NEUTRAL).sum())
    neutral_acts = np.empty((n_neutral, L, H), dtype=np.float16)
    nptr = 0

    with open(dat_path, "rb") as f:
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            buf = f.read((end - start) * rec)
            x = np.frombuffer(buf, dtype="<f2").reshape(end - start, L, H)
            lab = primary[start:end]
            for e in np.unique(lab):
                m = lab == e
                sums[e] += x[m].astype(np.float64).sum(axis=0)
                cnts[e] += int(m.sum())
            nm = lab == NEUTRAL
            if nm.any():
                k = int(nm.sum())
                neutral_acts[nptr:nptr + k] = x[nm]
                nptr += k
            if start % (chunk * 20) == 0:
                print(f"  [scan] {end}/{N}")
    assert nptr == n_neutral
    return sums, cnts, neutral_acts


def neutral_pcs(neutral_layer: np.ndarray, var_target: float):
    """Top principal components of neutral activations (n, H) capturing
    >= var_target of variance. Returns components V (k, H) and k."""
    from sklearn.decomposition import PCA
    n, H = neutral_layer.shape
    max_c = min(n, H)
    pca = PCA(n_components=max_c, svd_solver="full")
    pca.fit(neutral_layer.astype(np.float32))
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, var_target) + 1)
    return pca.components_[:k], k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--act", default="q", choices=["q", "a1", "a2"])
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    L, H, N = meta["L"], meta["H"], meta["N"]
    primary = np.load(run_dir / "labels_primary.npy").astype(int)
    dat = run_dir / f"{args.act}_acts.dat"
    out = run_dir / "emotion_vectors"
    out.mkdir(exist_ok=True)
    print(f"[setup] act={args.act} N={N} L={L} H={H} -> {out}")

    # ---- pass 1: per-emotion sums + neutral activations ----
    sums, cnts, neutral_acts = stream_accumulate(dat, primary, L, H, args.chunk)
    print(f"[scan] done. neutral examples: {len(neutral_acts)}; "
          f"per-emotion counts min/median/max: "
          f"{cnts[EMO_IDS].min()}/{int(np.median(cnts[EMO_IDS]))}/{cnts[EMO_IDS].max()}")

    # ---- per-emotion means, then center across emotions ----
    means = sums[EMO_IDS] / cnts[EMO_IDS][:, None, None]      # (E, L, H)
    grand = means.mean(axis=0, keepdims=True)                 # mean across emotions
    raw = (means - grand).astype(np.float32)                  # (E, L, H)
    np.save(out / "emotion_vectors_raw.npy", raw)

    # ---- confound removal: project out neutral PCs (per layer) ----
    clean = raw.copy()
    npcs = []
    for l in range(L):
        V, k = neutral_pcs(neutral_acts[:, l, :], VAR_TARGET)   # (k, H)
        npcs.append(k)
        proj = raw[:, l, :] @ V.T          # (E, k)
        clean[:, l, :] = raw[:, l, :] - proj @ V
        print(f"  [deconfound] layer {l}: {k} PCs for {int(VAR_TARGET*100)}% var")
    np.save(out / "emotion_vectors_clean.npy", clean)

    meta_out = {
        "act": args.act, "emotion_ids": EMO_IDS, "emotion_names": EMO_NAMES,
        "L": L, "H": H, "var_target": VAR_TARGET,
        "n_neutral": int(len(neutral_acts)),
        "counts": {GOEMOTIONS_LABELS[i]: int(cnts[i]) for i in EMO_IDS},
        "npcs_per_layer": npcs,
        "centering": "subtract mean across the 27 non-neutral emotion means",
    }
    (out / "meta.json").write_text(json.dumps(meta_out, indent=2))

    # ---- validation viz ----
    _heatmaps(out, clean)
    _npcs_plot(out, npcs)
    print(f"[done] emotion vectors + viz in {out}")


def _heatmaps(out: Path, vecs: np.ndarray):
    """Per-layer emotion x emotion cosine-similarity heatmap (clean vectors)."""
    E, L, H = vecs.shape
    with PdfPages(out / "emotion_cos_heatmaps.pdf") as pdf:
        for l in range(L):
            X = vecs[:, l, :]
            Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
            C = Xn @ Xn.T
            fig, ax = plt.subplots(figsize=(8, 7), dpi=110)
            im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(E)); ax.set_yticks(range(E))
            ax.set_xticklabels(EMO_NAMES, rotation=90, fontsize=5)
            ax.set_yticklabels(EMO_NAMES, fontsize=5)
            ax.set_title(f"emotion-vector cosine sim — layer {l}/{L-1}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _npcs_plot(out: Path, npcs):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=130)
    ax.plot(range(len(npcs)), npcs, "-o", ms=3)
    ax.set_xlabel("layer"); ax.set_ylabel(f"# neutral PCs for {int(VAR_TARGET*100)}% var")
    ax.set_title("Confound subspace size by layer"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "npcs_for_50pct_var.png"); plt.close(fig)


if __name__ == "__main__":
    main()
