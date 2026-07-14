"""Divider-basis decomposition: HOW is the distributed node representation laid out?

The node-mean representation is high-rank -- the 2-D coord-probe caught only ~1/4 of the causal
drive (see steer_normmatched / removal_probe). This asks how the REST is distributed by projecting
the representation onto an interpretable, COMPLETE basis of graph CUTS.

Two complementary bases, both over the n nodes:
  (A) Laplacian eigenmodes  -- eigenvectors of the graph Laplacian (built from the saved adjacency,
      so it is topology-correct for grid / ring / hex). Each is a standing wave over the nodes = a
      divider, ordered by spatial frequency (eigenvalue). The lowest non-trivial modes are the
      coordinate ramps (what coord_decode probes); higher modes are quarters / diagonals; the top
      mode of a bipartite graph is the checkerboard (parity). Complete + orthogonal, so the POWER
      SPECTRUM (fraction of node-mean variance per mode) partitions 100% of the structure.
  (B) Named dividers -- x, y, diagonals, checkerboard(parity), centre/edge, built from coords +
      graph 2-colouring. Greedy orthogonal matching pursuit -> ordered NAMED basis with cumulative
      variance: "how many human-nameable cuts reconstruct most of the map."

Also: data-driven PCA node-loadings (the model's OWN axes) matched to eigenmodes, and a node-label
shuffle null.

Reads either (a) a node-mean npz from capture_nodemeans.py (layer_* = (n,d) + adjacency + coords),
or (b) a per-occurrence acts_sub npz (layer_* + meta_node + meta_context_length) -- grid only.
CPU-only. Env: ACTS GRAPH TAG ROWS(4) COLS(4) CTXLO(100) OUTDIR
Out: <OUTDIR>/divider_basis_<TAG>_<graph>.json + .pdf
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ACTS = os.environ.get("ACTS", "runs/v2/square_grid/Llama_acts_sub.npz")
GRAPH = os.environ.get("GRAPH", "square_grid")
ROWS = int(os.environ.get("ROWS", "4")); COLS = int(os.environ.get("COLS", "4"))
CTXLO = int(os.environ.get("CTXLO", "100"))
TAG = os.environ.get("TAG", "Llama")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition/divider_basis")


def grid_adjacency(rows, cols):
    n = rows * cols; A = np.zeros((n, n))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    A[i, rr * cols + cc] = 1
    return A


def two_colour(A):
    """BFS 2-colouring; returns +-1 vector if bipartite else None."""
    n = A.shape[0]; col = np.full(n, 0);
    for s in range(n):
        if col[s] != 0: continue
        col[s] = 1; stack = [s]
        while stack:
            u = stack.pop()
            for v in np.where(A[u] > 0)[0]:
                if col[v] == 0: col[v] = -col[u]; stack.append(v)
                elif col[v] == col[u]: return None
    return col.astype(float)


def named_dividers(coords, A):
    x, y = coords[:, 0], coords[:, 1]
    d = {"x (coord 0)": x, "y (coord 1)": y,
         "diagonal (x+y)": x + y, "anti-diag (x-y)": x - y,
         "centre|edge": -(np.abs(x - x.mean()) + np.abs(y - y.mean()))}
    par = two_colour(A)
    if par is not None:
        d["checkerboard (parity)"] = par
    out = {}
    for k, v in d.items():
        v = v - v.mean(); nrm = np.linalg.norm(v)
        if nrm > 1e-9: out[k] = v / nrm
    return out


def band_power(V, w, Hc):
    coeff = V.T @ Hc; p = (coeff ** 2).sum(1); p[0] = 0.0
    frac = p / (p.sum() + 1e-12)
    bands = {}
    for k in range(1, len(w)):
        bands.setdefault(round(float(w[k]), 3), []).append(k)
    return frac, bands


def greedy_named(named, Hc):
    keys = list(named); B = np.stack([named[k] for k in keys], 0)
    tot = (Hc ** 2).sum() + 1e-12; R = Hc.copy(); used = []; order = []; cum = 0.0
    for _ in range(len(keys)):
        best, bk, bproj = -1, None, None
        for i, k in enumerate(keys):
            if k in used: continue
            u = B[i][:, None]; proj = u @ (u.T @ R); ev = (proj ** 2).sum() / tot
            if ev > best: best, bk, bproj = ev, k, proj
        used.append(bk); cum += best; order.append((bk, float(best), float(cum))); R = R - bproj
    return order


def load_reprs(npz):
    """Return (Hc_by_L, n, A, coords, rows, cols). Supports node-mean or per-occurrence npz."""
    files = set(npz.files); nL = sum(1 for k in files if k.startswith("layer_"))
    if "meta_node" in files:                                   # per-occurrence (grid only)
        node = npz["meta_node"]; ctx = npz["meta_context_length"]; m = ctx >= CTXLO
        n = ROWS * COLS; A = grid_adjacency(ROWS, COLS)
        coords = np.array([[r, c] for r in range(ROWS) for c in range(COLS)], float)
        rows, cols = ROWS, COLS
        Hc_by_L = {}
        for L in range(nL):
            H = npz[f"layer_{L}"].astype(np.float64)
            means = np.full((n, H.shape[1]), np.nan)
            for j in range(n):
                sel = m & (node == j)
                if sel.sum(): means[j] = H[sel].mean(0)
            ok = np.isfinite(means).all(1); Hc = np.zeros_like(means); Hc[ok] = means[ok] - means[ok].mean(0)
            Hc_by_L[L] = Hc
    else:                                                       # node-mean npz
        A = np.array(npz["adjacency"], float); coords = np.array(npz["coords"], float)
        rows = int(npz["rows"][0]); cols = int(npz["cols"][0]); n = A.shape[0]
        Hc_by_L = {}
        for L in range(nL):
            H = npz[f"layer_{L}"].astype(np.float64)
            Hc_by_L[L] = H - H.mean(0)
    return Hc_by_L, n, A, coords, rows, cols, nL


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    npz = np.load(ACTS, allow_pickle=True)
    Hc_by_L, n, A, coords, rows, cols, nL = load_reprs(npz)
    L = np.diag(A.sum(1)) - A; w, V = np.linalg.eigh(L)
    named = named_dividers(coords, A)

    spec = np.zeros((nL, n)); lowpow = np.zeros(nL)
    for Ly in range(nL):
        frac, bands = band_power(V, w, Hc_by_L[Ly]); spec[Ly] = frac
        lowbands = sorted(bands)[:1]
        lowpow[Ly] = sum(frac[k] for b in lowbands for k in bands[b])
    Lstar = int(np.argmax(lowpow))
    frac, bands = band_power(V, w, Hc_by_L[Lstar])
    band_rows = [{"eigenvalue": ev, "modes": bands[ev], "power_frac": float(sum(frac[k] for k in bands[ev])),
                  "degeneracy": len(bands[ev])} for ev in sorted(bands)]
    order = np.argsort(frac)[::-1]; cum = np.cumsum(frac[order])
    n80 = int(np.searchsorted(cum, 0.8) + 1); n90 = int(np.searchsorted(cum, 0.9) + 1)
    greedy = greedy_named(named, Hc_by_L[Lstar])

    Hc = Hc_by_L[Lstar]; Us, Ss, _ = np.linalg.svd(Hc, full_matrices=False)
    pcvar = (Ss ** 2) / (Ss ** 2).sum(); pc_match = []
    for i in range(min(6, Us.shape[1])):
        cos = np.abs(V.T @ Us[:, i]); k = int(np.argmax(cos[1:]) + 1)
        ncos = {nm: abs(float(named[nm] @ Us[:, i])) for nm in named}; bestnm = max(ncos, key=ncos.get)
        pc_match.append({"pc": i, "var": float(pcvar[i]), "best_mode": k, "cos_mode": float(cos[k]),
                         "best_named": bestnm, "cos_named": ncos[bestnm]})

    rng = np.random.default_rng(0); null_low = []
    for _ in range(200):
        f, b = band_power(V, w, Hc[rng.permutation(n)]); null_low.append(sum(f[k] for k in b[sorted(b)[0]]))
    null_low = np.array(null_low)

    out = {"graph": GRAPH, "tag": TAG, "n": n, "nL": nL, "Lstar": Lstar,
           "lowband_power_by_layer": lowpow.tolist(), "bands_at_Lstar": band_rows,
           "n_modes_for_80pct": n80, "n_modes_for_90pct": n90,
           "greedy_named": [{"name": nm, "marginal": mg, "cumulative": cu} for nm, mg, cu in greedy],
           "pca_vs_modes": pc_match, "lowband_null_mean": float(null_low.mean()),
           "lowband_null_p95": float(np.percentile(null_low, 95)), "lowband_observed": float(lowpow[Lstar])}
    tag_g = f"{TAG}_{GRAPH}"
    json.dump(out, open(f"{OUTDIR}/divider_basis_{tag_g}.json", "w"), indent=2)
    make_fig(out, spec, w, V, coords, rows, cols, Hc_by_L[Lstar], f"{OUTDIR}/divider_basis_{tag_g}.pdf")
    print(f"[{TAG}/{GRAPH}] L*={Lstar} lowband={lowpow[Lstar]:.2f} (null {null_low.mean():.2f}) "
          f"#modes->80%={n80} ->90%={n90}", flush=True)
    for nm, mg, cu in greedy[:6]:
        print(f"    {nm:24s} +{mg:.2f} -> {cu:.2f}", flush=True)
    print(f"DONE -> {OUTDIR}/divider_basis_{tag_g}.json", flush=True)


def thumb(ax, vec, rows, cols, coords, title):
    n = len(vec); lim = abs(vec).max() + 1e-9
    if rows * cols == n and rows > 0 and cols > 0:
        ax.imshow(vec.reshape(rows, cols), cmap="RdBu_r", vmin=-lim, vmax=lim)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c=vec, cmap="RdBu_r", vmin=-lim, vmax=lim, s=120, edgecolors="k", lw=.3)
        ax.set_aspect("equal")
    ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])


def make_fig(out, spec, w, V, coords, rows, cols, Hc, path):
    Lstar = out["Lstar"]; n = out["n"]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(15, 4.4))
        ax1 = fig.add_subplot(1, 3, 1); frac = spec[Lstar]
        ax1.bar(np.arange(1, n), frac[1:], color="tab:purple")
        ax1.set_title(f"{out['tag']} {out['graph']} L{Lstar}: power per Laplacian mode", fontsize=8)
        ax1.set_xlabel("eigenmode (freq →)"); ax1.set_ylabel("frac of node-mean variance")
        ax2 = fig.add_subplot(1, 3, 2); order = np.argsort(frac)[::-1]; cum = np.cumsum(frac[order])
        ax2.plot(np.arange(1, len(cum) + 1), cum, "-o", ms=3, color="tab:purple")
        ax2.axhline(0.8, color=".7", lw=.6, ls="--"); ax2.axhline(0.9, color=".7", lw=.6, ls="--")
        ax2.axvline(2, color="tab:red", lw=.8, label="2D coord probe")
        ax2.set_title("cumulative variance vs #modes", fontsize=8); ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("# eigenmodes (ranked)"); ax2.legend(fontsize=7)
        ax3 = fig.add_subplot(1, 3, 3)
        im = ax3.imshow(spec[:, 1:].T, aspect="auto", cmap="magma", origin="lower")
        ax3.set_title("power: eigenmode × layer", fontsize=8); ax3.set_xlabel("layer"); ax3.set_ylabel("eigenmode")
        fig.colorbar(im, ax=ax3, fraction=.046); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        ncols = 5; nrows = int(np.ceil((n - 1) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.1 * nrows))
        for idx, ax in enumerate(np.array(axes).flat):
            k = idx + 1
            if k < n: thumb(ax, V[:, k], rows, cols, coords, f"mode {k} λ={w[k]:.2f} ({spec[Lstar][k]*100:.0f}%)")
            else: ax.axis("off")
        fig.suptitle(f"{out['tag']} {out['graph']}: Laplacian eigenmode DIVIDERS (% = variance share at L{Lstar})", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(13, 6)); axg = fig.add_subplot(1, 2, 1)
        nmn = [g["name"] for g in out["greedy_named"]]; cu = [g["cumulative"] for g in out["greedy_named"]]
        mg = [g["marginal"] for g in out["greedy_named"]]
        axg.barh(range(len(nmn)), mg, color="teal"); axg.set_yticks(range(len(nmn))); axg.set_yticklabels(nmn, fontsize=7)
        axg.invert_yaxis(); axg.set_xlabel("marginal variance explained")
        for i, c in enumerate(cu): axg.text(mg[i] + .005, i, f"Σ={c:.2f}", fontsize=6, va="center")
        axg.set_title("named cuts, greedy orthogonal (cumulative Σ)", fontsize=9)
        Us, Ss, _ = np.linalg.svd(Hc, full_matrices=False); pv = (Ss**2)/(Ss**2).sum()
        for i in range(min(4, Us.shape[1])):
            axp = fig.add_subplot(2, 4, (i % 2) + 3 + 4 * (i // 2)); m = out["pca_vs_modes"][i]
            thumb(axp, Us[:, i], rows, cols, coords, f"PC{i} ({pv[i]*100:.0f}%)\n≈ {m['best_named']}")
        fig.suptitle(f"{out['tag']} {out['graph']}: named-cut basis (left) & model's own top axes (right)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
