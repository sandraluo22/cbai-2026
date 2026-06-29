"""Cross-model context-length RSA: how do two models' in-context node geometries
align across context length, layer by layer?

The cross-model analogue of analysis/context_rsa.py (which is within-model). For a
fixed layer (paged by relative depth) and a model pair (A, B), build each model's
per-node-mean RDM at a series of context-length centers, then RSA (Spearman of
RDMs) between A@context_i and B@context_j -> a (context x context) cross-model
heatmap. The matrix is NOT symmetric: rows = model A context, cols = model B
context. Three pairs per slide: Qwen vs Llama, Qwen vs Gemma, Gemma vs Llama.
One slide per layer.

Two slideshows, same colour conventions as the other RSA figures:
  cross_context_by_layer.pdf      raw cross-model RSA          (viridis, vmin -0.2..1)
  cross_context_significance.pdf  raw - node-label perm-null95 (RdBu_r, 0-centred;
                                  red = above chance / real, blue = below)

Distance metric = Euclidean node-RDM; RSA = Spearman of RDM upper triangles (same
as heatmaps_with_null.py / context_rsa.py). The null shuffles node correspondence
of model B's RDM at context j, conditioning on each model's true geometry (the
standard RSA permutation test, as in heatmaps_with_null / context_rsa).

Version-aware (CM_VERSION); context centers extend to CTX_HI. B permutations
(env NPERM, default 200).

Per graph -> runs/<version>/<graph>/slides/cross_context_{by_layer,significance}.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace
from config import get_config
import graph as G
import paths as P

GRAPHS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
          ("ring", dict(graph_type="ring", ring_size=16)),
          ("hex", dict(graph_type="hex", hex_rows=4, hex_cols=4))]
# A vs B (A on the y-axis, B on the x-axis), in the order requested.
PAIRS = [("Qwen", "Llama"), ("Qwen", "Gemma"), ("Gemma", "Llama")]
# fine-grained centers; capped at the version's max context (matches context_rsa)
BASE_CENTERS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 650, 800, 1000,
                1250, 1500, 1750, 2000]
B = int(os.environ.get("NPERM", "200"))     # permutations for the null
RNG = np.random.default_rng(0)


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def centers():
    return [c for c in BASE_CENTERS if c <= P.CTX_HI]


def build_cache(npz, gr, C):
    """Per layer: per-context-center node RDM (full n x n) and its upper triangle."""
    n = gr.n_nodes
    iu = np.triu_indices(n, 1)
    z = np.load(npz); node = z["meta_node"]; cl = z["meta_context_length"]
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
    masks = [(np.abs(cl - c) <= max(0.2 * c, 20)) for c in C]
    counts = [int(m.sum()) for m in masks]
    out = {}
    for L in layers:
        X = z[f"layer_{L}"].astype(np.float64)
        rdm, tri = [], []
        for m in masks:
            ndm = node[m]; Xm = X[m]
            H = np.stack([Xm[ndm == q].mean(0) if (ndm == q).any()
                          else np.full(X.shape[1], np.nan) for q in range(n)])
            D = np.linalg.norm(H[:, None] - H[None], axis=2)
            rdm.append(D); tri.append(D[iu])
        del X
        out[L] = dict(rdm=rdm, tri=tri)
    return dict(layers=layers, by_layer=out, counts=counts)


def pair_matrices(cA, La, cB, Lb, k, iu, perms):
    """Cross-model RSA (k x k) and its node-label perm-null95, A rows / B cols."""
    triA = cA["by_layer"][La]["tri"]
    triB = cB["by_layer"][Lb]["tri"]
    rdmB = cB["by_layer"][Lb]["rdm"]
    # shuffled B triangles depend only on (j, perm), reuse across all A contexts
    shufB = [[RB[np.ix_(p, p)][iu] for p in perms] for RB in rdmB]
    obs = np.zeros((k, k)); nl95 = np.zeros((k, k))
    for i in range(k):
        ai = triA[i]
        for j in range(k):
            obs[i, j] = sp(ai, triB[j])
            nl95[i, j] = np.percentile([sp(ai, s) for s in shufB[j]], 95)
    return obs, nl95


def main():
    C = centers(); lab = [str(c) for c in C]; k = len(C)
    for gname, kw in GRAPHS:
        if not all(os.path.exists(P.acts_path(gname, m)) for m in P.MODELS):
            print(f"skip {gname}: no acts for {P.VERSION}", flush=True); continue
        gr = G.build_graph(replace(get_config("gemma_qwen"), **kw))
        n = gr.n_nodes; iu = np.triu_indices(n, 1)
        perms = [RNG.permutation(n) for _ in range(B)]
        data = {m: build_cache(P.acts_path(gname, m), gr, C) for m in P.MODELS}
        Npg = max(len(data[m]["layers"]) for m in P.MODELS)

        def layer_at(m, p):
            ls = data[m]["layers"]; return ls[round(p / (Npg - 1) * (len(ls) - 1))]

        slides = f"{P.gdir(gname)}/slides"; os.makedirs(slides, exist_ok=True)
        raw_pdf = f"{slides}/cross_context_by_layer.pdf"
        sig_pdf = f"{slides}/cross_context_significance.pdf"
        with PdfPages(raw_pdf) as praw, PdfPages(sig_pdf) as psig:
            for p in range(Npg):
                f_raw, ax_raw = plt.subplots(1, 3, figsize=(16, 5))
                f_sig, ax_sig = plt.subplots(1, 3, figsize=(16, 5))
                for col, (A, Bm) in enumerate(PAIRS):
                    La = layer_at(A, p); Lb = layer_at(Bm, p)
                    obs, nl95 = pair_matrices(data[A], La, data[Bm], Lb, k, iu, perms)

                    a = ax_raw[col]
                    im = a.imshow(obs, vmin=-0.2, vmax=1, cmap="viridis", origin="lower")
                    bi, bj = np.unravel_index(int(np.nanargmax(obs)), obs.shape)
                    a.plot(bj, bi, "r*", ms=12)
                    a.set_xticks(range(k)); a.set_xticklabels(lab, rotation=90, fontsize=5)
                    a.set_yticks(range(k)); a.set_yticklabels(lab, fontsize=5)
                    a.set_xlabel(f"{Bm} context length"); a.set_ylabel(f"{A} context length")
                    a.set_title(f"{A} L{La}  vs  {Bm} L{Lb}   (max {obs[bi, bj]:.2f})", fontsize=10)
                    f_raw.colorbar(im, ax=a, fraction=0.046, label="cross-model RSA")

                    S = obs - nl95
                    vmax = max(0.05, float(np.nanmax(np.abs(S))))
                    a = ax_sig[col]
                    im = a.imshow(S, origin="lower", cmap="RdBu_r",
                                  norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
                    a.set_xticks(range(k)); a.set_xticklabels(lab, rotation=90, fontsize=5)
                    a.set_yticks(range(k)); a.set_yticklabels(lab, fontsize=5)
                    a.set_xlabel(f"{Bm} context length"); a.set_ylabel(f"{A} context length")
                    frac = float(np.nansum(S > 0)) / float(np.sum(~np.isnan(S))) * 100
                    a.set_title(f"{A} L{La}  vs  {Bm} L{Lb}   ({frac:.0f}% > null)", fontsize=10)
                    f_sig.colorbar(im, ax=a, fraction=0.046, label="RSA - null95")

                rd = p / (Npg - 1)
                f_raw.suptitle(f"{gname} [{P.VERSION}] cross-model context-vs-context RSA "
                               f"-- relative depth {rd:.2f}\n"
                               f"rows = model A context, cols = model B context  (viridis)")
                f_sig.suptitle(f"{gname} [{P.VERSION}] cross-model context-RSA significance "
                               f"(observed - perm-null95) -- relative depth {rd:.2f}\n"
                               f"red = above chance (real)")
                f_raw.tight_layout(); praw.savefig(f_raw); plt.close(f_raw)
                f_sig.tight_layout(); psig.savefig(f_sig); plt.close(f_sig)
        print(f"wrote {raw_pdf}  ({Npg} pages, {k} centers, B={B})", flush=True)
        print(f"wrote {sig_pdf}", flush=True)


if __name__ == "__main__":
    main()
