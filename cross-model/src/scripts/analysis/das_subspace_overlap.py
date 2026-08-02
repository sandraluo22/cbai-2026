"""Is a DAS subspace a SHARED coordinate code, or a per-counterfactual slice?

The patch is linear with no learned mapping beyond the projection: patch = (D @ R^T) @ R. So if R spanned
the space in which the node means live, EVERY permutation's delta would pass through untouched and
transfer would be near-perfect. Observed transfer is weak. That leaves two possibilities, and they are
geometrically distinguishable without any further training:

  (a) per-perm slice   — R_A and R_B (trained on independent remaps) are near-orthogonal, and each
                         captures only a small fraction of the node-mean variance. Low DAS rank then does
                         NOT imply a compact shared code; it means one remap needs few dimensions.
  (b) shared code      — R_A and R_B overlap strongly and both capture most of the node-mean variance,
                         in which case weak transfer has some other cause (e.g. sign/scale of the delta).

Reports, per grid and rank:
  principal_angles     cos of principal angles between R_A and R_B (mean, and how many are > 0.9)
  overlap_frac         ||P_A M||_F^2 / ||M||_F^2, M = centred node-mean matrix — fraction of node-mean
                       variance inside R_A's span (and same for R_B)
  rand_overlap_frac    same for a random subspace of equal rank = the chance floor (r/d_eff, not r/4096)

Env: GEN_MODEL(Llama) NPZ_A NPZ_B LAYER(24) LAZY(0.5) GRIDS(4x4,8x8) RDIMS(1,2,4,8,16,32,64)
     K_FROM_KEY(1) NWALKS(3) WLEN(1200) CTXLO(800) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/das_subspace_overlap<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
from graph import Walk
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, two_colour

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
GRAPH_TYPE = os.environ.get("GRAPH_TYPE", "grid")     # "grid" | "ring"
NPZ_A = os.environ["NPZ_A"]; NPZ_B = os.environ["NPZ_B"]
LAYER = int(os.environ.get("LAYER", "24")); LAZY = float(os.environ.get("LAZY", "0.5"))
GRIDS = [int(g.split("x")[0]) for g in os.environ.get("GRIDS", "4x4,8x8").split(",")]
RDIMS = [int(x) for x in os.environ.get("RDIMS", "1,2,4,8,16,32,64").split(",")]
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", "")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/coordperm")


def orth(R):
    """rows of R span the subspace -> orthonormal basis as columns [d, r]"""
    q, _ = np.linalg.qr(np.asarray(R, np.float64).T)
    return q[:, :R.shape[0]]


@torch.no_grad()
def node_means(model, tok, dev, k):
    """centred per-node mean residual at LAYER on lazy walks -> M [n, d]"""
    n = k if GRAPH_TYPE == "ring" else k * k
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = (replace(get_config("gemma_qwen"), graph_type="ring", ring_size=n,
                   n_walks=NWALKS, walk_length=WLEN, device=dev) if GRAPH_TYPE == "ring" else
           replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=k, grid_cols=k,
                   n_walks=NWALKS, walk_length=WLEN, device=dev))
    graph = G.build_graph(cfg); words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    rng = np.random.default_rng(SEED)
    walks = []
    for w in range(NWALKS):
        cur = w % n; nodes = [cur]
        for _ in range(WLEN - 1):
            if LAZY <= 0 or rng.random() >= LAZY: cur = int(rng.choice(graph.neighbors(cur)))
            nodes.append(cur)
        walks.append(Walk(walk_id=w, nodes=nodes, words=[words[x] for x in nodes]))
    d = model.config.hidden_size
    s = np.zeros((n, d)); c = np.zeros(n)
    for wk in walks:
        ids = torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev)
        o = model(input_ids=ids, output_hidden_states=True)
        H = o.hidden_states[LAYER + 1][0].float().cpu().numpy()
        for st in range(len(wk.nodes) - 1):
            if st + 1 < CTXLO: continue
            s[wk.nodes[st]] += H[st + 1]; c[wk.nodes[st]] += 1
    M_ = s / np.maximum(c, 1)[:, None]
    return M_ - M_.mean(0, keepdims=True)


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    zA, zB = np.load(NPZ_A), np.load(NPZ_B)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    rng = np.random.default_rng(SEED)
    out = {"model": tag, "layer": LAYER, "lazy": LAZY, "npz_a": NPZ_A, "npz_b": NPZ_B, "grids": {}}
    for k in GRIDS:
        Mn = node_means(model, tok, dev, k)                       # [n, d]
        fro = float((Mn ** 2).sum())
        # effective dimension of the node-mean cloud: how many singular values carry the variance
        sv = np.linalg.svd(Mn, compute_uv=False)
        eff = float((sv.sum() ** 2) / (sv ** 2).sum())            # PR of singular values (generous)
        var = sv ** 2; cum = np.cumsum(var) / var.sum()
        pr_var = float((var.sum() ** 2) / (var ** 2).sum())       # PR of VARIANCE (strict)
        d50, d90 = int(np.searchsorted(cum, 0.50) + 1), int(np.searchsorted(cum, 0.90) + 1)
        if GRAPH_TYPE == "ring":
            nn = Mn.shape[0]
            top2 = float(var[:2].sum() / var.sum())
            D = np.linalg.norm(Mn[:, None, :] - Mn[None, :, :], axis=-1)
            ring_d = np.array([[min((i - j) % nn, (j - i) % nn) for j in range(nn)] for i in range(nn)])
            iu = np.triu_indices(nn, 1)
            dv, rv = D[iu], ring_d[iu]
            loc = rv <= 2
            rsa_all = float(np.corrcoef(dv, rv)[0, 1])
            rsa_loc = float(np.corrcoef(dv[loc], rv[loc])[0, 1]) if loc.sum() > 2 else float("nan")
            # a PLANAR circle would put ~100% of variance in 2 dims and have D = 2R*sin(pi*d/n)
            print(f"  RING GEOMETRY: var in top-2 dims = {top2:.3f} (planar circle -> ~1.000)")
            print(f"    RSA(repr dist, ring dist) all pairs = {rsa_all:+.3f} | local (d<=2) = {rsa_loc:+.3f}")
            print(f"    mean repr dist at ring d=1: {dv[rv == 1].mean():.3f}  "
                  f"d=2: {dv[rv == 2].mean():.3f}  d=n/2: {dv[rv == nn // 2].mean():.3f}")
        res = {"n": (k if GRAPH_TYPE == "ring" else k * k), "node_mean_eff_dim": round(eff, 2), "node_mean_pr_var": round(pr_var, 2),
               "dims_for_50pct_var": d50, "dims_for_90pct_var": d90, "ranks": {},
               **({"ring_top2_var": round(top2, 4), "ring_rsa_all": round(rsa_all, 4),
                   "ring_rsa_local": round(rsa_loc, 4)} if GRAPH_TYPE == "ring" else {})}
        gl = f"ring{k}" if GRAPH_TYPE == "ring" else f"{k}x{k}"
        print(f"\n[{gl}] node-mean cloud: {Mn.shape[0]} nodes | PR(sv)={eff:.2f} PR(var)={pr_var:.2f} "
              f"| dims for 50% var={d50}, for 90% var={d90}", flush=True)
        print(f"  {'r':>3} {'cos_ang_mean':>12} {'n_aligned':>10} {'ovl_A':>7} {'ovl_B':>7} {'ovl_rand':>9}")
        for r in RDIMS:
            key = f"{gl}_r{r}"
            if key not in zA.files or key not in zB.files: continue
            A, B = orth(zA[key]), orth(zB[key])
            cs = np.linalg.svd(A.T @ B, compute_uv=False)         # cosines of principal angles
            def ovl(Q): return float(((Q.T @ Mn.T) ** 2).sum() / fro)
            Rr = np.linalg.qr(rng.standard_normal((Mn.shape[1], A.shape[1])))[0]
            row = {"cos_angles_mean": round(float(cs.mean()), 4),
                   "cos_angles_max": round(float(cs.max()), 4),
                   "n_aligned_gt0.9": int((cs > 0.9).sum()),
                   "overlap_frac_A": round(ovl(A), 4), "overlap_frac_B": round(ovl(B), 4),
                   "overlap_frac_rand": round(ovl(Rr), 4)}
            res["ranks"][str(r)] = row
            print(f"  {r:3} {row['cos_angles_mean']:12.3f} {row['n_aligned_gt0.9']:10} "
                  f"{row['overlap_frac_A']:7.3f} {row['overlap_frac_B']:7.3f} {row['overlap_frac_rand']:9.3f}",
                  flush=True)
        out["grids"][gl] = res
    p = f"{OUTDIR}/das_subspace_overlap{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
