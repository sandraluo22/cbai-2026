"""Remove the geometry found by a PROBE, not by PCA — and verify it is actually gone.

Why this supersedes the PC version: the "coordinate plane" there was the 2 PCs whose scores best track
(row, col), so it can only find geometry aligned with a high-variance direction. In this model that is
unreliable — the coordinate directions are PC1/PC0 at L10 but PC2/PC1 at L14+, best |r| is only 0.72-0.86,
and rsa_pc2 collapses after L16 while rsa_full does not, i.e. the geometry LEAVES the top PCs with depth.
So projecting out 2 PCs may remove only a fraction of the coordinate code and make "the geometry is not
used" look true for a trivial reason.

Instead: fit a RIDGE PROBE from the residual to (row, col) on per-readout samples (thousands, not the 16
node means — the underdetermination that broke three earlier coordinate metrics), validate it with
LEAVE-ONE-NODE-OUT CV (leaving out random tokens leaks, since the same node recurs), take the two weight
vectors, orthonormalise, and project THAT plane out.

**The control the PC version lacked**: after projecting, RE-FIT a fresh probe on the projected
representations. If coordinates are still decodable, the intervention did not remove the geometry and any
null result about the task is meaningless. So we iterate: remove rank 2, 4, 8... until coordinate
decodability is at floor, and report task accuracy at each step.

Reported per layer and rank:
    cv_r2_before / cv_r2_after   leave-one-node-out R^2 for (row,col) — did we actually delete it?
    nbr_valid                    task accuracy
    concept_decode               node identity (must survive, or we deleted the input not the geometry)
    vs random rank-k and vs the PC plane, for comparison

Env: GEN_MODEL(Llama) K(4) LAYERS("10,14,18,24") RANKS("2,4,8,16") ALPHA(1e3) NWALKS(3) WLEN(1200)
     CTXLO(800) NRAND(3) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/probe_axes_projection<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
K = int(os.environ.get("K", "4"))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "10,14,18,24").split(",")]
RANKS = [int(x) for x in os.environ.get("RANKS", "2,4,8,16").split(",")]
ALPHA = float(os.environ.get("ALPHA", "1e3"))
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); NRAND = int(os.environ.get("NRAND", "3"))
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


def ridge_lono(X, Y, nodes, n, alpha):
    """leave-one-NODE-out ridge. Returns (cv_R2, full-fit weight matrix [d, 2]).
    Node-level folds because token-level folds leak: the same node recurs thousands of times."""
    d = X.shape[1]
    preds = np.zeros_like(Y)
    for c in range(n):
        te = nodes == c
        if not te.any() or te.all(): continue
        tr = ~te
        Xt, Yt = X[tr], Y[tr]
        mu = Xt.mean(0); Xc = Xt - mu
        A = Xc.T @ Xc + alpha * np.eye(d)
        W = np.linalg.solve(A, Xc.T @ (Yt - Yt.mean(0)))
        preds[te] = (X[te] - mu) @ W + Yt.mean(0)
    ss = ((Y - preds) ** 2).sum(0); sv = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = float(np.mean(1 - ss / np.maximum(sv, 1e-12)))
    mu = X.mean(0); Xc = X - mu
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ (Y - Y.mean(0)))
    return r2, W


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); coords = np.array(graph.coords, float)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    rng = np.random.default_rng(SEED)

    data = []
    for wk in G.generate_walks(graph, cfg):
        steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        if steps:
            data.append((torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev),
                         torch.tensor([s + 1 for s in steps], device=dev),
                         [wk.nodes[s] for s in steps]))

    st = {"proj": None, "layer": None}
    hooks = []
    for l in range(cm.num_hidden_layers):
        def mk(l):
            def rh(_m, _i, out):
                if st["proj"] is None or st["layer"] != l: return out
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone(); P = st["proj"]; f = h[0].float()
                h[0] = (f - (f @ P.t()) @ P).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return rh
        hooks.append(blocks[l].register_forward_hook(mk(l)))

    def collect(L, sub=1200):
        X, Yn, ok, tot = [], [], 0, 0
        for ids, rp, nds in data:
            o = model(input_ids=ids, output_hidden_states=True)
            H = o.hidden_states[L + 1][0, rp].float().cpu().numpy()
            top = o.logits[0][rp][:, cand_t].float().argmax(1).tolist()
            for i, u in enumerate(nds):
                X.append(H[i]); Yn.append(u)
                ok += int(top[i] in graph.adjacency[u]); tot += 1
        X = np.stack(X); Yn = np.array(Yn)
        sel = rng.permutation(len(X))[:sub]
        return X[sel], Yn[sel], ok / tot

    def decode_id(X, Yn):
        d = 0
        for i in range(len(X)):
            m = np.ones(len(X), bool); m[i] = False
            cen = np.stack([X[m & (Yn == c)].mean(0) if (m & (Yn == c)).any()
                            else np.full(X.shape[1], 1e9) for c in range(n)])
            d += int(int(np.argmin(np.linalg.norm(cen - X[i], axis=1))) == Yn[i])
        return d / len(X)

    res = {"model": tag, "alpha": ALPHA, "layers": {}}
    print(f"{'L':>3} {'plane':<14} {'k':>3} {'cv_r2_after':>12} {'nbr_valid':>10} {'id_decode':>10}")
    for L in LAYERS:
        X, Yn, base_nbr = collect(L)
        Y = coords[Yn]
        r2_0, W = ridge_lono(X, Y, Yn, n, ALPHA)
        Mn = np.stack([X[Yn == c].mean(0) for c in range(n)]); Mn -= Mn.mean(0, keepdims=True)
        U, S_, Vt = np.linalg.svd(Mn, full_matrices=False)
        sc = U * S_
        r0 = [abs(np.corrcoef(sc[:, j], coords[:, 0])[0, 1]) for j in range(min(8, sc.shape[1]))]
        r1 = [abs(np.corrcoef(sc[:, j], coords[:, 1])[0, 1]) for j in range(min(8, sc.shape[1]))]
        j0 = int(np.argmax(r0)); j1 = int(np.argmax(r1))
        if j1 == j0: j1 = int(np.argsort(r1)[-2])
        row = {"cv_r2_baseline": round(r2_0, 4), "baseline_nbr": round(base_nbr, 4),
               "baseline_id": round(decode_id(X, Yn), 4), "pc_coord_dirs": [j0, j1], "planes": {}}
        pw = W / np.linalg.norm(W, axis=0, keepdims=True)
        cospc = [round(float(abs(pw[:, a] @ Vt[b])), 3) for a, b in ((0, j0), (1, j1))]
        print(f"{L:3} {'(none)':<14} {0:3} {r2_0:12.4f} {base_nbr:10.4f} {row['baseline_id']:10.4f}"
              f"   probe-vs-PC |cos| = {cospc}", flush=True)

        def test(nm, P):
            st["layer"] = L; st["proj"] = torch.tensor(P, dtype=torch.float32, device=dev)
            Xp, Ynp, nbr = collect(L); st["proj"] = None; st["layer"] = None
            r2a, _ = ridge_lono(Xp, coords[Ynp], Ynp, n, ALPHA)
            idd = decode_id(Xp, Ynp)
            row["planes"][nm] = {"cv_r2_after": round(r2a, 4), "nbr": round(nbr, 4),
                                 "id_decode": round(idd, 4)}
            print(f"{L:3} {nm:<14} {P.shape[0]:3} {r2a:12.4f} {nbr:10.4f} {idd:10.4f}", flush=True)

        # probe plane at increasing rank: augment the 2 probe axes with the top PCs of the RESIDUAL
        # coordinate signal, so rank grows toward "all decodable coordinate information"
        for k_ in RANKS:
            B = np.linalg.qr(W)[0].T                                   # 2 probe axes
            if k_ > 2:
                resid = Mn - (Mn @ B.T) @ B
                extra = np.linalg.svd(resid, full_matrices=False)[2][:k_ - 2]
                B = np.linalg.qr(np.vstack([B, extra]).T)[0].T
            test(f"probe{k_}", B.astype(np.float32))
        test("pc_coord2", np.linalg.qr(Vt[[j0, j1]].T)[0].T.astype(np.float32))
        Rr = np.linalg.qr(rng.standard_normal((cm.hidden_size, 2)))[0].T
        test("random2", Rr.astype(np.float32))
        res["layers"][str(L)] = row
        res["layers"][str(L)]["probe_vs_pc_cos"] = cospc
    for h in hooks: h.remove()
    p_ = f"{OUTDIR}/probe_axes_projection{OUTTAG}_{tag}.json"
    json.dump(res, open(p_, "w"), indent=2); print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
