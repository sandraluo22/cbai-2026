"""HYPOTHESIS: induction heads write LOW-frequency graph eigenmodes; a different set of heads writes
HIGH-frequency modes (the parity/checkerboard end of the spectrum).

For every head (l,h) we take its exact additive contribution to the residual stream,
    write_lh(t) = z_lh(t) @ W_O[h]        (h_L = emb + sum_l (attn_l + mlp_l), attn_l splits per head)
average it per NODE, centre across nodes, and decompose it in the graph-Laplacian eigenbasis
(combinatorial L = D - A, ascending lambda — same convention as cross_eigenmode_ablation). Mode 0 is the
constant and carries nothing after centring; the TOP mode of a bipartite grid is the parity checkerboard.

Per head we report:
    centroid     sum_m p_m * lambda_m  with p_m = energy_m / sum energy   -> "how high-frequency is this
                 head's write", on the same axis as the eigenvalues
    lo_frac      energy in the bottom third of the spectrum
    hi_frac      energy in the top third
    par_frac     energy in the single highest mode (parity, for a bipartite grid)
    write_norm   mean ||write||, so near-silent heads can be filtered

CONTROL that makes the centroid meaningful: the same decomposition with W_O[h] replaced by a RANDOM
[hd, dm] matrix of matched scale. A head's z has its own node-structure, so the random-projection centroid
is the null for THAT head — `centroid_rel` = centroid - centroid_random is what the hypothesis is about.
Reporting raw centroids alone would confound "this head writes high frequencies" with "this head's
activations happen to vary sharply across nodes".

Then, against Olsson prev-token / induction scores (loaded from olsson_head_scores_<model>.json):
    Spearman rho over ALL heads, and over the write_norm>thresh subset
    the top-K induction heads' mean centroid_rel vs the top-K high-frequency heads', with a permutation
    test, plus the overlap between the two sets

Env: GEN_MODEL(Llama) K(4) GRAPH(grid) NWALKS(3) WLEN(1200) CTXLO(800) LAZY(0) TOPK(20)
     OLSSON(runs/axes/4_circuits/parity/olsson_head_scores_<model>.json) NPERM(10000)
     MINNORM(0.02) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/head_mode_frequency<OUTTAG>_<model>.json
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
from grid_parity_compare import build_word_pool, two_colour, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
K = int(os.environ.get("K", "4")); GRAPH = os.environ.get("GRAPH", "grid")
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); LAZY = float(os.environ.get("LAZY", "0"))
TOPK = int(os.environ.get("TOPK", "20")); NPERM = int(os.environ.get("NPERM", "10000"))
MINNORM = float(os.environ.get("MINNORM", "0.02")); SEED = int(os.environ.get("SEED", "0"))
P = "runs/axes/4_circuits/parity"
OLSSON = os.environ.get("OLSSON", f"{P}/olsson_head_scores_{GEN_MODEL}.json")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", f"{P}")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else 0.0


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    dm = cm.hidden_size; hd = getattr(cm, "head_dim", None) or dm // nH
    rng = np.random.default_rng(SEED)

    n = K * K if GRAPH == "grid" else K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "grid", "grid_rows": K, "grid_cols": K} if GRAPH == "grid"
                     else {"graph_type": "ring", "ring_size": K}),
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg)
    words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]

    # ---- graph Laplacian eigenbasis, ascending lambda (mode 0 = constant, top mode = parity) ----
    A = np.zeros((n, n))
    for u in range(n):
        for v in graph.adjacency[u]: A[u, v] = 1.0
    lam, V = np.linalg.eigh(np.diag(A.sum(1)) - A)          # V columns are mode vectors over nodes
    col = two_colour(graph) if GRAPH == "grid" or n % 2 == 0 else np.ones(n)
    par_align = abs(V[:, -1] @ (col / np.linalg.norm(col)))
    print(f"[{tag}] {GRAPH} n={n}: lambda range {lam[0]:.3f}..{lam[-1]:.3f}; "
          f"|<top mode, parity>| = {par_align:.3f}", flush=True)
    # exact parity direction: on a 4x4 GRID the checkerboard is NOT a Laplacian eigenvector (degrees are
    # 2/3/4, not regular) so the top mode is only ~0.85 aligned with it and parity smears across modes.
    # Project on the parity vector itself for an exact read. On a ring (vertex-transitive) they coincide.
    cvec = torch.tensor(col / np.linalg.norm(col), dtype=torch.float32, device=dev)
    Vt = torch.tensor(V.T, dtype=torch.float32, device=dev)   # [n_modes, n]
    lam_t = torch.tensor(lam, dtype=torch.float32, device=dev)
    third = n // 3

    # ---- walks ----
    walks = G.generate_walks(graph, cfg)
    if LAZY > 0:
        lr = np.random.default_rng(SEED); lz = []
        for w in range(NWALKS):
            cur = w % n; nodes = [cur]
            for _ in range(WLEN - 1):
                if lr.random() >= LAZY: cur = int(lr.choice(graph.neighbors(cur)))
                nodes.append(cur)
            lz.append(Walk(walk_id=w, nodes=nodes, words=[words[x] for x in nodes]))
        walks = lz

    # ---- capture per-head z, accumulate per-node means ----
    caps = {}
    hooks = []
    for l in range(nL):
        def mk(l):
            def hh(_m, args): caps[l] = args[0].detach()
            return hh
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk(l)))
    zsum = torch.zeros(nL, nH, n, hd, device=dev); zcnt = torch.zeros(n, device=dev)
    for wk in walks:
        ids = torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev)
        steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        if not steps: continue
        rp = torch.tensor([s + 1 for s in steps], device=dev)
        nd = torch.tensor([wk.nodes[s] for s in steps], device=dev)
        caps.clear(); model(input_ids=ids)
        oh = torch.zeros(len(steps), n, device=dev); oh[torch.arange(len(steps)), nd] = 1.0
        for l in range(nL):
            Zl = caps[l][0, rp].float().view(len(steps), nH, hd)          # [T, nH, hd]
            zsum[l] += torch.einsum("tn,tkh->knh", oh, Zl)
        zcnt += oh.sum(0)
    for h in hooks: h.remove()
    zmean = zsum / zcnt.clamp(min=1).view(1, 1, n, 1)                     # [nL, nH, n, hd]

    # ---- per-head spectral decomposition of the WRITE ----
    rows = []
    for l in range(nL):
        Wl = attn_proj(blocks[l], cm)[0].weight.detach().float().t()      # [nH*hd, dm]
        for h in range(nH):
            Wh = Wl[h * hd:(h + 1) * hd]                                  # [hd, dm]
            Z = zmean[l, h]                                               # [n, hd]
            def spec(Wmat):
                Wr = Z @ Wmat                                             # [n, dm]
                Wr = Wr - Wr.mean(0, keepdim=True)                        # centre -> mode 0 empty
                e = ((Vt @ Wr) ** 2).sum(1)                               # [n_modes] energy per mode
                tot = e.sum()
                if tot < 1e-12: return None, 0.0, 0.0
                p = e / tot
                pex = float(((cvec @ Wr) ** 2).sum() / tot)                # exact parity fraction
                return p, float((p * lam_t).sum()), pex
            p, cen, pex = spec(Wh)
            nrm = float((Z @ Wh).norm(dim=1).mean())
            if p is None:
                rows.append({"head": f"L{l}H{h}", "layer": l, "write_norm": 0.0, "centroid": 0.0,
                             "centroid_rel": 0.0, "lo_frac": 0.0, "hi_frac": 0.0, "par_frac": 0.0})
                continue
            g = torch.tensor(rng.standard_normal((hd, dm)) / np.sqrt(hd), dtype=torch.float32, device=dev)
            _, cen0, pex0 = spec(g * Wh.norm() / g.norm())
            rows.append({"head": f"L{l}H{h}", "layer": l, "write_norm": round(nrm, 5),
                         "centroid": round(cen, 4), "centroid_rand": round(cen0, 4),
                         "centroid_rel": round(cen - cen0, 4),
                         "lo_frac": round(float(p[1:third + 1].sum()), 4),
                         "hi_frac": round(float(p[-third:].sum()), 4),
                         "par_frac": round(float(p[-1]), 4),
                         "par_exact": round(pex, 4), "par_exact_rel": round(pex - pex0, 4)})
        if l % 8 == 0: print(f"  layer {l}/{nL} done", flush=True)

    # ---- against Olsson scores ----
    out = {"model": tag, "graph": GRAPH, "n": n, "lazy": LAZY, "lambda": [round(float(x), 4) for x in lam],
           "top_mode_parity_alignment": round(float(par_align), 4), "heads": rows}
    if os.path.exists(OLSSON):
        oz = json.load(open(OLSSON))["per_head"]
        names = [r["head"] for r in rows]
        ind = np.array([oz[h]["ind"] if h in oz else np.nan for h in names])
        prv = np.array([oz[h]["prev"] if h in oz else np.nan for h in names])
        cen = np.array([r["centroid_rel"] for r in rows])
        nrm = np.array([r["write_norm"] for r in rows])
        hi = np.array([r["hi_frac"] for r in rows]); lo = np.array([r["lo_frac"] for r in rows])
        ok = ~np.isnan(ind) & (nrm > MINNORM)
        print(f"\nheads with write_norm > {MINNORM}: {ok.sum()} of {len(rows)}")
        cors = {}
        for nm, v in (("centroid_rel", cen), ("hi_frac", hi), ("lo_frac", lo)):
            cors[f"rho_induction_{nm}"] = round(spearman(ind[ok], v[ok]), 4)
            cors[f"rho_prevtoken_{nm}"] = round(spearman(prv[ok], v[ok]), 4)
        for k_, v_ in cors.items(): print(f"  {k_:34} {v_:+.4f}")
        out["correlations"] = cors

        idx = np.where(ok)[0]
        top_ind = idx[np.argsort(-ind[idx])][:TOPK]
        top_hi = idx[np.argsort(-hi[idx])][:TOPK]
        ov = len(set(top_ind) & set(top_hi))
        mi, mh = float(cen[top_ind].mean()), float(cen[top_hi].mean())
        # permutation test on the induction-set centroid
        pool = cen[idx]
        perm = np.array([pool[rng.permutation(len(pool))[:TOPK]].mean() for _ in range(NPERM)])
        pval = float((perm <= mi).mean())
        print(f"\ntop-{TOPK} INDUCTION heads : mean centroid_rel = {mi:+.4f}  (perm p(<=) = {pval:.4f})")
        print(f"top-{TOPK} HIGH-FREQ heads : mean centroid_rel = {mh:+.4f}")
        print(f"overlap between the two sets: {ov}/{TOPK}")
        print(f"\n  {'induction heads':<18} {'ind':>7} {'cen_rel':>8} {'lo':>6} {'hi':>6} {'par':>6}")
        for i in top_ind[:10]:
            r = rows[i]
            print(f"  {r['head']:<18} {ind[i]:7.3f} {r['centroid_rel']:+8.3f} {r['lo_frac']:6.3f} "
                  f"{r['hi_frac']:6.3f} {r['par_frac']:6.3f}")
        print(f"\n  {'high-freq heads':<18} {'ind':>7} {'cen_rel':>8} {'lo':>6} {'hi':>6} {'par':>6}")
        for i in top_hi[:10]:
            r = rows[i]
            print(f"  {r['head']:<18} {ind[i]:7.3f} {r['centroid_rel']:+8.3f} {r['lo_frac']:6.3f} "
                  f"{r['hi_frac']:6.3f} {r['par_frac']:6.3f}")
        out["top_induction"] = [rows[i]["head"] for i in top_ind]
        out["top_highfreq"] = [rows[i]["head"] for i in top_hi]
        out["overlap"] = ov
        out["mean_centroid_rel_induction"] = round(mi, 4)
        out["mean_centroid_rel_highfreq"] = round(mh, 4)
        out["perm_p_induction_lower"] = round(pval, 4)
    else:
        print(f"[warn] no Olsson scores at {OLSSON}; wrote spectra only")
    p_ = f"{OUTDIR}/head_mode_frequency{OUTTAG}_{tag}.json"
    json.dump(out, open(p_, "w"), indent=2); print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
