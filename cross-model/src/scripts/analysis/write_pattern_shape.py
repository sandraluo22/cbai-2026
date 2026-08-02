"""Is what these heads write STRICTLY the checkerboard, or does it have other spatial structure?

For each head we take its mean write to the parity direction PER NODE (a 16-vector on the 4x4 grid) and
decompose it in the graph-Laplacian eigenbasis. If the write is a pure parity pattern, all power sits on
the top mode (lambda=2, the checkerboard). Power on low-lambda modes = coordinate structure mixed in;
power spread over mid modes = something else entirely.

Also separates the STATIC/LEXICAL component from the in-context one: with every attention head
mean-ablated the residual at a position contains only the current token's embedding + MLPs, so any
parity-looking pattern there is a fixed word->class lookup, not in-context computation. We report the
per-node coefficient pattern in both conditions and their eigen-decompositions, so the two can be
compared directly.

Env: GEN_MODEL(Llama) HEADS("14:26,14:19,14:17,16:20,10:2,21:10,2:26,25:7") LAYER(14)
     PAR_NPY(...seed_stable_r1_<model>.npy) K(4) NWALKS(3) WLEN(1200) CTXLO(800) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/write_pattern_shape_<model>.json
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
from grid_parity_compare import build_word_pool, two_colour, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEADS = [tuple(int(x) for x in h.split(":")) for h in
         os.environ.get("HEADS", "14:26,14:19,14:17,16:20,10:2,21:10,2:26,25:7").split(",")]
LAYER = int(os.environ.get("LAYER", "14"))
K = int(os.environ.get("K", "4")); NWALKS = int(os.environ.get("NWALKS", "3"))
WLEN = int(os.environ.get("WLEN", "1200")); CTXLO = int(os.environ.get("CTXLO", "800"))
SEED = int(os.environ.get("SEED", "0"))
P = "runs/axes/4_circuits/parity"
PAR_NPY = os.environ.get("PAR_NPY", f"{P}/seed_stable_r1_{GEN_MODEL}.npy")
OUTDIR = os.environ.get("OUTDIR", P)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH

    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph); coords = np.array(graph.coords, int)
    words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]

    A = np.zeros((n, n))
    for u in range(n):
        for v_ in graph.adjacency[u]: A[u, v_] = 1.0
    dg = A.sum(1); di = 1 / np.sqrt(dg); Lap = np.eye(n) - di[:, None] * A * di[None, :]
    eigw, eigU = np.linalg.eigh(Lap)
    parity_mode = int(np.argmax(eigw))
    lo_modes = [i for i in range(n) if 0 < eigw[i] < 0.7]          # coordinate-like
    mid_modes = [i for i in range(n) if 0.7 <= eigw[i] <= 1.5]

    u = np.load(PAR_NPY).astype(np.float32); u = u / np.linalg.norm(u)
    ut = torch.tensor(u, device=dev)
    W = {l: attn_proj(blocks[l], cm)[0].weight.detach().float() for l in {h[0] for h in HEADS}}
    wh = {(l, h): (W[l].t() @ ut).view(nH, hd)[h] for (l, h) in HEADS}

    caps = {}
    hooks = []
    for l in {h[0] for h in HEADS}:
        def mk(l):
            def hh(_m, args): caps[l] = args[0].detach()
            return hh
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk(l)))
    abl = {"on": False}
    def killer(l):
        def hh(_m, args):
            if not abl["on"]: return
            x = args[0].clone(); x[0] = x[0].mean(0, keepdim=True)
            return (x,) + tuple(args[1:])
        return hh
    hooks += [attn_proj(blocks[l], cm)[0].register_forward_pre_hook(killer(l)) for l in range(nL)]

    walks = G.generate_walks(graph, cfg)
    def collect(ablate):
        abl["on"] = ablate
        hs = np.zeros((n, )); hc = np.zeros((n,))
        ws = {k: np.zeros(n) for k in wh}; wc = np.zeros(n)
        for wk in walks:
            ids = torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev)
            caps.clear()
            o = model(input_ids=ids, output_hidden_states=True)
            h = o.hidden_states[LAYER + 1][0].float() @ ut
            for s in range(len(wk.nodes) - 1):
                if s + 1 < CTXLO: continue
                t = s + 1; nd = wk.nodes[s]
                hs[nd] += float(h[t]); hc[nd] += 1
                if not ablate:
                    for (l, hh_) in wh:
                        z = caps[l][0, t].float().view(nH, hd)[hh_]
                        ws[(l, hh_)][nd] += float(z @ wh[(l, hh_)])
                    wc[nd] += 1
        abl["on"] = False
        return hs / np.maximum(hc, 1), {k: v / np.maximum(wc, 1) for k, v in ws.items()}

    full_coef, head_writes = collect(False)
    static_coef, _ = collect(True)
    for hk in hooks: hk.remove()

    def decomp(vec):
        c = vec - vec.mean()
        p = np.array([float(np.dot(c, eigU[:, m])) ** 2 for m in range(n)])
        tot = p.sum() + 1e-12
        chk = eigU[:, parity_mode] * np.sign(eigU[0, parity_mode])
        return {"parity_mode_frac": round(float(p[parity_mode] / tot), 3),
                "low_lambda_frac": round(float(p[lo_modes].sum() / tot), 3),
                "mid_lambda_frac": round(float(p[mid_modes].sum() / tot), 3),
                "corr_with_exact_checkerboard": round(float(abs(np.corrcoef(c, col)[0, 1])), 3),
                "corr_with_row": round(float(abs(np.corrcoef(c, coords[:, 0])[0, 1])), 3),
                "corr_with_col": round(float(abs(np.corrcoef(c, coords[:, 1])[0, 1])), 3)}

    out = {"model": tag, "layer": LAYER, "parity_mode_lambda": round(float(eigw[parity_mode]), 3),
           "residual_full": decomp(full_coef), "residual_attention_ablated": decomp(static_coef),
           "residual_full_per_node": [round(float(x), 3) for x in full_coef],
           "residual_static_per_node": [round(float(x), 3) for x in static_coef],
           "in_context_only": decomp(full_coef - static_coef),
           "heads": {f"L{l}H{h}": dict(decomp(head_writes[(l, h)]),
                                       per_node=[round(float(x), 3) for x in head_writes[(l, h)]])
                     for (l, h) in wh}}
    print(f"parity mode lambda={eigw[parity_mode]:.3f}\n")
    print(f"{'source':22} {'chkbd_r':>8} {'par_mode':>9} {'low_lam':>8} {'mid_lam':>8} {'row_r':>6} {'col_r':>6}")
    for nm, dd in [("residual FULL", out["residual_full"]),
                   ("residual ATTN-ABLATED", out["residual_attention_ablated"]),
                   ("residual IN-CONTEXT", out["in_context_only"])] + \
                  [(k, v) for k, v in out["heads"].items()]:
        print(f"{nm:22} {dd['corr_with_exact_checkerboard']:8.3f} {dd['parity_mode_frac']:9.3f}"
              f" {dd['low_lambda_frac']:8.3f} {dd['mid_lambda_frac']:8.3f}"
              f" {dd['corr_with_row']:6.3f} {dd['corr_with_col']:6.3f}")
    p = f"{OUTDIR}/write_pattern_shape_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
