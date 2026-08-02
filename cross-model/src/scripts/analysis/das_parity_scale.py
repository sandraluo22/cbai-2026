"""Does the DAS rank of the parity subspace grow with grid size? For each EVEN square grid, run the global
parity-flip interchange in L14H26 (pi = 90-deg rotation = automorphism that inverts every node's parity for
even k; delta_t = znode[pi(X_t)] - znode[X_t] injected into the aligned rank-r subspace at every node token),
train an orthogonal rotation R, sweep r, and record the parity-flip margin vs r. If parity is a fixed low-rank
feature the curve saturates at the same small r for all sizes; if it becomes distributed the effective rank
grows with n.

Env: GEN_MODEL(Llama) HEAD_LAYER(14) HEAD_IDX(26) GRIDS(4x4,6x6,8x8,10x10,12x12) RDIMS(1,2,4,8,16,32,128)
     NWALKS(10) SAMPLES_PER_NODE(80) WLEN_CAP(650) CTXLO(80) STEPS(45) BATCH(3) LR(0.02) SEED(0) OUTDIR DEVICE
     HOLDOUT(0)     last HOLDOUT walks are eval-only: R trains on the rest, margins reported on held-out walks,
                    and znode/prototypes (patch sources) come from train walks only. 0 = legacy train=eval.
     CTXLO_PER_N(0) if >0, readout threshold scales with graph size: ctxlo = max(CTXLO, CTXLO_PER_N * n),
                    so large grids are not read out before the walk could have covered the graph.
Out: <OUTDIR>/das_parity_scale_<PATCH><_hoK><_ctxN>_<model>.json  (legacy name when HOLDOUT=0, CTXLO_PER_N=0,
     PATCH=prototype)
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch
import torch.nn as nn

import config as _config
from config import get_config
import graph as G
import models as M
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, two_colour, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEAD_LAYER = int(os.environ.get("HEAD_LAYER", "14")); HEAD_IDX = int(os.environ.get("HEAD_IDX", "26"))
GRIDS = [tuple(int(x) for x in g.split("x")) for g in os.environ.get("GRIDS", "4x4,6x6,8x8,10x10,12x12").split(",")]
RDIMS = [int(x) for x in os.environ.get("RDIMS", "1,2,4,8,16,32,128").split(",")]
NWALKS = int(os.environ.get("NWALKS", "10")); SPN = int(os.environ.get("SAMPLES_PER_NODE", "80"))
WLEN_CAP = int(os.environ.get("WLEN_CAP", "650")); CTXLO = int(os.environ.get("CTXLO", "80"))
STEPS = int(os.environ.get("STEPS", "45")); BATCH = int(os.environ.get("BATCH", "3"))
LR = float(os.environ.get("LR", "0.02")); SEED = int(os.environ.get("SEED", "0"))
PATCH = os.environ.get("PATCH_MODE", "prototype")   # 'prototype' = pure parity flip (no coord confound) | 'rotation'
HOLDOUT = int(os.environ.get("HOLDOUT", "0"))
CTXLO_PER_N = float(os.environ.get("CTXLO_PER_N", "0"))
SAVE_R = os.environ.get("SAVE_R", "0") == "1"   # save each grid's trained rotations (rows 0..r-1) to an npz
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


def rot90_perm(coords, k):
    idx = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
    return np.array([idx[(int(c), int(k - 1 - r))] for (r, c) in coords], int)


@torch.no_grad()
def capture_znode(model, tok, blocks, cm, graph, cfg, dev, csl, hd, n, ctxlo, n_src):
    """n_src: number of leading walks that contribute to znode/prototypes (patch sources); with a
    holdout the eval walks must not leak into the source estimates."""
    zc = {}
    def cap(_m, args): zc["z"] = args[0].detach()
    proj = attn_proj(blocks[HEAD_LAYER], cm)[0]; hk = proj.register_forward_pre_hook(cap)
    walks = G.generate_walks(graph, cfg); wdata = []; zsum = np.zeros((n, hd)); zcnt = np.zeros(n)
    for wi, wk in enumerate(walks):
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); zc.clear()
        model(input_ids=ids); seqlen = ids.shape[1]; ntok = []; readpos = []; readnode = []
        for s in range(len(nodes)):
            t = spans[s][-1]; nd = nodes[s]; ntok.append((t, nd))
            if cl[s] >= ctxlo and s < len(nodes) - 1:
                readpos.append(t); readnode.append(nd)
                if wi < n_src:
                    zsum[nd] += zc["z"][0, t, csl].float().cpu().numpy(); zcnt[nd] += 1
        wdata.append({"ids": ids, "ntok": ntok, "readpos": readpos, "readnode": readnode, "seqlen": seqlen})
    hk.remove()
    return wdata, zsum / np.maximum(zcnt, 1)[:, None]


def run_grid(model, tok, blocks, cm, dev, k, rng):
    n = k * k
    ctxlo = max(CTXLO, int(round(CTXLO_PER_N * n)))
    wl = min(WLEN_CAP, ctxlo + int(np.ceil(n * SPN / NWALKS)))
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=k, grid_cols=k, n_walks=NWALKS, walk_length=wl, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph); coords = np.array(graph.coords, int)
    pi = rot90_perm(coords, k) if PATCH == "rotation" else None
    if PATCH == "rotation":
        assert all(col[pi[i]] == -col[i] for i in range(n)), "rotation must invert parity (even k only)"
    proj, hd = attn_proj(blocks[HEAD_LAYER], cm); csl = slice(HEAD_IDX * hd, (HEAD_IDX + 1) * hd)
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    pos_idx = torch.tensor(np.where(col > 0)[0], device=dev); neg_idx = torch.tensor(np.where(col < 0)[0], device=dev)
    n_train = len(range(NWALKS)) - HOLDOUT if HOLDOUT else NWALKS
    assert n_train >= 1, "HOLDOUT leaves no training walks"
    wdata, znode = capture_znode(model, tok, blocks, cm, graph, cfg, dev, csl, hd, n, ctxlo, n_train)
    zt = torch.tensor(znode, dtype=torch.float32, device=dev)
    tgt_sign = col.copy()                                                # after flipping X to opposite colour, predict X's OWN colour
    if PATCH == "rotation":
        delta_of = lambda nd: zt[pi[nd]] - zt[nd]                        # per-node interchange (parity + coord remap)
    else:                                                                # PURE parity: opposite-minus-same COLOUR prototype (coord-free)
        pd = torch.tensor(znode[col > 0].mean(0) - znode[col < 0].mean(0), dtype=torch.float32, device=dev)
        delta_of = lambda nd: (-float(col[nd])) * pd
    for w in wdata:
        D = torch.zeros(w["seqlen"], hd, device=dev)
        for t, nd in w["ntok"]: D[t] = delta_of(nd)
        w["delta"] = D
        w["rp_t"] = torch.tensor(w["readpos"], device=dev, dtype=torch.long)
        w["tgt_t"] = torch.tensor([tgt_sign[nd] for nd in w["readnode"]], device=dev)

    state = {"delta": None, "Rr": None}
    def patch_pre(_m, args):
        if state["delta"] is not None and state["Rr"] is not None:
            x = args[0].clone(); Rr = state["Rr"]; x[0, :, csl] = x[0, :, csl] + ((state["delta"] @ Rr.t()) @ Rr).to(x.dtype)
            return (x,) + tuple(args[1:])
    ph = proj.register_forward_pre_hook(patch_pre)

    def eval_walk(w, Rr):
        state["delta"] = w["delta"] if Rr is not None else None; state["Rr"] = Rr
        logits = model(input_ids=w["ids"]).logits[0][w["rp_t"]][:, cand_t].float(); state["delta"] = None
        lsm = torch.log_softmax(logits, 1); tp = w["tgt_t"] > 0
        same = torch.where(tp[:, None], lsm[:, pos_idx], lsm[:, neg_idx]); opp = torch.where(tp[:, None], lsm[:, neg_idx], lsm[:, pos_idx])
        sm = torch.logsumexp(same, 1); om = torch.logsumexp(opp, 1)
        return -sm.mean(), (sm - om).mean().item()

    train_w = wdata[:n_train]; eval_w = wdata[n_train:] if HOLDOUT else wdata

    def margin(Rr, ws):
        return float(np.mean([eval_walk(w, Rr)[1] for w in ws]))
    curve = {}; curve_tr = {}; saved_R = {}
    m_base = margin(None, eval_w)                                              # unpatched baseline
    m_rand = margin(torch.linalg.qr(torch.randn(hd, hd, device=dev))[0][:4], eval_w)   # random rank-4 baseline
    for r in RDIMS:
        if r >= hd:
            with torch.no_grad():
                curve[r] = margin(torch.eye(hd, device=dev), eval_w); curve_tr[r] = margin(torch.eye(hd, device=dev), train_w)
            continue
        lin = nn.Linear(hd, hd, bias=False).to(dev); nn.utils.parametrizations.orthogonal(lin)
        opt = torch.optim.Adam(lin.parameters(), lr=LR)
        for step in range(STEPS):
            opt.zero_grad(); bs = [train_w[i] for i in rng.choice(len(train_w), min(BATCH, len(train_w)), replace=False)]
            loss = sum(eval_walk(w, lin.weight[:r])[0] for w in bs) / len(bs); loss.backward(); opt.step()
        with torch.no_grad():
            curve[r] = margin(lin.weight[:r].detach(), eval_w); curve_tr[r] = margin(lin.weight[:r].detach(), train_w)
            if SAVE_R: saved_R[r] = lin.weight[:r].detach().float().cpu().numpy()
    ph.remove()
    # effective rank = smallest r reaching 90% of (best - r1) improvement over the r1 margin
    xs = sorted(curve); m1 = curve[xs[0]]; mbest = max(curve.values()); thr = m1 + 0.9 * (mbest - m1)
    eff = next((r for r in xs if curve[r] >= thr), xs[-1])
    print(f"[{k}x{k}] n={n} wl={wl} ctxlo={ctxlo} margins " + " ".join(f"r{r}={curve[r]:+.2f}" for r in xs)
          + f"  base={m_base:+.2f} rand4={m_rand:+.2f}  eff_rank={eff}"
          + (f"  (train " + " ".join(f"r{r}={curve_tr[r]:+.2f}" for r in xs) + ")" if HOLDOUT else ""), flush=True)
    out = {"k": k, "n": n, "wl": wl, "ctxlo": ctxlo, "margins": {str(r): round(curve[r], 3) for r in xs},
           "baseline": round(m_base, 3), "rand4": round(m_rand, 3), "eff_rank": int(eff)}
    if HOLDOUT: out["margins_train"] = {str(r): round(curve_tr[r], 3) for r in xs}
    return out, saved_R


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    for p in model.parameters(): p.requires_grad_(False)
    cm = model.config; blocks = M._decoder_blocks(model)
    need = max(k * k for k, _ in GRIDS)
    if need > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, need)
    out = {"model": tag, "head": [HEAD_LAYER, HEAD_IDX], "rdims": RDIMS, "patch_mode": PATCH,
           "holdout": HOLDOUT, "ctxlo_per_n": CTXLO_PER_N, "grids": {}}
    Rsave = {}
    for k, _ in GRIDS:
        out["grids"][f"{k}x{k}"], saved_R = run_grid(model, tok, blocks, cm, dev, k, rng)
        for r, w in saved_R.items(): Rsave[f"{k}x{k}_R{r}"] = w
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    suffix = ("" if PATCH == "prototype" else f"_{PATCH}") + (f"_ho{HOLDOUT}" if HOLDOUT else "") \
        + (f"_ctx{CTXLO_PER_N:g}" if CTXLO_PER_N else "") + os.environ.get("OUTTAG", "")
    if Rsave:
        rp = f"{OUTDIR}/das_parity_scale_R{suffix}_{tag}.npz"
        np.savez_compressed(rp, **Rsave); print(f"rotations -> {rp}", flush=True)
    p = f"{OUTDIR}/das_parity_scale{suffix}_{tag}.json"; json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
