"""Behavioural transfer between DAS subspaces trained on DIFFERENT ring counterfactuals.

Geometric overlap (principal angles) says whether two subspaces are the same subspace. It does NOT say
whether one DOES the other's job — a subspace can share most of its energy and still not carry the
intervention. This evaluates every (subspace, counterfactual) pair at each layer with no retraining.

Every cell is normalised by what that counterfactual achieves with its OWN trained subspace at that
layer:   recovery = (margin_cross - base_perm) / (margin_self - base_perm)
because raw flips are not comparable across perms — they have different baseline difficulty and, for
swapk1, ~9x fewer scorable readouts (only the 2 moved nodes are scorable). This is the same
normalisation the ring transfer matrix used; unnormalised cross-perm flips are uninterpretable.

Env: GEN_MODEL(Llama) K(16) LAYERS("14,16,20") RNPZ("cyc1=path.npz,swapk1=path.npz") PERMS("cyc1,swapk1")
     LAZY(0) NWALKS(8) HOLDOUT(3) SPN(300) CTXLO(400) WLEN_CAP(1600) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/xfer_perm_das<OUTTAG>_<model>.json
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
from grid_parity_compare import build_word_pool, attn_proj
from layer_sweep_attn_das import ring_perm

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama"); K = int(os.environ.get("K", "16"))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "14,16,20").split(",")]
RNPZ = dict(s.split("=") for s in os.environ.get("RNPZ", "").split(",") if "=" in s)
PERMS = [p for p in os.environ.get("PERMS", "cyc1,swapk1").split(",") if p]
LAZY = float(os.environ.get("LAZY", "0"))
NWALKS = int(os.environ.get("NWALKS", "8")); HOLDOUT = int(os.environ.get("HOLDOUT", "3"))
SPN = int(os.environ.get("SPN", "300")); CTXLO = int(os.environ.get("CTXLO", "400"))
WLEN_CAP = int(os.environ.get("WLEN_CAP", "1600")); SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/coordperm")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nH = cm.num_attention_heads; hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    n = K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    wl = min(WLEN_CAP, CTXLO + int(np.ceil(n * SPN / NWALKS)))
    cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=n,
                  n_walks=NWALKS, walk_length=wl, device=dev)
    graph = G.build_graph(cfg); words = list(graph.words)
    nbr = [set(graph.adjacency[u]) for u in range(n)]
    PI, TS = {}, {}
    for p in PERMS:
        pi = ring_perm(p, n); PI[p] = pi; ts = []
        for u in range(n):
            T = sorted(nbr[pi[u]] - nbr[u] - {u, int(pi[u])}); S = sorted(nbr[u] - nbr[pi[u]] - {u, int(pi[u])})
            ts.append(None if (pi[u] == u or not T or not S) else
                      (torch.tensor(T, device=dev), torch.tensor(S, device=dev)))
        TS[p] = ts
        print(f"[{tag}] perm {p}: moves {int((pi != np.arange(n)).sum())}/{n}, "
              f"{sum(x is not None for x in ts)} usable nodes", flush=True)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)

    lr_ = np.random.default_rng(SEED); wdata = []
    for w in range(NWALKS):
        cur = w % n; nodes = [cur]
        for _ in range(wl - 1):
            if LAZY <= 0 or lr_.random() >= LAZY: cur = int(lr_.choice(graph.neighbors(cur)))
            nodes.append(cur)
        ids = torch.tensor([[bos] + [wid[x] for x in nodes]], device=dev)
        steps = [s for s in range(len(nodes) - 1) if s + 1 >= CTXLO]
        wdata.append({"ids": ids, "ntok": [(t + 1, nodes[t]) for t in range(len(nodes))],
                      "rp": torch.tensor([s + 1 for s in steps], device=dev),
                      "rn": [nodes[s] for s in steps], "L": ids.shape[1]})
    ntr = NWALKS - HOLDOUT
    Rs = {k: np.load(v) for k, v in RNPZ.items()}

    cap, st = {}, {"R": None, "w": None}
    def pre(_m, args):
        x = args[0]; cap["z"] = x.detach()
        if st["R"] is not None and st["w"] is not None:
            x = x + ((st["w"]["delta"] @ st["R"].t()) @ st["R"]).to(x.dtype).unsqueeze(0)
        return (x,) + tuple(args[1:])

    out = {"model": tag, "n": n, "lazy": LAZY, "perms": PERMS, "rnpz": RNPZ, "layers": {}}
    for L in LAYERS:
        hk = attn_proj(blocks[L], cm)[0].register_forward_pre_hook(pre)
        zs = torch.zeros(n, nH * hd, device=dev); zc = torch.zeros(n, device=dev)
        for wi, w in enumerate(wdata):
            st["R"] = st["w"] = None; model(input_ids=w["ids"])
            if wi < ntr:
                Z = cap["z"][0].float()
                for t, nd in w["ntok"]:
                    if t < Z.shape[0]: zs[nd] += Z[t]; zc[nd] += 1
        znode = zs / zc.clamp(min=1)[:, None]

        def ev(Rr, perm):
            pi = PI[perm]; ts_ = TS[perm]; tot = fl = m = 0.0
            for w in wdata[ntr:]:
                Dm = torch.zeros(w["L"], nH * hd, device=dev)
                for t, nd in w["ntok"]:
                    if t < w["L"]: Dm[t] = znode[pi[nd]] - znode[nd]
                w["delta"] = Dm
                st["R"], st["w"] = Rr, w
                lg = model(input_ids=w["ids"]).logits[0][w["rp"]][:, cand_t].float()
                st["R"] = st["w"] = None
                lsm = torch.log_softmax(lg, 1)
                for j, u in enumerate(w["rn"]):
                    if ts_[u] is None: continue
                    T, S = ts_[u]
                    d = float(torch.logsumexp(lsm[j, T], 0) - torch.logsumexp(lsm[j, S], 0))
                    tot += d; fl += int(d > 0); m += 1
            return tot / max(m, 1), fl / max(m, 1)

        row = {"base": {}, "cells": {}}
        for p in PERMS:
            mm, ff = ev(None, p); row["base"][p] = {"margin": round(mm, 3), "flip": round(ff, 3)}
        self_m = {}
        for rn in Rs:
            Rr = torch.tensor(Rs[rn][f"L{L}"], dtype=torch.float32, device=dev)
            for p in PERMS:
                mm, ff = ev(Rr, p)
                row["cells"][f"{rn}->{p}"] = {"margin": round(mm, 3), "flip": round(ff, 3)}
                if rn == p: self_m[p] = mm
        for k, v in row["cells"].items():
            rn, p = k.split("->")
            b = row["base"][p]["margin"]; c = self_m.get(p)
            v["recovery"] = round((v["margin"] - b) / (c - b), 3) if c is not None and abs(c - b) > 1e-6 else None
        hk.remove()
        out["layers"][str(L)] = row
        print(f"\nL{L}  base " + "  ".join(f"{p}={row['base'][p]['margin']:+.2f}(flip {row['base'][p]['flip']:.2f})"
                                           for p in PERMS), flush=True)
        for k, v in row["cells"].items():
            print(f"   {k:<20} margin {v['margin']:+7.2f}  flip {v['flip']:.3f}  "
                  f"recovery {v['recovery']}", flush=True)
        json.dump(out, open(f"{OUTDIR}/xfer_perm_das{OUTTAG}_{tag}.json", "w"), indent=2)
    print(f"\nDONE -> {OUTDIR}/xfer_perm_das{OUTTAG}_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
