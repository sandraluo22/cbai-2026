"""Direct-logit attribution per head to the NEIGHBOUR contrast, on the ring walk.

Writers and readers are invisible to each other's assay. A head that builds the position variable shows
up in axis attribution but need not touch the logits; a head that reads the variable out contributes to
the logits directly and may write no axis content at all. This measures the second edge.

For head h at layer L, its additive contribution to the residual stream at a readout position is
    o_h = z[:, h*hd:(h+1)*hd] @ W_O[:, h*hd:(h+1)*hd]^T
and its DIRECT contribution to the logit of candidate word c (the path that skips all later layers) is
    dla_h(c) = ((o_h / rms_final) * g_final) . W_U[c]
with rms_final and g_final the final RMSNorm's scale, captured per position from the real forward pass
(the usual frozen-LayerNorm approximation — the scale is treated as constant w.r.t. removing o_h).

Score = mean over readouts of  [ mean over TRUE NEIGHBOURS of u ] - [ mean over non-neighbours ].
Reported next to the same quantity for every head, so a head's rank among all 1024 is available and
"large" means large relative to the population, not relative to nothing.

Env: GEN_MODEL(Llama) K(16) GRAPH(ring) LAZY(0) HEADS("L14H19,...") NWALKS(6) WLEN(1200) CTXLO(800)
     SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/head_dla_nbr<OUTTAG>_<model>.json
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
from grid_parity_compare import build_word_pool, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama"); K = int(os.environ.get("K", "16"))
GRAPH = os.environ.get("GRAPH", "ring"); LAZY = float(os.environ.get("LAZY", "0"))
HEADS = [h for h in os.environ.get("HEADS", "").split(",") if h]
NWALKS = int(os.environ.get("NWALKS", "6")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    core = model.model if hasattr(model, "model") else model
    fnorm = core.norm
    g_final = fnorm.weight.detach().float()
    eps = getattr(fnorm, "variance_epsilon", getattr(fnorm, "eps", 1e-6))
    W_U = model.lm_head.weight.detach().float()

    n = K if GRAPH == "ring" else K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "ring", "ring_size": K} if GRAPH == "ring"
                     else {"graph_type": "grid", "grid_rows": K, "grid_cols": K}),
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    WUc = W_U[torch.tensor(wid, device=dev)]                       # [n, hidden] candidate unembeddings

    cap = {}
    hooks = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(
        (lambda L: (lambda _m, a: cap.__setitem__(L, a[0].detach())))(L)) for L in range(nL)]
    resid = {}
    hooks.append(fnorm.register_forward_pre_hook(lambda _m, a: resid.__setitem__("x", a[0].detach())))

    lr_ = np.random.default_rng(SEED)
    tot = np.zeros((nL, nH)); cnt = 0
    for w in range(NWALKS):
        cur = w % n; nodes = [cur]
        for _ in range(WLEN - 1):
            if LAZY <= 0 or lr_.random() >= LAZY: cur = int(lr_.choice(graph.neighbors(cur)))
            nodes.append(cur)
        ids = torch.tensor([[bos] + [wid[x] for x in nodes]], device=dev)
        cap.clear(); resid.clear(); model(input_ids=ids)
        rp = [s for s in range(len(nodes) - 1) if s + 1 >= CTXLO]
        pos = torch.tensor([s + 1 for s in rp], device=dev)
        x = resid["x"][0, pos].float()                              # [P, hidden] pre-final-norm residual
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)      # [P, 1]
        nbmask = torch.zeros(len(rp), n, device=dev)
        for j, s in enumerate(rp): nbmask[j, list(graph.adjacency[nodes[s]])] = 1.0
        nb = nbmask / nbmask.sum(1, keepdim=True)
        nn_ = (1 - nbmask) / (1 - nbmask).sum(1, keepdim=True)
        for L in range(nL):
            proj, _ = attn_proj(blocks[L], cm)
            W_O = proj.weight.detach().float()                      # [hidden, nH*hd]
            z = cap[L][0, pos].float()                              # [P, nH*hd]
            for h in range(nH):
                sl = slice(h * hd, (h + 1) * hd)
                o = z[:, sl] @ W_O[:, sl].t()                       # [P, hidden]
                d = ((o / rms) * g_final) @ WUc.t()                 # [P, n] direct logit contribution
                tot[L, h] += float(((d * nb).sum(1) - (d * nn_).sum(1)).sum())
        cnt += len(rp)
    for hk in hooks: hk.remove()
    dla = tot / max(cnt, 1)
    flat = dla.ravel(); order = np.argsort(-flat)
    rank = {f"L{i//nH}H{i%nH}": int(np.where(order == i)[0][0]) + 1 for i in range(nL * nH)}
    res = {"model": tag, "graph": GRAPH, "n": n, "lazy": LAZY, "readouts": cnt,
           "dla": {f"L{L}H{h}": round(float(dla[L, h]), 5) for L in range(nL) for h in range(nH)},
           "rank": rank,
           "population": {"mean": round(float(flat.mean()), 5), "sd": round(float(flat.std()), 5),
                          "max": round(float(flat.max()), 5), "min": round(float(flat.min()), 5)}}
    print(f"[{tag}] {GRAPH}{n} readouts={cnt}  population mean {flat.mean():+.4f} sd {flat.std():.4f}",
          flush=True)
    print(f"\n{'head':<9}{'DLA(nbr-nonnbr)':>17}{'rank/1024':>11}{'z vs pop':>10}")
    for h in (HEADS or [f"L{i//nH}H{i%nH}" for i in order[:15]]):
        v = res["dla"][h]
        print(f"{h:<9}{v:>17.4f}{rank[h]:>11}{(v-flat.mean())/flat.std():>10.1f}", flush=True)
    print(f"\ntop-8 of all 1024: " + ", ".join(f"L{i//nH}H{i%nH}={flat[i]:.3f}" for i in order[:8]),
          flush=True)
    p = f"{OUTDIR}/head_dla_nbr{OUTTAG}_{tag}.json"
    json.dump(res, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
