"""Per-layer node-mean attention-output cloud, for interpreting DAS subspace overlaps.

The DAS patch is (D R^T) R with D[t] = z[pi(u)] - z[u], so ONLY the component of R inside the span of
the node-mean differences does anything. With n nodes that span is at most n-1 dimensional (15 for
ring16) inside a 4096-d concat space. Two rank-16 subspaces both fitted to move node identity therefore
have to overlap inside that small space no matter which counterfactual trained them — which means a raw
principal-angle overlap between two DAS subspaces cannot be read as "they found the same thing" until
it is compared against that constraint. This dumps znode per layer so the overlap can be recomputed
restricted to (and matched against) the node-mean span.

Env: GEN_MODEL(Llama) K(16) GRAPH(ring) LAZY(0) NWALKS(8) SPN(300) CTXLO(400) WLEN_CAP(1600) SEED(0)
     OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/znode_layers<OUTTAG>_<model>.npz   keys L0..L{nL-1}, each [n, nH*hd]
"""
from __future__ import annotations
import os
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
NWALKS = int(os.environ.get("NWALKS", "8")); SPN = int(os.environ.get("SPN", "300"))
CTXLO = int(os.environ.get("CTXLO", "400")); WLEN_CAP = int(os.environ.get("WLEN_CAP", "1600"))
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/coordperm")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    n = K if GRAPH == "ring" else K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    wl = min(WLEN_CAP, CTXLO + int(np.ceil(n * SPN / NWALKS)))
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "ring", "ring_size": K} if GRAPH == "ring"
                     else {"graph_type": "grid", "grid_rows": K, "grid_cols": K}),
                  n_walks=NWALKS, walk_length=wl, device=dev)
    graph = G.build_graph(cfg); words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]

    cap = {}
    hooks = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(
        (lambda L: (lambda _m, a: cap.__setitem__(L, a[0].detach())))(L)) for L in range(nL)]

    lr_ = np.random.default_rng(SEED)
    zs = {L: torch.zeros(n, nH * hd, device=dev, dtype=torch.float32) for L in range(nL)}
    zc = torch.zeros(n, device=dev)
    for w in range(NWALKS):
        cur = w % n; nodes = [cur]
        for _ in range(wl - 1):
            if LAZY <= 0 or lr_.random() >= LAZY: cur = int(lr_.choice(graph.neighbors(cur)))
            nodes.append(cur)
        ids = torch.tensor([[bos] + [wid[x] for x in nodes]], device=dev)
        cap.clear(); model(input_ids=ids)
        idx = torch.tensor([t + 1 for t in range(len(nodes))], device=dev)
        nd = torch.tensor(nodes, device=dev)
        oh = torch.zeros(len(nodes), n, device=dev); oh[torch.arange(len(nodes)), nd] = 1.0
        for L in range(nL):
            zs[L] += oh.t() @ cap[L][0, idx].float()
        zc += oh.sum(0)
    for h in hooks: h.remove()
    out = {f"L{L}": (zs[L] / zc.clamp(min=1)[:, None]).cpu().numpy() for L in range(nL)}
    p = f"{OUTDIR}/znode_layers{OUTTAG}_{tag}.npz"
    np.savez_compressed(p, **out)
    print(f"[{tag}] {GRAPH}{n} lazy={LAZY} wl={wl} -> {p}  ({nL} layers, {n} x {nH*hd})", flush=True)


if __name__ == "__main__":
    main()
