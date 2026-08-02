"""WHICH HEADS write into a rank-r DAS subspace? Exact additive attribution, rank-aware.

Motivation: coordinates are not a rank-1 variable (residual DAS flips coords ~1% at r=1 but 91% at r=4),
so scalar rank-1 attribution is the wrong tool for finding coordinate heads. And the readout LAYER must sit
DOWNSTREAM of the writers — attributing a layer-14 subspace can never see L21H10, the main coordinate
writer.

The residual is an exact sum, h_L = emb + sum_{l<=L}(attn_l + mlp_l), and attn_l splits per head as
z_lh @ W_O[h]. So each head's contribution to the r-dim subspace is exactly

    c_lh(t) = (z_lh(t) @ W_O[h]) @ R^T          in R^r

For each head we report:
  write_norm   mean ||c_lh(t)||                       — how much it writes into the subspace at all
  coord_R2     R^2 of predicting the node's (row,col) from c_lh(t)   — is what it writes coordinate info
  parity_r     |corr| of the leading component with parity           — contamination check
Ranked by coord_R2, this is coordinate-head identification at the correct rank and the correct depth.

Env: GEN_MODEL(Llama) NPZ(...das_multihead_resid_rot180_L<LAYER>_lazy_save_<model>.npz) RKEY(4x4_r8)
     LAYER(24) K(4) NWALKS(3) WLEN(1200) CTXLO(800) LAZY(0.5) TOPK(30) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/subspace_write_attribution<OUTTAG>_<model>.json
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
LAYER = int(os.environ.get("LAYER", "24"))
NPZ = os.environ.get("NPZ", "")
RKEY = os.environ.get("RKEY", "4x4_r8")
K = int(os.environ.get("K", "4")); NWALKS = int(os.environ.get("NWALKS", "3"))
WLEN = int(os.environ.get("WLEN", "1200")); CTXLO = int(os.environ.get("CTXLO", "800"))
LAZY = float(os.environ.get("LAZY", "0.5")); TOPK = int(os.environ.get("TOPK", "30"))
SEED = int(os.environ.get("SEED", "0"))
P = "runs/axes/4_circuits/parity"
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", P)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    dm = cm.hidden_size; hd = getattr(cm, "head_dim", None) or dm // nH

    npz = NPZ or f"{P}/das_multihead_resid_rot180{'_lazy' if LAZY > 0 else ''}_L{LAYER}_save_{GEN_MODEL}.npz"
    z = np.load(npz)
    R = z[RKEY].astype(np.float32)
    q, _ = np.linalg.qr(R.T); R = q.T[:R.shape[0]]                     # orthonormal rows, [r, dm]
    Rt = torch.tensor(R, dtype=torch.float32, device=dev)
    r = R.shape[0]
    print(f"[{tag}] subspace {npz}:{RKEY} rank={r}, attributing heads in layers 0..{LAYER}", flush=True)

    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph); coords = np.array(graph.coords, float)
    words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]

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

    # per-head projection matrices: P_lh = W_O[l][:, h] -> [hd, dm]; then into subspace: [hd, r]
    PR = {}
    for l in range(LAYER + 1):
        W = attn_proj(blocks[l], cm)[0].weight.detach().float()        # [dm, nH*hd]
        PR[l] = (W.t() @ Rt.t()).view(nH, hd, r)                       # [nH, hd, r]

    caps = {}
    hooks = []
    for l in range(LAYER + 1):
        def mk(l):
            def hh(_m, args): caps[l] = args[0].detach()
            return hh
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk(l)))

    C = {(l, h): [] for l in range(LAYER + 1) for h in range(nH)}
    Zraw = {(l, h): [] for l in range(LAYER + 1) for h in range(nH)}
    Y = []
    for wk in walks:
        ids = torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev)
        steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        rp = torch.tensor([s + 1 for s in steps], device=dev)
        caps.clear(); model(input_ids=ids)
        Y += [wk.nodes[s] for s in steps]
        for l in range(LAYER + 1):
            Zl = caps[l][0, rp].float().view(len(steps), nH, hd)       # [T, nH, hd]
            cc = torch.einsum("tnh,nhr->tnr", Zl, PR[l]).cpu().numpy() # [T, nH, r]
            for h in range(nH):
                C[(l, h)].append(cc[:, h, :]); Zraw[(l, h)].append(Zl[:, h, :].cpu().numpy())
    for hk in hooks: hk.remove()
    Y = np.array(Y); XY = coords[Y]                                    # [T, 2] true (row,col)
    par = col[Y]

    # RSA instead of a fitted map. With r=8 predictors and only 16 distinct nodes, BOTH in-sample R^2
    # (overfits: ~0.85 for any projection) and held-out-node CV (exactly-determined on 8 training nodes,
    # so R^2 << 0) are uninterpretable. Representational similarity fits NOTHING: build the head's
    # per-node mean contribution in the subspace, take all 120 pairwise distances, and correlate with the
    # true coordinate distances. Same treatment for a random r-dim projection of the same head = control.
    iu = np.triu_indices(n, 1)
    Dcoord = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1)[iu]     # Manhattan on the grid

    def rsa(Cm):
        pn = np.zeros((n, Cm.shape[1]))
        for nd in range(n):
            m = Y == nd
            if m.sum(): pn[nd] = Cm[m].mean(0)
        D = np.linalg.norm(pn[:, None, :] - pn[None, :, :], axis=-1)[iu]
        if D.std() < 1e-9: return 0.0
        return float(np.corrcoef(D, Dcoord)[0, 1])

    rows = []
    for (l, h), lst in C.items():
        Cm = np.concatenate(lst, 0)
        nrm = float(np.linalg.norm(Cm, axis=1).mean())
        if nrm < 1e-8:
            rows.append({"head": f"L{l}H{h}", "layer": l, "write_norm": 0.0, "coord_R2": 0.0, "parity_r": 0.0})
            continue
        lead = Cm[:, 0] if r == 1 else Cm @ np.linalg.svd(Cm - Cm.mean(0), full_matrices=False)[2][0]
        Zc = np.concatenate(Zraw[(l, h)], 0)
        Rr = np.linalg.qr(np.random.default_rng(l * 100 + h).standard_normal((Zc.shape[1], r)))[0]
        rows.append({"head": f"L{l}H{h}", "layer": l, "write_norm": round(nrm, 4),
                     "coord_rsa": round(rsa(Cm), 4),
                     "coord_rsa_randproj": round(rsa(Zc @ Rr), 4),
                     "parity_r": round(float(abs(np.corrcoef(lead, par)[0, 1])), 3)})
    rows.sort(key=lambda d: -d["coord_rsa"])
    print(f"\nTOP {TOPK} heads writing COORDINATE info into the rank-{r} subspace (lazy={LAZY}):")
    print(f"  {'head':9} {'layer':>5} {'rsa':>7} {'randproj':>9} {'lift':>7} {'norm':>7} {'par_r':>6}")
    for d in rows[:TOPK]:
        print(f"  {d['head']:9} {d['layer']:5} {d['coord_rsa']:7.3f} {d['coord_rsa_randproj']:9.3f}"
              f" {d['coord_rsa']-d['coord_rsa_randproj']:7.3f} {d['write_norm']:7.3f} {d['parity_r']:6.3f}")
    keep = [d["head"] for d in rows[:22]]
    print(f"\nKEEP={','.join(keep)}")
    out = {"model": tag, "layer": LAYER, "rank": r, "rkey": RKEY, "lazy": LAZY, "npz": npz,
           "heads": rows, "suggested_keep": keep}
    p = f"{OUTDIR}/subspace_write_attribution{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
