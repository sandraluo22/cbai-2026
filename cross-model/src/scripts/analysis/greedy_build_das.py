"""(1) DAS on the residual stream -> causally-validated coordinate axes. (2) Greedy forward selection of
the heads that BUILD those axes.

Why this supersedes the behavioural greedy: selecting on a logit margin is gameable. On the ring that
search reported 181% "recovery" — a margin 1.75x the intact model — because heavy ablation concentrates
the output distribution and a ring node has only 2 neighbours among 16 candidates, so piling mass on them
inflates a logsumexp margin without representing position at all. A subspace-reconstruction objective has
no such failure mode: sharpening the output distribution does not change what a head WRITES into a fixed
subspace.

The axes come from DAS, so they are causal BY CONSTRUCTION — the subspace had to support an interchange
intervention (delta = znode[pi(u)] - znode[u] patched at every node token, margin scored on held-out
walks). That is the guarantee a probe cannot give.

Objective. With R the orthonormal DAS basis [r, d], H the per-node mean residual [n, d], and w_h the
per-node mean write of head h:
    Y      = H  @ R.T           the residual's coordinate-subspace content, per node   [n, r]
    y_h    = w_h @ R.T          head h's contribution to it                            [n, r]
    score(S) = cos^2( vec(Y), vec(sum_{h in S} y_h) )
Scale-invariant and bounded [0,1], so a head cannot score by writing a large vector in the wrong
direction. Note the ceiling is below 1 by construction: MLPs and the embedding also write into R, and no
set of attention heads can account for their share.

Everything after the single activation capture is linear algebra, so the greedy loop needs no forward
passes and can search ALL 1024 heads rather than a pre-filtered pool — removing the pool-selection bias
of the behavioural version.

Then the selected set is VALIDATED causally: keep only those heads, mean-ablate the rest, and re-measure
both the subspace content and neighbour validity, against random same-size sets.

Env: GEN_MODEL(Llama) GRAPH(grid|ring) K(4) LAYER(24) DAS_NPZ DAS_KEY(4x4_r4) KMAX(30) NWALKS(3)
     WLEN(1200) CTXLO(800) NRAND(5) VALIDATE(1) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/greedy_build_das<OUTTAG>_<model>.json
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

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
GRAPH = os.environ.get("GRAPH", "grid"); K = int(os.environ.get("K", "4"))
LAYER = int(os.environ.get("LAYER", "24"))
DAS_NPZ = os.environ.get("DAS_NPZ", ""); DAS_KEY = os.environ.get("DAS_KEY", "4x4_r4")
KMAX = int(os.environ.get("KMAX", "30"))
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); NRAND = int(os.environ.get("NRAND", "5"))
VALIDATE = os.environ.get("VALIDATE", "1") == "1"; SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")

PARITY21 = {"L15H30", "L16H20", "L2H22", "L16H1", "L13H18", "L25H7", "L14H26", "L9H11", "L1H20",
            "L21H10", "L4H12", "L3H17", "L21H2", "L14H19", "L14H17", "L10H2", "L7H25", "L8H11",
            "L4H16", "L1H21", "L2H26"}


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    dm = cm.hidden_size; hd = getattr(cm, "head_dim", None) or dm // nH
    rng = np.random.default_rng(SEED)

    z = np.load(DAS_NPZ)
    assert DAS_KEY in z.files, f"{DAS_KEY} not in {z.files}"
    R = z[DAS_KEY].astype(np.float64)
    q, _ = np.linalg.qr(R.T); R = q.T[:R.shape[0]]
    Rt = torch.tensor(R, dtype=torch.float32, device=dev)
    print(f"[{tag}] DAS axes {DAS_KEY} rank={R.shape[0]} from {os.path.basename(DAS_NPZ)}", flush=True)

    n = K * K if GRAPH == "grid" else K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "grid", "grid_rows": K, "grid_cols": K} if GRAPH == "grid"
                     else {"graph_type": "ring", "ring_size": K}),
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    data = []
    for w in G.generate_walks(graph, cfg):
        steps = [s for s in range(len(w.nodes) - 1) if s + 1 >= CTXLO]
        if steps:
            data.append((torch.tensor([[bos] + [wid[x] for x in w.nodes]], device=dev),
                         torch.tensor([s + 1 for s in steps], device=dev),
                         [w.nodes[s] for s in steps]))

    # ---- one capture pass: per-node mean residual, and per-node mean z for every head <= LAYER ----
    caps = {}
    hks = []
    for l in range(LAYER + 1):
        def mk(l):
            def hh(_m, args): caps[l] = args[0].detach()
            return hh
        hks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk(l)))
    Hs = torch.zeros(n, dm, device=dev); C = torch.zeros(n, device=dev)
    Zs = torch.zeros(LAYER + 1, nH, n, hd, device=dev)
    for ids, rp, nds in data:
        caps.clear()
        o = model(input_ids=ids, output_hidden_states=True)
        H = o.hidden_states[LAYER + 1][0, rp].float()
        oh = torch.zeros(len(nds), n, device=dev)
        oh[torch.arange(len(nds)), torch.tensor(nds, device=dev)] = 1.0
        Hs += oh.t() @ H; C += oh.sum(0)
        for l in range(LAYER + 1):
            Zl = caps[l][0, rp].float().view(len(nds), nH, hd)
            Zs[l] += torch.einsum("tn,tkh->knh", oh, Zl)
    for h in hks: h.remove()
    cn = C.clamp(min=1)[:, None]
    Hn = Hs / cn; Hn = Hn - Hn.mean(0, keepdim=True)
    Y = (Hn @ Rt.t()).cpu().numpy()                                  # [n, r] target

    names, ycontrib = [], []
    for l in range(LAYER + 1):
        Wl = attn_proj(blocks[l], cm)[0].weight.detach().float().t()  # [nH*hd, dm]
        for h in range(nH):
            Zm = Zs[l, h] / cn
            Zm = Zm - Zm.mean(0, keepdim=True)
            wr = Zm @ Wl[h * hd:(h + 1) * hd]                         # [n, dm]
            names.append(f"L{l}H{h}"); ycontrib.append((wr @ Rt.t()).cpu().numpy())
    Yc = np.stack(ycontrib)                                           # [nheads, n, r]
    yv = Y.reshape(-1); nyv = float(yv @ yv)
    print(f"[{tag}] {len(names)} heads at layers 0..{LAYER}; target ||Y||^2 = {nyv:.3f}", flush=True)

    def cos2(vec):
        d = float(vec @ vec)
        return 0.0 if d < 1e-12 else float((vec @ yv) ** 2 / (d * nyv))

    # ---- greedy forward over ALL heads (pure linear algebra, no forward passes) ----
    cur, acc, sel = [], np.zeros_like(yv), []
    remaining = set(range(len(names)))
    print(f"\n{'k':>3} {'added':<10} {'cos2':>8} {'delta':>8} {'rand_cos2':>10}  cumulative")
    Yflat = Yc.reshape(len(names), -1)
    for k in range(1, KMAX + 1):
        best, bs, bv = None, -1.0, None
        for i in remaining:
            v = acc + Yflat[i]; s = cos2(v)
            if s > bs: best, bs, bv = i, s, v
        prev = cos2(acc) if k > 1 else 0.0
        acc = bv; remaining.discard(best); sel.append(names[best])
        rc = []
        for _ in range(NRAND):
            idx = rng.choice(len(names), k, replace=False)
            rc.append(cos2(Yflat[idx].sum(0)))
        mark = "*" if names[best] in PARITY21 else " "
        print(f"{k:3} {names[best]+mark:<10} {bs:8.4f} {bs-prev:+8.4f} {np.mean(rc):10.4f}  "
              f"{'/'.join(sel[-3:])}", flush=True)
    res = {"model": tag, "graph": GRAPH, "n": n, "layer": LAYER, "das_key": DAS_KEY,
           "das_rank": int(R.shape[0]), "selected": sel,
           "curve": [{"k": i + 1, "head": sel[i]} for i in range(len(sel))]}

    # ---- causal validation: keep only the selected heads, re-measure ----
    if VALIDATE:
        st = {"keep": None}
        hooks = []
        for l in range(nL):
            def mk2(l):
                def ph(_m, args):
                    kp = st["keep"]
                    if kp is None: return
                    s2 = [h for h in range(nH) if (l, h) not in kp]
                    if not s2: return
                    x = args[0].clone()
                    for h in s2:
                        sl = slice(h * hd, (h + 1) * hd)
                        x[0, :, sl] = x[0, :, sl].mean(0, keepdim=True)
                    return (x,) + tuple(args[1:])
                return ph
            hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk2(l)))

        def keeponly(spec):
            st["keep"] = None if spec is None else {(int(h.split("H")[0][1:]), int(h.split("H")[1]))
                                                    for h in spec}
            S2 = torch.zeros(n, dm, device=dev); C2 = torch.zeros(n, device=dev)
            ok = tot = 0
            for ids, rp, nds in data:
                o = model(input_ids=ids, output_hidden_states=True)
                Hh = o.hidden_states[LAYER + 1][0, rp].float()
                top = o.logits[0][rp][:, cand_t].float().argmax(1).tolist()
                oh = torch.zeros(len(nds), n, device=dev)
                oh[torch.arange(len(nds)), torch.tensor(nds, device=dev)] = 1.0
                S2 += oh.t() @ Hh; C2 += oh.sum(0)
                for j, u in enumerate(nds):
                    ok += int(top[j] in graph.adjacency[u]); tot += 1
            Mn = S2 / C2.clamp(min=1)[:, None]; Mn = Mn - Mn.mean(0, keepdim=True)
            yy = (Mn @ Rt.t()).cpu().numpy().reshape(-1)
            st["keep"] = None
            return ok / tot, cos2(yy), float(np.linalg.norm(yy) / (np.linalg.norm(yv) + 1e-12))
        print(f"\n{'set':<14} {'nbr_valid':>10} {'cos2_to_Y':>11} {'norm_ratio':>11}")
        for nm, spec in (("full", None), ("floor", [])):
            a, b, c = keeponly(spec); print(f"{nm:<14} {a:10.4f} {b:11.4f} {c:11.4f}", flush=True)
            res[nm] = {"nbr": round(a, 4), "cos2": round(b, 4), "norm_ratio": round(c, 4)}
        for kk in (8, 16, min(KMAX, 24)):
            a, b, c = keeponly(sel[:kk])
            rr = []
            for _ in range(3):
                rr.append(keeponly([names[j] for j in rng.choice(len(names), kk, replace=False)]))
            print(f"{'greedy'+str(kk):<14} {a:10.4f} {b:11.4f} {c:11.4f}   "
                  f"random {np.mean([x[0] for x in rr]):.4f}/{np.mean([x[1] for x in rr]):.4f}",
                  flush=True)
            res[f"greedy{kk}"] = {"nbr": round(a, 4), "cos2": round(b, 4), "norm_ratio": round(c, 4),
                                  "rand_nbr": round(float(np.mean([x[0] for x in rr])), 4),
                                  "rand_cos2": round(float(np.mean([x[1] for x in rr])), 4)}
        for h in hooks: h.remove()
    p_ = f"{OUTDIR}/greedy_build_das{OUTTAG}_{tag}.json"
    json.dump(res, open(p_, "w"), indent=2); print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
