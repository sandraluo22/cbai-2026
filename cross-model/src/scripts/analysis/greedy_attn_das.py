"""Greedy selection of the heads that build an ATTENTION-ONLY DAS subspace — with no attribution gap.

Why attention-only. Residual-stream DAS gave axes that attention barely builds: with all 1024 heads
mean-ablated, cos^2 to those axes was still 0.558 (grid) / 0.727 (ring), i.e. most of the coordinate
subspace is written by MLPs and the embedding. No head-selection method can recover what attention does
not produce. DAS in MODE=concat over all heads at one layer patches the CONCATENATED head-output space,
so the learned subspace is one attention can express by construction.

Why single-layer makes it exact. The earlier greedy had a catastrophic attribution/intervention gap
(offline cos^2 0.972 -> causal 0.235, WORSE than random) because ablating heads changes the surviving
heads' inputs. Here both the subspace and the selection live at ONE layer L, and heads within a layer are
computed in parallel from the same layer input — so mean-ablating some heads at L leaves the others' z
untouched. Centred per-node contributions of ablated heads go to exactly zero, and

    sum over ALL heads at L of (their contribution) == the target, exactly

so the floor is 0, the ceiling is 1, and offline attribution and causal intervention must agree. Any
residual disagreement is a bug, which makes this a self-checking design.

Reported per k: cos^2 to the target, plus random k-subsets of the same 32 heads, plus (at selected sizes)
a causal keep-only check that ablates the OTHER heads at layer L and re-measures both the subspace
content and neighbour validity.

Env: GEN_MODEL(Llama) GRAPH(grid|ring) K(4) LAYER(24) DAS_NPZ DAS_KEY KMAX(16) NWALKS(3) WLEN(1200)
     CTXLO(800) NRAND(5) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/greedy_attn_das<OUTTAG>_<model>.json
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
KMAX = int(os.environ.get("KMAX", "16"))
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); NRAND = int(os.environ.get("NRAND", "5"))
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    rng = np.random.default_rng(SEED)

    z = np.load(DAS_NPZ)
    assert DAS_KEY in z.files, f"{DAS_KEY} not in {list(z.files)}"
    R = z[DAS_KEY].astype(np.float64)
    q, _ = np.linalg.qr(R.T); R = q.T[:R.shape[0]]                    # [r, nH*hd] concat space
    assert R.shape[1] == nH * hd, f"expected concat-space DAS of width {nH*hd}, got {R.shape[1]}"
    Rt = torch.tensor(R, dtype=torch.float32, device=dev)
    print(f"[{tag}] attention-only DAS {DAS_KEY} rank={R.shape[0]} at L{LAYER} "
          f"(concat space {R.shape[1]} = {nH} heads x {hd})", flush=True)

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

    cap = {}
    hk = attn_proj(blocks[LAYER], cm)[0].register_forward_pre_hook(
        lambda _m, a: cap.__setitem__("z", a[0].detach()))
    Zs = torch.zeros(n, nH * hd, device=dev); C = torch.zeros(n, device=dev)
    for ids, rp, nds in data:
        cap.clear(); model(input_ids=ids)
        Zl = cap["z"][0, rp].float()
        oh = torch.zeros(len(nds), n, device=dev)
        oh[torch.arange(len(nds)), torch.tensor(nds, device=dev)] = 1.0
        Zs += oh.t() @ Zl; C += oh.sum(0)
    hk.remove()
    Zn = Zs / C.clamp(min=1)[:, None]
    Zn = Zn - Zn.mean(0, keepdim=True)                                 # [n, nH*hd]
    Y = (Zn @ Rt.t()).cpu().numpy().reshape(-1)
    nY = float(Y @ Y)
    per = []
    for h in range(nH):
        m = torch.zeros(nH * hd, device=dev); m[h * hd:(h + 1) * hd] = 1.0
        per.append(((Zn * m) @ Rt.t()).cpu().numpy().reshape(-1))
    per = np.stack(per)
    chk = float(np.abs(per.sum(0) - Y).max())
    print(f"[{tag}] ||Y||^2 = {nY:.3f}; sum-of-heads reconstruction error = {chk:.2e} "
          f"(must be ~0 -- the decomposition is exact within a layer)", flush=True)

    def cos2(v):
        d = float(v @ v)
        return 0.0 if d < 1e-12 else float((v @ Y) ** 2 / (d * nY))

    sel, acc, rem = [], np.zeros_like(Y), set(range(nH))
    print(f"\n{'k':>3} {'head':<8} {'cos2':>8} {'delta':>8} {'rand_cos2':>10}")
    curve = []
    for k in range(1, min(KMAX, nH) + 1):
        best, bs, bv = None, -1.0, None
        for h in rem:
            v = acc + per[h]; s = cos2(v)
            if s > bs: best, bs, bv = h, s, v
        prev = cos2(acc) if k > 1 else 0.0
        acc = bv; rem.discard(best); sel.append(best)
        rc = [cos2(per[rng.choice(nH, k, replace=False)].sum(0)) for _ in range(NRAND)]
        curve.append({"k": k, "head": f"L{LAYER}H{best}", "cos2": round(bs, 4),
                      "delta": round(bs - prev, 4), "rand_cos2": round(float(np.mean(rc)), 4)})
        print(f"{k:3} L{LAYER}H{best:<4} {bs:8.4f} {bs-prev:+8.4f} {np.mean(rc):10.4f}", flush=True)

    # ---- causal check: ablate the OTHER heads AT LAYER L only ----
    st = {"keep": None}
    hk2 = None
    def ph(_m, args):
        kp = st["keep"]
        if kp is None: return
        x = args[0].clone()
        for h in range(nH):
            if h not in kp:
                sl = slice(h * hd, (h + 1) * hd)
                x[0, :, sl] = x[0, :, sl].mean(0, keepdim=True)
        return (x,) + tuple(args[1:])
    hk2 = attn_proj(blocks[LAYER], cm)[0].register_forward_pre_hook(ph)

    def keeponly(hs):
        st["keep"] = set(hs)
        S2 = torch.zeros(n, nH * hd, device=dev); C2 = torch.zeros(n, device=dev)
        ok = tot = 0
        cap2 = {}
        h3 = attn_proj(blocks[LAYER], cm)[0].register_forward_pre_hook(
            lambda _m, a: cap2.__setitem__("z", a[0].detach()))
        for ids, rp, nds in data:
            cap2.clear()
            out = model(input_ids=ids)
            Zl = cap2["z"][0, rp].float()
            oh = torch.zeros(len(nds), n, device=dev)
            oh[torch.arange(len(nds)), torch.tensor(nds, device=dev)] = 1.0
            S2 += oh.t() @ Zl; C2 += oh.sum(0)
            top = out.logits[0][rp][:, cand_t].float().argmax(1).tolist()
            for j, u in enumerate(nds):
                ok += int(top[j] in graph.adjacency[u]); tot += 1
        h3.remove(); st["keep"] = None
        Zc = S2 / C2.clamp(min=1)[:, None]; Zc = Zc - Zc.mean(0, keepdim=True)
        return ok / tot, cos2((Zc @ Rt.t()).cpu().numpy().reshape(-1))
    print(f"\n{'set':<12} {'nbr_valid':>10} {'cos2 (causal)':>14} {'cos2 (offline)':>15}")
    st["keep"] = None
    base_nbr, _ = keeponly(set(range(nH)))
    print(f"{'all 32':<12} {base_nbr:10.4f} {1.0:14.4f} {1.0:15.4f}")
    res = {"model": tag, "graph": GRAPH, "layer": LAYER, "das_key": DAS_KEY,
           "selected": [f"L{LAYER}H{h}" for h in sel], "curve": curve,
           "recon_err": chk, "all32_nbr": round(base_nbr, 4), "causal": {}}
    for kk in (1, 2, 4, 8, min(KMAX, 16)):
        a, b = keeponly(sel[:kk])
        off = curve[kk - 1]["cos2"]
        res["causal"][str(kk)] = {"nbr": round(a, 4), "cos2_causal": round(b, 4), "cos2_offline": off}
        print(f"{'greedy'+str(kk):<12} {a:10.4f} {b:14.4f} {off:15.4f}", flush=True)
    hk2.remove()
    p_ = f"{OUTDIR}/greedy_attn_das{OUTTAG}_{tag}.json"
    json.dump(res, open(p_, "w"), indent=2); print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
