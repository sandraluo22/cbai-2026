"""Attention-only DAS at EVERY layer, then greedy selection of the heads that build each layer's axes.

Motivation. Residual DAS cannot separate "attention writes this variable here" from "this arrived from
upstream": at L24 residual DAS reaches flip 1.00 at rank 4, but attention-only DAS at the same layer tops
out at 0.28 — the coordinate variable is in the stream but no L24 head is producing it. Restricting DAS
to the concatenated head-output space makes the question layer-local, and a sweep says where attention
actually writes.

Per layer L:
  1. DAS on the concatenated output of all nH heads at L (attention-only by construction; the patch
     cannot reach MLPs or the embedding). Interchange objective: delta_t = znode[pi(X_t)] - znode[X_t]
     injected at every node token, neighbourhood margin scored on HELD-OUT walks. Reports flip rate.
  2. Greedy over the nH heads at L to reconstruct that layer's DAS subspace, objective
     cos^2(vec(Y), vec(sum_{h in S} y_h)) with y_h head h's contribution. Heads are added until cos^2
     reaches TARGET (default 0.9), so the reported set is "the heads that build this layer's axes", not
     an arbitrary fixed count — k varies by layer and is reported.
     Within a layer heads run in parallel from the same input, so the decomposition is EXACT
     (sum over all heads == target; the script asserts it) and offline attribution equals causal
     intervention — the property that failed catastrophically for the cross-layer residual version
     (offline 0.972 -> causal 0.235, worse than random).
  3. With SAVE_R=1 each layer's learned rotation R is written to an npz, so the SUBSPACES from two
     different counterfactuals can be compared directly (principal angles), not just their head sets.

Output per layer: flip rate, top head, cos^2 at k=1/2/4, and the random-k=1 baseline. A layer where the
flip rate is at baseline has no attention-expressed coordinate variable, and its greedy row is
meaningless — reported but flagged.

PERM (ring). The default cyc1 is a ROTATION: every node moves one step, coherently. To ask whether the
layer profile is about rotational structure or just about local displacement, the alternative is a
transposition family — swapk<m> applies m DISJOINT adjacent transpositions, (0 1)(2 3)...:
  swapk1     ABCDE -> ABCED, the literal single adjacent swap. Only 2 of n nodes move, so the
             counterfactual delta is nonzero at 2/n of node tokens and the flip rate is scored on
             readouts at those 2 nodes only — low power BY CONSTRUCTION. n_eval is reported per layer.
  swapk<n/2> = swappairs, ABCDEF -> BADCFE. Same displacement 1 per node as cyc1, same number of moved
             nodes, but alternating rather than coherent — the displacement-matched non-rotation.
The ladder m = 1, 2, 4, n/2 separates "not a rotation" from "few nodes move", which a single swapk1 run
cannot. Also: refl, randbij (control), cyc<k>.
Flip rates are NOT comparable across perms (different baseline difficulty — see the transfer-matrix
normalisation in CHECKLIST). What IS comparable is the SHAPE of the layer profile and the identity of
the greedy heads, since every layer is trained on its own perm.

Env: GEN_MODEL(Llama) GRAPH(grid|ring) K(4) PERM(ring: cyc1) RANK(16) LAYERS(all) LAZY(0) NWALKS(8)
     HOLDOUT(2) SPN(60) STEPS(80) LR(0.02) BATCH(3) CTXLO(400) WLEN_CAP(1000) NRAND(3) TARGET(0.9)
     SAVE_R(0) SEED(0) OUTDIR DEVICE OUTTAG
Out: <OUTDIR>/layer_sweep_attn_das<OUTTAG>_<model>.json
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch
import torch.nn as nn

import config as _config
from config import get_config
import graph as G
from graph import Walk
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from grid_parity_compare import build_word_pool, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
GRAPH = os.environ.get("GRAPH", "grid"); K = int(os.environ.get("K", "4"))
RANK = int(os.environ.get("RANK", "16"))
LAZY = float(os.environ.get("LAZY", "0"))
TARGET = float(os.environ.get("TARGET", "0.9"))   # greedy stops once cos^2 to the layer's axes hits this
TRAIN_SUB = int(os.environ.get("TRAIN_SUB", "0"))  # cap readouts per walk in the GRADIENT (0 = all)
SAVE_R = os.environ.get("SAVE_R", "0") == "1"
NWALKS = int(os.environ.get("NWALKS", "8")); HOLDOUT = int(os.environ.get("HOLDOUT", "2"))
SPN = int(os.environ.get("SPN", "60")); STEPS = int(os.environ.get("STEPS", "80"))
LR = float(os.environ.get("LR", "0.02")); BATCH = int(os.environ.get("BATCH", "3"))
CTXLO = int(os.environ.get("CTXLO", "400")); WLEN_CAP = int(os.environ.get("WLEN_CAP", "1000"))
NRAND = int(os.environ.get("NRAND", "3")); SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/coordperm")
LAYERS_ENV = os.environ.get("LAYERS", "all")
PERM = os.environ.get("PERM", "")                 # ring only; default cyc1. grid is always rot180 here.


def ring_perm(name, n):
    """Bijection on ring nodes. The input sequence is never modified — the counterfactual enters only as
    node-mean activations z[pi(u)] - z[u] — so pi need only be a bijection, not an automorphism."""
    if name.startswith("swapk") or name in ("swapadj", "swappairs"):
        m = {"swapadj": 1, "swappairs": n // 2}.get(name) or int(name[5:])
        assert 1 <= m <= n // 2, f"swapk{m}: need 1 <= m <= n/2"
        pi = np.arange(n)
        for i in range(m):                        # m DISJOINT adjacent transpositions: (0 1)(2 3)...
            a, b = 2 * i, 2 * i + 1
            pi[a], pi[b] = b, a
        return pi
    if name.startswith("cyc"):
        return np.array([(i + int(name[3:] or 1)) % n for i in range(n)], int)
    if name == "refl":
        return np.array([(-i) % n for i in range(n)], int)
    if name == "randbij":
        return np.random.default_rng(4242 + 17 * n).permutation(n)
    raise ValueError(f"unknown ring perm {name}")


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    for p in model.parameters(): p.requires_grad_(False)
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    LAYERS = list(range(nL)) if LAYERS_ENV == "all" else [int(x) for x in LAYERS_ENV.split(",")]

    n = K * K if GRAPH == "grid" else K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    wl = min(WLEN_CAP, CTXLO + int(np.ceil(n * SPN / NWALKS)))
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "grid", "grid_rows": K, "grid_cols": K} if GRAPH == "grid"
                     else {"graph_type": "ring", "ring_size": K}),
                  n_walks=NWALKS, walk_length=wl, device=dev)
    graph = G.build_graph(cfg)
    if GRAPH == "grid":
        co = np.array(graph.coords, int); ix = {(int(r), int(c)): i for i, (r, c) in enumerate(co)}
        pi = np.array([ix[(K - 1 - int(r), K - 1 - int(c))] for (r, c) in co], int); pname = "rot180"
    else:
        pname = PERM or "cyc1"; pi = ring_perm(pname, n)
    nbr = [set(graph.adjacency[u]) for u in range(n)]
    TS = []
    for u in range(n):
        T = sorted(nbr[pi[u]] - nbr[u] - {u, int(pi[u])}); S = sorted(nbr[u] - nbr[pi[u]] - {u, int(pi[u])})
        TS.append(None if (pi[u] == u or not T or not S) else
                  (torch.tensor(T, device=dev), torch.tensor(S, device=dev)))
    words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    lr_ = np.random.default_rng(SEED); walks = []
    for w in range(NWALKS):
        cur = w % n; nodes = [cur]
        for _ in range(wl - 1):
            if LAZY <= 0 or lr_.random() >= LAZY: cur = int(lr_.choice(graph.neighbors(cur)))
            nodes.append(cur)
        walks.append(Walk(walk_id=w, nodes=nodes, words=[words[x] for x in nodes]))
    wdata = []
    for wk in walks:
        ids = torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev)
        steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        wdata.append({"ids": ids, "ntok": [(t + 1, wk.nodes[t]) for t in range(len(wk.nodes))],
                      "rp": torch.tensor([s + 1 for s in steps], device=dev),
                      "rn": [wk.nodes[s] for s in steps], "L": ids.shape[1]})
    ntr = NWALKS - HOLDOUT
    # Scored readouts on the held-out walks. For a sparse perm (swapk1 moves 2 of n nodes) only readouts
    # AT a moved node are scorable, so this can be ~2/n of the total — the flip rate's binomial sd is
    # 0.5/sqrt(n_eval) and must be quoted with any cross-perm comparison.
    n_eval = sum(1 for w in wdata[ntr:] for u in w["rn"] if TS[u] is not None)
    n_moved = int((pi != np.arange(n)).sum())
    print(f"[{tag}] {GRAPH} n={n} perm={pname} rank={RANK} lazy={LAZY} wl={wl} "
          f"moves={n_moved}/{n} usable={sum(x is not None for x in TS)}/{n} "
          f"n_eval={n_eval} (flip sd ~{0.5 / max(n_eval, 1) ** 0.5:.3f})", flush=True)

    cap, st = {}, {"R": None, "w": None, "keep": None}
    def pre(_m, args):
        x = args[0]
        cap["z"] = x.detach()
        if st["R"] is not None and st["w"] is not None:
            D = st["w"]["delta"]; Rr = st["R"]
            x = x + ((D @ Rr.t()) @ Rr).to(x.dtype).unsqueeze(0)
        if st["keep"] is not None:
            x = x.clone()
            for h in range(nH):
                if h not in st["keep"]:
                    sl = slice(h * hd, (h + 1) * hd)
                    x[0, :, sl] = x[0, :, sl].mean(0, keepdim=True)
        return (x,) + tuple(args[1:])

    results, basev, Rsave = {}, {}, {}
    print(f"\n{'L':>3} {'flip':>7} {'margin':>9} {'top head':<10} {'cos2@1':>8} {'k@'+str(TARGET):>5} "
          f"{'cos2@k':>8} {'rand@k':>8} {'rand@1':>8}")
    for L in LAYERS:
        hk = attn_proj(blocks[L], cm)[0].register_forward_pre_hook(pre)
        # per-node mean z on training walks
        zs = torch.zeros(n, nH * hd, device=dev); zc = torch.zeros(n, device=dev)
        with torch.no_grad():
            for wi, w in enumerate(wdata):
                st["R"] = st["w"] = st["keep"] = None
                model(input_ids=w["ids"])
                if wi < ntr:
                    Z = cap["z"][0].float()
                    for t, nd in w["ntok"]:
                        if t < Z.shape[0]: zs[nd] += Z[t]; zc[nd] += 1
        znode = zs / zc.clamp(min=1)[:, None]
        for w in wdata:
            Dm = torch.zeros(w["L"], nH * hd, device=dev)
            for t, nd in w["ntok"]:
                if t < w["L"]: Dm[t] = znode[pi[nd]] - znode[nd]
            w["delta"] = Dm
        def ev(Rr, ws, grad=False):
            tot, fl, ls, m = 0.0, 0, [], 0
            for w in ws:
                st["R"], st["w"] = Rr, w
                lg = model(input_ids=w["ids"]).logits[0][w["rp"]][:, cand_t].float()
                st["R"] = st["w"] = None
                lsm = torch.log_softmax(lg, 1)
                # TRAIN_SUB caps how many readouts contribute to the GRADIENT (the per-readout python
                # loop dominates runtime, and it is 8x longer for a perm that moves all 16 nodes than for
                # one that moves 2). Evaluation always uses every scorable readout, so the reported flip
                # and margin are unaffected — only the optimiser sees a subsample.
                idx = range(len(w["rn"]))
                if grad and TRAIN_SUB and len(w["rn"]) > TRAIN_SUB:
                    idx = w.setdefault("sub", rng.choice(len(w["rn"]), TRAIN_SUB, replace=False))
                for j in idx:
                    u = w["rn"][j]; ts = TS[u]
                    if ts is None: continue
                    T, S = ts
                    mt = torch.logsumexp(lsm[j, T], 0); ms = torch.logsumexp(lsm[j, S], 0)
                    if grad: ls.append(-mt)
                    tot += float(mt - ms); fl += int(float(mt - ms) > 0); m += 1
            if grad: return torch.stack(ls).mean() if ls else None
            return tot / max(m, 1), fl / max(m, 1)
        if not basev:                             # unpatched control: what the flip rate is worth 0
            with torch.no_grad(): bm, bf = ev(None, wdata[ntr:])
            basev.update(flip=round(bf, 4), margin=round(bm, 4), n_eval=n_eval)
            print(f"BASE {bf:7.3f} {bm:+9.3f}   (no patch; every layer row is against this)", flush=True)
        ln = nn.Linear(nH * hd, RANK, bias=False).to(dev)
        nn.utils.parametrizations.orthogonal(ln)
        opt = torch.optim.Adam(ln.parameters(), lr=LR)
        for _ in range(STEPS):
            opt.zero_grad()
            bs = [wdata[i] for i in rng.choice(ntr, min(BATCH, ntr), replace=False)]
            loss = ev(ln.weight[:RANK], bs, grad=True)
            if loss is not None: loss.backward(); opt.step()
        with torch.no_grad():
            Rr = ln.weight[:RANK].detach()
            marg, flip = ev(Rr, wdata[ntr:])
            # greedy: contributions of each head to this layer's DAS subspace
            Zn = znode - znode.mean(0, keepdim=True)
            Y = (Zn @ Rr.t()).cpu().numpy().reshape(-1); nY = float(Y @ Y)
            per = []
            for h in range(nH):
                mk_ = torch.zeros(nH * hd, device=dev); mk_[h * hd:(h + 1) * hd] = 1.0
                per.append(((Zn * mk_) @ Rr.t()).cpu().numpy().reshape(-1))
            per = np.stack(per)
            err = float(np.abs(per.sum(0) - Y).max())
            def c2(v):
                d = float(v @ v)
                return 0.0 if d < 1e-12 else float((v @ Y) ** 2 / (d * nY))
            # greedy until the reconstruction of THIS layer's axes reaches TARGET — k is an output, not
            # a fixed budget, so "the heads that build the axes here" is defined by recovery, not by rank
            sel, acc, rem, cs = [], np.zeros_like(Y), set(range(nH)), []
            while rem and (not cs or cs[-1] < TARGET):
                b, bs_, bv = None, -1.0, None
                for h in rem:
                    v = acc + per[h]; s = c2(v)
                    if s > bs_: b, bs_, bv = h, s, v
                acc = bv; rem.discard(b); sel.append(b); cs.append(bs_)
            # random k-subsets of the SAME layer, matched to the k this layer needed
            kk = len(sel)
            rk = []
            for _ in range(NRAND * 4):
                idx = rng.choice(nH, kk, replace=False)
                rk.append(c2(per[idx].sum(0)))
            r1 = float(np.mean([c2(per[rng.choice(nH)]) for _ in range(NRAND * 4)]))
            rkm, rksd = float(np.mean(rk)), float(np.std(rk))
        hk.remove()
        if SAVE_R: Rsave[f"L{L}"] = Rr.cpu().numpy()
        results[str(L)] = {"flip": round(flip, 4), "margin": round(marg, 4),
                           "top_heads": [f"L{L}H{h}" for h in sel], "cos2": [round(x, 4) for x in cs],
                           "k_target": kk, "rand_cos2_k": round(rkm, 4), "rand_cos2_k_sd": round(rksd, 4),
                           "rand_cos2_k1": round(r1, 4), "recon_err": err}
        print(f"{L:3} {flip:7.3f} {marg:+9.3f} {'L'+str(L)+'H'+str(sel[0]):<10} {cs[0]:8.4f} "
              f"{kk:5} {cs[-1]:8.4f} {rkm:8.4f} {r1:8.4f}", flush=True)
        json.dump({"model": tag, "graph": GRAPH, "n": n, "perm": pname, "rank": RANK,
                   "lazy": LAZY, "n_moved": n_moved, "n_eval": n_eval, "baseline": basev,
                   "steps": STEPS, "seed": SEED, "target": TARGET, "layers": results},
                  open(f"{OUTDIR}/layer_sweep_attn_das{OUTTAG}_{tag}.json", "w"), indent=2)
        if SAVE_R:
            np.savez_compressed(f"{OUTDIR}/layer_sweep_attn_das{OUTTAG}_{tag}.npz", **Rsave)
    print(f"\nDONE -> {OUTDIR}/layer_sweep_attn_das{OUTTAG}_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
