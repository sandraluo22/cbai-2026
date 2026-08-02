"""Find the ring circuit the way the grid circuit was found: causal pool -> greedy on BEHAVIOUR -> whittle.

The grid's bench9 was not found by attribution. It came from a causally-derived candidate pool, then a
greedy/whittle search whose objective was the task metric itself. That is exactly the step axis
attribution cannot imitate: cos^2 scores a head by how much of the DAS axes it writes, which selects
writers and is blind to routers, movers and read-out heads. So this reruns the pipeline natively on the
ring rather than importing the grid's answer.

Stage A (pool)     mean-ablate each of the 1024 heads ALONE; score = drop in neighbour mass. Causal, and
                   cheap because only one layer is touched.
Stage B (forward)  greedy keep-only over the top-POOL candidates: repeatedly add the head that most
                   raises keep-only neighbour mass (every other head mean-ablated, MLPs intact).
                   Random same-size keep-sets are scored at each k so "it grew" means something.
Stage C (whittle)  backward pass over the selected set, dropping the cheapest head each time, to expose
                   the minimal sufficient subset and each member's marginal value.

Search uses short walks for speed; the winning set must be re-scored at full settings against bench9
(keep_set_nbr.py) before any comparison is made — search-time numbers are not comparable to those.

Env: GEN_MODEL(Llama) K(16) GRAPH(ring) LAZY(0) NWALKS(3) WLEN(500) CTXLO(250) POOL(30) KMAX(14)
     NRAND(3) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/ring_circuit_search<OUTTAG>_<model>.json
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
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "500"))
CTXLO = int(os.environ.get("CTXLO", "250")); POOL = int(os.environ.get("POOL", "30"))
KMAX = int(os.environ.get("KMAX", "0"))   # forward greedy steps; 0 = skip (backward is primary)
WHITTLE_FROM_POOL = os.environ.get("WHITTLE_FROM_POOL", "1") == "1"; NRAND = int(os.environ.get("NRAND", "3"))
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

    n = K if GRAPH == "ring" else K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "ring", "ring_size": K} if GRAPH == "ring"
                     else {"graph_type": "grid", "grid_rows": K, "grid_cols": K}),
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)

    lr_ = np.random.default_rng(SEED); seqs = []
    for w in range(NWALKS):
        cur = w % n; nodes = [cur]
        for _ in range(WLEN - 1):
            if LAZY <= 0 or lr_.random() >= LAZY: cur = int(lr_.choice(graph.neighbors(cur)))
            nodes.append(cur)
        rp = [s for s in range(len(nodes) - 1) if s + 1 >= CTXLO]
        mask = torch.zeros(len(rp), n, device=dev)
        for j, s in enumerate(rp): mask[j, list(graph.adjacency[nodes[s]])] = 1.0
        seqs.append((torch.tensor([[bos] + [wid[x] for x in nodes]], device=dev),
                     torch.tensor([s + 1 for s in rp], device=dev), mask))

    st = {"keep": None, "abl": None, "means": None}
    def mk(L):
        def pre(_m, a):
            x = a[0]
            if st["means"] is None:
                st.setdefault("cap", {})[L] = x.detach(); return
            drop = ([h for h in range(nH) if (L, h) not in st["keep"]] if st["keep"] is not None
                    else [h for (l, h) in (st["abl"] or []) if l == L])
            if drop:
                x = x.clone()
                for h in drop:
                    sl = slice(h * hd, (h + 1) * hd)
                    x[0, :, sl] = st["means"][L][sl].to(x.dtype)
                return (x,) + tuple(a[1:])
        return pre
    hooks = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mk(L)) for L in range(nL)]

    def ev():
        ms = ac = tot = 0.0
        for ids, pos, mask in seqs:
            p = torch.softmax(model(input_ids=ids).logits[0][pos][:, cand_t].float(), 1)
            ms += float((p * mask).sum()); ac += float(mask.gather(1, p.argmax(1, keepdim=True)).sum())
            tot += mask.shape[0]
        return ms / tot, ac / tot

    st["means"] = None; full = ev()
    st["means"] = {L: st["cap"][L][0].float().mean(0) for L in st["cap"]}
    st["keep"] = set(); floor = ev(); st["keep"] = None
    print(f"[{tag}] {GRAPH}{n} lazy={LAZY} walks={NWALKS}x{WLEN} ctxlo={CTXLO}", flush=True)
    print(f"  full  mass {full[0]:.4f} acc {full[1]:.4f}   floor  mass {floor[0]:.4f} acc {floor[1]:.4f}",
          flush=True)
    rec = lambda v: (v - floor[0]) / (full[0] - floor[0])

    # ---- Stage A: single-head ablation necessity over all 1024 ----
    sc = np.zeros((nL, nH))
    for L in range(nL):
        for h in range(nH):
            st["abl"] = [(L, h)]; sc[L, h] = full[0] - ev()[0]
        print(f"  stageA L{L:<2} best {sc[L].max():+.4f} (H{int(sc[L].argmax())})", flush=True)
    st["abl"] = None
    flat = [(float(sc[L, h]), f"L{L}H{h}") for L in range(nL) for h in range(nH)]
    flat.sort(reverse=True)
    pool = [h for _, h in flat[:POOL]]
    print(f"\n  pool (top {POOL} by solo-ablation damage): " +
          ", ".join(f"{h}({v:+.3f})" for v, h in flat[:POOL]), flush=True)

    def parse(x): return {(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x}

    # ---- Stage B: forward greedy on keep-only behaviour ----
    sel, curve = [], []
    for k in range(1, KMAX + 1):
        best, bh = -1e9, None
        for h in pool:
            if h in sel: continue
            st["keep"] = parse(sel + [h]); m, _ = ev()
            if m > best: best, bh = m, h
        sel.append(bh)
        st["keep"] = parse(sel); m, a = ev()
        rnd = []
        for _ in range(NRAND):
            rs = set()
            while len(rs) < len(sel): rs.add((int(rng.integers(nL)), int(rng.integers(nH))))
            st["keep"] = rs; rnd.append(ev()[0])
        st["keep"] = None
        curve.append({"k": k, "added": bh, "mass": round(m, 4), "acc": round(a, 4),
                      "recov": round(rec(m), 4), "rand_mass": round(float(np.mean(rnd)), 4),
                      "rand_recov": round(rec(float(np.mean(rnd))), 4)})
        print(f"  k={k:<3} +{bh:<8} mass {m:.4f} acc {a:.4f} recov {rec(m):+.3f}   "
              f"random-{k} recov {rec(float(np.mean(rnd))):+.3f}", flush=True)

    # ---- Stage C: backward whittle ----
    # Backward is the PRIMARY search here (and is how the grid's set was obtained). Forward greedy is
    # unreliable on this objective because keep-only sits at the floor until several heads are present
    # (k=1 recovers ~0.003), so its early picks are made on noise. Whittling starts from the whole pool,
    # where the objective is well above floor, and every drop is a real comparison.
    cur = list(pool) if WHITTLE_FROM_POOL else list(sel)
    whittle = []
    st["keep"] = parse(cur); m0, a0 = ev(); st["keep"] = None
    print(f"\n  whittle start k={len(cur)} mass {m0:.4f} acc {a0:.4f} recov {rec(m0):+.3f}", flush=True)
    while len(cur) > 1:
        best, bh = -1e9, None
        for h in cur:
            st["keep"] = parse([x for x in cur if x != h]); m, _ = ev()
            if m > best: best, bh = m, h
        cur = [x for x in cur if x != bh]
        st["keep"] = parse(cur); m, a = ev(); st["keep"] = None
        rnd = []
        for _ in range(NRAND):
            rs = set()
            while len(rs) < len(cur): rs.add((int(rng.integers(nL)), int(rng.integers(nH))))
            st["keep"] = rs; rnd.append(ev()[0])
        st["keep"] = None
        whittle.append({"k": len(cur), "dropped": bh, "mass": round(m, 4), "acc": round(a, 4),
                        "recov": round(rec(m), 4), "set": list(cur),
                        "rand_recov": round(rec(float(np.mean(rnd))), 4)})
        print(f"  drop {bh:<8} -> k={len(cur):<3} mass {m:.4f} acc {a:.4f} recov {rec(m):+.3f}   "
              f"random-{len(cur)} recov {rec(float(np.mean(rnd))):+.3f}", flush=True)
    for hk in hooks: hk.remove()
    res = {"model": tag, "graph": GRAPH, "n": n, "lazy": LAZY, "full": full, "floor": floor,
           "solo_ablation": {f"L{L}H{h}": round(float(sc[L, h]), 5) for L in range(nL) for h in range(nH)},
           "pool": pool, "forward": curve, "backward": whittle, "selected": sel}
    p = f"{OUTDIR}/ring_circuit_search{OUTTAG}_{tag}.json"
    json.dump(res, open(p, "w"), indent=2); print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
