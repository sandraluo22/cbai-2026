"""Greedy forward selection for a COORDINATE circuit — optimising the objective directly.

Every previous coordinate-circuit attempt picked heads by a PROXY score and then tested the result:
parity-derived interchange carriers (19.5%), lazy-walk carriers (8.5%), RSA subspace attribution (-1.0%).
None of them searched over the coordinate objective itself, so "no method finds a coordinate circuit" was
a claim about three particular proxies, not about whether such a set exists.

Objective (causal, not correlational): the rot180 neighbourhood margin
    coord_margin = logsumexp over nbrs(pi(u)) \\ nbrs(u)  -  logsumexp over nbrs(u) \\ nbrs(pi(u))
with pi = rot180. The intact model strongly prefers its ACTUAL neighbours, so this is large and negative;
with everything ablated it sits at ~0. More negative = more coordinate discrimination preserved. Greedy
maximises |coord_margin| in the correct direction.

Two stages, because greedy over all 1024 heads is infeasible:
  1. POOL SCAN — ablate each head singly from the intact model, rank by coordinate damage, keep top POOL.
     This is a necessity score over the whole model, not a hand-picked shortlist.
  2. GREEDY FORWARD — keep-only search within the pool, adding whichever head most improves the objective.
     At every size k, NRAND random k-subsets OF THE POOL give the control: if greedy tracks random, the
     pool has no privileged members and the count is all that matters.

Parity margin is reported alongside but never optimised, so the parity/coordinate comparison at matched
set size is out-of-objective for coordinates and directly comparable to the 47.6% parity number.

Env: GEN_MODEL(Llama) K(4) POOL(60) KMAX(24) NWALKS(2) WLEN(1000) CTXLO(700) NRAND(3)
     SCAN_NWALKS(1) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/greedy_coord_circuit<OUTTAG>_<model>.json
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
GRAPH = os.environ.get("GRAPH", "grid")        # "grid" (K x K, rot180) | "ring" (K nodes, see PERM)
# PERM for the ring. rot180 on a grid is the maximal-displacement automorphism that preserves parity;
# the ring analogue is the ANTIPODAL rotation i -> i + n/2 (no fixed points, maximal displacement, and
# parity-preserving when n/2 is even). `reflect` (i -> -i) is the other dihedral generator but fixes two
# nodes, so it yields fewer usable readouts.
PERM = os.environ.get("PERM", "antipodal")
K = int(os.environ.get("K", "4")); POOL = int(os.environ.get("POOL", "60"))
KMAX = int(os.environ.get("KMAX", "24"))
NWALKS = int(os.environ.get("NWALKS", "2")); WLEN = int(os.environ.get("WLEN", "1000"))
CTXLO = int(os.environ.get("CTXLO", "700")); NRAND = int(os.environ.get("NRAND", "3"))
SCAN_NWALKS = int(os.environ.get("SCAN_NWALKS", "1")); SEED = int(os.environ.get("SEED", "0"))
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
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    rng = np.random.default_rng(SEED)

    n = K * K if GRAPH == "grid" else K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"),
                  **({"graph_type": "grid", "grid_rows": K, "grid_cols": K} if GRAPH == "grid"
                     else {"graph_type": "ring", "ring_size": K}),
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg)
    col = two_colour(graph) if (GRAPH == "grid" or n % 2 == 0) else np.ones(n, int)
    if GRAPH == "grid":
        coords = np.array(graph.coords, int)
        idx = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
        pi = np.array([idx[(K - 1 - int(r), K - 1 - int(c))] for (r, c) in coords], int)   # rot180
        pname = "rot180"
    elif PERM == "reflect":
        pi = np.array([(-i) % n for i in range(n)], int); pname = "reflect"
    else:
        pi = np.array([(i + n // 2) % n for i in range(n)], int); pname = "antipodal"
    assert len(set(pi.tolist())) == n, "perm must be a bijection"
    print(f"[{tag}] {GRAPH} n={n}, perm={pname}, parity preserved="
          f"{all(col[pi[i]] == col[i] for i in range(n))}", flush=True)
    nbr = [set(graph.adjacency[u]) for u in range(n)]
    TS = []
    for u in range(n):
        T = sorted(nbr[pi[u]] - nbr[u] - {u, int(pi[u])})
        S = sorted(nbr[u] - nbr[pi[u]] - {u, int(pi[u])})
        TS.append(None if (pi[u] == u or not T or not S) else
                  (torch.tensor(T, device=dev), torch.tensor(S, device=dev)))
    print(f"[{tag}] {sum(x is not None for x in TS)}/{n} usable readout nodes", flush=True)

    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    pos_i = torch.tensor(np.where(col > 0)[0], device=dev)
    neg_i = torch.tensor(np.where(col < 0)[0], device=dev)

    def mkdata(nw):
        c2 = replace(cfg, n_walks=nw)
        out = []
        for w in G.generate_walks(graph, c2):
            steps = [s for s in range(len(w.nodes) - 1) if s + 1 >= CTXLO]
            if steps:
                out.append((torch.tensor([[bos] + [wid[x] for x in w.nodes]], device=dev),
                            torch.tensor([s + 1 for s in steps], device=dev),
                            [w.nodes[s] for s in steps]))
        return out
    data_scan, data = mkdata(SCAN_NWALKS), mkdata(NWALKS)

    st = {"keep": None, "abl": None}
    hooks = []
    for l in range(nL):
        def mk(l):
            def ph(_m, args):
                kp, ab = st["keep"], st["abl"]
                if kp is not None: sel = [h for h in range(nH) if (l, h) not in kp]
                elif ab is not None: sel = [h for (ll, h) in ab if ll == l]
                else: return
                if not sel: return
                x = args[0].clone()
                for h in sel:
                    sl = slice(h * hd, (h + 1) * hd)
                    x[0, :, sl] = x[0, :, sl].mean(0, keepdim=True)
                return (x,) + tuple(args[1:])
            return ph
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk(l)))

    def measure(dat):
        cm_, pm_, tot = 0.0, 0.0, 0
        for ids, rp, nds in dat:
            lsm = torch.log_softmax(model(input_ids=ids).logits[0][rp][:, cand_t].float(), 1)
            cur = torch.tensor([col[u] for u in nds], device=dev)
            opp = torch.where((cur > 0)[:, None], lsm[:, neg_i], lsm[:, pos_i])
            sam = torch.where((cur > 0)[:, None], lsm[:, pos_i], lsm[:, neg_i])
            pm_ += float((torch.logsumexp(opp, 1) - torch.logsumexp(sam, 1)).sum())
            for j, u in enumerate(nds):
                ts = TS[u]
                if ts is None: continue
                T, S = ts
                cm_ += float(torch.logsumexp(lsm[j, T], 0) - torch.logsumexp(lsm[j, S], 0)); tot += 1
        return cm_ / max(tot, 1), pm_ / max(len(nds) * len(dat), 1)

    def parse(x): return {(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x}
    names = [f"L{l}H{h}" for l in range(nL) for h in range(nH)]

    st["keep"] = st["abl"] = None; full = measure(data)
    st["keep"] = set(); floor = measure(data); st["keep"] = None
    print(f"[{tag}] coord_margin  full={full[0]:+.4f}  floor={floor[0]:+.4f}   "
          f"(more NEGATIVE = coordinate discrimination preserved)", flush=True)

    # ---- stage 1: single-head ablation necessity scan over ALL heads ----
    print(f"\nscanning all {len(names)} heads (ablate-one, coordinate damage)...", flush=True)
    st["keep"] = None
    base_scan = None
    st["abl"] = None; base_scan = measure(data_scan)[0]
    dmg = np.zeros(len(names))
    for i, nm in enumerate(names):
        st["abl"] = parse([nm]); dmg[i] = measure(data_scan)[0] - base_scan; st["abl"] = None
        if i % 256 == 255: print(f"   {i+1}/{len(names)}", flush=True)
    order = np.argsort(-dmg)                      # most POSITIVE damage = margin moved toward 0
    pool = [names[i] for i in order[:POOL]]
    npar = sum(1 for h in pool if h in PARITY21)
    print(f"\npool = top {POOL} by ablation damage; {npar} of them are in the parity-21 set")
    print("   top 12: " + ", ".join(f"{names[i]}({dmg[i]:+.3f})" for i in order[:12]), flush=True)

    # ---- stage 2: greedy forward within the pool ----
    res = {"model": tag, "full": full, "floor": floor, "pool": pool,
           "pool_damage": {names[i]: round(float(dmg[i]), 4) for i in order[:POOL]},
           "forward": []}
    print(f"\n{'k':>3} {'added':<10} {'coord_marg':>11} {'recov%':>8} {'rand_marg':>10} {'rand_sd':>8} "
          f"{'parity_marg':>12}")
    cur, rest = [], list(pool)
    rng2 = np.random.default_rng(SEED)
    for k in range(1, KMAX + 1):
        best, bestv = None, None
        for h in rest:
            st["keep"] = parse(cur + [h]); v = measure(data); st["keep"] = None
            if bestv is None or v[0] < bestv[0]: best, bestv = h, v      # more negative is better
        cur.append(best); rest.remove(best)
        rv = []
        for _ in range(NRAND):
            sub = [pool[j] for j in rng2.choice(len(pool), k, replace=False)]
            st["keep"] = parse(sub); rv.append(measure(data)[0]); st["keep"] = None
        rm, rs = float(np.mean(rv)), float(np.std(rv))
        rec = 100 * (bestv[0] - floor[0]) / (full[0] - floor[0])
        res["forward"].append({"k": k, "added": best, "coord_margin": round(bestv[0], 4),
                               "recovery_pct": round(rec, 2), "rand_margin": round(rm, 4),
                               "rand_sd": round(rs, 4), "parity_margin": round(bestv[1], 4),
                               "set": list(cur)})
        mark = "*" if best in PARITY21 else " "
        print(f"{k:3} {best+mark:<10} {bestv[0]:+11.4f} {rec:7.1f}% {rm:+10.4f} {rs:8.4f} "
              f"{bestv[1]:+12.4f}", flush=True)
    for h in hooks: h.remove()
    print("\n(* = member of the parity-21 set)")
    res["graph"] = GRAPH; res["n"] = n; res["perm"] = pname
    p_ = f"{OUTDIR}/greedy_coord_circuit{OUTTAG}_{tag}.json"
    json.dump(res, open(p_, "w"), indent=2); print(f"DONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
