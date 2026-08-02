"""Multi-head / multi-site DAS for the parity variable. Three MODEs, sharing the rotation-interchange
objective of das_parity_scale (delta_t = znode[pi(X_t)] - znode[X_t], pi = 90-degree grid rotation that
inverts every parity for even k; injected at every node token; margin evaluated on HELD-OUT walks):

  block  — one rotation per head over its 128-d output space, all heads patched simultaneously and
           trained JOINTLY (rank r per head). Heads may sit at different layers (deltas are clean-run
           node means; downstream interaction flows through the patched forward — the simple form of
           multi-layer patching, not full causal scrubbing).
  concat — a single rotation over the CONCATENATED K*128-d space of all heads: the learned subspace may
           mix heads (rank r total).
  resid  — DAS directly on the residual stream at the output of block LAYER (4096-d, rank r total):
           captures the summed write of every head at that depth.

Also reports flip_rate: fraction of held-out readouts where the patched margin is positive (the model
actually predicts the flipped colour class), which single-head DAS never achieved.

Env: GEN_MODEL(Llama) MODE(block) HEADS("14:26,14:19") LAYER(14, resid only) GRIDS(4x4,8x8)
     RDIMS(block: per-head "1,4,8,16" | concat "4,8,16,32,48" | resid "1,4,8,16,32,64")
     NWALKS(10) SAMPLES_PER_NODE(80) CTXLO(1000) WLEN_CAP(1600) HOLDOUT(3) STEPS(45) BATCH(3) LR(0.02)
     SEED(0) SAVE_R(0) OUTTAG("") OUTDIR DEVICE
Out: <OUTDIR>/das_multihead_<MODE>_<heads-or-layer><OUTTAG>_<model>.json
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
MODE = os.environ.get("MODE", "block")
# PERM: which grid automorphism supplies the interchange counterfactual.
#   rot90     — inverts every node's parity (the parity experiments; colour-class margin)
#   rot180    — reverses BOTH coordinates, preserves parity (coordinate experiments)
#   transpose — swaps row<->col, preserves parity
#   swap2     — identity except two same-colour corners exchanged (local coordinate patch)
# For non-rot90 perms the margin is neighbourhood-based: logsumexp over nbrs(pi(X))\nbrs(X)
# minus logsumexp over nbrs(X)\nbrs(pi(X)) at each readout (skipped where pi(X)=X).
#   rowperm/colperm/rowcolperm — random permutation of the row and/or column INDEX.
#     These are NOT graph automorphisms (only reversal preserves path adjacency), and they do not need
#     to be: the input sequence is never modified, the counterfactual enters purely as node-mean
#     activations z[pi(u)] - z[u]. pi only has to be a bijection on nodes.
#     This matters because parity = (r+c) mod 2 is a FUNCTION of the coordinates, so no automorphism can
#     vary coordinates while holding parity fixed. A bijection can: with PARITY_SAFE=1 the permutation is
#     restricted to act within same-parity index classes (r -> r+-2), moving coordinates a long way while
#     leaving every node's parity untouched. That is the dissociation D4 structurally could not provide.
PERM = os.environ.get("PERM", "rot90")
# PERM_EVAL: extra perms to evaluate the TRAINED subspace on (transfer). A subspace that encodes
# coordinates should support a remap it was never trained on; one that merely looks up node identity
# should not. This is the discriminator between a coordinate code and a memorised identity table.
PERM_EVAL = [x for x in os.environ.get("PERM_EVAL", "").split(",") if x]
PERM_SEED = int(os.environ.get("PERM_SEED", "0"))
PARITY_SAFE = os.environ.get("PARITY_SAFE", "1") == "1"
# context must scale with node count or big grids are starved (16x16 = 256 nodes)
CTXLO_PER_N = float(os.environ.get("CTXLO_PER_N", "0"))
# WORDPERM: index of a random word->node assignment. The lexical identity of the parity classes changes
# with it, so a direction averaged over several WORDPERMs keeps only the structural (in-context) part.
WORDPERM = int(os.environ.get("WORDPERM", "-1"))
LAZY = float(os.environ.get("LAZY", "0"))   # self-loop prob: kills parity persistence, keeps coords
HEADS_SPEC = os.environ.get("HEADS", "14:26,14:19")       # "L:H,..." ; "L:*" expands to every head at L
LAYER = int(os.environ.get("LAYER", "14"))
GRAPH_TYPE = os.environ.get("GRAPH_TYPE", "grid")         # "grid" | "ring"
# grid: "8x8,2x32" -> [(8,8),(2,32)] (non-square OK).  ring: "16,32" -> [(16,),(32,)]
GRIDS = [tuple(int(x) for x in g.split("x")) for g in os.environ.get("GRIDS", "4x4,8x8").split(",")]
_defr = {"block": "1,4,8,16", "concat": "4,8,16,32,48", "resid": "1,4,8,16,32,64"}[MODE]
RDIMS = [int(x) for x in os.environ.get("RDIMS", _defr).split(",")]
NWALKS = int(os.environ.get("NWALKS", "10")); SPN = int(os.environ.get("SAMPLES_PER_NODE", "80"))
CTXLO = int(os.environ.get("CTXLO", "1000")); WLEN_CAP = int(os.environ.get("WLEN_CAP", "1600"))
HOLDOUT = int(os.environ.get("HOLDOUT", "3")); STEPS = int(os.environ.get("STEPS", "45"))
BATCH = int(os.environ.get("BATCH", "3")); LR = float(os.environ.get("LR", "0.02"))
SEED = int(os.environ.get("SEED", "0")); SAVE_R = os.environ.get("SAVE_R", "0") == "1"
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")
CUR = {"ctxlo": CTXLO}                                    # per-grid context floor (set in run_grid)


def _perm_axis(k, rng, parity_safe):
    """Random permutation of 0..k-1. If parity_safe, only permute within same-parity index classes, so
    every index moves by an even amount and node parity (r+c) mod 2 is preserved exactly."""
    for _ in range(200):
        s = np.arange(k)
        if parity_safe:
            for cls in (0, 1):
                ix = np.where(s % 2 == cls)[0]
                if len(ix) > 1: s[ix] = rng.permutation(s[ix])
        else:
            s = rng.permutation(s)
        if not np.array_equal(s, np.arange(k)): return s
    raise ValueError(f"no non-identity {'parity-safe ' if parity_safe else ''}permutation exists for k={k}")


def build_perm(coords, kk, col, name=None):
    kr, kc = (kk, kk) if isinstance(kk, int) else kk
    PERM = name or globals()["PERM"]
    pseed = PERM_SEED
    if "#" in PERM:                    # "rowcolperm#2" = an INDEPENDENT random draw of the same family,
        PERM, sfx = PERM.split("#")    # so transfer can be tested against a remap never trained on
        pseed = PERM_SEED + int(sfx)
    if GRAPH_TYPE == "ring":
        # Ring position is a SINGLE cyclic coordinate, so "compact code" has a sharp prediction: a 2-d
        # (cos t, sin t) embedding suffices, and every rotation is a rotation WITHIN that same 2-d plane.
        #   cyc<k>   rotate by k steps — a genuine automorphism that shifts position by k
        #   refl     reflection i -> -i — the other dihedral generator
        #   swapadj  ABCDE -> ABCED: transpose two ADJACENT nodes. Minimal local change; only 2 nodes move
        #   swapfar  transpose two ANTIPODAL nodes — same 2-node budget, maximal position change
        nn = len(coords)
        if PERM == "randbij":
            return np.random.default_rng(4242 + 7919 * pseed + 17 * nn).permutation(nn)
        if PERM.startswith("cyc"):
            kk_ = int(PERM[3:] or 1)
            return np.array([(i + kk_) % nn for i in range(nn)], int)
        if PERM == "refl":
            return np.array([(-i) % nn for i in range(nn)], int)
        if PERM == "swappairs":
            # ABCDEF -> BADCFE: apply the adjacent transposition EVERYWHERE, (0 1)(2 3)(4 5)...
            # Every node moves by exactly 1 — the SAME displacement as cyc1 — but alternating rather
            # than coherent. A 2-d circular position code can express cyc1 (it is a rotation in that
            # plane) but NOT this, which is a high-frequency perturbation. Displacement-matched contrast.
            pi = np.arange(nn)
            for i in range(0, nn - 1, 2): pi[i], pi[i + 1] = i + 1, i
            return pi
        if PERM.startswith("swapadj") or PERM.startswith("swapfar"):
            a = int(PERM[7:] or 0)
            b = (a + 1) % nn if PERM.startswith("swapadj") else (a + nn // 2) % nn
            pi = np.arange(nn); pi[a], pi[b] = b, a
            return pi
        raise ValueError(f"unknown ring perm {PERM}")
    idx = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
    if PERM == "randbij":
        # control: a uniformly random bijection with NO row/col structure. A subspace that encodes
        # coordinates should transfer worse to this than to a row/col remap; one that is really a
        # node-identity lookup should transfer to it just as well.
        return np.random.default_rng(4242 + 7919 * pseed + 17 * (100 * kr + kc)).permutation(len(coords))
    if PERM in ("rowperm", "colperm", "rowcolperm"):
        rr = np.random.default_rng(1234 + 7919 * pseed + 17 * (100 * kr + kc))
        sr = _perm_axis(kr, rr, PARITY_SAFE) if PERM in ("rowperm", "rowcolperm") else np.arange(kr)
        sc = _perm_axis(kc, rr, PARITY_SAFE) if PERM in ("colperm", "rowcolperm") else np.arange(kc)
        pi = np.array([idx[(int(sr[r]), int(sc[c]))] for (r, c) in coords], int)
        assert len(set(pi.tolist())) == len(pi), "perm must be a bijection"
        if PARITY_SAFE:
            assert all(col[pi[i]] == col[i] for i in range(len(pi))), "PARITY_SAFE perm must preserve parity"
        return pi
    if PERM == "rot90":
        assert kr == kc, "rot90 needs a square grid"
        pi = np.array([idx[(int(c), int(kr - 1 - r))] for (r, c) in coords], int)
        assert all(col[pi[i]] == -col[i] for i in range(len(pi))), "rot90 must invert parity (even k only)"
    elif PERM == "rot180":
        pi = np.array([idx[(int(kr - 1 - r), int(kc - 1 - c))] for (r, c) in coords], int)
        assert all(col[pi[i]] == col[i] for i in range(len(pi))), "rot180 must preserve parity"
    elif PERM == "transpose":
        assert kr == kc, "transpose needs a square grid"
        pi = np.array([idx[(int(c), int(r))] for (r, c) in coords], int)
        assert all(col[pi[i]] == col[i] for i in range(len(pi))), "transpose must preserve parity"
    elif PERM == "swap2":
        a, b = idx[(0, 0)], idx[(kr - 1, kc - 1)]         # same colour iff kr+kc even
        assert col[a] == col[b]
        pi = np.arange(len(coords)); pi[a], pi[b] = b, a
    else:
        raise ValueError(PERM)
    return pi


@torch.no_grad()
def capture(model, tok, blocks, cm, graph, cfg, dev, sites, n, n_src):
    """sites: list of dicts {kind:'head', layer, csl, hd} or {kind:'resid', layer, dim}. Returns wdata +
    per-site znode. Only the first n_src walks feed the node means."""
    caps = {}
    hooks = []
    layers_h = sorted({s["layer"] for s in sites if s["kind"] == "head"})
    for L in layers_h:
        proj = attn_proj(blocks[L], cm)[0]
        def mk(L):
            def hh(_m, args): caps[("h", L)] = args[0].detach()
            return hh
        hooks.append(proj.register_forward_pre_hook(mk(L)))
    for s in sites:
        if s["kind"] == "resid":
            def rh(_m, _i, out): caps[("r", s["layer"])] = (out[0] if isinstance(out, tuple) else out).detach()
            hooks.append(blocks[s["layer"]].register_forward_hook(rh))
    walks = G.generate_walks(graph, cfg); wdata = []
    zsum = [np.zeros((n, s["dim"])) for s in sites]; zcnt = np.zeros(n)
    for wi, wk in enumerate(walks):
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); caps.clear()
        model(input_ids=ids); seqlen = ids.shape[1]; ntok = []; readpos = []; readnode = []
        for st in range(len(nodes)):
            t = spans[st][-1]; nd = nodes[st]; ntok.append((t, nd))
            if cl[st] >= CUR["ctxlo"] and st < len(nodes) - 1:
                readpos.append(t); readnode.append(nd)
                if wi < n_src:
                    for si, s in enumerate(sites):
                        src = caps[("h", s["layer"])][0, t, s["csl"]] if s["kind"] == "head" else caps[("r", s["layer"])][0, t]
                        zsum[si][nd] += src.float().cpu().numpy()
                    zcnt[nd] += 1
        wdata.append({"ids": ids, "ntok": ntok, "readpos": readpos, "readnode": readnode, "seqlen": seqlen})
    for h in hooks: h.remove()
    cn = np.maximum(zcnt, 1)[:, None]
    return wdata, [zs / cn for zs in zsum]


def run_grid(model, tok, blocks, cm, dev, kk, rng, sites):
    if GRAPH_TYPE == "ring":
        n = kk[0] if isinstance(kk, tuple) else kk
        kr, kc = n, 1
    else:
        kr, kc = (kk, kk) if isinstance(kk, int) else kk
        n = kr * kc
    k = kr                                                # legacy alias (square-grid call sites)
    gl = f"ring{n}" if GRAPH_TYPE == "ring" else f"{kr}x{kc}"
    CUR["ctxlo"] = max(CTXLO, int(CTXLO_PER_N * n))
    wl = min(WLEN_CAP, CUR["ctxlo"] + int(np.ceil(n * SPN / NWALKS)))
    if wl - CUR["ctxlo"] < 40:
        print(f"[{gl}] SKIP: WLEN_CAP={WLEN_CAP} leaves only {wl - CUR['ctxlo']} readouts "
              f"after ctxlo={CUR['ctxlo']}; raise WLEN_CAP", flush=True)
        return None, {}
    if GRAPH_TYPE == "ring":
        cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=n,
                      n_walks=NWALKS, walk_length=wl, device=dev)
        graph = G.build_graph(cfg); coords = np.array(graph.coords, float)
        col = two_colour(graph) if n % 2 == 0 else np.ones(n, int)   # odd ring is not bipartite
    else:
        cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=kr, grid_cols=kc, n_walks=NWALKS, walk_length=wl, device=dev)
        graph = G.build_graph(cfg); col = two_colour(graph); coords = np.array(graph.coords, int)
    if WORDPERM >= 0:
        _rw = np.random.default_rng(9000 + WORDPERM)
        _sel = _rw.permutation(len(_config.WORDS))[:n]
        graph = replace(graph, words=[_config.WORDS[i] for i in _sel])
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    pos_idx = torch.tensor(np.where(col > 0)[0], device=dev); neg_idx = torch.tensor(np.where(col < 0)[0], device=dev)
    # one entry per perm: the bijection, plus neighbourhood target/source sets for the coordinate margin
    nbrset = [set(graph.adjacency[u]) for u in range(n)]
    PNAMES = [PERM] + [p for p in PERM_EVAL if p != PERM]
    PIS, TGTSRC = {}, {}
    for pn in PNAMES:
        p_i = build_perm(coords, (kr, kc), col, pn); ts = []
        for u in range(n):
            if p_i[u] == u: ts.append(None); continue
            # Drop the current node u and the target pi(u) from BOTH sides. On lazy walks the model puts
            # ~p of its mass on the current node, and when pi(u) is adjacent to u that self-mass lands
            # inside nbrs(pi(u)) -> the margin is won with no patching at all (cyc1 baseline was +0.30,
            # flip 0.67, before this fix). Symmetrically, after a correct patch the model should sit at
            # pi(u), so pi(u) in S would penalise correct behaviour. Neither is evidence about neighbour
            # structure. Only matters for SHORT-displacement perms (cyc1/swappairs/swapadj); far perms
            # such as cyc2 or grid rowcolperm are unaffected (their T/S never contain u or pi(u)).
            excl = {int(u), int(p_i[u])}
            T = sorted(nbrset[p_i[u]] - nbrset[u] - excl); S = sorted(nbrset[u] - nbrset[p_i[u]] - excl)
            ts.append(None if (not T or not S) else
                      (torch.tensor(T, device=dev), torch.tensor(S, device=dev)))
        PIS[pn], TGTSRC[pn] = p_i, ts
        nmov = int((p_i != np.arange(n)).sum()); nusable = sum(x is not None for x in ts)
        print(f"[{gl}] perm {pn}: moves {nmov}/{n} nodes, {nusable} usable readout nodes, "
              f"parity preserved={all(col[p_i[i]] == col[i] for i in range(n))}", flush=True)
    pi = PIS[PERM]
    n_train = NWALKS - HOLDOUT
    if LAZY > 0:
        _lr = np.random.default_rng(SEED)
        def _lazy_walks(_g, _cfg):
            from graph import Walk as _W
            out = []
            for w in range(_cfg.n_walks):
                cur = w % _g.n_nodes; nodes = [cur]
                for _ in range(_cfg.walk_length - 1):
                    if _lr.random() >= LAZY: cur = int(_lr.choice(_g.neighbors(cur)))
                    nodes.append(cur)
                out.append(_W(walk_id=w, nodes=nodes, words=[_g.words[x] for x in nodes]))
            return out
        _orig = G.generate_walks; G.generate_walks = _lazy_walks
        wdata, znodes = capture(model, tok, blocks, cm, graph, cfg, dev, sites, n, n_train)
        G.generate_walks = _orig
    else:
        wdata, znodes = capture(model, tok, blocks, cm, graph, cfg, dev, sites, n, n_train)
    zts = [torch.tensor(z, dtype=torch.float32, device=dev) for z in znodes]
    tgt_sign = col.copy()
    for w in wdata:                                                   # per-perm, per-site delta [seqlen, dim]
        w["delta"] = {}
        for pn in PNAMES:
            p_i = PIS[pn]; Ds = []
            for zt in zts:
                D = torch.zeros(w["seqlen"], zt.shape[1], device=dev)
                for t, nd in w["ntok"]: D[t] = zt[p_i[nd]] - zt[nd]
                Ds.append(D)
            w["delta"][pn] = Ds
        w["rp_t"] = torch.tensor(w["readpos"], device=dev, dtype=torch.long)
        w["tgt_t"] = torch.tensor([tgt_sign[nd] for nd in w["readnode"]], device=dev)

    # ---- patch hooks: per unique head-layer one o_proj pre-hook; resid gets a block forward hook ----
    state = {"w": None, "Rr": None, "pn": PERM}                       # Rr: list per site (block) | single (concat/resid)
    def site_patch(x_slice, si):
        D = state["w"]["delta"][state["pn"]][si]
        if MODE == "block": Rr = state["Rr"][si]; return (D @ Rr.t()) @ Rr
        if MODE == "resid": Rr = state["Rr"]; return (D @ Rr.t()) @ Rr
        return None
    hooks = []
    if MODE in ("block",):
        for L in sorted({s["layer"] for s in sites}):
            proj = attn_proj(blocks[L], cm)[0]
            def mk(L):
                def ph(_m, args):
                    if state["Rr"] is None: return
                    x = args[0].clone()
                    for si, s in enumerate(sites):
                        if s["layer"] != L: continue
                        x[0, :, s["csl"]] = x[0, :, s["csl"]] + site_patch(None, si).to(x.dtype)
                    return (x,) + tuple(args[1:])
                return ph
            hooks.append(proj.register_forward_pre_hook(mk(L)))
    elif MODE == "concat":
        offs = np.cumsum([0] + [s["dim"] for s in sites])
        def cat_patch():                                              # [seqlen, sum dims] projected
            Dcat = torch.cat(state["w"]["delta"][state["pn"]], dim=1); Rr = state["Rr"]
            return (Dcat @ Rr.t()) @ Rr
        for L in sorted({s["layer"] for s in sites}):
            proj = attn_proj(blocks[L], cm)[0]
            def mk(L):
                def ph(_m, args):
                    if state["Rr"] is None: return
                    x = args[0].clone(); P = state["catP"]
                    for si, s in enumerate(sites):
                        if s["layer"] != L: continue
                        x[0, :, s["csl"]] = x[0, :, s["csl"]] + P[:, offs[si]:offs[si + 1]].to(x.dtype)
                    return (x,) + tuple(args[1:])
                return ph
            hooks.append(proj.register_forward_pre_hook(mk(L)))
        state["cat_patch"] = cat_patch
    else:                                                             # resid
        def rh(_m, _i, out):
            if state["Rr"] is None: return out
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone(); h[0] = h[0] + site_patch(None, 0).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hooks.append(blocks[sites[0]["layer"]].register_forward_hook(rh))

    def eval_walk(w, Rr, pn=None):
        state["w"] = w; state["Rr"] = Rr; state["pn"] = pn = pn or PERM
        if MODE == "concat" and Rr is not None: state["catP"] = state["cat_patch"]()
        logits = model(input_ids=w["ids"]).logits[0][w["rp_t"]][:, cand_t].float()
        state["Rr"] = None
        lsm = torch.log_softmax(logits, 1)
        if pn == "rot90":                                         # colour-class margin (parity)
            tp = w["tgt_t"] > 0
            same = torch.where(tp[:, None], lsm[:, pos_idx], lsm[:, neg_idx]); opp = torch.where(tp[:, None], lsm[:, neg_idx], lsm[:, pos_idx])
            sm = torch.logsumexp(same, 1); om = torch.logsumexp(opp, 1)
            return -sm.mean(), (sm - om).mean().item(), ((sm - om) > 0).float().mean().item()
        # neighbourhood margin (coordinates): want mass to move onto nbrs(pi(X)) and off nbrs(X)
        loss = []; marg = []
        for j, nd in enumerate(w["readnode"]):
            ts = TGTSRC[pn][nd]
            if ts is None: continue
            T, S = ts
            mt = torch.logsumexp(lsm[j, T], 0); ms = torch.logsumexp(lsm[j, S], 0)
            loss.append(-mt); marg.append((mt - ms).item())
        if not marg: return torch.tensor(0.0, device=dev, requires_grad=Rr is not None), 0.0, 0.0
        return torch.stack(loss).mean(), float(np.mean(marg)), float(np.mean([m > 0 for m in marg]))

    train_w, eval_w = wdata[:n_train], wdata[n_train:]
    def metrics(Rr, ws, pn=None):
        ms = [eval_walk(w, Rr, pn)[1:] for w in ws]
        return float(np.mean([m[0] for m in ms])), float(np.mean([m[1] for m in ms]))
    total_dim = sum(s["dim"] for s in sites) if MODE == "concat" else (sites[0]["dim"] if MODE == "resid" else None)
    base = {pn: metrics(None, eval_w, pn) for pn in PNAMES}
    base_m, base_f = base[PERM]
    curve = {}; flip = {}; saved = {}; xfer = {pn: {} for pn in PNAMES}
    for r in RDIMS:
        if MODE == "block":
            lins = [nn.Linear(s["dim"], s["dim"], bias=False).to(dev) for s in sites]
            for ln in lins: nn.utils.parametrizations.orthogonal(ln)
            params = [p for ln in lins for p in ln.parameters()]
            getR = lambda: [ln.weight[:r] for ln in lins]
        else:
            dim = total_dim
            ln = nn.Linear(dim, max(r, 1), bias=False).to(dev)
            nn.utils.parametrizations.orthogonal(ln)
            params = list(ln.parameters())
            getR = lambda: ln.weight[:r]
        opt = torch.optim.Adam(params, lr=LR)
        for _ in range(STEPS):
            opt.zero_grad()
            bs = [train_w[i] for i in rng.choice(len(train_w), min(BATCH, len(train_w)), replace=False)]
            loss = sum(eval_walk(w, getR(), PERM)[0] for w in bs) / len(bs); loss.backward(); opt.step()
        with torch.no_grad():
            Rr = [x.detach() for x in getR()] if MODE == "block" else getR().detach()
            curve[r], flip[r] = metrics(Rr, eval_w, PERM)
            for pn in PNAMES:                                   # transfer: same R, unseen remap
                xfer[pn][r] = metrics(Rr, eval_w, pn)
            if SAVE_R:
                saved[r] = [x.cpu().numpy() for x in Rr] if MODE == "block" else Rr.cpu().numpy()
    for h in hooks: h.remove()
    xs = sorted(curve)
    print(f"[{gl}] n={n} wl={wl} ctxlo={CUR['ctxlo']} base={base_m:+.2f}(flip {base_f:.2f})  "
          + "  ".join(f"r{r}={curve[r]:+.2f}(flip {flip[r]:.2f})" for r in xs), flush=True)
    for pn in PNAMES[1:]:
        print(f"          transfer -> {pn:12} base={base[pn][0]:+.2f}  "
              + "  ".join(f"r{r}={xfer[pn][r][0]:+.2f}(flip {xfer[pn][r][1]:.2f})" for r in xs), flush=True)
    return {"k": kr, "kr": kr, "kc": kc, "n": n, "wl": wl, "ctxlo": CUR["ctxlo"],
            "baseline": round(base_m, 3), "baseline_flip": round(base_f, 3),
            "margins": {str(r): round(curve[r], 3) for r in xs},
            "flip_rate": {str(r): round(flip[r], 3) for r in xs},
            "transfer": {pn: {"baseline": round(base[pn][0], 3),
                              "margins": {str(r): round(xfer[pn][r][0], 3) for r in xs},
                              "flip_rate": {str(r): round(xfer[pn][r][1], 3) for r in xs}}
                         for pn in PNAMES}}, saved


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    for p in model.parameters(): p.requires_grad_(False)
    cm = model.config; blocks = M._decoder_blocks(model)
    if MODE == "resid":
        sites = [{"kind": "resid", "layer": LAYER, "dim": cm.hidden_size, "csl": None}]
        name = f"L{LAYER}"
    else:
        heads = []
        for h in HEADS_SPEC.split(","):
            L, H = h.split(":")
            heads += [(int(L), x) for x in range(cm.num_attention_heads)] if H == "*" else [(int(L), int(H))]
        _, hd = attn_proj(blocks[heads[0][0]], cm)
        sites = [{"kind": "head", "layer": L, "csl": slice(H * hd, (H + 1) * hd), "dim": hd} for L, H in heads]
        star = sorted({L for L, H in heads if f"{L}:*" in HEADS_SPEC.split(",")})
        name = ("-".join(f"L{L}Hall" for L in star) if star and len(heads) == len(star) * cm.num_attention_heads
                else "-".join(f"L{L}H{H}" for L, H in heads))
        print(f"[{tag}] {len(heads)} head sites -> {name}", flush=True)
    need = max(int(np.prod(t)) for t in GRIDS)
    if need > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, need)
    out = {"model": tag, "mode": MODE, "sites": name, "rdims": RDIMS, "ctxlo": CTXLO, "holdout": HOLDOUT, "grids": {}}
    Rsave = {}
    for gspec in GRIDS:
        res, saved = run_grid(model, tok, blocks, cm, dev, gspec, rng, sites)
        gname = f"ring{gspec[0]}" if GRAPH_TYPE == "ring" else f"{gspec[0]}x{gspec[1]}"
        if res is None: continue
        out["grids"][gname] = res
        for r, w in saved.items(): Rsave[f"{gname}_r{r}"] = np.concatenate([x.reshape(1, -1) for x in w]) if isinstance(w, list) else w
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    out["graph_type"] = GRAPH_TYPE; out["perm"] = PERM; out["perm_eval"] = PERM_EVAL; out["parity_safe"] = PARITY_SAFE
    out["perm_seed"] = PERM_SEED; out["lazy"] = LAZY
    ptag = ("_ring" if GRAPH_TYPE == "ring" else "") + ("" if PERM == "rot90" else f"_{PERM}") + (f"_wp{WORDPERM}" if WORDPERM >= 0 else "") + ("_lazy" if LAZY > 0 else "")
    if PERM in ("rowperm", "colperm", "rowcolperm"):
        ptag += ("_ps" if PARITY_SAFE else "_free") + (f"_s{PERM_SEED}" if PERM_SEED else "")
    p = f"{OUTDIR}/das_multihead_{MODE}{ptag}_{name}{OUTTAG}_{tag}.json"
    if Rsave: np.savez_compressed(p.replace(".json", ".npz"), **Rsave)
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
