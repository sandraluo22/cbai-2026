"""Is the discovered circuit SUFFICIENT? Mean-ablate every attention head EXCEPT a keep-set and see how
much of the behaviour survives. (MLPs are always left intact — ablating them destroys the model outright,
so this measures the sufficiency of an ATTENTION-head circuit, not of the whole computation.)

Conditions:
  full          nothing ablated
  circuit       keep only KEEP heads, mean-ablate the other ~1000
  random-k      keep a random set of the same size (NRAND draws)  <- the control that makes "sufficient" mean something
  circuit-minus every leave-one-out of the keep set, to get each member's marginal contribution
  none          ablate every head (floor)

Metrics: parity margin (opposite-colour minus same-colour log-mass at readouts), neighbour validity
(top-1 prediction is a graph neighbour), and coordinate margin when PERM=rot180.

Env: GEN_MODEL(Llama) KEEP("L14H26,L14H19,L16H20,...") METRIC(parity|coord) K(4) NWALKS(3) WLEN(1200)
     CTXLO(800) NRAND(8) LAZY(0) LOO(1) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/circuit_sufficiency_<METRIC><OUTTAG>_<model>.json
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
PAR_NPY = os.environ.get("PAR_NPY", "runs/axes/4_circuits/parity/seed_stable_r1_Llama.npy")
PROBE_LAYER = int(os.environ.get("PROBE_LAYER", "14"))

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
KEEP = [h for h in os.environ.get("KEEP", "").split(",") if h]
METRIC = os.environ.get("METRIC", "parity")
K = int(os.environ.get("K", "4")); NWALKS = int(os.environ.get("NWALKS", "3"))
WLEN = int(os.environ.get("WLEN", "1200")); CTXLO = int(os.environ.get("CTXLO", "800"))
NRAND = int(os.environ.get("NRAND", "8")); LAZY = float(os.environ.get("LAZY", "0"))
LOO = os.environ.get("LOO", "1") == "1"; SEED = int(os.environ.get("SEED", "0"))
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

    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph); coords = np.array(graph.coords, int)
    words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    pos_i = torch.tensor(np.where(col > 0)[0], device=dev); neg_i = torch.tensor(np.where(col < 0)[0], device=dev)
    nbr = [set(graph.adjacency[u]) for u in range(n)]
    idx = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
    pi180 = np.array([idx[(int(K - 1 - r), int(K - 1 - c))] for (r, c) in coords], int)
    tgt_src = []
    for u in range(n):
        T = sorted(nbr[pi180[u]] - nbr[u]); S = sorted(nbr[u] - nbr[pi180[u]])
        tgt_src.append(None if (pi180[u] == u or not T or not S) else
                       (torch.tensor(T, device=dev), torch.tensor(S, device=dev)))

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

    keepset = set()
    for h in KEEP:
        l, hh_ = h[1:].split("H"); keepset.add((int(l), int(hh_)))
    print(f"[{tag}] keep-set has {len(keepset)} heads of {nL*nH}", flush=True)

    pv = None
    if os.path.exists(PAR_NPY):
        _v = np.load(PAR_NPY).astype(np.float32); pv = torch.tensor(_v/np.linalg.norm(_v), device=dev)

    active = {"keep": None}
    def patcher(l):
        def hh(_m, args):
            kp = active["keep"]
            if kp is None: return
            x = args[0].clone()
            for h in range(nH):
                if (l, h) in kp: continue
                s = slice(h * hd, (h + 1) * hd)
                x[0, :, s] = x[0, :, s].mean(0, keepdim=True)
            return (x,) + tuple(args[1:])
        return hh
    hooks = [attn_proj(blocks[l], cm)[0].register_forward_pre_hook(patcher(l)) for l in range(nL)]

    data = []
    for wk in walks:
        ids = torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev)
        steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        data.append((ids, [s + 1 for s in steps], [wk.nodes[s] for s in steps],
                     [wk.nodes[s + 1] for s in steps]))

    def evaluate(keep):
        active["keep"] = keep
        pm, cmg, val, lval, kl, coef, cpar = [], [], [], [], [], [], []
        for ids, rp, rn, nxt in data:
            o = model(input_ids=ids, output_hidden_states=pv is not None)
            if pv is not None:
                h = o.hidden_states[PROBE_LAYER + 1][0, torch.tensor(rp, device=dev)].float()
                coef += (h @ pv).cpu().numpy().tolist(); cpar += [float(col[nd]) for nd in rn]
            lg = o.logits[0][torch.tensor(rp, device=dev)][:, cand_t].float()
            lsm = torch.log_softmax(lg, 1)
            for j, nd in enumerate(rn):
                same = lsm[j, pos_i] if col[nd] > 0 else lsm[j, neg_i]
                opp = lsm[j, neg_i] if col[nd] > 0 else lsm[j, pos_i]
                pm.append(float(torch.logsumexp(opp, 0) - torch.logsumexp(same, 0)))
                ts = tgt_src[nd]
                if ts is not None:
                    cmg.append(float(torch.logsumexp(lsm[j, ts[0]], 0) - torch.logsumexp(lsm[j, ts[1]], 0)))
            top = lsm.argmax(1).cpu().numpy()
            val += [bool(int(t) in nbr[nd]) for t, nd in zip(top, rn)]
            # LAZY-AWARE validity: on a lazy walk, staying put is a legal transition, so a top-1 equal to
            # the current node is correct, not a failure. Plain neighbour-validity is meaningless at p>0.
            lval += [bool(int(t) in nbr[nd] or int(t) == nd) for t, nd in zip(top, rn)]
            # KL( true lazy transition distribution || model ), the metric that is well-defined for any p
            pr = lsm.exp().cpu().numpy()
            for j, nd in enumerate(rn):
                q = np.zeros(n); q[nd] = LAZY
                for b in nbr[nd]: q[b] = (1 - LAZY) / len(nbr[nd])
                m_ = pr[j] / max(pr[j].sum(), 1e-9)
                kl.append(float((q[q > 0] * np.log(q[q > 0] / np.maximum(m_[q > 0], 1e-12))).sum()))
        active["keep"] = None
        out = {"parity_margin": round(float(np.mean(pm)), 4),
               "coord_margin": round(float(np.mean(cmg)), 4) if cmg else None,
               "neighbour_validity": round(float(np.mean(val)), 4),
               "lazy_aware_validity": round(float(np.mean(lval)), 4),
               "kl_to_true_transition": round(float(np.mean(kl)), 4)}
        if coef:
            c = np.array(coef); pp = np.array(cpar)
            out["parity_coef_r"] = round(float(np.corrcoef(c, pp)[0, 1]), 4)
            out["parity_coef_sep"] = round(float(c[pp > 0].mean() - c[pp < 0].mean()), 4)
        return out

    res = {}
    res["full"] = evaluate(None)
    print(f"  full        {res['full']}", flush=True)
    res["circuit"] = evaluate(keepset)
    print(f"  circuit     {res['circuit']}", flush=True)
    res["none"] = evaluate(set())
    print(f"  none        {res['none']}", flush=True)
    rnd = []
    allh = [(l, h) for l in range(nL) for h in range(nH)]
    for i in range(NRAND):
        pick = set(map(tuple, np.array(allh)[rng.choice(len(allh), len(keepset), replace=False)].tolist()))
        rnd.append(evaluate(pick))
        print(f"  random-{i}    {rnd[-1]}", flush=True)
    res["random_same_size"] = rnd
    res["random_mean"] = {k: round(float(np.mean([r[k] for r in rnd if r[k] is not None])), 4)
                          for k in ("parity_margin", "coord_margin", "neighbour_validity")}
    if LOO:
        loo = {}
        for h in sorted(keepset):
            nm = f"L{h[0]}H{h[1]}"
            loo[nm] = evaluate(keepset - {h})
            print(f"  minus {nm:8} {loo[nm]}", flush=True)
        res["leave_one_out"] = loo
    for hk in hooks: hk.remove()

    key = "parity_margin" if METRIC == "parity" else "coord_margin"
    f, c, z = res["full"][key], res["circuit"][key], res["none"][key]
    res["fraction_recovered"] = round(float((c - z) / (f - z + 1e-9)), 3) if f != z else None
    res["random_fraction_recovered"] = round(float((res["random_mean"][key] - z) / (f - z + 1e-9)), 3) if f != z else None
    out = {"model": tag, "metric": METRIC, "keep": sorted(f"L{l}H{h}" for l, h in keepset),
           "n_keep": len(keepset), "lazy": LAZY, "ctxlo": CTXLO, "results": res}
    p = f"{OUTDIR}/circuit_sufficiency_{METRIC}{OUTTAG}_{tag}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n{METRIC}: full={f:+.3f}  circuit={c:+.3f}  all-ablated={z:+.3f}")
    print(f"  circuit recovers {res['fraction_recovered']} of the full-vs-floor range "
          f"(random same-size set recovers {res['random_fraction_recovered']})")
    print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
