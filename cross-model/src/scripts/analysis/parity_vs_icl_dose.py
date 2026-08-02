"""Is parity damaged BEYOND what the in-context-learning collapse already explains?

Ablating the 21 heads destroys ICL (repeated-random-token induction 0.987 -> 0.049) and also destroys the
grid task. Parity necessarily falls with it, so "does ablating the ICL heads hurt parity" is trivially
yes and tells us nothing. The real question is whether parity falls FASTER than ICL damage predicts.

Design: a DOSE-RESPONSE curve. Ablate head sets of many sizes and kinds, and for each one measure both
    icl        induction accuracy on repeated random tokens (no graph, no parity, no semantics)
    parity     parity margin + parity validity on the grid walk
    nbr        neighbour validity on the grid walk
then plot parity against icl. Every set traces out a curve; if the parity-derived set lies ON the curve
swept by RANDOM sets, its parity damage is fully accounted for by ICL damage and there is nothing
parity-specific. If it lies BELOW, there is parity-specific damage on top.

Random 21-head sets barely dent ICL (0.9865), so the random arm sweeps k = 21..300 to trace the whole
curve and give the parity-21 point something to be compared against at matched ICL.

Sets: nested prefixes of parity-21 (ordered by leave-one-out importance), nested prefixes of the Olsson
top-20 induction heads, and random sets at several sizes (NREP draws each).

Parity convention: on a bipartite graph the next node always has the OPPOSITE colour, so
parity_margin = logsumexp(opposite-colour words) - logsumexp(same-colour words), higher is better.

Env: GEN_MODEL(Llama) K(4) NWALKS(3) WLEN(1200) CTXLO(800) SEQLEN(60) NSEQ(8)
     RANDK("21,50,100,200,300") NREP(3) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/parity_vs_icl_dose<OUTTAG>_<model>.json
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
K = int(os.environ.get("K", "4"))
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800"))
SEQLEN = int(os.environ.get("SEQLEN", "60")); NSEQ = int(os.environ.get("NSEQ", "8"))
RANDK = [int(x) for x in os.environ.get("RANDK", "21,50,100,200,300").split(",")]
NREP = int(os.environ.get("NREP", "3")); SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")

# parity-21 ordered by leave-one-out importance on the induction probe (most important first)
P21_ORDER = ["L4H16", "L14H26", "L8H11", "L10H2", "L14H19", "L16H1", "L14H17", "L1H20", "L21H10",
             "L13H18", "L7H25", "L16H20", "L1H21", "L4H12", "L9H11", "L2H22", "L15H30", "L3H17",
             "L2H26", "L25H7", "L21H2"]
OLSSON20 = ["L15H30", "L8H1", "L16H20", "L15H1", "L2H22", "L10H14", "L5H8", "L20H14", "L24H27",
            "L5H11", "L20H1", "L19H3", "L26H15", "L13H6", "L27H6", "L16H1", "L26H13", "L27H5",
            "L22H14", "L27H7"]


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    hd = getattr(cm, "head_dim", None) or cm.hidden_size // nH
    rng = np.random.default_rng(SEED)
    Vsz = model.get_input_embeddings().weight.shape[0]

    st = {"heads": None}
    hooks = []
    for l in range(nL):
        def mk(l):
            def ph(_m, args):
                hs = st["heads"]
                if not hs: return
                sel = [h for (ll, h) in hs if ll == l]
                if not sel: return
                x = args[0].clone()
                for h in sel:
                    sl = slice(h * hd, (h + 1) * hd)
                    x[0, :, sl] = x[0, :, sl].mean(0, keepdim=True)
                return (x,) + tuple(args[1:])
            return ph
        hooks.append(attn_proj(blocks[l], cm)[0].register_forward_pre_hook(mk(l)))

    ind_seqs = []
    for _ in range(NSEQ):
        t = rng.integers(1000, min(Vsz, 30000), size=SEQLEN).tolist()
        ind_seqs.append((torch.tensor([[tok.bos_token_id or t[0]] + t + t], device=dev),
                         torch.arange(1 + SEQLEN, 1 + 2 * SEQLEN - 1, device=dev),
                         torch.tensor(t[1:], device=dev)))

    n = K * K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=K, grid_cols=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph); words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)
    pos_i = torch.tensor(np.where(col > 0)[0], device=dev)
    neg_i = torch.tensor(np.where(col < 0)[0], device=dev)
    data = []
    for wk in G.generate_walks(graph, cfg):
        steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        if steps:
            data.append((torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev),
                         torch.tensor([s + 1 for s in steps], device=dev),
                         [wk.nodes[s] for s in steps]))

    def probes():
        ok = m = 0
        for ids, q, gold in ind_seqs:
            ok += int((model(input_ids=ids).logits[0][q].argmax(-1) == gold).sum()); m += len(gold)
        icl = ok / m
        nb = nbok = pm = pv = tot = 0
        for ids, rp, nds in data:
            lsm = torch.log_softmax(model(input_ids=ids).logits[0][rp][:, cand_t].float(), 1)
            top = lsm.argmax(1).tolist()
            cur = torch.tensor([col[u] for u in nds], device=dev)
            # next node has OPPOSITE colour on a bipartite graph
            opp = torch.where((cur > 0)[:, None], lsm[:, neg_i], lsm[:, pos_i])
            sam = torch.where((cur > 0)[:, None], lsm[:, pos_i], lsm[:, neg_i])
            pm += float((torch.logsumexp(opp, 1) - torch.logsumexp(sam, 1)).sum())
            for i, (t_, u) in enumerate(zip(top, nds)):
                nbok += int(t_ in graph.adjacency[u]); pv += int(col[t_] == -col[u]); tot += 1
        return {"icl": round(icl, 4), "nbr": round(nbok / tot, 4),
                "parity_valid": round(pv / tot, 4), "parity_margin": round(pm / tot, 4)}

    def parse(x): return [(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x]
    rows = []
    base = probes(); base["set"] = "baseline"; base["k"] = 0; rows.append(base)
    print(f"[{tag}] baseline  icl={base['icl']:.4f}  nbr={base['nbr']:.4f}  "
          f"par_valid={base['parity_valid']:.4f}  par_margin={base['parity_margin']:+.4f}", flush=True)
    print(f"\n{'set':<16} {'k':>4} {'icl':>8} {'nbr':>8} {'par_valid':>10} {'par_margin':>11}")

    def run(nm, heads):
        st["heads"] = parse(heads); r = probes(); st["heads"] = None
        r["set"] = nm; r["k"] = len(heads); rows.append(r)
        print(f"{nm:<16} {len(heads):4} {r['icl']:8.4f} {r['nbr']:8.4f} {r['parity_valid']:10.4f} "
              f"{r['parity_margin']:+11.4f}", flush=True)

    for k_ in (1, 2, 3, 5, 8, 13, 21): run("parity21", P21_ORDER[:k_])
    for k_ in (1, 2, 3, 5, 8, 13, 20): run("olsson20", OLSSON20[:k_])
    allh = [f"L{l}H{h}" for l in range(nL) for h in range(nH)]
    for k_ in RANDK:
        for r_ in range(NREP):
            run(f"random{k_}", [allh[j] for j in rng.choice(len(allh), k_, replace=False)])
    for h in hooks: h.remove()

    # ---- is parity-21 an outlier off the random curve at matched ICL? ----
    R = [r for r in rows if r["set"].startswith("random")]
    P = [r for r in rows if r["set"] == "parity21"]
    print("\nparity vs ICL, compared at matched ICL (interpolating the RANDOM curve):")
    print(f"{'set':<12} {'k':>4} {'icl':>8} {'par_margin':>11} {'rand@icl':>10} {'excess':>9}")
    xs = np.array([r["icl"] for r in R]); ys = np.array([r["parity_margin"] for r in R])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    for r in P + [x for x in rows if x["set"] == "olsson20"]:
        pred = float(np.interp(r["icl"], xs, ys))
        print(f"{r['set']:<12} {r['k']:4} {r['icl']:8.4f} {r['parity_margin']:+11.4f} "
              f"{pred:+10.4f} {r['parity_margin'] - pred:+9.4f}")
        r["rand_at_matched_icl"] = round(pred, 4)
        r["parity_excess_vs_icl"] = round(r["parity_margin"] - pred, 4)
    p_ = f"{OUTDIR}/parity_vs_icl_dose{OUTTAG}_{tag}.json"
    json.dump({"model": tag, "baseline": base, "rows": rows}, open(p_, "w"), indent=2)
    print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
