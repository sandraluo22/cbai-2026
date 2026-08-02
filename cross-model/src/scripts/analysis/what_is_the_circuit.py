"""What is the "parity-21" circuit actually doing? It is named after the task it was DERIVED from
(interchange patching on the grid parity variable), and that name may be wrong.

Facts to reconcile: ablating it destroys the toy in-context ring walk (1.000 -> 0.213) while random
20-head sets cost 0.0001; but it is inert on the Engels day/month task; and it overlaps the top-20
induction heads at only 4/21. The toy task needs IN-CONTEXT LEARNING (arbitrary word->node mapping,
learned from the window) but its metric — neighbour validity — has nothing to do with parity.

So: is this a parity circuit, a graph circuit, or a generic in-context-learning circuit?
Three probes, no graph structure shared between them:

  induction   REPEATED RANDOM TOKENS, no graph, no parity, no semantics: [BOS] t_1..t_L t_1..t_L, and we
              score next-token accuracy inside the second copy. Pure prefix-matching. If ablation kills
              THIS, the circuit is doing in-context copying and "parity" is a misnomer.
  ring        toy ring walk, neighbour validity (parity is live on a non-lazy walk over an even ring)
  ring_lazy   same walk with self-loops p=0.5, which nulls parity autocorrelation but leaves the graph.
              If ablation still kills it here, the damage is not parity-mediated.

Plus leave-one-out over the 21 heads on whichever probe moves most, to see whether the effect is carried
by a few heads or spread.

Env: GEN_MODEL(Llama) HEADS(the 21) K(16) LAYERSEQ(60) NSEQ(8) NWALKS(4) WLEN(1200) CTXLO(800)
     NRAND(5) RANDMODE(all|layer) LOO(1) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/what_is_the_circuit<OUTTAG>_<model>.json
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
from grid_parity_compare import build_word_pool, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
P21 = ("L21H2,L14H19,L14H17,L10H2,L2H22,L7H25,L8H11,L1H20,L4H16,L13H18,L16H1,L16H20,L14H26,"
       "L15H30,L4H12,L9H11,L1H21,L21H10,L3H17,L2H26,L25H7")
HEADS = [h for h in os.environ.get("HEADS", P21).split(",") if h]
K = int(os.environ.get("K", "16"))
SEQLEN = int(os.environ.get("SEQLEN", "60")); NSEQ = int(os.environ.get("NSEQ", "8"))
NWALKS = int(os.environ.get("NWALKS", "4")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); NRAND = int(os.environ.get("NRAND", "5"))
LOO = os.environ.get("LOO", "1") == "1"; SEED = int(os.environ.get("SEED", "0"))
RANDMODE = os.environ.get("RANDMODE", "all")      # "all" = any of the 1024 heads | "layer" = depth-matched
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

    # ---- probe 1: pure induction on repeated random tokens (no graph, no parity, no semantics) ----
    ind_seqs = []
    for _ in range(NSEQ):
        t = rng.integers(1000, min(Vsz, 30000), size=SEQLEN).tolist()
        ids = torch.tensor([[tok.bos_token_id or t[0]] + t + t], device=dev)
        q = torch.arange(1 + SEQLEN, 1 + 2 * SEQLEN - 1, device=dev)   # positions in the 2nd copy
        gold = torch.tensor(t[1:], device=dev)                          # what should follow each
        ind_seqs.append((ids, q, gold))

    def induction_acc():
        ok = m = 0
        for ids, q, gold in ind_seqs:
            lg = model(input_ids=ids).logits[0]
            pred = lg[q].argmax(-1)
            ok += int((pred == gold).sum()); m += len(gold)
        return ok / m

    # ---- probes 2/3: toy ring walk, non-lazy and lazy ----
    n = K
    if n > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, n)
    cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=K,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); words = list(graph.words)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)

    def mkdata(lazy):
        if lazy <= 0:
            wks = G.generate_walks(graph, cfg)
        else:
            lr = np.random.default_rng(SEED); wks = []
            for w in range(NWALKS):
                cur = w % n; nodes = [cur]
                for _ in range(WLEN - 1):
                    if lr.random() >= lazy: cur = int(lr.choice(graph.neighbors(cur)))
                    nodes.append(cur)
                wks.append(Walk(walk_id=w, nodes=nodes, words=[words[x] for x in nodes]))
        out = []
        for wk in wks:
            steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
            if steps:
                out.append((torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev),
                            torch.tensor([s + 1 for s in steps], device=dev),
                            [wk.nodes[s] for s in steps]))
        return out
    d_plain, d_lazy = mkdata(0.0), mkdata(0.5)

    def ring_acc(data, lazy):
        ok = m = 0
        for ids, rp, nds in data:
            top = model(input_ids=ids).logits[0][rp][:, cand_t].float().argmax(1).tolist()
            for t_, u in zip(top, nds):
                ok += int(t_ in graph.adjacency[u] or (lazy and t_ == u)); m += 1
        return ok / m

    def probes():
        return {"induction": round(induction_acc(), 4),
                "ring": round(ring_acc(d_plain, False), 4),
                "ring_lazy": round(ring_acc(d_lazy, True), 4)}

    def parse(x): return [(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x]
    base = probes()
    print(f"[{tag}] baseline: " + "  ".join(f"{k}={v:.4f}" for k, v in base.items()), flush=True)
    st["heads"] = parse(HEADS); abl = probes(); st["heads"] = None
    print(f"[{tag}] ablate {len(HEADS)} heads: " + "  ".join(f"{k}={v:.4f}" for k, v in abl.items()),
          flush=True)
    allh = [(l, h) for l in range(nL) for h in range(nH)]
    # RANDMODE=layer draws each control head at the SAME LAYER as the head it replaces. A circuit picked
    # by "top head at every layer" is depth-structured by construction, and heads at different depths have
    # very different generic importance, so an all-1024 random draw is not a matched control for it.
    rnd = []
    for _ in range(NRAND):
        if RANDMODE == "layer":
            st["heads"] = [(l, int(rng.choice([x for x in range(nH) if x != h]))) for l, h in parse(HEADS)]
        else:
            st["heads"] = [allh[j] for j in rng.choice(len(allh), len(HEADS), replace=False)]
        rnd.append(probes()); st["heads"] = None
    rmean = {k: float(np.mean([r[k] for r in rnd])) for k in base}
    rsd = {k: float(np.std([r[k] for r in rnd])) for k in base}
    print(f"[{tag}] random ({RANDMODE}) {len(HEADS)}-head sets: "
          + "  ".join(f"{k}={rmean[k]:.4f}+-{rsd[k]:.4f}" for k in base), flush=True)
    print(f"\n{'probe':<12} {'base':>8} {'ablated':>9} {'random':>9} {'excess':>9}")
    for k in base:
        print(f"{k:<12} {base[k]:8.4f} {abl[k]:9.4f} {rmean[k]:9.4f} {abl[k]-rmean[k]:+9.4f}")

    res = {"model": tag, "heads": HEADS, "randmode": RANDMODE, "nrand": NRAND,
           "baseline": base, "ablated": abl,
           "random_mean": {k: round(v, 4) for k, v in rmean.items()},
           "random_sd": {k: round(v, 4) for k, v in rsd.items()}}
    if LOO:
        key = max(base, key=lambda k: base[k] - abl[k])
        print(f"\nleave-one-out on '{key}' (higher = that head mattered more):")
        loo = {}
        for h in HEADS:
            st["heads"] = parse([x for x in HEADS if x != h])
            v = probes()[key]; st["heads"] = None
            loo[h] = round(v, 4)
        for h, v in sorted(loo.items(), key=lambda x: -x[1])[:21]:
            print(f"   drop {h:<8} -> {key}={v:.4f}   (full ablation {abl[key]:.4f}, "
                  f"baseline {base[key]:.4f})")
        res["loo_probe"] = key; res["loo"] = loo
    for h in hooks: h.remove()
    p_ = f"{OUTDIR}/what_is_the_circuit{OUTTAG}_{tag}.json"
    json.dump(res, open(p_, "w"), indent=2); print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
