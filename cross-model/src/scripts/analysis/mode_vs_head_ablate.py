"""Why does MODE ablation hurt the toy ring task while HEAD ablation does not?

Three candidate answers, all tested here on ONE task with ONE metric so the numbers are comparable:

  (a) different tasks — per_mode_ablate ran on the TOY in-context ring walk (neighbour validity), while
      the null head-ablation results were on the ENGELS day/month task (pretrained calendar knowledge).
      Fixed here by running head ablation on the toy task.
  (b) wrong heads — the ablated set was the grid-derived PARITY circuit. The heads that actually write
      the high-frequency modes were never ablated. Fixed here by ablating them by name.
  (c) dilution — mode ablation removes a mode from the WHOLE residual (every head, every MLP, the
      embedding) at once; head ablation removes one head's share. Quantified here as `energy_share`:
      the fraction of each mode's residual energy contributed by a given head set.

Conditions, all on the toy ring walk, metric = neighbour validity (top-1 next-token among the node words
is a graph neighbour):
    baseline / random rank-1 direction
    per-mode ablation (project the mode direction out of the residual at LAYER)
    head-set mean ablation for: top-K high-frequency heads, top-K low-frequency heads, top-K induction
    heads, the 21-head parity circuit, and NRAND random same-size sets

Mode direction: modes are functions on NODES, so mode m's direction in activation space is
d_m = sum_nodes V[node,m] * mean_resid[node], normalised. Projecting d_m out removes that mode's
component from every contributor simultaneously — which is exactly what a head ablation cannot do.

Env: GEN_MODEL(Llama) K(16) GRAPH(ring) LAYER(14) NWALKS(4) WLEN(1200) CTXLO(800) LAZY(0)
     TOPK(20) NRAND(5) MODEFREQ(...head_mode_frequency_ring16_<model>.json) SEED(0) OUTDIR DEVICE OUTTAG
Out: <OUTDIR>/mode_vs_head_ablate<OUTTAG>_<model>.json
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
K = int(os.environ.get("K", "16")); GRAPH = os.environ.get("GRAPH", "ring")
LAYER = int(os.environ.get("LAYER", "14"))
NWALKS = int(os.environ.get("NWALKS", "4")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); LAZY = float(os.environ.get("LAZY", "0"))
TOPK = int(os.environ.get("TOPK", "20")); NRAND = int(os.environ.get("NRAND", "5"))
SEED = int(os.environ.get("SEED", "0"))
P = "runs/axes/4_circuits/parity"
MODEFREQ = os.environ.get("MODEFREQ", f"{P}/head_mode_frequency_ring16_{GEN_MODEL}.json")
PARITY21 = ("L21H2,L14H19,L14H17,L10H2,L2H22,L7H25,L8H11,L1H20,L4H16,L13H18,L16H1,L16H20,L14H26,"
            "L15H30,L4H12,L9H11,L1H21,L21H10,L3H17,L2H26,L25H7")
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", P)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    dm = cm.hidden_size; hd = getattr(cm, "head_dim", None) or dm // nH
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

    A = np.zeros((n, n))
    for u in range(n):
        for v in graph.adjacency[u]: A[u, v] = 1.0
    lam, V = np.linalg.eigh(np.diag(A.sum(1)) - A)

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
    data = []
    for wk in walks:
        ids = torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev)
        steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        if steps:
            data.append((ids, torch.tensor([s + 1 for s in steps], device=dev),
                         [wk.nodes[s] for s in steps]))

    # ---- hooks: head mean-ablation, and residual direction projection at LAYER ----
    st = {"heads": None, "proj": None}
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

    def rh(_m, _i, out):
        d = st["proj"]
        if d is None: return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        f = h[0].float()
        h[0] = (f - torch.outer(f @ d, d)).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hooks.append(blocks[LAYER].register_forward_hook(rh))

    def nbr_validity():
        ok = m = 0
        for ids, rp, nds in data:
            lg = model(input_ids=ids).logits[0][rp][:, cand_t].float()
            top = lg.argmax(1).tolist()
            for t, u in zip(top, nds):
                ok += int(t in graph.adjacency[u]); m += 1
        return ok / m

    base = nbr_validity()
    print(f"[{tag}] toy {GRAPH}{n} walk, LAYER={LAYER}: baseline neighbour validity = {base:.4f}",
          flush=True)

    # ---- per-node mean residual at LAYER -> mode directions, and per-mode residual energy ----
    Hs = torch.zeros(n, dm, device=dev); cnt = torch.zeros(n, device=dev)
    for ids, rp, nds in data:
        o = model(input_ids=ids, output_hidden_states=True)
        Hh = o.hidden_states[LAYER + 1][0, rp].float()
        for i, u in enumerate(nds): Hs[u] += Hh[i]; cnt[u] += 1
    Hn = Hs / cnt.clamp(min=1)[:, None]
    Hn = Hn - Hn.mean(0, keepdim=True)
    Vt = torch.tensor(V.T, dtype=torch.float32, device=dev)
    mode_vec = Vt @ Hn                                          # [n_modes, dm]
    mode_energy = (mode_vec ** 2).sum(1)                        # residual energy per mode

    res = {"model": tag, "graph": GRAPH, "n": n, "layer": LAYER, "lazy": LAZY,
           "baseline_nbr_validity": round(base, 4),
           "lambda": [round(float(x), 4) for x in lam],
           "residual_mode_energy": [round(float(x), 3) for x in mode_energy],
           "modes": {}, "heads": {}}

    print(f"\n--- MODE ablation (project the mode direction out of the residual at L{LAYER}) ---")
    print(f"{'mode':>5} {'lambda':>8} {'nbr_valid':>10} {'drop':>8} {'resid_energy_share':>19}")
    tot_e = float(mode_energy.sum())
    for m_ in range(1, n):
        d = mode_vec[m_] / mode_vec[m_].norm().clamp(min=1e-9)
        st["proj"] = d; v = nbr_validity(); st["proj"] = None
        sh = float(mode_energy[m_]) / tot_e
        res["modes"][str(m_)] = {"lambda": round(float(lam[m_]), 4), "nbr_validity": round(v, 4),
                                 "drop": round(base - v, 4), "energy_share": round(sh, 4)}
        print(f"{m_:5} {lam[m_]:8.3f} {v:10.4f} {base - v:+8.4f} {sh:19.4f}", flush=True)
    rd = torch.tensor(rng.standard_normal(dm), dtype=torch.float32, device=dev)
    st["proj"] = rd / rd.norm(); res["random_dir_nbr"] = round(nbr_validity(), 4); st["proj"] = None
    print(f"random rank-1 direction: {res['random_dir_nbr']:.4f}")

    # ---- head sets ----
    def parse(x): return [(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x]
    sets = {}
    if os.path.exists(MODEFREQ):
        mf = json.load(open(MODEFREQ))["heads"]
        good = [r for r in mf if r["write_norm"] > 0.02]
        sets["top_highfreq"] = [r["head"] for r in sorted(good, key=lambda r: -r["hi_frac"])[:TOPK]]
        sets["top_lowfreq"] = [r["head"] for r in sorted(good, key=lambda r: -r["lo_frac"])[:TOPK]]
        oz_p = f"{P}/olsson_head_scores_{tag}.json"
        if os.path.exists(oz_p):
            oz = json.load(open(oz_p))["per_head"]
            sets["top_induction"] = [r["head"] for r in
                                     sorted(good, key=lambda r: -oz.get(r["head"], {"ind": 0})["ind"])[:TOPK]]
    sets["parity21"] = PARITY21.split(",")

    print(f"\n--- HEAD-SET mean ablation (same task, same metric) ---")
    print(f"{'set':<16} {'k':>3} {'nbr_valid':>10} {'drop':>8}   members")
    allh = [(l, h) for l in range(nL) for h in range(nH)]
    for nm, hs in sets.items():
        st["heads"] = parse(hs); v = nbr_validity(); st["heads"] = None
        res["heads"][nm] = {"k": len(hs), "nbr_validity": round(v, 4), "drop": round(base - v, 4),
                            "members": hs}
        print(f"{nm:<16} {len(hs):3} {v:10.4f} {base - v:+8.4f}   {','.join(hs[:6])}...", flush=True)
    rv = []
    for i in range(NRAND):
        sel = [allh[j] for j in rng.choice(len(allh), TOPK, replace=False)]
        st["heads"] = sel; rv.append(nbr_validity()); st["heads"] = None
    res["random_heads"] = {"k": TOPK, "mean": round(float(np.mean(rv)), 4),
                           "sd": round(float(np.std(rv)), 4), "draws": [round(x, 4) for x in rv]}
    print(f"{'random':<16} {TOPK:3} {np.mean(rv):10.4f} {base - np.mean(rv):+8.4f}   "
          f"(sd {np.std(rv):.4f})")
    for h in hooks: h.remove()
    p_ = f"{OUTDIR}/mode_vs_head_ablate{OUTTAG}_{tag}.json"
    json.dump(res, open(p_, "w"), indent=2); print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
