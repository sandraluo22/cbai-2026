"""Does ablating the heads destroy the GEOMETRY? (the edge Arditi tested and we never did)

Arditi tested heads -> geometry. We tested heads -> task, and geometry -> task. The missing edge is
heads -> geometry: if the grid/ring structure SURVIVES ablation while the task dies, his mechanism is
wrong even though his deflationary conclusion is right.

Ablate a head set, then measure the geometry of the per-node mean representation at LAYER:
    rsa          Pearson r between representational distance and GRAPH distance over all node pairs
                 (the standard "the geometry is there" measure; fit-free)
    top2_var     fraction of node-mean variance in the top-2 PCs
    coord_r      |r| of PC1 with row and PC2 with col, best assignment (grid only)
    nbr_valid    task performance under the same ablation, so geometry and task are read off the SAME
                 forward passes and cannot drift apart for trivial reasons

Head sets (the 21 split by their STANDARD scores, which is the whole question):
    induction12  the 6 top-100-induction + 6 top-100-prev-token heads inside the 21
    neither9     the 9 that are top-100 on NEITHER (the same-token aggregators)
    all21        the full set
    plus NRAND random sets matched to each size

On a RING there is no row/col to correlate a PC with, so coord_r is instead the circular-circular
concentration of the top-2 PC angle against the true ring angle (1 = the plane still carries the cycle).

Env: GEN_MODEL(Llama) GRAPH(grid|ring) K(4) LAYER(14) SETS("nm=L14H19+L16H3,...") RANDMODE(all|layer)
     NWALKS(3) WLEN(1200) CTXLO(800) NRAND(5) SEED(0) OUTDIR DEVICE OUTTAG("")
Out: <OUTDIR>/geometry_under_ablation<OUTTAG>_<model>.json
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
GRAPH = os.environ.get("GRAPH", "grid"); K = int(os.environ.get("K", "4"))
LAYER = int(os.environ.get("LAYER", "14"))
NWALKS = int(os.environ.get("NWALKS", "3")); WLEN = int(os.environ.get("WLEN", "1200"))
CTXLO = int(os.environ.get("CTXLO", "800")); NRAND = int(os.environ.get("NRAND", "5"))
SEED = int(os.environ.get("SEED", "0"))
OUTTAG = os.environ.get("OUTTAG", ""); OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")

# SETS: "name=L14H19+L16H3,other=L1H2+L3H4" overrides the built-in three. Used to ask the same
# heads->geometry question of a set derived some other way (e.g. the greedy builders of the DAS axes).
SETS = {s.split("=")[0]: s.split("=")[1].split("+") for s in os.environ.get("SETS", "").split(",") if "=" in s}
RANDMODE = os.environ.get("RANDMODE", "all")      # "all" = any upstream head | "layer" = depth-matched

# the 21, split by their Olsson ranks (top-100 on either metric vs neither)
INDUCTION12 = ["L15H30", "L16H20", "L2H22", "L16H1", "L13H18", "L25H7",      # top-100 induction
               "L14H26", "L9H11", "L1H20", "L21H10", "L4H12", "L3H17"]       # top-100 prev-token
NEITHER9 = ["L21H2", "L14H19", "L14H17", "L10H2", "L7H25", "L8H11", "L4H16", "L1H21", "L2H26"]
ALL21 = INDUCTION12 + NEITHER9


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
    graph = G.build_graph(cfg); words = list(graph.words)
    coords = np.array(graph.coords, float)
    wid = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok("a")["input_ids"][0]
    cand_t = torch.tensor(wid, device=dev)

    # graph (shortest-path) distance
    INF = 10 ** 6
    D = np.full((n, n), INF); np.fill_diagonal(D, 0)
    for u in range(n):
        for v in graph.adjacency[u]: D[u, v] = 1
    for k_ in range(n):
        D = np.minimum(D, D[:, k_][:, None] + D[k_][None, :])
    iu = np.triu_indices(n, 1); gd = D[iu]

    data = []
    for wk in G.generate_walks(graph, cfg):
        steps = [s for s in range(len(wk.nodes) - 1) if s + 1 >= CTXLO]
        if steps:
            data.append((torch.tensor([[bos] + [wid[x] for x in wk.nodes]], device=dev),
                         torch.tensor([s + 1 for s in steps], device=dev),
                         [wk.nodes[s] for s in steps]))

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

    def measure():
        S = torch.zeros(n, cm.hidden_size, device=dev); C = torch.zeros(n, device=dev)
        ok = tot = 0
        for ids, rp, nds in data:
            o = model(input_ids=ids, output_hidden_states=True)
            H = o.hidden_states[LAYER + 1][0, rp].float()
            top = o.logits[0][rp][:, cand_t].float().argmax(1).tolist()
            for i, u in enumerate(nds):
                S[u] += H[i]; C[u] += 1
                ok += int(top[i] in graph.adjacency[u]); tot += 1
        Mn = (S / C.clamp(min=1)[:, None]); Mn = Mn - Mn.mean(0, keepdim=True)
        Mnp = Mn.cpu().numpy()
        rd = np.linalg.norm(Mnp[:, None, :] - Mnp[None, :, :], axis=-1)[iu]
        rsa = float(np.corrcoef(rd, gd)[0, 1]) if rd.std() > 1e-9 else 0.0
        sv = np.linalg.svd(Mnp, compute_uv=False); var = sv ** 2
        U = np.linalg.svd(Mnp, full_matrices=False)[0]
        cr = 0.0
        if GRAPH == "grid":
            a = abs(np.corrcoef(U[:, 0], coords[:, 0])[0, 1]) + abs(np.corrcoef(U[:, 1], coords[:, 1])[0, 1])
            b = abs(np.corrcoef(U[:, 0], coords[:, 1])[0, 1]) + abs(np.corrcoef(U[:, 1], coords[:, 0])[0, 1])
            cr = max(a, b) / 2
        if GRAPH == "ring":
            # A ring has ONE cyclic coordinate, so the grid's PC1-vs-row test does not apply. The
            # matched question is whether the top-2 PC plane still carries the cyclic angle: correlate
            # the node's angle in that plane with its true ring angle, up to rotation/reflection, via
            # the circular-circular correlation of (theta_hat - theta) being constant.
            th = 2 * np.pi * np.arange(n) / n
            ph = np.arctan2(U[:, 1], U[:, 0])
            cr = max(abs(float(np.abs(np.mean(np.exp(1j * (ph - s * th)))))) for s in (1, -1))
        return {"rsa": round(rsa, 4), "top2_var": round(float(var[:2].sum() / var.sum()), 4),
                "coord_r": round(float(cr), 4), "nbr_valid": round(ok / tot, 4)}

    def parse(x): return [(int(h.split("H")[0][1:]), int(h.split("H")[1])) for h in x]

    # A head at layer > LAYER has NO causal path to hidden_states[LAYER+1]. Ablating it cannot move the
    # geometry measured there, so both the real sets and the random controls must be restricted to
    # UPSTREAM heads and matched on the resulting count — otherwise the sweep confounds "geometry varies
    # with depth" with "the effective ablation set shrinks as LAYER drops", and random controls drawn
    # from all 1024 heads are systematically weaker than the real sets at low LAYER.
    def upstream(x): return [h for h in x if int(h.split("H")[0][1:]) <= LAYER]
    rows = {}
    rows["baseline"] = measure()
    print(f"[{tag}] {GRAPH}{n} L{LAYER}  baseline: " +
          "  ".join(f"{k}={v}" for k, v in rows["baseline"].items()), flush=True)
    print(f"\n{'ablated set':<16} {'k':>3} {'rsa':>8} {'top2_var':>9} {'coord_r':>8} {'nbr_valid':>10}")
    print(f"{'(none)':<16} {0:3} {rows['baseline']['rsa']:8.4f} {rows['baseline']['top2_var']:9.4f} "
          f"{rows['baseline']['coord_r']:8.4f} {rows['baseline']['nbr_valid']:10.4f}")
    eff, upsets = {}, {}
    for nm, hs in (SETS.items() if SETS else
                   (("all21", ALL21), ("induction12", INDUCTION12), ("neither9", NEITHER9))):
        up = upstream(hs); eff[nm] = len(up); upsets[nm] = up
        st["heads"] = parse(up); r = measure(); st["heads"] = None
        r["k_nominal"] = len(hs); r["k_upstream"] = len(up); rows[nm] = r
        print(f"{nm:<16} {len(up):3} {r['rsa']:8.4f} {r['top2_var']:9.4f} {r['coord_r']:8.4f} "
              f"{r['nbr_valid']:10.4f}   (of {len(hs)} nominal; "
              f"{len(hs)-len(up)} downstream of L{LAYER} and causally inert here)", flush=True)
    allh = [(l, h) for l in range(nL) for h in range(nH) if l <= LAYER]   # UPSTREAM pool only
    # RANDMODE=layer replaces each head with a random head AT ITS OWN LAYER. A set defined as "top head
    # at every layer" is depth-structured by construction, so a draw from the whole upstream pool is not
    # a matched control for it.
    draws = list(upsets.items()) if RANDMODE == "layer" else [(f"k{k_}", None) for k_ in sorted(set(eff.values()))]
    for nm, up in draws:
        k_ = len(up) if up is not None else int(nm[1:])
        acc = []
        for _ in range(NRAND):
            st["heads"] = ([(l, int(rng.choice([x for x in range(nH) if x != h]))) for l, h in parse(up)]
                           if up is not None else [allh[j] for j in rng.choice(len(allh), k_, replace=False)])
            acc.append(measure()); st["heads"] = None
        key = f"random_{nm}" if up is not None else f"random{k_}"
        rows[key] = {k: round(float(np.mean([a[k] for a in acc])), 4) for k in acc[0]}
        rows[key]["k_upstream"] = k_
        rows[f"{key}_sd"] = {k: round(float(np.std([a[k] for a in acc])), 4) for k in acc[0]}
        r = rows[key]
        print(f"{key:<16} {k_:3} {r['rsa']:8.4f} {r['top2_var']:9.4f} {r['coord_r']:8.4f} "
              f"{r['nbr_valid']:10.4f}   (+-{rows[f'{key}_sd']['rsa']:.3f} rsa, "
              f"+-{rows[f'{key}_sd']['nbr_valid']:.3f} valid over {NRAND} draws)", flush=True)
    for h in hooks: h.remove()
    p_ = f"{OUTDIR}/geometry_under_ablation{OUTTAG}_{tag}.json"
    json.dump({"model": tag, "graph": GRAPH, "n": n, "layer": LAYER, "rows": rows,
               "randmode": RANDMODE, "nrand": NRAND,
               "sets": SETS or {"all21": ALL21, "induction12": INDUCTION12, "neither9": NEITHER9}},
              open(p_, "w"), indent=2)
    print(f"\nDONE -> {p_}", flush=True)


if __name__ == "__main__":
    main()
