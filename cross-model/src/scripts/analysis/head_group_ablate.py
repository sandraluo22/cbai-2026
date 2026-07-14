"""Group-ablation control: do the top axis heads (from head_axis_sweep) really carry each axis, vs a
random set of K heads? Ablate a SET of heads together, recompute teacher-forced node-means, and
measure x/y/parity power. Compare: top-K parity heads, top-K coordinate heads, and NRAND random
draws of K heads (+ induction top-K and DLA top-K for reference).

Env: PRESET GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) WLEN(300) CTXLO(100) K(5) NRAND(12)
     SWEEPJSON INDJSON DLAJSON OUTDIR DEVICE
Out: <OUTDIR>/head_group_ablate_<model>_<graph>.json + .pdf
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import torch
except Exception:
    torch = None

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans

PRESET = os.environ.get("PRESET", "gemma_qwen")
ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"),
           "Qwen": ("Qwen/Qwen3-8B-Base", None), "distilgpt2": ("distilgpt2", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama" if PRESET != "smoke" else "distilgpt2")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "16")); WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100")); K = int(os.environ.get("K", "5")); NRAND = int(os.environ.get("NRAND", "12"))
SWEEPJSON = os.environ.get("SWEEPJSON", "runs/axes/4_circuits/head_axis_sweep/head_axis_sweep_Llama_square_grid.json")
INDJSON = os.environ.get("INDJSON", "runs/induction-head/1_circuits/induction_heads/induction.json")
DLAJSON = os.environ.get("DLAJSON", "runs/induction-head/1_circuits/attribution/head_attribution_square_grid.json")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/head_group_ablate")
RNG = np.random.default_rng(0)


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def two_colour(graph):
    n = graph.n_nodes; col = np.zeros(n)
    for s in range(n):
        if col[s] != 0: continue
        col[s] = 1; st = [s]
        while st:
            u = st.pop()
            for v in graph.adjacency[u]:
                if col[v] == 0: col[v] = -col[u]; st.append(v)
    return col.astype(float)


def unit(v): v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-9)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def by_layer(heads):
    d = {}
    for l, h in heads: d.setdefault(int(l), []).append(int(h))
    return d


def ablation_hooks(blocks, cm, dev, by_layer_map):
    handles = []
    for L, heads in by_layer_map.items():
        proj, hd = attn_proj(blocks[L], cm)
        ct = torch.tensor(np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in heads]), device=dev, dtype=torch.long)
        def pre(_m, args, ct=ct):
            x = args[0].clone(); x[..., ct] = 0
            return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    return handles


@torch.no_grad()
def node_means(model, tok, blocks, cm, walks, dev, n, layers, abl):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    caps = [blocks[L].register_forward_hook(mk(L)) for L in layers]
    handles = ablation_hooks(blocks, cm, dev, abl) if abl else []
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in layers}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1); first = layers[0]
            for L in layers:
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == first: ncnt[nodes[s]] += 1
    finally:
        for h in caps: h.remove()
        for h in handles: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in layers}


def axis_power(H, cuts):
    Hc = H - H.mean(0); tot = (Hc ** 2).sum() + 1e-12
    return {k: float(((Hc.T @ u) ** 2).sum() / tot) for k, u in cuts.items()}


def topk_from_map(M, k):
    idx = np.dstack(np.unravel_index(np.argsort(M, axis=None)[::-1], M.shape))[0][:k]
    return [(int(l), int(h)) for l, h in idx]


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    sw = json.load(open(SWEEPJSON)); nL = sw["nL"]; nH = sw["nH"]; Lpk = sw["peak_layer"]
    Dp = np.array(sw["damage"]["parity"]); Dc = (np.array(sw["damage"]["x"]) + np.array(sw["damage"]["y"])) / 2
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    cuts = {"x": unit(coords[:, 0]), "y": unit(coords[:, 1])}
    par = two_colour(graph)
    if not np.allclose(par, par[0]): cuts["parity"] = unit(par)
    axes = list(cuts)

    parity_heads = topk_from_map(Dp, K); coord_heads = topk_from_map(Dc, K)
    used = set(parity_heads) | set(coord_heads)
    ind = np.array(json.load(open(INDJSON))["models"][tag]["generic"]) if os.path.exists(INDJSON) else np.zeros((nL, nH))
    dla = np.array(json.load(open(DLAJSON))["models"][tag]["head_attr"]) if os.path.exists(DLAJSON) else np.zeros((nL, nH))
    groups = {f"parity_top{K}": parity_heads, f"coord_top{K}": coord_heads,
              f"induction_top{K}": topk_from_map(ind, K), f"dla_top{K}": topk_from_map(dla, K)}
    pool = [(l, h) for l in range(nL) for h in range(nH) if (l, h) not in used]
    rand_groups = [[pool[i] for i in RNG.choice(len(pool), K, replace=False)] for _ in range(NRAND)]

    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    walks = G.generate_walks(graph, cfg)
    read_layers = sorted(set(Lpk[a] for a in axes))

    def power_for(abl):
        mm = node_means(model, tok, blocks, cm, walks, dev, n, read_layers, abl)
        return {a: axis_power(mm[Lpk[a]], cuts)[a] for a in axes}

    clean = power_for(None)
    out = {"model": tag, "graph": GRAPH, "K": K, "clean": clean, "peak_layer": Lpk,
           "groups": {"clean": {"power": clean, "heads": []}}}
    for gname, heads in groups.items():
        p = power_for(by_layer(heads))
        out["groups"][gname] = {"power": p, "heads": [list(t) for t in heads]}
    rand_pows = [power_for(by_layer(g)) for g in rand_groups]
    out["random"] = {"draws": NRAND, "power_mean": {a: float(np.mean([rp[a] for rp in rand_pows])) for a in axes},
                     "power_std": {a: float(np.std([rp[a] for rp in rand_pows])) for a in axes},
                     "all": [{a: rp[a] for a in axes} for rp in rand_pows]}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n{'group':16} " + "  ".join(f"{a:>8}" for a in axes))
    print(f"{'clean':16} " + "  ".join(f"{clean[a]:8.3f}" for a in axes))
    for gname in list(groups) + ["__rand__"]:
        if gname == "__rand__":
            m = out["random"]["power_mean"]; s = out["random"]["power_std"]
            print(f"{'random(mean±sd)':16} " + "  ".join(f"{m[a]:.2f}±{s[a]:.2f}" for a in axes))
        else:
            p = out["groups"][gname]["power"]
            print(f"{gname:16} " + "  ".join(f"{p[a]:8.3f}" for a in axes))
    json.dump(out, open(f"{OUTDIR}/head_group_ablate_{tag}_{GRAPH}.json", "w"), indent=2)
    make_fig(out, axes, f"{OUTDIR}/head_group_ablate_{tag}_{GRAPH}.pdf")
    print(f"DONE -> {OUTDIR}/head_group_ablate_{tag}_{GRAPH}.json", flush=True)


def make_fig(out, axes, path):
    groups = ["clean"] + [g for g in out["groups"] if g != "clean"] + ["random"]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(len(axes)); w = 0.8 / len(groups)
        for i, g in enumerate(groups):
            if g == "random":
                vals = [out["random"]["power_mean"][a] for a in axes]; err = [out["random"]["power_std"][a] for a in axes]
                ax.bar(x + (i - len(groups) / 2) * w, vals, w, yerr=err, capsize=3, label=f"random×{out['random']['draws']}", color=".6")
            else:
                vals = [out["groups"][g]["power"][a] for a in axes]
                ax.bar(x + (i - len(groups) / 2) * w, vals, w, label=g)
        ax.set_xticks(x); ax.set_xticklabels(axes); ax.set_ylabel("axis power (variance fraction) after ablation")
        ax.set_title(f"{out['model']} {out['graph']} — group ablation of K={out['K']} heads: which set kills which axis "
                     "(lower = more damage). clean = leftmost; random = grey control.", fontsize=9)
        ax.legend(fontsize=7, ncol=3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
