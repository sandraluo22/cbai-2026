"""Single-head ablation sweep: which heads build each axis (x, y, parity)?

Ablate every attention head one at a time (zero its o_proj slice at all positions), recompute the
teacher-forced node-means, and measure the change in each axis's POWER (fraction of node-mean
variance along that cut) at the axis's clean peak layer. damage = clean_power - ablated_power
(positive = the head helps build that axis; negative = ablating it strengthens the axis).

Output: per-axis (nL x nH) damage map + the top heads for each axis. Reveals whether each axis has
a dedicated circuit and whether x/y/parity are built by distinct heads.

Env: PRESET GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) WLEN(300) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/head_axis_sweep_<model>_<graph>.json + .pdf
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
CTXLO = int(os.environ.get("CTXLO", "100"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/head_axis_sweep")


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


def ablate_head_hook(block, cm, dev, h):
    proj, hd = attn_proj(block, cm)
    ct = torch.arange(h * hd, (h + 1) * hd, device=dev, dtype=torch.long)
    def pre(_m, args, ct=ct):
        x = args[0].clone(); x[..., ct] = 0
        return (x,) + tuple(args[1:])
    return proj.register_forward_pre_hook(pre)


@torch.no_grad()
def node_means(model, tok, blocks, cm, walks, dev, n, layers):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in layers]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in layers}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            first = layers[0]
            for L in layers:
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == first: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in layers}


def axis_power(H, cuts):
    Hc = H - H.mean(0); tot = (Hc ** 2).sum() + 1e-12
    return {k: float(((Hc.T @ u) ** 2).sum() / tot) for k, u in cuts.items()}


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    cuts = {"x": unit(coords[:, 0]), "y": unit(coords[:, 1])}
    par = two_colour(graph)
    if not np.allclose(par, par[0]): cuts["parity"] = unit(par)
    axes = list(cuts)
    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    walks = G.generate_walks(graph, cfg)

    # clean pass over ALL layers -> per-axis peak layer
    meansL = node_means(model, tok, blocks, cm, walks, dev, n, list(range(nL)))
    powL = {ax: np.array([axis_power(meansL[L], cuts)[ax] for L in range(nL)]) for ax in axes}
    Lpk = {ax: int(powL[ax].argmax()) for ax in axes}
    clean = {ax: float(powL[ax][Lpk[ax]]) for ax in axes}
    read_layers = sorted(set(Lpk.values()))
    print(f"[{tag}] peak layers {Lpk}  clean power { {a: round(clean[a],3) for a in axes} }", flush=True)

    damage = {ax: np.zeros((nL, nH)) for ax in axes}
    for L in range(nL):
        for h in range(nH):
            hd = ablate_head_hook(blocks[L], cm, dev, h)
            try:
                mm = node_means(model, tok, blocks, cm, walks, dev, n, read_layers)
                for ax in axes:
                    p = axis_power(mm[Lpk[ax]], cuts)[ax]
                    damage[ax][L, h] = clean[ax] - p
            finally:
                hd.remove()
        if L % 4 == 0: print(f"[{tag}] swept layer {L}/{nL}", flush=True)

    out = {"model": tag, "graph": GRAPH, "nL": nL, "nH": nH, "axes": axes,
           "peak_layer": Lpk, "clean_power": clean,
           "damage": {ax: damage[ax].tolist() for ax in axes}, "top": {}}
    for ax in axes:
        flat = np.dstack(np.unravel_index(np.argsort(damage[ax], axis=None)[::-1], damage[ax].shape))[0]
        out["top"][ax] = [[int(l), int(h), round(float(damage[ax][l, h]), 4)] for l, h in flat[:12]]
        print(f"[{tag}] top heads for {ax}: " +
              ", ".join(f"L{l}H{h}({damage[ax][l,h]:+.3f})" for l, h, _ in out['top'][ax][:6]), flush=True)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    json.dump(out, open(f"{OUTDIR}/head_axis_sweep_{tag}_{GRAPH}.json", "w"), indent=2)
    make_fig(out, damage, f"{OUTDIR}/head_axis_sweep_{tag}_{GRAPH}.pdf")
    print(f"DONE -> {OUTDIR}/head_axis_sweep_{tag}_{GRAPH}.json", flush=True)


def make_fig(out, damage, path):
    axes = out["axes"]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, len(axes), figsize=(5.2 * len(axes), 5))
        if len(axes) == 1: ax = [ax]
        for i, a in enumerate(axes):
            D = damage[a]; lim = np.abs(D).max() + 1e-9
            im = ax[i].imshow(D, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[i].set_title(f"{a}: damage (clean−abl)\npeak L{out['peak_layer'][a]}, clean {out['clean_power'][a]:.2f}", fontsize=9)
            ax[i].set_xlabel("head"); ax[i].set_ylabel("layer")
            fig.colorbar(im, ax=ax[i], fraction=.046)
            for l, h, d in out["top"][a][:5]:
                ax[i].plot(h, l, "o", mfc="none", mec="k", ms=9)
        fig.suptitle(f"{out['model']} {out['graph']} — single-head ablation: damage to each axis "
                     "(red = ablating this head weakens the axis)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
