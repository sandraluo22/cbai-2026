"""Steer / remove the x,y,parity axes -- measured NEXT-TOKEN (teacher-forced) AND LONG-TERM (rollout).

Complements mode_ablate (removal, long-term only). For each axis cut we build the per-layer readout
direction v_L = normalise(Hc_L^T u) and intervene at ALL layers/positions:
  remove_<axis> : project v_L out               (h -= (h.v)v)
  steer_<axis>  : add +DOSE*std along v_L        (h += DOSE * proj_std_L * v_L)  [push toward + side]
where proj_std_L is the std over nodes of the clean projection onto v_L (so DOSE is in "axis std" units).

Two readouts per condition:
  NEXT-TOKEN (immediate): over real held-out walks, at every position with ctx>=CTXLO, the intervened
    model's neighbour mass + validity for the immediate next node (no autoregression). For steer we also
    split candidate mass by node parity (does steering +parity push predictions onto one sublattice?).
  LONG-TERM (autoregressive): seed then generate GSTEPS under the intervention; mean neighbour mass + validity.

Env: PRESET GEN_MODEL(Llama) GRAPH(square_grid) XCTX(150) GSTEPS(150) NSEED(4) NWALK_TF(12) CTXLO(100)
     DOSE(4.0) TEMP(1.0) OUTDIR DEVICE
Out: <OUTDIR>/axis_steer_<graph>.json + .pdf
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
from graph import Walk
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
XCTX = int(os.environ.get("XCTX", "150")); GSTEPS = int(os.environ.get("GSTEPS", "150"))
NSEED = int(os.environ.get("NSEED", "4")); NWALK_TF = int(os.environ.get("NWALK_TF", "12"))
CTXLO = int(os.environ.get("CTXLO", "100")); DOSE = float(os.environ.get("DOSE", "4.0"))
TEMP = float(os.environ.get("TEMP", "1.0"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/3_causal/axis_steer")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def mkwalk(nodes, graph): return Walk(walk_id=0, nodes=list(nodes), words=[graph.words[j] for j in nodes])


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


@torch.no_grad()
def clean_node_means(model, tok, blocks, cm, walks, dev, n):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    nL = cm.num_hidden_layers; hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(nL)}


def build_dirs(means, cuts, dev, nL, cm):
    """v_L (unit direction) and s_L (= std * v_L, one 'axis std' of steer) per layer per cut."""
    vmap = {c: {} for c in cuts}; smap = {c: {} for c in cuts}
    for L in range(nL):
        H = means[L]; Hc = H - H.mean(0)
        for c, u in cuts.items():
            v = Hc.T @ u; v = v / (np.linalg.norm(v) + 1e-9)
            proj = Hc @ v; std = float(proj.std())
            vmap[c][L] = torch.tensor(v, device=dev, dtype=torch.float32)
            smap[c][L] = torch.tensor(std * v, device=dev, dtype=torch.float32)
    return vmap, smap


def hooks(blocks, layers_vec, mode):
    """mode='remove' -> project out unit vec; mode='steer' -> add vec."""
    handles = []
    for L, vec in layers_vec.items():
        def hk(_m, _i, out, vec=vec, mode=mode):
            h = out[0] if isinstance(out, tuple) else out
            w = vec.to(h.dtype)
            h = (h - (h @ w).unsqueeze(-1) * w) if mode == "remove" else (h + w)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handles.append(blocks[L].register_forward_hook(hk))
    return handles


@torch.no_grad()
def next_token(model, tok, blocks, cm, graph, cand_t, dev, walks, layers_vec, mode, parity):
    handles = hooks(blocks, layers_vec, mode) if layers_vec else []
    nbr, val, ppos, pneg, cnt = 0.0, 0, 0.0, 0.0, 0
    pos_idx = np.where(parity > 0)[0]; neg_idx = np.where(parity < 0)[0]
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes
            logits = model(input_ids=ids).logits[0]
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                p = torch.softmax(logits[spans[s][-1]][cand_t].float() / TEMP, 0).cpu().numpy(); p = p / p.sum()
                nb = graph.neighbors(nodes[s]); nbr += float(p[nb].sum()); val += int(int(p.argmax()) in nb)
                ppos += float(p[pos_idx].sum()); pneg += float(p[neg_idx].sum()); cnt += 1
    finally:
        for h in handles: h.remove()
    c = max(cnt, 1)
    return {"nbr": nbr / c, "val": val / c, "mass_pos": ppos / c, "mass_neg": pneg / c}


@torch.no_grad()
def long_term(model, tok, blocks, cm, graph, cand_t, dev, seeds, layers_vec, mode, rng):
    nbr, val, cnt = 0.0, 0, 0
    for seed in seeds:
        nodes = list(seed.nodes); handles = hooks(blocks, layers_vec, mode) if layers_vec else []
        try:
            for t in range(GSTEPS):
                wk = mkwalk(nodes, graph)
                ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
                p = torch.softmax(model(input_ids=ids).logits[0, -1][cand_t].float() / TEMP, 0).cpu().numpy(); p = p / p.sum()
                nb = graph.neighbors(nodes[-1]); j = int(rng.choice(len(p), p=p))
                nbr += float(p[nb].sum()); val += int(j in nb); cnt += 1; nodes.append(j)
        finally:
            for h in handles: h.remove()
    c = max(cnt, 1)
    return {"nbr": nbr / c, "val": val / c}


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=max(NSEED, NWALK_TF, 24), walk_length=XCTX, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    parity = two_colour(graph)
    cuts = {"x": unit(coords[:, 0]), "y": unit(coords[:, 1])}
    if not np.allclose(parity, parity[0]): cuts["parity"] = unit(parity)
    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    walks = G.generate_walks(graph, cfg); tf_walks = walks[:NWALK_TF]; seeds = walks[:NSEED]

    means = clean_node_means(model, tok, blocks, cm, walks, dev, n)
    vmap, smap = build_dirs(means, cuts, dev, nL, cm)

    DOSES = os.environ.get("DOSES", "")
    conds = {"clean": (None, None)}
    if DOSES:                                    # dose sweep along one axis (default parity)
        swc = os.environ.get("SWEEP_CUT", "parity" if "parity" in cuts else "x")
        for d in [float(x) for x in DOSES.split(",")]:
            conds[f"steer_{swc}_{d:g}"] = ({L: d * smap[swc][L] for L in range(nL)}, "steer")
    else:
        for c in cuts:
            conds[f"remove_{c}"] = (vmap[c], "remove")
            conds[f"steer_{c}+"] = ({L: DOSE * smap[c][L] for L in range(nL)}, "steer")
    out = {"graph": GRAPH, "model": tag, "dose": DOSE, "nL": nL, "conds": {}}
    for cname, (lv, mode) in conds.items():
        nt = next_token(model, tok, blocks, cm, graph, cand_t, dev, tf_walks, lv, mode, parity)
        lt = long_term(model, tok, blocks, cm, graph, cand_t, dev, seeds, lv, mode, np.random.default_rng(0))
        out["conds"][cname] = {"next_token": nt, "long_term": lt}
        print(f"[{tag}/{GRAPH}/{cname:12s}] NT nbr={nt['nbr']:.2f} val={nt['val']:.2f} "
              f"(par+ {nt['mass_pos']:.2f}/par- {nt['mass_neg']:.2f}) | LT nbr={lt['nbr']:.2f} val={lt['val']:.2f}", flush=True)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    json.dump(out, open(f"{OUTDIR}/axis_steer_{GRAPH}.json", "w"), indent=2)
    make_fig(out, f"{OUTDIR}/axis_steer_{GRAPH}.pdf")
    print(f"DONE -> {OUTDIR}/axis_steer_{GRAPH}.json", flush=True)


def make_fig(out, path):
    conds = list(out["conds"]); x = np.arange(len(conds)); w = 0.35
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        for a, metric in enumerate(["val", "nbr"]):
            ax[a].bar(x - w / 2, [out["conds"][c]["next_token"][metric] for c in conds], w, label="next-token", color="tab:orange")
            ax[a].bar(x + w / 2, [out["conds"][c]["long_term"][metric] for c in conds], w, label="long-term", color="tab:purple")
            ax[a].set_xticks(x); ax[a].set_xticklabels(conds, rotation=35, ha="right", fontsize=7)
            ax[a].set_title(f"{'validity' if metric=='val' else 'neighbour mass'}: next-token vs long-term", fontsize=9)
            ax[a].set_ylim(0, 1.05); ax[a].legend(fontsize=8)
        fig.suptitle(f"[{out['graph']}] {out['model']} — remove / steer(+{out['dose']}std) the x,y,parity axes.\n"
                     "next-token (immediate) vs long-term (150-step rollout).", fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
