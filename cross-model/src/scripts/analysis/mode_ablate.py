"""Causal test of the divider basis: does removing the PARITY (checkerboard) direction hurt
long-horizon behaviour more than removing the x or y coordinate direction?

divider_basis found the node representation is dominated by three cuts -- x, y, and checkerboard
parity (the biggest, and the one the 2-D coord probe misses). Here we test their causal weight.

For each named cut u (a node-vector: x=coord0, y=coord1, parity=graph 2-colouring) we build, at EVERY
layer L, the rank-1 residual direction that carries it: v_L = normalise(Hc_L^T u), where Hc_L is the
centred clean node-means at L. Then we seed context and generate GSTEPS steps while projecting v_L
OUT of the residual at every layer/position. Conditions: clean | remove_x | remove_y | remove_parity
| remove_random (rank-1 per-layer control). Per window we track neighbour mass + validity, and we
verify each removal actually drops that cut's linear decodability.

If remove_parity tanks behaviour more than remove_x / remove_y / remove_random, parity is not just
the largest descriptive component but the most causally load-bearing cut.

Env: PRESET GEN_MODEL(Llama) GRAPH(square_grid) XCTX(150) GSTEPS(150) NSEED(4) NWIN(6) CTXLO(100)
     TEMP(1.0) OUTDIR DEVICE
Out: <OUTDIR>/mode_ablate_<graph>.json + .pdf
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
NSEED = int(os.environ.get("NSEED", "4")); NWIN = int(os.environ.get("NWIN", "6"))
CTXLO = int(os.environ.get("CTXLO", "100")); TEMP = float(os.environ.get("TEMP", "1.0"))
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]
OUTDIR = os.environ.get("OUTDIR", "runs/axes/3_causal/mode_ablate")


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


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def loo_r2_1d(H, y):
    ok = np.isfinite(H).all(1); H = H[ok]; y = y[ok]; nn = H.shape[0]
    if nn < 6: return float("nan")
    mu = H.mean(0); sd = H.std(0) + 1e-6; Xs = (H - mu) / sd; yc = y - y.mean()
    best = -9.0
    for a in ALPHAS:
        pred = np.zeros(nn)
        for k in range(nn):
            idx = [i for i in range(nn) if i != k]
            U, S, Vt = np.linalg.svd(Xs[idx], full_matrices=False)
            ytr = yc[idx]
            pred[k] = (Xs[k] @ Vt.T) @ ((S / (S ** 2 + a)) * (U.T @ ytr))
        best = max(best, _r2(yc, pred))
    return best


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


def cut_directions(means, cuts, dev, nL):
    """v_L[cut] = normalised Hc_L^T u_cut  (rank-1 residual direction carrying that node-cut)."""
    dirs = {c: {} for c in cuts}
    for L in range(nL):
        H = means[L]; Hc = H - H.mean(0)
        for c, u in cuts.items():
            v = Hc.T @ u; nrm = np.linalg.norm(v)
            dirs[c][L] = torch.tensor((v / (nrm + 1e-9)), device=dev, dtype=torch.float32)
    return dirs


def removal_hooks(blocks, cm, vlayers):
    """Project a per-layer unit direction OUT of the residual at every layer, all positions."""
    handles = []
    for L, v in vlayers.items():
        def hk(_m, _i, out, v=v):
            h = out[0] if isinstance(out, tuple) else out
            vv = v.to(h.dtype)
            h = h - (h @ vv).unsqueeze(-1) * vv
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handles.append(blocks[L].register_forward_hook(hk))
    return handles


@torch.no_grad()
def gen_track(model, tok, blocks, cm, graph, cand_t, dev, seed_nodes, vlayers, coords, parity, rng, acc, GBIN):
    NWIN = acc["nbr"].shape[0]; nodes = list(seed_nodes); nL = cm.num_hidden_layers
    handles = removal_hooks(blocks, cm, vlayers) if vlayers else []
    try:
        for t in range(GSTEPS):
            wk = mkwalk(nodes, graph)
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            last = model(input_ids=ids).logits[0, -1]
            p = torch.softmax(last[cand_t].float() / TEMP, 0).cpu().numpy(); p = p / p.sum()
            prev = nodes[-1]; nb = graph.neighbors(prev); j = int(rng.choice(len(p), p=p))
            b = min(t // GBIN, NWIN - 1)
            acc["nbr"][b] += float(p[nb].sum()); acc["val"][b] += int(j in nb); acc["cnt"][b] += 1
            nodes.append(j)
    finally:
        for h in handles: h.remove()
    # decodability of x/y/parity from generated-token node-means (removal still applied)
    gen = nodes[len(seed_nodes):]; gstart = len(seed_nodes); ng = len(gen); grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    caps = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    handles = removal_hooks(blocks, cm, vlayers) if vlayers else []
    wk = mkwalk(nodes, graph); spans = resolve_token_spans(tok, wk)
    ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev); grabbed.clear()
    model(input_ids=ids); single = [t[-1] for t in spans]
    for h in caps: h.remove()
    for h in handles: h.remove()
    Lprobe = acc["Lprobe"]
    rowsL = grabbed[Lprobe][0][[single[gstart + i] for i in range(ng)]].float().cpu().numpy()
    for i in range(ng):
        acc["gsum"][gen[i]] += rowsL[i]; acc["gcnt"][gen[i]] += 1


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=max(NSEED, 24), walk_length=XCTX, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    parity = two_colour(graph)
    cuts = {"x": coords[:, 0] - coords[:, 0].mean(), "y": coords[:, 1] - coords[:, 1].mean(),
            "parity": parity - parity.mean()}
    cuts = {c: u / (np.linalg.norm(u) + 1e-9) for c, u in cuts.items()}
    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    seeds = G.generate_walks(graph, cfg)[:NSEED]
    Lprobe = int(round(nL * 0.7))                      # late-ish layer for the decodability check

    means = clean_node_means(model, tok, blocks, cm, seeds, dev, n)
    dirs = cut_directions(means, cuts, dev, nL)
    rng0 = np.random.default_rng(0)
    randdir = {L: torch.tensor(rng0.standard_normal(cm.hidden_size), device=dev, dtype=torch.float32) for L in range(nL)}
    for L in randdir: randdir[L] = randdir[L] / randdir[L].norm()

    conds = {"clean": None, "remove_x": dirs["x"], "remove_y": dirs["y"],
             "remove_parity": dirs["parity"], "remove_random": randdir}
    GBIN = max(1, GSTEPS // NWIN); win_mid = [b * GBIN + GBIN // 2 for b in range(NWIN)]
    out = {"graph": GRAPH, "model": tag, "nL": nL, "Lprobe": Lprobe, "nwin": NWIN, "win_mid": win_mid, "conds": {}}
    for cname, vlayers in conds.items():
        acc = {"nbr": np.zeros(NWIN), "val": np.zeros(NWIN), "cnt": np.zeros(NWIN),
               "gsum": np.zeros((n, cm.hidden_size)), "gcnt": np.zeros(n), "Lprobe": Lprobe}
        for si, seed in enumerate(seeds):
            gen_track(model, tok, blocks, cm, graph, cand_t, dev, seed.nodes, vlayers, coords, parity,
                      np.random.default_rng(1000 + si), acc, GBIN)
        cnt = np.maximum(acc["cnt"], 1); val = acc["val"] / cnt; nbr = acc["nbr"] / cnt
        H = np.where(acc["gcnt"][:, None] > 0, acc["gsum"] / np.maximum(acc["gcnt"][:, None], 1), np.nan)
        dec = {"x": loo_r2_1d(H, coords[:, 0]), "y": loo_r2_1d(H, coords[:, 1]), "parity": loo_r2_1d(H, parity)}
        out["conds"][cname] = {"val": val.tolist(), "nbr": nbr.tolist(), "decode": dec}
        print(f"[{tag}/{GRAPH}/{cname}] val {val.mean():.2f}  nbr {nbr.mean():.2f}  "
              f"decode x={dec['x']:.2f} y={dec['y']:.2f} parity={dec['parity']:.2f}", flush=True)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    json.dump(out, open(f"{OUTDIR}/mode_ablate_{GRAPH}.json", "w"), indent=2)
    make_fig(out, f"{OUTDIR}/mode_ablate_{GRAPH}.pdf")
    print(f"DONE -> {OUTDIR}/mode_ablate_{GRAPH}.json", flush=True)


def make_fig(out, path):
    colors = {"clean": "k", "remove_x": "tab:blue", "remove_y": "tab:cyan",
              "remove_parity": "tab:red", "remove_random": ".6"}
    wm = out["win_mid"]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        for cn, c in colors.items():
            cd = out["conds"].get(cn)
            if not cd: continue
            ax[0].plot(wm, cd["nbr"], "-o", ms=3, color=c, label=cn)
            ax[1].plot(wm, cd["val"], "-o", ms=3, color=c, label=cn)
        ax[0].set_title("neighbour mass (behaviour)", fontsize=9); ax[0].set_ylim(0, 1.05)
        ax[1].set_title("validity (behaviour)", fontsize=9); ax[1].set_ylim(0, 1.05)
        for a in ax[:2]: a.set_xlabel("generation step"); a.legend(fontsize=7)
        labels = list(out["conds"]); xk = np.arange(len(labels)); w = 0.25
        for i, cut in enumerate(["x", "y", "parity"]):
            ax[2].bar(xk + (i - 1) * w, [out["conds"][l]["decode"][cut] for l in labels], w, label=f"decode {cut}")
        ax[2].set_xticks(xk); ax[2].set_xticklabels(labels, rotation=30, fontsize=6, ha="right")
        ax[2].set_title("linear decodability after removal", fontsize=9); ax[2].axhline(0, color=".7", lw=.6); ax[2].legend(fontsize=7)
        fig.suptitle(f"[{out['graph']}] {out['model']} — remove x / y / parity direction (rank-1, all layers) during generation.\n"
                     "red=parity. Does the biggest descriptive cut carry the most causal weight?", fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
