"""Trajectory of the geometry (best-2D RSA) over CONTEXT LENGTH, with vs without the
QK (prefix-matching) heads ablated. Does ablation change how the in-context geometry
EMERGES, even though the converged (ctx>=100) geometry survives?

For each model & graph, at the best-2D peak layer L*: run clean / ablate-QK (all heads
with QK>thresh) / ablate-random (matched size); for each condition, compute node-mean
best-2D RSA (and next-step neighbour mass) in context-length bins. Plot RSA & behaviour
vs context length, 3 conditions.

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(24) WLEN(320) QK_THRESH(0.2)
     INDJSON OUTDIR DEVICE
Out: <OUTDIR>/context_rsa_ablation_<graph>.json + .pdf
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
if PRESET == "smoke":
    MODELS = [("distilgpt2", "distilgpt2", None)]
else:
    MODELS = [("Llama", "meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
              ("Gemma", "google/gemma-2-9b", "unsloth/gemma-2-9b"),
              ("Qwen",  "Qwen/Qwen3-8B-Base", None)]
_mf = os.environ.get("MODELS_FILTER")
if _mf:
    MODELS = [m for m in MODELS if m[0] in set(_mf.split(","))]
GKW = {"days": dict(graph_type="ring", ring_size=7, word_set="days"),
       "square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "24"))
WLEN = int(os.environ.get("WLEN", "320"))
QK_THRESH = float(os.environ.get("QK_THRESH", "0.2"))
CENTERS = [int(x) for x in os.environ.get("CENTERS", "10,20,40,80,150,250").split(",")]
WINDOW = 0.3
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/context_traj")
RNG = np.random.default_rng(0)


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def best2d_rsa(H, Gc, GD, iu):
    if np.isnan(H).any():
        return float("nan")
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    P = Hc @ (Vt[:k].T @ W)
    return sp(np.linalg.norm(P[:, None] - P[None], axis=2)[iu], GD)


@torch.no_grad()
def run(model, tok, blocks, cm, walks, graph, cand_t, dev, cap_layers, centers, ablate_heads):
    n = graph.n_nodes; nb = len(centers)
    handles = []
    by_layer = {}
    for (l, h) in ablate_heads:
        by_layer.setdefault(l, []).append(h)
    for l, hs in by_layer.items():
        proj, hdim = attn_proj(blocks[l], cm)
        cols = torch.tensor(np.concatenate([np.arange(h * hdim, (h + 1) * hdim) for h in hs]), device=dev)
        def pre(_m, args, cols=cols):
            x = args[0].clone(); x[..., cols] = 0; return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    for L in cap_layers:
        handles.append(blocks[L].register_forward_hook(mk(L)))
    nsum = {L: np.zeros((nb, n, cm.hidden_size)) for L in cap_layers}
    ncnt = {L: np.zeros((nb, n)) for L in cap_layers}
    massc = np.zeros((nb, 2))
    lo = [c * (1 - WINDOW) for c in centers]; hi = [c * (1 + WINDOW) for c in centers]
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            logits = model(input_ids=ids).logits[0]
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            caprows = {L: grabbed[L][0][single].float().cpu().numpy() for L in cap_layers}
            for s in range(len(nodes)):
                for bi in range(nb):
                    if lo[bi] <= cl[s] <= hi[bi]:
                        for L in cap_layers:
                            nsum[L][bi, nodes[s]] += caprows[L][s]; ncnt[L][bi, nodes[s]] += 1
                        if s < len(nodes) - 1:
                            p = torch.softmax(logits[spans[s + 1][0] - 1][cand_t].float(), 0).cpu().numpy()
                            massc[bi, 0] += float(p[graph.neighbors(nodes[s])].sum()); massc[bi, 1] += 1
    finally:
        for hnd in handles:
            hnd.remove()
    means = {L: np.where(ncnt[L][:, :, None] > 0, nsum[L] / np.maximum(ncnt[L][:, :, None], 1), np.nan) for L in cap_layers}
    mass = [float(massc[bi, 0] / massc[bi, 1]) if massc[bi, 1] else float("nan") for bi in range(nb)]
    return means, mass


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "centers": CENTERS, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes
        iu = np.triu_indices(n, 1); GD = graph.distance_matrix()[iu]; Gc = np.array(graph.coords, float)
        walks = G.generate_walks(graph, cfg)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)

        # find L* from clean high-context node means
        gen = np.array(ind.get(tag, {}).get("generic", np.zeros((nL, nH))))
        qk_heads = [(int(y), int(x)) for y, x in zip(*np.where(gen > QK_THRESH))]
        allh = [(l, h) for l in range(nL) for h in range(nH)]
        rest = [x for x in allh if x not in set(qk_heads)]
        rand_heads = [rest[i] for i in RNG.choice(len(rest), min(len(qk_heads), len(rest)), replace=False)]
        print(f"[{tag}] ablating {len(qk_heads)} QK heads (QK>{QK_THRESH})", flush=True)

        means_clean, mass_clean = run(model, tok, blocks, cm, walks, graph, cand_t, dev, list(range(nL)), CENTERS, [])
        # L* = layer with max best-2D RSA at the largest context bin
        bi_last = len(CENTERS) - 1
        rsa_layer = {L: best2d_rsa(means_clean[L][bi_last], Gc, GD, iu) for L in range(nL)}
        Lstar = max(rsa_layer, key=lambda L: (rsa_layer[L] if rsa_layer[L] == rsa_layer[L] else -9))
        print(f"[{tag}] L*={Lstar}", flush=True)

        def traj(means, mass):
            return {"rsa": [best2d_rsa(means[Lstar][bi], Gc, GD, iu) for bi in range(len(CENTERS))], "mass": mass}
        rec = {"Lstar": Lstar, "n_qk_ablated": len(qk_heads), "clean": traj(means_clean, mass_clean)}
        for cname, heads in [("ablate_qk", qk_heads), ("ablate_random", rand_heads)]:
            m2, mass2 = run(model, tok, blocks, cm, walks, graph, cand_t, dev, [Lstar], CENTERS, heads)
            rec[cname] = traj(m2, mass2)
        out["models"][tag] = rec
        print(f"[{tag}] clean RSA traj: {[round(x,2) for x in rec['clean']['rsa']]}", flush=True)
        print(f"[{tag}] QK-abl RSA traj: {[round(x,2) for x in rec['ablate_qk']['rsa']]}", flush=True)
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    prev = f"{OUTDIR}/context_rsa_ablation_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/context_rsa_ablation_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    C = out["centers"]; colors = {"clean": "k", "ablate_qk": "tab:red", "ablate_random": "tab:blue"}
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(2, len(models), figsize=(5 * len(models), 8), squeeze=False)
        for j, m in enumerate(models):
            r = out["models"][m]
            for cname, c in colors.items():
                if cname in r:
                    ax[0, j].plot(C, r[cname]["rsa"], "-o", ms=4, color=c, label=cname)
                    ax[1, j].plot(C, r[cname]["mass"], "-o", ms=4, color=c, label=cname)
            ax[0, j].set_title(f"{m}  best-2D RSA vs context (L*={r['Lstar']})", fontsize=9)
            ax[0, j].set_xscale("log"); ax[0, j].set_xlabel("context length"); ax[0, j].set_ylabel("best-2D RSA"); ax[0, j].legend(fontsize=7)
            ax[1, j].set_title(f"{m}  next-step neighbour mass vs context", fontsize=9)
            ax[1, j].set_xscale("log"); ax[1, j].set_xlabel("context length"); ax[1, j].set_ylabel("neighbour mass"); ax[1, j].set_ylim(0, 1.05)
        fig.suptitle(f"[{out['graph']}] geometry (top) & behaviour (bottom) EMERGENCE vs context — "
                     "clean / ablate-QK / ablate-random", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
