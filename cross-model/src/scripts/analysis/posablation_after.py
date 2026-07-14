"""Ablate a head-set ONLY in early context (ctx < SPLIT), then read best-2D RSA in narrow
context WINDOWS just after the ablation boundary and at increasing distance, to see whether
the geometry is transiently damaged right after ablation and how fast it recovers.

Windows (absolute ctx): [SPLIT, SPLIT+10) [+10,+25) [+25,+50) [+50,+100) [+100, WLEN].
Conditions: clean, ablate_qk, ablate_dla, ablate_rsa, ablate_random (top-K each).

Env: PRESET MODELS_FILTER GRAPHS(square_grid,days) NWALKS(32) WLEN(400) SPLIT(100) KTOP(15)
     INDJSON DLAJSON RSAJSON OUTDIR DEVICE
Out: <OUTDIR>/posablation_after.json + .pdf
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
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
GRAPHS = os.environ.get("GRAPHS", "square_grid,days").split(",")
NWALKS = int(os.environ.get("NWALKS", "32"))
WLEN = int(os.environ.get("WLEN", "400"))
SPLIT = int(os.environ.get("SPLIT", "100"))
KTOP = int(os.environ.get("KTOP", "15"))
OFFS = [int(x) for x in os.environ.get("OFFS", "0,15,40,80,160").split(",")]   # window edges relative to SPLIT
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
DLAJSON = os.environ.get("DLAJSON", "/workspace/cross-model/runs/induction-head/attribution/head_attribution_square_grid.json")
RSAJSON = os.environ.get("RSAJSON", "/workspace/cross-model/runs/induction-head/patch_swap/patch_swap_metrics_12_15.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/posablation_after")
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


def topk(mat, K):
    m = np.array(mat); flat = np.argsort(m, axis=None)[::-1][:K]
    return [(int(i // m.shape[1]), int(i % m.shape[1])) for i in flat]


def win_edges():
    lo = [SPLIT + o for o in OFFS]; hi = [SPLIT + o for o in OFFS[1:]] + [WLEN + 1]
    return list(zip(lo, hi))


@torch.no_grad()
def run(model, tok, blocks, cm, walks, graph, dev, ablate_by_layer, Gc, GD, iu):
    n = graph.n_nodes; nL = cm.num_hidden_layers; wins = win_edges(); nW = len(wins)
    handles = []; state = {"cut": 0}
    for L, heads in ablate_by_layer.items():
        proj, hd = attn_proj(blocks[L], cm)
        cols = torch.tensor(np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in heads]), device=dev, dtype=torch.long)
        def pre(_m, args, cols=cols):
            if state["cut"] > 0:
                x = args[0].clone(); x[0, :state["cut"], cols] = 0; return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    for L in range(nL):
        handles.append(blocks[L].register_forward_hook(mk(L)))
    nsum = {L: np.zeros((nW, n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros((nW, n))
    try:
        for wk in walks:
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1)
            state["cut"] = spans[SPLIT][0] if SPLIT < len(spans) else 0
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            grabbed.clear(); model(input_ids=ids)
            for s in range(len(nodes)):
                c = cl[s]
                wi = next((k for k, (lo, hi) in enumerate(wins) if lo <= c < hi), None)
                if wi is None:
                    continue
                for L in range(nL):
                    nsum[L][wi, nodes[s]] += grabbed[L][0, spans[s][-1]].float().cpu().numpy()
                ncnt[wi, nodes[s]] += 1
    finally:
        for h in handles: h.remove()
    res = []
    for wi in range(nW):
        cn = np.maximum(ncnt[wi], 1)
        best = max(best2d_rsa(nsum[L][wi] / cn[:, None] if (ncnt[wi] > 0).all() else np.full((n, cm.hidden_size), np.nan), Gc, GD, iu)
                   for L in range(nL))
        res.append(best)
    return {"windows": [f"{lo}-{hi if hi <= WLEN else 'end'}" for lo, hi in wins], "rsa": res,
            "min_count": [int(ncnt[wi].min()) for wi in range(nW)]}


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    dla = json.load(open(DLAJSON))["models"] if os.path.exists(DLAJSON) else {}
    rsaj = json.load(open(RSAJSON))["models"] if os.path.exists(RSAJSON) else {}
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"split": SPLIT, "models": {}}
    for tag, hf, mirror in MODELS:
        print(f"[{tag}] loading", flush=True)
        base = replace(get_config("gemma_qwen"), n_walks=NWALKS, walk_length=WLEN, device=dev)
        model, tok = load_with_fallback(tag, hf, mirror, replace(base, **GKW[GRAPHS[0]]))
        cm = model.config; blocks = M._decoder_blocks(model)
        nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
        qk = topk(ind.get(tag, {}).get("generic", np.zeros((nL, nH))), KTOP)
        dl = topk(dla.get(tag, {}).get("head_attr", np.zeros((nL, nH))), KTOP)
        rs = topk(rsaj.get(tag, {}).get("restore_rsa", np.zeros((nL, nH))), KTOP)
        used = set(qk) | set(dl) | set(rs)
        pool = [(l, h) for l in range(nL) for h in range(nH) if (l, h) not in used]
        rnd = [pool[i] for i in RNG.choice(len(pool), KTOP, replace=False)]
        def by_layer(hs):
            d = {}
            for l, h in hs: d.setdefault(l, []).append(h)
            return d
        sets = {"clean": {}, "ablate_rsa": by_layer(rs), "ablate_qk": by_layer(qk),
                "ablate_dla": by_layer(dl), "ablate_random": by_layer(rnd)}
        rec = {"graphs": {}}
        for gname in GRAPHS:
            cfg = replace(base, **GKW[gname]); graph = G.build_graph(cfg); n = graph.n_nodes
            iu = np.triu_indices(n, 1); GD = graph.distance_matrix()[iu]; Gc = np.array(graph.coords, float)
            walks = G.generate_walks(graph, cfg)
            rec["graphs"][gname] = {c: run(model, tok, blocks, cm, walks, graph, dev, abl, Gc, GD, iu) for c, abl in sets.items()}
            r = rec["graphs"][gname]
            print(f"[{tag}/{gname}] windows {r['clean']['windows']}", flush=True)
            for c in sets:
                print(f"    {c:14} RSA {[round(x,2) for x in r[c]['rsa']]}", flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/posablation_after.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/posablation_after.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    conds = ["clean", "ablate_rsa", "ablate_qk", "ablate_dla", "ablate_random"]
    cols = {"clean": "k", "ablate_rsa": "tab:purple", "ablate_qk": "tab:green", "ablate_dla": "tab:red", "ablate_random": "tab:blue"}
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]; graphs = list(r["graphs"])
            fig, ax = plt.subplots(1, len(graphs), figsize=(6.5 * len(graphs), 4.6), squeeze=False)
            for gi, g in enumerate(graphs):
                gg = r["graphs"][g]; wins = gg["clean"]["windows"]; x = np.arange(len(wins))
                for c in conds:
                    ax[0, gi].plot(x, gg[c]["rsa"], "-o", ms=4, color=cols[c], label=c.replace("ablate_", ""),
                                   lw=2 if c == "clean" else 1)
                ax[0, gi].axvline(0, color=".8", lw=.6)
                ax[0, gi].set_xticks(x); ax[0, gi].set_xticklabels(wins, fontsize=7, rotation=30)
                ax[0, gi].set_xlabel(f"context window after ablation (split={out['split']})"); ax[0, gi].set_ylabel("best-2D RSA")
                ax[0, gi].set_title(f"{m} [{g}]", fontsize=9); ax[0, gi].legend(fontsize=7)
            fig.suptitle(f"{m}: ablate head-set early (ctx<{out['split']}); best-2D RSA vs distance AFTER ablation "
                         "(leftmost = tokens right after)", fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
