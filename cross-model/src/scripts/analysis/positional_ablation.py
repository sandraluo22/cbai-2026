"""Temporally-resolved ablation: zero a head-set ONLY in the EARLY context (ctx < SPLIT),
leave it active later, then measure the geometry (best-2D RSA) and behaviour on the LATER
tokens (ctx >= SPLIT). Tests whether these heads causally BUILD the map during in-context
accumulation, vs merely coexist with it.

Head-sets: rsa (ΔRSA swap heads), qk (prefix-match), dla (writers), random control -- each
top-K. Compared to clean.

Env: PRESET MODELS_FILTER GRAPHS(square_grid,days) NWALKS(16) WLEN(320) SPLIT(100) KTOP(15)
     INDJSON DLAJSON RSAJSON OUTDIR DEVICE
Out: <OUTDIR>/positional_ablation.json + .pdf
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
NWALKS = int(os.environ.get("NWALKS", "16"))
WLEN = int(os.environ.get("WLEN", "320"))
SPLIT = int(os.environ.get("SPLIT", "100"))
KTOP = int(os.environ.get("KTOP", "15"))
INDJSON = os.environ.get("INDJSON", "/workspace/cross-model/runs/induction-head/induction.json")
DLAJSON = os.environ.get("DLAJSON", "/workspace/cross-model/runs/induction-head/attribution/head_attribution_square_grid.json")
RSAJSON = os.environ.get("RSAJSON", "/workspace/cross-model/runs/induction-head/patch_swap/patch_swap_metrics_12_15.json")
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/positional_ablation")
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
    flat = np.argsort(np.array(mat), axis=None)[::-1][:K]
    nH = np.array(mat).shape[1]
    return [(int(i // nH), int(i % nH)) for i in flat]


@torch.no_grad()
def run(model, tok, blocks, cm, walks, graph, cand_t, dev, ablate_by_layer, Gc, GD, iu):
    n = graph.n_nodes; nL = cm.num_hidden_layers
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
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    massc = np.zeros(2)
    try:
        for wk in walks:
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1)
            state["cut"] = spans[SPLIT][0] if SPLIT < len(spans) else 0     # zero ablated heads before this token
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            grabbed.clear(); logits = model(input_ids=ids).logits[0]
            for L in range(nL):
                hs = grabbed[L][0]
                for s in range(len(nodes)):
                    if cl[s] >= SPLIT:                                       # read geometry on LATE tokens
                        nsum[L][nodes[s]] += hs[spans[s][-1]].float().cpu().numpy()
                        if L == 0: ncnt[nodes[s]] += 1
            for s in range(len(nodes) - 1):
                if cl[s] >= SPLIT:
                    p = torch.softmax(logits[spans[s + 1][0] - 1][cand_t].float(), 0).cpu().numpy()
                    massc[0] += float(p[graph.neighbors(nodes[s])].sum()); massc[1] += 1
    finally:
        for h in handles: h.remove()
    cn = np.maximum(ncnt, 1)
    means = {L: nsum[L] / cn[:, None] for L in range(nL)}
    b2 = max(best2d_rsa(means[L], Gc, GD, iu) for L in range(nL))
    raw = max(sp(np.linalg.norm((means[L]-means[L].mean(0))[:,None]-(means[L]-means[L].mean(0))[None],axis=2)[iu], GD) for L in range(nL))
    return {"best2d": b2, "raw": raw, "nbr_mass": float(massc[0] / max(massc[1], 1))}


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
        def by_layer(heads):
            d = {}
            for l, h in heads: d.setdefault(l, []).append(h)
            return d
        sets = {"clean": {}, "ablate_rsa": by_layer(rs), "ablate_qk": by_layer(qk),
                "ablate_dla": by_layer(dl), "ablate_random": by_layer(rnd)}
        rec = {"heads": {"qk": qk, "dla": dl, "rsa": rs}, "graphs": {}}
        for gname in GRAPHS:
            cfg = replace(base, **GKW[gname]); graph = G.build_graph(cfg); n = graph.n_nodes
            iu = np.triu_indices(n, 1); GD = graph.distance_matrix()[iu]; Gc = np.array(graph.coords, float)
            walks = G.generate_walks(graph, cfg)
            cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
            rec["graphs"][gname] = {}
            for cname, abl in sets.items():
                rec["graphs"][gname][cname] = run(model, tok, blocks, cm, walks, graph, cand_t, dev, abl, Gc, GD, iu)
            r = rec["graphs"][gname]
            print(f"[{tag}/{gname}] best2d: clean={r['clean']['best2d']:.2f} rsa={r['ablate_rsa']['best2d']:.2f} "
                  f"qk={r['ablate_qk']['best2d']:.2f} dla={r['ablate_dla']['best2d']:.2f} rand={r['ablate_random']['best2d']:.2f} "
                  f"| nbr clean={r['clean']['nbr_mass']:.2f} rsa={r['ablate_rsa']['nbr_mass']:.2f} qk={r['ablate_qk']['nbr_mass']:.2f} "
                  f"dla={r['ablate_dla']['nbr_mass']:.2f} rand={r['ablate_random']['nbr_mass']:.2f}", flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/positional_ablation.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/positional_ablation.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    conds = ["clean", "ablate_rsa", "ablate_qk", "ablate_dla", "ablate_random"]
    cols = {"clean": "k", "ablate_rsa": "tab:purple", "ablate_qk": "tab:green", "ablate_dla": "tab:red", "ablate_random": "tab:blue"}
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]; graphs = list(r["graphs"])
            fig, ax = plt.subplots(1, 2 * len(graphs), figsize=(6 * len(graphs), 4.5), squeeze=False)
            for gi, g in enumerate(graphs):
                gg = r["graphs"][g]; x = np.arange(len(conds))
                ax[0, 2*gi].bar(x, [gg[c]["best2d"] for c in conds], color=[cols[c] for c in conds])
                ax[0, 2*gi].set_title(f"{m} [{g}] best-2D RSA (late tokens)", fontsize=8); ax[0, 2*gi].set_ylim(0, 1)
                ax[0, 2*gi].set_xticks(x); ax[0, 2*gi].set_xticklabels([c.replace("ablate_", "") for c in conds], rotation=45, fontsize=7)
                ax[0, 2*gi+1].bar(x, [gg[c]["nbr_mass"] for c in conds], color=[cols[c] for c in conds])
                ax[0, 2*gi+1].set_title(f"{m} [{g}] neighbour mass (late)", fontsize=8); ax[0, 2*gi+1].set_ylim(0, 1)
                ax[0, 2*gi+1].set_xticks(x); ax[0, 2*gi+1].set_xticklabels([c.replace("ablate_", "") for c in conds], rotation=45, fontsize=7)
            fig.suptitle(f"{m}: ablate head-set in EARLY context (ctx<{out['split']}), read geometry/behaviour on LATE tokens", fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
