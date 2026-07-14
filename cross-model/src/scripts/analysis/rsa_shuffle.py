"""Control: does shuffling the word->cell assignment change the RSA? The graph (grid
adjacencies) and walk structure are identical; only which word labels each cell changes.
If the geometry is induced in-context from the walk, RSA should be ~invariant to shuffling.

Compares the default assignment vs NSHUF random permutations (same cell sequences, seeded).

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(16) WLEN(300) CTXLO(100) NSHUF(5) OUTDIR DEVICE
Out: <OUTDIR>/rsa_shuffle_<graph>.json + .pdf
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
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "16"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
NSHUF = int(os.environ.get("NSHUF", "5"))
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/rsa_shuffle")


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


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


def raw_rsa(H, GD, iu):
    Hc = H - H.mean(0)
    return sp(np.linalg.norm(Hc[:, None] - Hc[None], axis=2)[iu], GD)


@torch.no_grad()
def measure(model, tok, blocks, cm, walks, graph, dev, Gc, GD, iu):
    n = graph.n_nodes; nL = cm.num_hidden_layers
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
            model(input_ids=ids)
            for L in range(nL):
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += grabbed[L][0, spans[s][-1]].float().cpu().numpy()
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    means = {L: nsum[L] / cn[:, None] for L in range(nL)}
    b2 = max(best2d_rsa(means[L], Gc, GD, iu) for L in range(nL))
    raw = max(raw_rsa(means[L], GD, iu) for L in range(nL))
    return b2, raw


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes
        iu = np.triu_indices(n, 1); GD = graph.distance_matrix()[iu]; Gc = np.array(graph.coords, float)
        base_words = list(graph.words)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        conds = {}
        # default
        graph.words = list(base_words)
        conds["default"] = measure(model, tok, blocks, cm, G.generate_walks(graph, cfg), graph, dev, Gc, GD, iu)
        # shuffles (same cell sequences via same cfg seed; only word labels permuted)
        for s in range(NSHUF):
            perm = np.random.default_rng(1000 + s).permutation(n)
            graph.words = [base_words[p] for p in perm]
            conds[f"shuffle{s}"] = measure(model, tok, blocks, cm, G.generate_walks(graph, cfg), graph, dev, Gc, GD, iu)
        b2 = {k: v[0] for k, v in conds.items()}; raw = {k: v[1] for k, v in conds.items()}
        sh_b2 = [b2[f"shuffle{s}"] for s in range(NSHUF)]; sh_raw = [raw[f"shuffle{s}"] for s in range(NSHUF)]
        rec = {"best2d": b2, "raw": raw,
               "best2d_default": b2["default"], "best2d_shuffle_mean": float(np.mean(sh_b2)), "best2d_shuffle_std": float(np.std(sh_b2)),
               "raw_default": raw["default"], "raw_shuffle_mean": float(np.mean(sh_raw)), "raw_shuffle_std": float(np.std(sh_raw))}
        out["models"][tag] = rec
        print(f"[{tag}] best-2D RSA: default={b2['default']:.3f}  shuffled={np.mean(sh_b2):.3f}±{np.std(sh_b2):.3f} "
              f"| raw: default={raw['default']:.3f} shuffled={np.mean(sh_raw):.3f}±{np.std(sh_raw):.3f}", flush=True)
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/rsa_shuffle_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/rsa_shuffle_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(models)); w = 0.35
        for k, (metric, a) in enumerate([("best2d", ax[0]), ("raw", ax[1])]):
            dv = [out["models"][m][f"{metric}_default"] for m in models]
            sm = [out["models"][m][f"{metric}_shuffle_mean"] for m in models]
            ss = [out["models"][m][f"{metric}_shuffle_std"] for m in models]
            a.bar(x - w/2, dv, w, label="default", color="tab:blue")
            a.bar(x + w/2, sm, w, yerr=ss, label="shuffled (mean±std)", color="tab:orange", capsize=4)
            a.set_xticks(x); a.set_xticklabels(models); a.set_ylim(0, 1); a.set_ylabel(f"{metric} RSA"); a.legend(fontsize=8)
            a.set_title(f"{metric} RSA: default vs shuffled word->cell assignment", fontsize=9)
        fig.suptitle(f"[{out['graph']}] does shuffling the word->cell names change RSA?", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
