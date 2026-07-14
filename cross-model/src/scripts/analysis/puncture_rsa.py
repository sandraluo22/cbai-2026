"""Remove a node from the grid; measure the model's in-context geometry against TWO ground
truths:
  GT_full : distances as if the grid were COMPLETE (Manhattan from grid coords) -> nodes keep
            their original grid positions.
  GT_punc : geodesic on the actual PUNCTURED graph (shortest paths detour around the hole).

Reports RSA(GT_full, GT_punc) (how distinguishable the two ground truths are), and the model's
raw + best-2D RSA vs each. If raw RSA(model, GT_punc) > raw RSA(model, GT_full), the model built
the true traversable graph; if GT_full wins, it retained ambient grid positions.

Env: PRESET MODELS_FILTER REMOVE("1,0,5") NWALKS(16) WLEN(300) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/puncture_rsa.json + .pdf   (REMOVE nodes are indices into the 4x4 grid)
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
from graph import Graph
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
REMOVE = [int(x) for x in os.environ.get("REMOVE", "1,0,5").split(",")]
NWALKS = int(os.environ.get("NWALKS", "16"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/puncture_rsa")


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def best2d_rsa(H, Gc, GDu, iu):
    if np.isnan(H).any():
        return float("nan")
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    P = Hc @ (Vt[:k].T @ W)
    return sp(np.linalg.norm(P[:, None] - P[None], axis=2)[iu], GDu)


def raw_rsa(H, GDu, iu):
    Hc = H - H.mean(0)
    return sp(np.linalg.norm(Hc[:, None] - Hc[None], axis=2)[iu], GDu)


def build_punctured(fg, k):
    keep = [i for i in range(fg.n_nodes) if i != k]
    idx = {old: new for new, old in enumerate(keep)}
    adj = [[idx[v] for v in fg.adjacency[old] if v != k] for old in keep]
    coords = [fg.coords[old] for old in keep]; words = [fg.words[old] for old in keep]
    return Graph(n_nodes=len(keep), words=words, adjacency=adj, coords=coords)


@torch.no_grad()
def geometry(model, tok, blocks, cm, walks, n, dev):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(cm.num_hidden_layers)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(cm.num_hidden_layers)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
            model(input_ids=ids)
            for L in range(cm.num_hidden_layers):
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += grabbed[L][0, spans[s][-1]].float().cpu().numpy()
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(cm.num_hidden_layers)}


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"remove": REMOVE, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=4, grid_cols=4,
                      n_walks=NWALKS, walk_length=WLEN, device=dev)
        full = G.build_graph(cfg)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        rec = {}
        for k in REMOVE:
            pg = build_punctured(full, k); n = pg.n_nodes
            iu = np.triu_indices(n, 1)
            Gc = np.array(pg.coords, float)
            GT_full = (np.abs(Gc[:, None, 0] - Gc[None, :, 0]) + np.abs(Gc[:, None, 1] - Gc[None, :, 1]))[iu]  # complete-grid geodesic
            GT_punc = pg.distance_matrix()[iu]                                                                 # detours around hole
            rsa_gt = sp(GT_full, GT_punc)
            deg = len(full.adjacency[k])
            walks = G.generate_walks(pg, cfg)
            means = geometry(model, tok, blocks, cm, walks, n, dev)
            raw_full = max(raw_rsa(means[L], GT_full, iu) for L in range(cm.num_hidden_layers))
            raw_punc = max(raw_rsa(means[L], GT_punc, iu) for L in range(cm.num_hidden_layers))
            b2_full = max(best2d_rsa(means[L], Gc, GT_full, iu) for L in range(cm.num_hidden_layers))
            b2_punc = max(best2d_rsa(means[L], Gc, GT_punc, iu) for L in range(cm.num_hidden_layers))
            rec[str(k)] = {"degree": deg, "n_pairs_differ": int((GT_full != GT_punc).sum() // 2),
                           "rsa_gt_full_vs_punc": rsa_gt,
                           "raw_full": raw_full, "raw_punc": raw_punc, "best2d_full": b2_full, "best2d_punc": b2_punc}
            print(f"[{tag}] remove node {k} (deg {deg}): RSA(GT_full,GT_punc)={rsa_gt:.3f} | "
                  f"raw model vs full={raw_full:.3f} punc={raw_punc:.3f} | best2d vs full={b2_full:.3f} punc={b2_punc:.3f}", flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/puncture_rsa.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/puncture_rsa.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    ks = [str(k) for k in out["remove"]]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.8), squeeze=False)
        for j, m in enumerate(models):
            r = out["models"][m]; x = np.arange(len(ks)); w = 0.2
            ax[0, j].bar(x - 1.5*w, [r[k]["raw_full"] for k in ks], w, label="raw vs GT_full", color="tab:blue")
            ax[0, j].bar(x - 0.5*w, [r[k]["raw_punc"] for k in ks], w, label="raw vs GT_punc", color="tab:cyan")
            ax[0, j].bar(x + 0.5*w, [r[k]["best2d_full"] for k in ks], w, label="best2d vs GT_full", color="tab:red")
            ax[0, j].bar(x + 1.5*w, [r[k]["best2d_punc"] for k in ks], w, label="best2d vs GT_punc", color="tab:orange")
            for i, k in enumerate(ks):
                ax[0, j].annotate(f"GTsim={r[k]['rsa_gt_full_vs_punc']:.2f}", (i, 1.01), ha="center", fontsize=6)
            ax[0, j].set_xticks(x); ax[0, j].set_xticklabels([f"node {k}\n(deg {r[k]['degree']})" for k in ks], fontsize=7)
            ax[0, j].set_ylim(0, 1.08); ax[0, j].set_ylabel("RSA"); ax[0, j].set_title(m, fontsize=10); ax[0, j].legend(fontsize=6)
        fig.suptitle("Remove a node: model geometry vs GT_full (grid positions) vs GT_punc (routes around hole)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
