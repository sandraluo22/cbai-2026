"""Causal read-out test: is the geometry USED for next-step prediction?

Steer node X's representation to node Y's location in the map (at the geometry-peak layer
L*), then read the next-step prediction at X's readout tokens. If the model now predicts
Y's neighbours instead of X's, the behaviour circuit reads the geometry; if unchanged, the
map is epiphenomenal for prediction.

Interventions (add a vector at X's readout positions, layer L*):
  full   : v = mean_L*(Y) - mean_L*(X)                 (move everything X->Y)
  plane  : project v onto the best-2D geometry plane   (move ONLY the map coordinate)
  rand   : random vector of the same norm as `plane`   (control)

Metric: mass on neighbours(X) vs neighbours(Y) at X's readout, clean vs steered, over
far-apart pairs (disjoint neighbourhoods). square_grid.

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(12) WLEN(300) CTXLO(100) ALPHA(1.0)
     PAIRS("0-12,3-15,0-15,5-10") OUTDIR DEVICE
Out: <OUTDIR>/steer_<graph>.json + .pdf
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
NWALKS = int(os.environ.get("NWALKS", "12"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
ALPHA = float(os.environ.get("ALPHA", "1.0"))
PAIRS = [tuple(int(x) for x in p.split("-")) for p in os.environ.get("PAIRS", "0-12,3-15,0-15,5-10").split(",")]
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/steer")
RNG = np.random.default_rng(0)


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def best2d_plane(H, Gc):
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    B = Vt[:k].T @ W
    Q, _ = np.linalg.qr(B)                       # [d,2] orthonormal plane basis
    return Q


@torch.no_grad()
def node_means_all(model, tok, blocks, cm, walks, dev, n):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(cm.num_hidden_layers)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(cm.num_hidden_layers)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(cm.num_hidden_layers):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0:
                            ncnt[nodes[s]] += 1
    finally:
        for h in hs:
            h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(cm.num_hidden_layers)}


@torch.no_grad()
def measure(model, tok, blocks, graph, cand_t, dev, Lstar, walks, X, steervec):
    """At X's readout tokens (ctx>=CTXLO), add steervec at layer L*; return mean P over nodes."""
    n = graph.n_nodes
    state = {"pos": None, "v": None if steervec is None else torch.tensor(steervec, device=dev)}
    def hook(_m, _i, out):
        if state["pos"]:
            hsd = (out[0] if isinstance(out, tuple) else out).clone()
            hsd[0, state["pos"], :] += state["v"].to(hsd.dtype)
            return (hsd,) + tuple(out[1:]) if isinstance(out, tuple) else hsd
    h = blocks[Lstar].register_forward_hook(hook) if steervec is not None else None
    Psum = np.zeros(n); cnt = 0
    try:
        for wk in walks:
            nodes = wk.nodes; spans = resolve_token_spans(tok, wk); cl = np.arange(1, len(nodes) + 1)
            pos = [spans[s + 1][0] - 1 for s in range(len(nodes) - 1) if nodes[s] == X and cl[s] >= CTXLO]
            if not pos:
                continue
            state["pos"] = pos
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            logits = model(input_ids=ids).logits[0]
            for p in pos:
                Psum += torch.softmax(logits[p][cand_t].float(), 0).cpu().numpy(); cnt += 1
    finally:
        if h:
            h.remove()
    return Psum / max(cnt, 1)


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "pairs": PAIRS, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes; Gc = np.array(graph.coords, float)
        iu = np.triu_indices(n, 1); GD = graph.distance_matrix()[iu]
        walks = G.generate_walks(graph, cfg)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)

        means = node_means_all(model, tok, blocks, cm, walks, dev, n)
        best2d = {L: sp(np.linalg.norm((means[L] - means[L].mean(0))[:, None] - (means[L] - means[L].mean(0))[None], axis=2)[iu], GD)
                  for L in range(cm.num_hidden_layers)}
        Lstar = max(best2d, key=best2d.get); H = means[Lstar]; B = best2d_plane(H, Gc)
        print(f"[{tag}] L*={Lstar} best2dRSA={best2d[Lstar]:.2f}", flush=True)

        rec = {"Lstar": Lstar, "pairs": {}}
        for (X, Y) in PAIRS:
            nbrX = graph.neighbors(X); nbrY = graph.neighbors(Y)
            v_full = (H[Y] - H[X]) * ALPHA
            v_plane = (B @ (B.T @ (H[Y] - H[X]))) * ALPHA
            v_rand = RNG.standard_normal(H.shape[1]); v_rand *= np.linalg.norm(v_plane) / (np.linalg.norm(v_rand) + 1e-9)
            # NORM-MATCHED-TO-FULL variants: push ONLY in the geometry plane, but scaled to |v_full|
            # (plane above is norm-limited to the low-variance in-plane component -> unfair sufficiency test).
            fn = np.linalg.norm(v_full)
            vp = B @ (B.T @ (H[Y] - H[X]))
            v_plane_big = vp * (fn / (np.linalg.norm(vp) + 1e-9))
            v_rand_big = RNG.standard_normal(H.shape[1]); v_rand_big *= fn / (np.linalg.norm(v_rand_big) + 1e-9)
            res = {"nbrX": nbrX, "nbrY": nbrY}
            for cname, vec in [("clean", None), ("full", v_full), ("plane", v_plane), ("rand", v_rand),
                               ("plane_big", v_plane_big), ("rand_big", v_rand_big)]:
                P = measure(model, tok, blocks, graph, cand_t, dev, Lstar, walks, X, vec)
                res[cname] = {"mass_nbrX": float(P[nbrX].sum()), "mass_nbrY": float(P[nbrY].sum())}
            rec["pairs"][f"{X}->{Y}"] = res
            print(f"[{tag}] {X}->{Y}: clean(Y={res['clean']['mass_nbrY']:.2f}) "
                  f"| full(Y={res['full']['mass_nbrY']:.2f}) "
                  f"| plane(Y={res['plane']['mass_nbrY']:.2f}) "
                  f"| plane_big(Y={res['plane_big']['mass_nbrY']:.2f}) "
                  f"| rand_big(Y={res['rand_big']['mass_nbrY']:.2f})  [|plane|/|full|={np.linalg.norm(v_plane)/(np.linalg.norm(v_full)+1e-9):.2f}]", flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    prev = f"{OUTDIR}/steer_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/steer_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    conds = ["clean", "full", "plane", "rand"]
    with PdfPages(path) as pdf:
        for m in models:
            r = out["models"][m]; pairs = list(r["pairs"])
            fig, ax = plt.subplots(1, 2, figsize=(13, 5))
            x = np.arange(len(pairs)); w = 0.2
            for i, cn in enumerate(conds):
                ax[0].bar(x + (i - 1.5) * w, [r["pairs"][p][cn]["mass_nbrX"] for p in pairs], w, label=cn)
                ax[1].bar(x + (i - 1.5) * w, [r["pairs"][p][cn]["mass_nbrY"] for p in pairs], w, label=cn)
            ax[0].set_title(f"{m}: mass on X's TRUE neighbours (should DROP if geometry used)", fontsize=9)
            ax[1].set_title(f"{m}: mass on Y's (steered-target) neighbours (should RISE)", fontsize=9)
            for a in ax:
                a.set_xticks(x); a.set_xticklabels(pairs, fontsize=7); a.set_ylim(0, 1.0); a.legend(fontsize=7); a.set_ylabel("prob mass")
            fig.suptitle(f"{m} [{out['graph']}] L*={r['Lstar']}: steer X->Y in the map, read next-step prediction", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
