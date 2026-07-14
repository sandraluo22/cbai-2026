"""Removal / necessity test: project the coord-PROBE subspace OUT of the residual stream at
layer L_rem, then measure whether downstream geometry and behaviour survive. This is the
complement of steer_probe (which pushes ALONG the probe direction) -- here we delete it.

Conditions:
  clean          -- no intervention
  remove_probe   -- project out the 2-D probe readout subspace at L_rem (all positions)
  remove_random  -- project out a random rank-2 subspace at L_rem (rank-matched control)

Measured at layers downstream of L_rem:
  coord-probe LOO R² per layer   (does the geometry re-form / survive?)
  next-step neighbour mass        (does behaviour survive?)
If remove_probe tanks geometry+behaviour while remove_random does not, the probe subspace is
causally necessary (not an epiphenomenal readout).

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(20) WLEN(300) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/removal_probe_<graph>.json + .pdf
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
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "20"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/removal_probe")
CKPTS = [20, 60, 150, 250]
RNG = np.random.default_rng(0)


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def coord_loo_r2(H, coords):
    n = H.shape[0]
    if np.isnan(H).any(): return float("nan")
    mu = H.mean(0); sd = H.std(0) + 1e-6; Xs = (H - mu) / sd; Yc = coords - coords.mean(0)
    folds = []
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        U, S, Vt = np.linalg.svd(Xs[idx], full_matrices=False)
        folds.append((np.array(idx), (Xs[k] @ Vt.T), U.T.copy(), S))
    best = -9.0
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for k, (idx, proj, UT, S) in enumerate(folds):
            ytr = Yc[idx]; ymu = ytr.mean(0)
            pred[k] = proj @ ((S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))) + ymu
        best = max(best, 0.5 * (_r2(Yc[:, 0], pred[:, 0]) + _r2(Yc[:, 1], pred[:, 1])))
    return float(best)


def probe_basis(H, coords):
    """LOO-ridge coord-probe readout dirs in raw activation space (d x 2) + LOO R²."""
    n, d = H.shape
    mu = H.mean(0); sd = H.std(0) + 1e-6; Xs = (H - mu) / sd; Yc = coords - coords.mean(0)
    folds = []
    for kf in range(n):
        idx = [i for i in range(n) if i != kf]
        U, S, Vt = np.linalg.svd(Xs[idx], full_matrices=False)
        folds.append((np.array(idx), (Xs[kf] @ Vt.T), U.T.copy(), S))
    best = (-9.0, ALPHAS[0])
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for kf, (idx, proj, UT, S) in enumerate(folds):
            ytr = Yc[idx]; ymu = ytr.mean(0)
            pred[kf] = proj @ ((S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))) + ymu
        sc = 0.5 * (_r2(Yc[:, 0], pred[:, 0]) + _r2(Yc[:, 1], pred[:, 1]))
        if sc > best[0]: best = (sc, a)
    a = best[1]
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    coef_std = Vt.T @ ((S / (S ** 2 + a))[:, None] * (U.T @ Yc))
    Q, _ = np.linalg.qr(coef_std / sd[:, None])                     # d x 2 orthonormal
    return Q, float(best[0])


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
            model(input_ids=ids); single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(cm.num_hidden_layers):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(cm.num_hidden_layers)}


@torch.no_grad()
def run_condition(model, tok, blocks, cm, walks, graph, cand_t, dev, L_rem, Q, cap_layers):
    """Project subspace Q (d x2 orthonormal, or None) out of the residual at L_rem for ALL
    positions; capture downstream node-means (cap_layers) + next-step neighbour behaviour."""
    n = graph.n_nodes; handles = []
    if Q is not None:
        Qt = torch.tensor(Q, device=dev, dtype=torch.float32)
        def rem(_m, _i, out):
            h = (out[0] if isinstance(out, tuple) else out)
            hf = h.float()
            hf = hf - (hf @ Qt) @ Qt.T
            hf = hf.to(h.dtype)
            return (hf,) + tuple(out[1:]) if isinstance(out, tuple) else hf
        handles.append(blocks[L_rem].register_forward_hook(rem))
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    for L in cap_layers:
        handles.append(blocks[L].register_forward_hook(mk(L)))
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in cap_layers}; ncnt = {L: np.zeros(n) for L in cap_layers}
    acc = {C: {"mass": 0.0, "total": 0} for C in CKPTS}
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            logits = model(input_ids=ids).logits[0]
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in cap_layers:
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]; ncnt[L][nodes[s]] += 1
            for C in CKPTS:
                s = C - 1
                if 0 <= s <= len(nodes) - 2:
                    p = torch.softmax(logits[spans[s + 1][0] - 1][cand_t].float(), 0).cpu().numpy()
                    acc[C]["mass"] += float(p[graph.neighbors(nodes[s])].sum()); acc[C]["total"] += 1
    finally:
        for h in handles: h.remove()
    coords = np.array(graph.coords, float)
    coordp = {}
    for L in cap_layers:
        H = np.where(ncnt[L][:, None] > 0, nsum[L] / np.maximum(ncnt[L][:, None], 1), np.nan)
        coordp[L] = coord_loo_r2(H, coords)
    beh = {C: (acc[C]["mass"] / acc[C]["total"] if acc[C]["total"] else float("nan")) for C in CKPTS}
    return coordp, beh


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
        walks = G.generate_walks(graph, cfg)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
        cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
        means = node_means_all(model, tok, blocks, cm, walks, dev, n)
        r2byL = {L: coord_loo_r2(means[L], coords) for L in range(nL)}
        # remove at the peak-geometry layer within first 85% depth (near the true peak, still
        # leaving a few downstream layers to test whether the map re-forms)
        cand = [L for L in range(nL) if L <= 0.85 * nL] or list(range(nL))
        L_rem = max(cand, key=lambda L: r2byL[L])
        Qp, pr2 = probe_basis(means[L_rem], coords)
        Qr, _ = np.linalg.qr(RNG.standard_normal((cm.hidden_size, 2)))       # rank-2 random control
        cap_layers = sorted(set(int(round(x)) for x in np.linspace(L_rem, nL - 1, 8)))
        print(f"[{tag}] L_rem={L_rem} probeR²={pr2:.2f} cap={cap_layers}", flush=True)
        rec = {"L_rem": int(L_rem), "probe_r2": pr2, "cap_layers": cap_layers, "conds": {}}
        for cname, Q in [("clean", None), ("remove_probe", Qp), ("remove_random", Qr)]:
            coordp, beh = run_condition(model, tok, blocks, cm, walks, graph, cand_t, dev, L_rem, Q, cap_layers)
            rec["conds"][cname] = {"coordprobe_by_layer": {str(k): v for k, v in coordp.items()},
                                   "neighbor_mass": {str(k): v for k, v in beh.items()}}
            pk = max((v for v in coordp.values() if np.isfinite(v)), default=float("nan"))
            print(f"[{tag}/{GRAPH}/{cname}] downstream peak coordProbeR²={pk:+.2f} nbr_mass@250={beh[250]:.2f}", flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/removal_probe_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/removal_probe_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    colors = {"clean": "k", "remove_probe": "tab:red", "remove_random": "tab:blue"}
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(2, len(models), figsize=(5.2 * len(models), 8.4), squeeze=False)
        for j, m in enumerate(models):
            r = out["models"][m]
            for cname, c in colors.items():
                cd = r["conds"][cname]
                Ls = sorted(int(k) for k in cd["coordprobe_by_layer"])
                ax[0, j].plot(Ls, [cd["coordprobe_by_layer"][str(L)] for L in Ls], "-o", ms=3, color=c, label=cname)
                Cs = sorted(int(k) for k in cd["neighbor_mass"])
                ax[1, j].plot(Cs, [cd["neighbor_mass"][str(C)] for C in Cs], "-o", ms=3, color=c, label=cname)
            ax[0, j].axvline(r["L_rem"], color=".6", ls=":", lw=1)
            ax[0, j].set_title(f"{m}  downstream coord-probe R² (L_rem={r['L_rem']})", fontsize=8)
            ax[0, j].set_xlabel("layer"); ax[0, j].set_ylabel("coord-probe R²"); ax[0, j].axhline(0, color=".7", lw=.6); ax[0, j].set_ylim(-0.6, 1.0); ax[0, j].legend(fontsize=6)
            ax[1, j].set_title(f"{m}  next-step neighbour mass", fontsize=8)
            ax[1, j].set_xlabel("context length"); ax[1, j].set_ylabel("neighbour mass"); ax[1, j].set_ylim(0, 1.05); ax[1, j].legend(fontsize=6)
        fig.suptitle(f"[{out['graph']}] REMOVAL: project probe subspace out at L_rem -> downstream geometry & behaviour\n"
                     "black=clean, red=remove probe subspace, blue=remove random rank-2 (control)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
