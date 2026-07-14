"""Causal test of the COORD-PROBE directions: if we push the residual stream along the
linear direction the probe reads a coordinate off of, do the model's next-token logits
move toward nodes with that coordinate?

At the peak-geometry layer L* we fit the leave-one-node-out coordinate probe (as in
coord_decode.py) and recover, per axis a in {row/col or cos/sin}, the readout direction
w_a in RAW activation space (decoded coord_a = (x-mu)·w_a). We steer the residual at every
next-step prediction position by ±s·dev·û_a (dev = typical node-mean deviation norm, s a
dose), and read the mean candidate-node distribution P. The summary statistic is the
expected coordinate of the prediction  E_a(P) = Σ_v P[v]·coord_a(v).

  dE_a(s) = E_a(P[+s·û_a]) − E_a(P[−s·û_a])        # does the probe axis move outputs?
Compared against a norm-matched RANDOM direction (should give ~0), and cross-axis leakage
E_{1-a} under axis-a steering (specificity). If dE_a > 0 and grows with dose while the
random control stays flat, the probe direction is causal on downstream logits.

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(12) WLEN(300) CTXLO(100)
     SCALES(1,2,4,8) OUTDIR DEVICE
Out: <OUTDIR>/steer_probe_<graph>.json + .pdf
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
NWALKS = int(os.environ.get("NWALKS", "12"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
SCALES = [float(x) for x in os.environ.get("SCALES", "1,2,4,8").split(",")]
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/steer_probe")
RNG = np.random.default_rng(0)


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


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


def probe_dirs(H, coords):
    """Fit LOO coord probe on node-means H (n x d); pick alpha by mean-axis LOO R²; return
    unit readout directions in RAW activation space (d x 2), the LOO R², and alpha."""
    n, d = H.shape
    mu = H.mean(0); sd = H.std(0) + 1e-6; Xs = (H - mu) / sd
    Yc = coords - coords.mean(0)
    folds = []
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        U, S, Vt = np.linalg.svd(Xs[idx], full_matrices=False)
        folds.append((np.array(idx), (Xs[k] @ Vt.T), U.T.copy(), S))
    best = (-9.0, ALPHAS[0])
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for k, (idx, proj, UT, S) in enumerate(folds):
            ytr = Yc[idx]; ymu = ytr.mean(0)
            pred[k] = proj @ ((S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))) + ymu
        sc = 0.5 * (_r2(Yc[:, 0], pred[:, 0]) + _r2(Yc[:, 1], pred[:, 1]))
        if sc > best[0]: best = (sc, a)
    a = best[1]
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    coef_std = Vt.T @ ((S / (S ** 2 + a))[:, None] * (U.T @ Yc))     # d x 2 (standardized space)
    W = coef_std / sd[:, None]                                       # raw-space readout, d x 2
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)       # unit directions
    return Wn, float(best[0]), float(a)


@torch.no_grad()
def measure(model, tok, blocks, cm, walks, cand_t, dev, L_steer, steervec):
    """Mean candidate-node softmax over all next-step prediction positions (ctx>=CTXLO),
    optionally adding steervec to the residual at those positions at layer L_steer."""
    n = len(cand_t); st = {"pos": None}
    v = None if steervec is None else torch.tensor(steervec, device=dev)
    def shook(_m, _i, out):
        if st["pos"] is not None:
            h = (out[0] if isinstance(out, tuple) else out).clone()
            h[0, st["pos"], :] += v.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    handle = blocks[L_steer].register_forward_hook(shook) if steervec is not None else None
    Psum = np.zeros(n); cnt = 0
    try:
        for wk in walks:
            nodes = wk.nodes; spans = resolve_token_spans(tok, wk); cl = np.arange(1, len(nodes) + 1)
            pos = [spans[s][-1] for s in range(len(nodes) - 1) if cl[s] >= CTXLO]
            if not pos: continue
            st["pos"] = pos
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            logits = model(input_ids=ids).logits[0]
            for p in pos:
                Psum += torch.softmax(logits[p][cand_t].float(), 0).cpu().numpy(); cnt += 1
    finally:
        if handle is not None: handle.remove()
    return Psum / max(cnt, 1)


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "scales": SCALES, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes
        coords = np.array(graph.coords, float); axes = ("axis0", "axis1")
        iu = np.triu_indices(n, 1); GD = graph.distance_matrix()[iu]
        walks = G.generate_walks(graph, cfg)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
        cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
        means = node_means_all(model, tok, blocks, cm, walks, dev, n)
        # choose L*: peak coord-probe R² in the first ~80% of depth (leaves readout downstream)
        r2byL = {}
        for L in range(nL):
            _, r2L, _ = probe_dirs(means[L], coords); r2byL[L] = r2L
        cand = [L for L in range(nL) if L <= 0.8 * nL] or list(range(nL))
        L_steer = max(cand, key=lambda L: r2byL[L])
        U2, probe_r2, alpha = probe_dirs(means[L_steer], coords)         # d x 2 unit dirs
        H = means[L_steer]; dev_norm = float(np.mean(np.linalg.norm(H - H.mean(0), axis=1)))
        print(f"[{tag}] L*={L_steer} probeR²={probe_r2:.2f} alpha={alpha:g} dev={dev_norm:.1f}", flush=True)

        P0 = measure(model, tok, blocks, cm, walks, cand_t, dev, L_steer, None)
        def E(P, a): return float((P * coords[:, a]).sum() / P.sum())
        rec = {"L_steer": int(L_steer), "probe_r2": probe_r2, "alpha": alpha, "dev_norm": dev_norm,
               "axes": list(axes), "E0": [E(P0, 0), E(P0, 1)], "coord_range": [
                   [float(coords[:, 0].min()), float(coords[:, 0].max())],
                   [float(coords[:, 1].min()), float(coords[:, 1].max())]],
               "by_scale": {}}
        rdir = RNG.standard_normal(H.shape[1]); rdir /= np.linalg.norm(rdir)  # norm-matched random control
        for s in SCALES:
            amp = s * dev_norm
            entry = {"probe": {}, "random": {}}
            # probe axes
            dE = [[0.0, 0.0], [0.0, 0.0]]                 # dE[a][b] = shift in E_b under axis-a steering
            for a in (0, 1):
                u = U2[:, a]
                Pp = measure(model, tok, blocks, cm, walks, cand_t, dev, L_steer, +amp * u)
                Pm = measure(model, tok, blocks, cm, walks, cand_t, dev, L_steer, -amp * u)
                for b in (0, 1):
                    dE[a][b] = E(Pp, b) - E(Pm, b)
            entry["probe"] = {"dE": dE}                   # diagonal = on-axis effect, off-diag = leakage
            # random control (axis-agnostic direction, same amplitude)
            Pp = measure(model, tok, blocks, cm, walks, cand_t, dev, L_steer, +amp * rdir)
            Pm = measure(model, tok, blocks, cm, walks, cand_t, dev, L_steer, -amp * rdir)
            entry["random"] = {"dE": [E(Pp, 0) - E(Pm, 0), E(Pp, 1) - E(Pm, 1)]}
            rec["by_scale"][str(s)] = entry
            print(f"[{tag}/{GRAPH}] s={s:g} probe dE(on-axis)=[{dE[0][0]:+.3f},{dE[1][1]:+.3f}] "
                  f"leak=[{dE[0][1]:+.3f},{dE[1][0]:+.3f}] random=[{entry['random']['dE'][0]:+.3f},{entry['random']['dE'][1]:+.3f}]",
                  flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/steer_probe_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/steer_probe_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    scales = out["scales"]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(2, len(models), figsize=(5.2 * len(models), 8.4), squeeze=False)
        for j, m in enumerate(models):
            r = out["models"][m]; bs = r["by_scale"]
            for a in (0, 1):
                on = [bs[str(s)]["probe"]["dE"][a][a] for s in scales]
                leak = [bs[str(s)]["probe"]["dE"][a][1 - a] for s in scales]
                rnd = [bs[str(s)]["random"]["dE"][a] for s in scales]
                ax[a, j].plot(scales, on, "-o", color="tab:red", label="probe axis (on-axis)")
                ax[a, j].plot(scales, leak, "-o", color="tab:orange", label="probe (cross-axis leak)")
                ax[a, j].plot(scales, rnd, "-o", color="tab:blue", label="random dir")
                ax[a, j].axhline(0, color=".7", lw=.6)
                ax[a, j].set_xlabel("steer dose (×node-dev norm)")
                ax[a, j].set_ylabel(f"Δ E[{r['axes'][a]}] of prediction")
                ax[a, j].set_title(f"{m}  axis {a} (L*={r['L_steer']}, probeR²={r['probe_r2']:.2f})", fontsize=8)
                ax[a, j].legend(fontsize=6)
        fig.suptitle(f"[{out['graph']}] steering along COORD-PROBE readout directions → shift in expected coordinate of prediction\n"
                     "red = steer along probe axis (should move that coordinate), blue = norm-matched random (should stay flat)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
