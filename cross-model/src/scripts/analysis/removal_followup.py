"""Removal follow-ups (two modes).

MODE=alllayers -- project the probe geometry subspace out at EVERY layer (each layer's own
  coord-probe 2-D subspace), not just one, then measure behaviour. Tests whether continuous
  removal (which single-layer removal couldn't) finally breaks next-step prediction.
  Conditions: clean | remove_probe_all | remove_random_all. Reports neighbour mass at context
  checkpoints + per-layer coord-probe R² (should be crushed for remove_probe_all).

MODE=generate -- project the probe subspace out at ONE layer (L_rem) for the CONTEXT tokens
  only (first X steps), then let the model keep GENERATING autoregressively. Track over
  generation time: (1) does geometry re-form in the generated tokens' representations,
  (2) does behaviour (validity = generated step is a true graph neighbour; neighbour logprob)
  drop. Conditions clean | remove_probe | remove_random. Default model Llama (env-settable).

Env: PRESET MODE(alllayers|generate) MODELS_FILTER GRAPH(square_grid) NWALKS(20) WLEN(300)
     CTXLO(100)  [generate:] GEN_MODEL(Llama) XCTX(150) GSTEPS(150) NSEED(4) GWIN(60)
     OUTDIR DEVICE
Out: <OUTDIR>/removal_<mode>_<graph>.json + .pdf
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
ALLSPEC = [("Llama", "meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           ("Gemma", "google/gemma-2-9b", "unsloth/gemma-2-9b"),
           ("Qwen",  "Qwen/Qwen3-8B-Base", None)]
if PRESET == "smoke":
    ALLSPEC = [("distilgpt2", "distilgpt2", None)]
MODE = os.environ.get("MODE", "alllayers")
_mf = os.environ.get("MODELS_FILTER")
MODELS = [m for m in ALLSPEC if (not _mf or m[0] in set(_mf.split(",")))]
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama" if PRESET != "smoke" else "distilgpt2")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "20"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
XCTX = int(os.environ.get("XCTX", "150"))
GSTEPS = int(os.environ.get("GSTEPS", "150"))
TEMP = float(os.environ.get("TEMP", "1.0"))          # sampling temperature (>0 samples; diversifies the walk)
NSEED = int(os.environ.get("NSEED", "4"))
GWIN = int(os.environ.get("GWIN", "60"))
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
CKPTS = [20, 60, 150, 250]
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/removal_followup")
RNG = np.random.default_rng(0)


def load_with_fallback(hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def mkwalk(nodes, graph):
    return Walk(walk_id=0, nodes=list(nodes), words=[graph.words[j] for j in nodes])


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def norm_and_head(model):
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    norm = getattr(base, "norm", None) or getattr(base, "ln_f", None)
    return norm, model.get_output_embeddings().weight


def coord_loo_r2(H, coords):
    ok = np.isfinite(H).all(1)                          # drop nodes missing from this window
    H = H[ok]; coords = coords[ok]
    n = H.shape[0]
    if n < 6: return float("nan")
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


def probe_Q(H, coords):
    """orthonormal d x2 probe readout basis + LOO R²."""
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
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    coef = Vt.T @ ((S / (S ** 2 + best[1]))[:, None] * (U.T @ Yc))
    Q, _ = np.linalg.qr(coef / sd[:, None])
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


# ---------------- MODE = alllayers ----------------
@torch.no_grad()
def run_alllayers(model, tok, blocks, cm, walks, graph, cand_t, dev, QL, cap_layers):
    """Project QL[L] (d x2 orthonormal) out of every layer L's output (QL None = clean).
    Return per-cap-layer coord-probe R² + neighbour mass at CKPTS."""
    n = graph.n_nodes; nL = cm.num_hidden_layers; handles = []
    if QL is not None:
        Qts = {L: torch.tensor(QL[L], device=dev, dtype=torch.float32) for L in range(nL)}
        def mkrem(L):
            Qt = Qts[L]
            def rem(_m, _i, out):
                h = (out[0] if isinstance(out, tuple) else out); hf = h.float()
                hf = (hf - (hf @ Qt) @ Qt.T).to(h.dtype)
                return (hf,) + tuple(out[1:]) if isinstance(out, tuple) else hf
            return rem
        for L in range(nL):
            handles.append(blocks[L].register_forward_hook(mkrem(L)))
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
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
            logits = model(input_ids=ids).logits[0]; single = [t[-1] for t in spans]
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
    coords = np.array(graph.coords, float); coordp = {}
    for L in cap_layers:
        H = np.where(ncnt[L][:, None] > 0, nsum[L] / np.maximum(ncnt[L][:, None], 1), np.nan)
        coordp[L] = coord_loo_r2(H, coords)
    beh = {C: (acc[C]["mass"] / acc[C]["total"] if acc[C]["total"] else float("nan")) for C in CKPTS}
    return coordp, beh


def main_alllayers():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "mode": "alllayers", "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
        walks = G.generate_walks(graph, cfg)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
        cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
        means = node_means_all(model, tok, blocks, cm, walks, dev, n)
        QL = {}; r2L = {}
        for L in range(nL):
            q, r = probe_Q(means[L], coords); QL[L] = q; r2L[L] = r
        QLr = {L: np.linalg.qr(RNG.standard_normal((cm.hidden_size, 2)))[0] for L in range(nL)}
        cap_layers = sorted(set(int(round(x)) for x in np.linspace(0.2 * nL, nL - 1, 8)))
        rec = {"n_layers": nL, "cap_layers": cap_layers, "conds": {}}
        for cname, Q in [("clean", None), ("remove_probe_all", QL), ("remove_random_all", QLr)]:
            coordp, beh = run_alllayers(model, tok, blocks, cm, walks, graph, cand_t, dev, Q, cap_layers)
            rec["conds"][cname] = {"coordprobe_by_layer": {str(k): v for k, v in coordp.items()},
                                   "neighbor_mass": {str(k): v for k, v in beh.items()}}
            pk = max((v for v in coordp.values() if np.isfinite(v)), default=float("nan"))
            print(f"[{tag}/{GRAPH}/{cname}] peak coordProbeR²={pk:+.2f} nbr_mass@250={beh[250]:.2f}", flush=True)
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    prev = f"{OUTDIR}/removal_alllayers_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    fig_alllayers(out, f"{OUTDIR}/removal_alllayers_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def fig_alllayers(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    colors = {"clean": "k", "remove_probe_all": "tab:red", "remove_random_all": "tab:blue"}
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
            ax[0, j].set_title(f"{m}  per-layer coord-probe R² (all-layer removal)", fontsize=8)
            ax[0, j].set_xlabel("layer"); ax[0, j].set_ylabel("R²"); ax[0, j].axhline(0, color=".7", lw=.6); ax[0, j].set_ylim(-0.6, 1.0); ax[0, j].legend(fontsize=6)
            ax[1, j].set_title(f"{m}  next-step neighbour mass", fontsize=8)
            ax[1, j].set_xlabel("context length"); ax[1, j].set_ylabel("neighbour mass"); ax[1, j].set_ylim(0, 1.05); ax[1, j].legend(fontsize=6)
        fig.suptitle(f"[{out['graph']}] ALL-LAYER removal: project probe subspace out at every layer\n"
                     "black=clean, red=remove probe (all layers), blue=remove random (all layers)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


# ---------------- MODE = generate ----------------
@torch.no_grad()
def generate_track(model, tok, blocks, cm, graph, cand_t, dev, seed_nodes, QLt, coords, rng, acc, GBIN):
    """Project each layer's probe subspace QLt[L] out at EVERY layer, on the SEED/context tokens
    only, then sample GSTEPS steps (freely). Record the REAL downstream next-step neighbour mass +
    validity per generation window (from the actual output logits), and — from one final forward
    (same all-layer seed projection) — the coord-probe node-means per (layer, window). No logit
    lens. Mutates `acc` (keys nbr/val/cnt/gsum/gcnt)."""
    nL = cm.num_hidden_layers; state = {"seed_end": None}
    def proj_hooks():
        hh = []
        if QLt is None: return hh
        for L in range(nL):
            Qt = QLt[L]
            def rem(_m, _i, out, Qt=Qt):
                se = state["seed_end"]; h = (out[0] if isinstance(out, tuple) else out); hf = h.float()
                hf = hf.clone(); hf[:, :se, :] = hf[:, :se, :] - (hf[:, :se, :] @ Qt) @ Qt.T
                return (hf.to(h.dtype),) + tuple(out[1:]) if isinstance(out, tuple) else hf.to(h.dtype)
            hh.append(blocks[L].register_forward_hook(rem))
        return hh
    NWIN = acc["nbr"].shape[0]; nodes = list(seed_nodes)
    handles = proj_hooks()
    try:
        for t in range(GSTEPS):
            wk = mkwalk(nodes, graph); spans = resolve_token_spans(tok, wk)
            state["seed_end"] = spans[len(seed_nodes) - 1][-1] + 1
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            last = model(input_ids=ids).logits[0, -1]
            p = torch.softmax(last[cand_t].float() / TEMP, 0).cpu().numpy(); p = p / p.sum()
            prev = nodes[-1]; nb = graph.neighbors(prev); j = int(rng.choice(len(p), p=p))
            b = min(t // GBIN, NWIN - 1)
            acc["nbr"][b] += float(p[nb].sum()); acc["val"][b] += int(j in nb); acc["cnt"][b] += 1   # REAL output
            nodes.append(j)
    finally:
        for h in handles: h.remove()
    gen = nodes[len(seed_nodes):]; gstart = len(seed_nodes); ng = len(gen)
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    caps = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    handles = proj_hooks()
    wk = mkwalk(nodes, graph); spans = resolve_token_spans(tok, wk); state["seed_end"] = spans[gstart - 1][-1] + 1
    ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev); grabbed.clear()
    model(input_ids=ids); single = [t[-1] for t in spans]
    for h in caps: h.remove()
    for h in handles: h.remove()
    bins = [min(i // GBIN, NWIN - 1) for i in range(ng)]
    for L in range(nL):
        rows = grabbed[L][0][[single[gstart + i] for i in range(ng)]].float().cpu().numpy()
        for i in range(ng):
            acc["gsum"][L, bins[i], gen[i]] += rows[i]
    for i in range(ng):
        acc["gcnt"][bins[i], gen[i]] += 1


def main_generate():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = {t: (h, m) for t, h, m in ALLSPEC}[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=max(NSEED, 8), walk_length=XCTX, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    seeds = G.generate_walks(graph, cfg)[:NSEED]
    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    means = node_means_all(model, tok, blocks, cm, seeds, dev, n)
    r2L = {L: coord_loo_r2(means[L], coords) for L in range(nL)}
    L_rem = int(max((L for L in range(nL) if L <= 0.85 * nL), key=lambda L: r2L[L]))
    pr2 = probe_Q(means[L_rem], coords)[1]
    print(f"[{tag}] peak-geom L{L_rem} probeR²={pr2:.2f} — ALL-LAYER context ablation, {GSTEPS} steps x {NSEED} seeds x 3 conds", flush=True)
    QLp = {L: torch.tensor(probe_Q(means[L], coords)[0], device=dev, dtype=torch.float32) for L in range(nL)}
    QLr = {L: torch.tensor(np.linalg.qr(RNG.standard_normal((cm.hidden_size, 2)))[0], device=dev, dtype=torch.float32) for L in range(nL)}
    NWIN = int(os.environ.get("NWIN", "6")); GBIN = max(1, GSTEPS // NWIN)
    win_mid = [b * GBIN + GBIN // 2 for b in range(NWIN)]
    conds = {"clean": None, "remove_probe": QLp, "remove_random": QLr}
    out = {"graph": GRAPH, "mode": "generate", "model": tag, "L_rem": int(L_rem), "probe_r2": pr2,
           "xctx": XCTX, "gsteps": GSTEPS, "ablation": "all_layers_context", "nL": nL, "nwin": NWIN,
           "win_mid": win_mid, "conds": {}}
    for cname, QLt in conds.items():
        acc = {"nbr": np.zeros(NWIN), "val": np.zeros(NWIN), "cnt": np.zeros(NWIN),
               "gsum": np.zeros((nL, NWIN, n, cm.hidden_size)), "gcnt": np.zeros((NWIN, n))}
        for si, seed in enumerate(seeds):
            generate_track(model, tok, blocks, cm, graph, cand_t, dev, seed.nodes, QLt, coords,
                           np.random.default_rng(1000 + si), acc, GBIN)
        cnt = np.maximum(acc["cnt"], 1)
        val = (acc["val"] / cnt); nbr = (acc["nbr"] / cnt)                          # real output, per window
        geom = np.full((nL, NWIN), np.nan)
        for L in range(nL):
            for w in range(NWIN):
                H = np.where(acc["gcnt"][w][:, None] > 0, acc["gsum"][L, w] / np.maximum(acc["gcnt"][w][:, None], 1), np.nan)
                geom[L, w] = coord_loo_r2(H, coords)
        out["conds"][cname] = {"val": val.tolist(), "nbr": nbr.tolist(), "geom": geom.tolist()}
        print(f"[{tag}/{GRAPH}/{cname}] downstream nbr {nbr[0]:.2f}->{nbr[-1]:.2f}  geom peak {np.nanmax(geom):.2f}", flush=True)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    prev = f"{OUTDIR}/removal_generate_{GRAPH}.json"
    json.dump(out, open(prev, "w"), indent=2)
    fig_generate(out, f"{OUTDIR}/removal_generate_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def fig_generate(out, path):
    """One slide PER LAYER: the 3 trends (validity | neighbour mass | coord-probe R²) vs
    generation step, with the 3 conditions overlaid."""
    colors = {"clean": "k", "remove_probe": "tab:red", "remove_random": "tab:blue"}
    nL = out["nL"]; wm = out["win_mid"]
    with PdfPages(path) as pdf:
        for L in range(nL):
            fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
            for cn, c in colors.items():
                cd = out["conds"][cn]
                ax[0].plot(wm, cd["val"], "-o", ms=3, color=c, label=cn)               # real output (same each slide)
                ax[1].plot(wm, cd["nbr"], "-o", ms=3, color=c, label=cn)               # real output (same each slide)
                ax[2].plot(wm, np.array(cd["geom"])[L], "-o", ms=3, color=c, label=cn)  # per-layer geometry
            ax[0].set_title("generated-step validity (real output)", fontsize=9); ax[0].set_ylim(0, 1.05)
            ax[1].set_title("downstream neighbour mass (real output)", fontsize=9); ax[1].set_ylim(0, 1.05)
            ax[2].set_title(f"coord-probe R² @ LAYER {L}", fontsize=9); ax[2].set_ylim(-0.6, 1.0); ax[2].axhline(0, color=".7", lw=.6)
            for a in ax: a.set_xlabel("generation step"); a.legend(fontsize=7)
            fig.suptitle(f"[{out['graph']}] {out['model']} — LAYER {L}: validity | downstream neighbour mass (both output-level, "
                         "identical across layers) | per-layer geometry.  ALL-LAYER context ablation.\n"
                         "black=clean, red=remove probe, blue=remove random", fontsize=9)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    (main_generate if MODE == "generate" else main_alllayers)()
