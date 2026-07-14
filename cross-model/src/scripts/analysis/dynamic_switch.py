"""DYNAMIC graph switch mid-context: how does the in-context map update when the graph changes
partway through the walk?

SWITCH=remove -- a node R is deleted at step T: the walk never visits R again (re-routed to
  avoid it). Track over time-since-switch:
    P(R)  -- the probability the model still assigns to the removed node R as the next step,
             measured every time the walk lands on a NEIGHBOUR of R (does belief in R decay?)
    geom  -- coord-probe R² over the 15 surviving nodes in sliding windows (does the rest of
             the map stay intact?)
SWITCH=swap -- two nodes u,v with DISJOINT neighbourhoods swap word labels at step T (node u
  now emits v's word and vice versa; graph structure unchanged). Track in sliding windows:
    R²_orig    -- coord-probe R² of the 16 word-representations against the ORIGINAL coords
    R²_swapped -- ... against the SWAPPED coords (u<->v). If the model relearns, orig falls and
                  swapped rises over time.

New sibling dir: runs/dynamic.
Env: PRESET SWITCH(remove|swap) MODELS_FILTER GRAPH(square_grid) NW(12) T(150) POST(220)
     WIN(60) STRIDE(30) OUTDIR DEVICE
Out: <OUTDIR>/dynamic_<switch>_<graph>.json + .pdf
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
_mf = os.environ.get("MODELS_FILTER")
MODELS = [m for m in ALLSPEC if (not _mf or m[0] in set(_mf.split(",")))]
SWITCH = os.environ.get("SWITCH", "remove")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
NW = int(os.environ.get("NW", "12"))
T = int(os.environ.get("T", "150"))
POST = int(os.environ.get("POST", "220"))
WIN = int(os.environ.get("WIN", "60"))
STRIDE = int(os.environ.get("STRIDE", "30"))
NDEPTH = int(os.environ.get("NDEPTH", "12"))         # number of depth slides (relative-depth, aligned across models)
SUB = float(os.environ.get("SUB", "0.5"))            # occurrence subsample fraction for the scatter
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/dynamic")


def load_with_fallback(hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def mkwalk(nodes, words):
    return Walk(walk_id=0, nodes=list(nodes), words=list(words))


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


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
    """orthonormal d x2 probe readout basis (for projecting occurrences into probe axes)."""
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
    return Q, mu


def peak_layer(model, tok, blocks, cm, graph, dev):
    """quick normal-walk capture to pick the read layer (peak coord-probe)."""
    n = graph.n_nodes; nL = cm.num_hidden_layers
    cfg2 = None
    walks = G.generate_walks(graph, replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=8, walk_length=T, device=dev))
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    with torch.no_grad():
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            model(input_ids=ids); single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= 40:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    for h in hs: h.remove()
    cn = np.maximum(ncnt, 1); coords = np.array(graph.coords, float)
    means = {L: nsum[L] / cn[:, None] for L in range(nL)}
    r2 = {L: coord_loo_r2(means[L], coords) for L in range(nL)}
    return int(max(r2, key=lambda L: r2[L])), means


def build_walk_remove(graph, R, rng):
    """normal walk that includes R for the first T steps, then re-routes to avoid R."""
    nodes = [rng.integers(graph.n_nodes)]
    for step in range(1, T + POST):
        nbrs = graph.neighbors(nodes[-1])
        if step >= T:
            nbrs = [x for x in nbrs if x != R] or nbrs
        nodes.append(int(rng.choice(nbrs)))
    words = [graph.words[j] for j in nodes]
    return nodes, words


def build_walk_swap(graph, u, v, rng):
    nodes = [rng.integers(graph.n_nodes)]
    for _ in range(1, T + POST):
        nodes.append(int(rng.choice(graph.neighbors(nodes[-1]))))
    swap = {u: v, v: u}
    words = [graph.words[nodes[s]] if s < T else graph.words[swap.get(nodes[s], nodes[s])] for s in range(len(nodes))]
    disp = [nodes[s] if s < T else swap.get(nodes[s], nodes[s]) for s in range(len(nodes))]   # displayed word-id
    return nodes, words, disp


@torch.no_grad()
def capture(model, tok, blocks, cm, L_read, slide_layers, QL, muL, walkobjs, cand_t, dev):
    """Run each walk once. Return per walk: read-layer residuals [nsteps,d] (for geometry),
    candidate logprobs at each prediction position [nsteps-1,nnode], and per-occurrence 2-D
    projections onto each slide layer's probe axes {slide_layer: [nsteps,2]}."""
    caps = sorted(set(slide_layers) | {L_read}); grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in caps]
    res = []
    try:
        for wk in walkobjs:
            spans = resolve_token_spans(tok, wk); single = [t[-1] for t in spans]; grabbed.clear()
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            logits = model(input_ids=ids).logits[0]
            rows = {L: grabbed[L][0][single].float().cpu().numpy() for L in caps}      # [nsteps, d] per capped layer
            lp = np.stack([torch.log_softmax(logits[spans[s + 1][0] - 1][cand_t].float(), 0).cpu().numpy()
                           for s in range(len(wk.nodes) - 1)])
            projs = {L: (rows[L] - muL[L]) @ QL[L] for L in slide_layers}              # [nsteps, 2]
            res.append((rows[L_read], lp, projs))
    finally:
        for h in hs: h.remove()
    return res


def windows():
    return [(w0, w0 + WIN) for w0 in range(0, POST - WIN + 1, STRIDE)]


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "switch": SWITCH, "T": T, "win": WIN, "stride": STRIDE, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NW, walk_length=T + POST, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
        rng = np.random.default_rng(0)
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model)
        cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
        nL = cm.num_hidden_layers
        L_read, meansN = peak_layer(model, tok, blocks, cm, graph, dev)
        fracs = np.linspace(0.1, 0.95, NDEPTH)
        frac_layers = [int(round(f * (nL - 1))) for f in fracs]                       # depth-slide layer per fraction
        QL = {}; muL = {}
        for L in sorted(set(frac_layers)):
            QL[L], muL[L] = probe_Q(meansN[L], coords)                                # per-layer probe axes for the scatter
        wins = windows(); wmid = [T + (a + b) // 2 - T for a, b in wins]              # steps since switch (window mid)
        print(f"[{tag}] L_read={L_read} switch={SWITCH} nL={nL}", flush=True)

        if SWITCH == "remove":
            deg = [len(graph.neighbors(i)) for i in range(n)]
            R = int(np.argmax(deg))                                                   # highest-degree node (most neighbours to test)
            walks = [build_walk_remove(graph, R, rng) for _ in range(NW)]
            wobjs = [mkwalk(nodes, words) for nodes, words in walks]
            cap = capture(model, tok, blocks, cm, L_read, sorted(set(frac_layers)), QL, muL, wobjs, cand_t, dev)
            nbrR = set(graph.neighbors(R)); keep = [i for i in range(n) if i != R]
            d = cap[0][0].shape[1]
            # P(R) at neighbour-of-R landings, binned by step-since-switch
            pbins = {w: [] for w in wmid}
            for (nodes, words), (reps, lp, _) in zip(walks, cap):
                for s in range(T, len(nodes) - 1):
                    if nodes[s] in nbrR:
                        wsel = min(wmid, key=lambda w: abs(w - (s - T)))
                        pbins[wsel].append(float(np.exp(lp[s, R])))
            # geometry over surviving nodes, POOLED across walks per window
            geo = {}
            for (a, b), wm in zip(wins, wmid):
                nsum = np.zeros((n, d)); ncnt = np.zeros(n)
                for (nodes, words), (reps, lp, _) in zip(walks, cap):
                    for s in range(T + a, min(T + b, len(nodes))):
                        nsum[nodes[s]] += reps[s]; ncnt[nodes[s]] += 1
                H = np.where(ncnt[keep, None] > 0, nsum[keep] / np.maximum(ncnt[keep, None], 1), np.nan)
                geo[wm] = coord_loo_r2(H, coords[keep])
            # baseline P(R) before switch
            base = []
            for (nodes, words), (reps, lp, _) in zip(walks, cap):
                for s in range(40, T - 1):
                    if nodes[s] in nbrR: base.append(float(np.exp(lp[s, R])))
            rec = {"L_read": L_read, "removed": R, "removed_word": graph.words[R],
                   "wmid": wmid, "P_removed_base": float(np.mean(base)) if base else None,
                   "P_removed": [float(np.nanmean(pbins[w])) if pbins[w] else float("nan") for w in wmid],
                   "geom_survivors": [geo[w] for w in wmid]}
            print(f"[{tag}] P(R) base={rec['P_removed_base']:.3f} -> {rec['P_removed'][0]:.3f}..{rec['P_removed'][-1]:.3f}"
                  f"  geom {rec['geom_survivors'][0]:+.2f}->{rec['geom_survivors'][-1]:+.2f}", flush=True)
        else:  # swap
            # find u,v: non-adjacent, disjoint neighbourhoods
            pair = None
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in graph.neighbors(u) and not (set(graph.neighbors(u)) & set(graph.neighbors(v))):
                        pair = (u, v); break
                if pair: break
            u, v = pair
            swapc = coords.copy(); swapc[[u, v]] = swapc[[v, u]]
            walks = [build_walk_swap(graph, u, v, rng) for _ in range(NW)]
            wobjs = [mkwalk(nodes, words) for nodes, words, _ in walks]
            cap = capture(model, tok, blocks, cm, L_read, sorted(set(frac_layers)), QL, muL, wobjs, cand_t, dev)
            d = cap[0][0].shape[1]; go = {}; gs = {}
            for (a, b), wm in zip(wins, wmid):                                    # POOL across walks per window
                nsum = np.zeros((n, d)); ncnt = np.zeros(n)
                for (nodes, words, disp), (reps, lp, _) in zip(walks, cap):
                    for s in range(T + a, min(T + b, len(nodes))):
                        nsum[disp[s]] += reps[s]; ncnt[disp[s]] += 1
                H = np.where(ncnt[:, None] > 0, nsum / np.maximum(ncnt[:, None], 1), np.nan)
                go[wm] = coord_loo_r2(H, coords); gs[wm] = coord_loo_r2(H, swapc)
            rec = {"L_read": L_read, "swap": [u, v], "swap_words": [graph.words[u], graph.words[v]], "wmid": wmid,
                   "R2_orig": [go[w] for w in wmid],
                   "R2_swapped": [gs[w] for w in wmid]}
            print(f"[{tag}] swap {graph.words[u]}<->{graph.words[v]}: orig {rec['R2_orig'][0]:+.2f}->{rec['R2_orig'][-1]:+.2f}"
                  f"  swapped {rec['R2_swapped'][0]:+.2f}->{rec['R2_swapped'][-1]:+.2f}", flush=True)
        # per-occurrence projections onto each depth-slide layer's probe axes (colour = node id)
        occ_by_frac = {}
        for fi, L in enumerate(frac_layers):
            pts = []; rngp = np.random.default_rng(100 + fi)
            for wtuple, (reps, lp, projs) in zip(walks, cap):
                nodes = wtuple[0]; disp = wtuple[2] if (SWITCH == "swap" and len(wtuple) > 2) else None
                P = projs[L]
                for s in range(len(nodes)):
                    if rngp.random() > SUB: continue
                    nid = disp[s] if disp is not None else nodes[s]
                    pts.append([int(nid), round(float(P[s, 0]), 3), round(float(P[s, 1]), 3), 0 if s < T else 1])
            occ_by_frac[str(fi)] = pts
        rec["occ_by_frac"] = occ_by_frac; rec["frac_layers"] = frac_layers
        rec["fracs"] = [round(float(f), 3) for f in fracs]; rec["nL"] = nL
        # river-resample: after a long absence, force the removed node again at 5 late contexts and
        # project where its activation lands in the probe axes (per slide layer)
        if SWITCH == "remove":
            Rn = rec["removed"]; rpos = [T + POST - 1 - k * 12 for k in range(5)]
            gg = {}
            def mkg(L):
                def hh(_m, _i, out): gg[L] = (out[0] if isinstance(out, tuple) else out).detach()
                return hh
            hsr = [blocks[L].register_forward_hook(mkg(L)) for L in sorted(set(frac_layers))]
            rr = {str(fi): [] for fi in range(len(frac_layers))}
            with torch.no_grad():
                for (nodes, words) in walks:
                    for posk in rpos:
                        seq = list(nodes[:posk]) + [Rn]
                        wk2 = mkwalk(seq, [graph.words[j] for j in seq]); spans2 = resolve_token_spans(tok, wk2)
                        gg.clear()
                        ids2 = tok(wk2.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
                        model(input_ids=ids2); rtok = spans2[-1][-1]
                        for fi, L in enumerate(frac_layers):
                            xy = (gg[L][0][rtok].float().cpu().numpy() - muL[L]) @ QL[L]
                            rr[str(fi)].append([round(float(xy[0]), 3), round(float(xy[1]), 3)])
            for h in hsr: h.remove()
            rec["river_resample"] = rr
        out["models"][tag] = rec
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    prev = f"{OUTDIR}/dynamic_{SWITCH}_{GRAPH}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/dynamic_{SWITCH}_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    sw = out["switch"]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.8), squeeze=False)
        for j, m in enumerate(models):
            r = out["models"][m]; wm = r["wmid"]
            if sw == "remove":
                ax[0, j].plot(wm, r["P_removed"], "-o", ms=3, color="tab:red", label="P(removed node) at nbr landings")
                if r["P_removed_base"] is not None:
                    ax[0, j].axhline(r["P_removed_base"], color="tab:red", ls=":", lw=1, label="P(removed) pre-switch")
                ax2 = ax[0, j].twinx()
                ax2.plot(wm, r["geom_survivors"], "-o", ms=3, color="tab:green", label="survivor coord-probe R²")
                ax2.set_ylim(-0.3, 1.0); ax2.set_ylabel("survivor R²", color="tab:green")
                ax[0, j].set_ylabel("P(removed node)", color="tab:red")
                ax[0, j].set_title(f"{m}  remove '{r['removed_word']}' (L{r['L_read']})", fontsize=8)
                ax[0, j].legend(fontsize=6, loc="upper right")
            else:
                ax[0, j].plot(wm, r["R2_orig"], "-o", ms=3, color="tab:blue", label="R² vs ORIGINAL coords")
                ax[0, j].plot(wm, r["R2_swapped"], "-o", ms=3, color="tab:red", label="R² vs SWAPPED coords")
                ax[0, j].set_ylim(-0.3, 1.0); ax[0, j].set_ylabel("coord-probe R²")
                ax[0, j].set_title(f"{m}  swap {r['swap_words'][0]}<->{r['swap_words'][1]} (L{r['L_read']})", fontsize=8)
                ax[0, j].legend(fontsize=6)
            ax[0, j].set_xlabel("steps since switch")
        title = ("removed node: does belief in it decay & does the rest of the map survive?"
                 if sw == "remove" else "swapped nodes: does the map relearn (orig R² falls, swapped R² rises)?")
        fig.suptitle(f"[{out['graph']}] DYNAMIC {sw} switch @ step {out['T']} — {title}", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # slideshow: one 2x3 slide per depth fraction (rows: PRE / POST switch, cols: models)
        cmap = plt.get_cmap("tab20")
        fracs = out["models"][models[0]].get("fracs", [])
        rmword = out["models"][models[0]].get("removed_word", "")
        for fi, f in enumerate(fracs):
            fig, ax = plt.subplots(2, len(models), figsize=(4.8 * len(models), 8.8), squeeze=False)
            for j, m in enumerate(models):
                r = out["models"][m]; occ = np.array(r.get("occ_by_frac", {}).get(str(fi), []))
                Lm = r.get("frac_layers", [-1] * (fi + 1))[fi]
                if occ.size == 0:
                    continue
                nid = occ[:, 0].astype(int); x = occ[:, 1]; y = occ[:, 2]; ph = occ[:, 3].astype(int)
                xl = (np.percentile(x, 1), np.percentile(x, 99)); yl = (np.percentile(y, 1), np.percentile(y, 99))
                for row, phase in [(0, 0), (1, 1)]:
                    sel = ph == phase
                    ax[row, j].scatter(x[sel], y[sel], c=[cmap(k % 20) for k in nid[sel]], s=13, alpha=0.7, edgecolors="none")
                    if sw == "remove":
                        selR = sel & (nid == r["removed"])
                        ax[row, j].scatter(x[selR], y[selR], facecolors="none", edgecolors="k", s=42, lw=1.2,
                                           label=f"'{r['removed_word']}' (pre)" if row == 0 else None)
                        if row == 1:                                            # POST: resampled node + pre centroid
                            rr = np.array(r.get("river_resample", {}).get(str(fi), []))
                            if rr.size:
                                ax[row, j].scatter(rr[:, 0], rr[:, 1], marker="X", c="magenta", s=75, edgecolors="k", lw=.5,
                                                   label=f"'{r['removed_word']}' resampled")
                            pre = (nid == r["removed"]) & (ph == 0)
                            if pre.any():
                                ax[row, j].scatter(x[pre].mean(), y[pre].mean(), marker="D", facecolors="none",
                                                   edgecolors="k", s=95, lw=1.6, label="pre-switch centroid")
                            ax[row, j].legend(fontsize=5, loc="best")
                    else:
                        swcol = ["black", "magenta"]
                        for k, nodeid in enumerate(r["swap"]):
                            selS = sel & (nid == nodeid)
                            ax[row, j].scatter(x[selS], y[selS], marker="*", c=swcol[k], s=64, edgecolors="white", lw=.4,
                                               label=r["swap_words"][k])
                        if j == 0: ax[row, j].legend(fontsize=6, loc="best")
                    ax[row, j].set_xlim(xl); ax[row, j].set_ylim(yl)
                    ax[row, j].set_title(f"{m}  L{Lm}  {'PRE' if phase == 0 else 'POST'}", fontsize=8)
                    ax[row, j].set_xlabel("probe axis 0"); ax[row, j].set_ylabel("probe axis 1")
            note = (f"removed '{rmword}' ringed" if sw == "remove" else "swapped words starred")
            fig.suptitle(f"[{out['graph']}] {sw} @ depth {f:.2f} — probe-axis activations (top=PRE-switch, bottom=POST); "
                         f"colour = node; {note}", fontsize=9)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
