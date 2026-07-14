"""Cross-model COMPOSITION (sampled): feed a real walk prefix to A, let A SAMPLE n next
node-steps, hand the full sequence to B. As n grows, B's recent context becomes A-authored.

Tracks, per n:
  A gen validity            : fraction of A's sampled steps that are true graph neighbours
  B coord-probe R² PER LAYER: over node-means from the last WINDOW steps (layer x n heatmap)
  B neighbour mass PER LAYER: logit-lens next-step neighbour mass at the generated tail
                              (layer x n heatmap = layer-by-layer behavioural accuracy)
  B final neighbour mass    : from the real output logits (scalar per n)
  A coord-probe R² PER LAYER: A's own geometry on the generated sequence (reference vs ground truth)
Geometry targets are the ground-truth graph coords; A-curve and B-curves are directly comparable.

Single process, default A=Llama -> B=Qwen (both in /workspace/hf).
Env: PRESET A_TAG(Llama) B_TAG(Qwen) GRAPH(square_grid) NWALKS(8) PREFIX(120) NMAX(128)
     WINDOW(96) NS(0,8,16,32,64,128) CTXLO(40) TEMP(1.0) OUTDIR DEVICE
Out: <OUTDIR>/composition_<A>_to_<B>_<graph>.json + .pdf
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

SPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
        "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"),
        "Qwen":  ("Qwen/Qwen3-8B-Base", None)}
A_TAG = os.environ.get("A_TAG", "Llama"); B_TAG = os.environ.get("B_TAG", "Qwen")
if os.environ.get("PRESET") == "smoke":
    SPEC = {"Llama": ("distilgpt2", None), "Qwen": ("distilgpt2", None)}; A_TAG, B_TAG = "Llama", "Qwen"
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "8"))
PREFIX = int(os.environ.get("PREFIX", "120"))
NMAX = int(os.environ.get("NMAX", "128"))
WINDOW = int(os.environ.get("WINDOW", "96"))
NS = [int(x) for x in os.environ.get("NS", "0,8,16,32,64,128").split(",")]
CTXLO = int(os.environ.get("CTXLO", "40"))
TEMP = float(os.environ.get("TEMP", "1.0"))
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/composition")


def load_with_fallback(tag, cfg):
    hf, mirror = SPEC[tag]
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def mkwalk(nodes, graph):
    return Walk(walk_id=0, nodes=list(nodes), words=[graph.words[j] for j in nodes])


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def coord_loo_r2(H, coords):
    ok = np.isfinite(H).all(1); H = H[ok]; coords = coords[ok]; n = H.shape[0]
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


def norm_and_head(model):
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    norm = getattr(base, "norm", None) or getattr(base, "ln_f", None)
    W = model.get_output_embeddings().weight
    return norm, W


@torch.no_grad()
def generate_tail(model, tok, graph, cand_t, dev, seed_nodes, n_gen, rng):
    nodes = list(seed_nodes)
    for _ in range(n_gen):
        wk = mkwalk(nodes, graph)
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        last = model(input_ids=ids).logits[0, -1]
        p = torch.softmax(last[cand_t].float() / TEMP, 0).cpu().numpy(); p = p / p.sum()
        nodes.append(int(rng.choice(len(p), p=p)))
    return nodes[len(seed_nodes):]


@torch.no_grad()
def read_seq(model, tok, blocks, cm, seq_nodes, n_prefix, graph, cand_t, dev, want_lens):
    """One forward over seq_nodes. Returns per-layer window node-mean accumulators (gsum,gcnt),
    per-layer logit-lens tail neighbour-mass accumulators (lsum,lcnt), and final-logit tail mass."""
    n = graph.n_nodes; nL = cm.num_hidden_layers
    wk = mkwalk(seq_nodes, graph); spans = resolve_token_spans(tok, wk); nodes = wk.nodes
    single = [t[-1] for t in spans]; lo = max(0, len(nodes) - WINDOW)
    tail_lo = max(n_prefix, CTXLO)
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    gsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; gcnt = np.zeros(n)
    lsum = {L: 0.0 for L in range(nL)}; lcnt = 0; fmass = 0.0; fcnt = 0
    norm, W = norm_and_head(model) if want_lens else (None, None)
    Wc = W[cand_t] if want_lens else None
    try:
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev); grabbed.clear()
        logits = model(input_ids=ids).logits[0]
        for L in range(nL):
            hs_L = grabbed[L][0]
            rows = hs_L[single].float().cpu().numpy()
            for s in range(len(nodes)):
                if s >= lo:
                    gsum[L][nodes[s]] += rows[s]
                    if L == 0: gcnt[nodes[s]] += 1
            if want_lens:                                                # logit-lens neighbour mass at tail predict positions
                pos = [spans[s + 1][0] - 1 for s in range(tail_lo, len(nodes) - 1)]
                if pos:
                    x = norm(hs_L[pos])
                    lp = torch.softmax((x.float() @ Wc.float().T), -1).cpu().numpy()
                    for i, s in enumerate(range(tail_lo, len(nodes) - 1)):
                        lsum[L] += float(lp[i][graph.neighbors(nodes[s])].sum())
                    if L == 0: lcnt += len(pos)
        for s in range(tail_lo, len(nodes) - 1):
            p = torch.softmax(logits[spans[s + 1][0] - 1][cand_t].float(), 0).cpu().numpy()
            fmass += float(p[graph.neighbors(nodes[s])].sum()); fcnt += 1
    finally:
        for h in hs: h.remove()
    return gsum, gcnt, lsum, lcnt, fmass, fcnt


def geom_per_layer(gsum, gcnt, coords, nL):
    out = []
    for L in range(nL):
        H = np.where(gcnt[:, None] > 0, gsum[L] / np.maximum(gcnt[:, None], 1), np.nan)
        out.append(coord_loo_r2(H, coords))
    return out


def main():
    dev = os.environ.get("DEVICE", "cpu" if os.environ.get("PRESET") == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=PREFIX, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    walks = G.generate_walks(graph, cfg)

    print(f"[A={A_TAG}] loading", flush=True)
    mA, tA = load_with_fallback(A_TAG, cfg); cmA = mA.config; bA = M._decoder_blocks(mA); nLA = cmA.num_hidden_layers
    candA = torch.tensor([tA(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    gens = []; validity = {str(nn): [] for nn in NS}
    for wi, wk in enumerate(walks):
        tail = generate_tail(mA, tA, graph, candA, dev, wk.nodes, NMAX, np.random.default_rng(100 + wi))
        gens.append(tail)
        for nn in NS:
            if nn > 0:
                validity[str(nn)].append(float(np.mean([tail[i] in graph.neighbors((wk.nodes + tail)[len(wk.nodes) + i - 1])
                                                        for i in range(nn)])))
    print(f"[A] sampled gen validity (n=max) ~ {np.mean(validity[str(NS[-1])]):.2f}", flush=True)
    # A geometry per layer at each n (reference vs ground truth)
    A_geom = {}
    for nn in NS:
        gsum = {L: np.zeros((n, cmA.hidden_size)) for L in range(nLA)}; gcnt = np.zeros(n)
        for wi, wk in enumerate(walks):
            seq = list(wk.nodes) + gens[wi][:nn]
            gs, gc_, _, _, _, _ = read_seq(mA, tA, bA, cmA, seq, len(wk.nodes), graph, candA, dev, False)
            for L in range(nLA): gsum[L] += gs[L]
            gcnt += gc_
        A_geom[str(nn)] = geom_per_layer(gsum, gcnt, coords, nLA)
    del mA, tA; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"[B={B_TAG}] loading", flush=True)
    mB, tB = load_with_fallback(B_TAG, cfg); cmB = mB.config; bB = M._decoder_blocks(mB); nLB = cmB.num_hidden_layers
    candB = torch.tensor([tB(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    B_geom = {}; B_lens = {}; B_final = {}
    for nn in NS:
        gsum = {L: np.zeros((n, cmB.hidden_size)) for L in range(nLB)}; gcnt = np.zeros(n)
        lsum = {L: 0.0 for L in range(nLB)}; lcnt = 0; fmass = 0.0; fcnt = 0
        for wi, wk in enumerate(walks):
            seq = list(wk.nodes) + gens[wi][:nn]
            gs, gc_, ls, lc, fm, fc = read_seq(mB, tB, bB, cmB, seq, len(wk.nodes), graph, candB, dev, True)
            for L in range(nLB): gsum[L] += gs[L]; lsum[L] += ls[L]
            gcnt += gc_; lcnt += lc; fmass += fm; fcnt += fc
        B_geom[str(nn)] = geom_per_layer(gsum, gcnt, coords, nLB)
        B_lens[str(nn)] = [lsum[L] / lcnt if lcnt else float("nan") for L in range(nLB)]
        B_final[str(nn)] = fmass / fcnt if fcnt else float("nan")
        pk = np.nanmax(B_geom[str(nn)])
        print(f"[compose n={nn}] B peak geomR²={pk:+.2f} B_final_nbr={B_final[str(nn)] if fcnt else float('nan'):.2f} "
              f"Aval={np.mean(validity[str(nn)]) if nn>0 else float('nan'):.2f}", flush=True)
    del mB, tB; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    rec = {"A": A_TAG, "B": B_TAG, "graph": GRAPH, "ns": NS, "temp": TEMP, "window": WINDOW,
           "A_validity": {k: float(np.mean(v)) if v else None for k, v in validity.items()},
           "A_geom_by_layer": A_geom, "B_geom_by_layer": B_geom, "B_lens_by_layer": B_lens, "B_final_nbr": B_final}
    prev = f"{OUTDIR}/composition_{A_TAG}_to_{B_TAG}_{GRAPH}.json"
    json.dump(rec, open(prev, "w"), indent=2)
    make_fig(rec, f"{OUTDIR}/composition_{A_TAG}_to_{B_TAG}_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(rec, path):
    ns = rec["ns"]
    Bg = np.array([rec["B_geom_by_layer"][str(nn)] for nn in ns])                 # [n_n, nLB]
    Bl = np.array([rec["B_lens_by_layer"][str(nn)] for nn in ns])
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        im0 = ax[0].imshow(Bg, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                           extent=[0, Bg.shape[1], 0, len(ns) - 1]); fig.colorbar(im0, ax=ax[0], fraction=.046)
        ax[0].set_yticks(range(len(ns))); ax[0].set_yticklabels(ns); ax[0].set_ylabel("n (A-generated steps)")
        ax[0].set_xlabel(f"{rec['B']} layer"); ax[0].set_title(f"{rec['B']} coord-probe R² (layer × n)", fontsize=9)
        im1 = ax[1].imshow(Bl, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                           extent=[0, Bl.shape[1], 0, len(ns) - 1]); fig.colorbar(im1, ax=ax[1], fraction=.046)
        ax[1].set_yticks(range(len(ns))); ax[1].set_yticklabels(ns); ax[1].set_ylabel("n")
        ax[1].set_xlabel(f"{rec['B']} layer"); ax[1].set_title(f"{rec['B']} logit-lens neighbour mass (layer × n)", fontsize=9)
        # summary curves vs n
        Bpk = [np.nanmax(rec["B_geom_by_layer"][str(nn)]) for nn in ns]
        Apk = [np.nanmax(rec["A_geom_by_layer"][str(nn)]) for nn in ns]
        Bf = [rec["B_final_nbr"][str(nn)] for nn in ns]
        Av = [rec["A_validity"][str(nn)] if rec["A_validity"][str(nn)] is not None else np.nan for nn in ns]
        ax[2].plot(ns, Bpk, "-o", color="tab:red", label=f"{rec['B']} peak geom R²")
        ax[2].plot(ns, Apk, "--o", color="tab:orange", label=f"{rec['A']} peak geom R² (ref)")
        ax[2].plot(ns, Bf, "-s", color="tab:green", label=f"{rec['B']} final nbr mass")
        ax[2].plot(ns, Av, ":^", color="tab:gray", label=f"{rec['A']} gen validity")
        ax[2].set_xlabel("n (A-generated steps)"); ax[2].set_ylim(-0.1, 1.05); ax[2].legend(fontsize=7)
        ax[2].set_title("peak geometry (A vs B vs ground truth) & behaviour vs n", fontsize=9)
        fig.suptitle(f"[{rec['graph']}] composition {rec['A']}→{rec['B']} (sampled, T={rec['temp']}): "
                     "does B keep the map & behaviour as context becomes A-generated?", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
