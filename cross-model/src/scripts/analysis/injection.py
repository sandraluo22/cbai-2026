"""Cross-model INJECTION swept over L_A x L_B, paired with the cross-model probe R² heatmap.

LEFT  = cross-model probe alignment R² (predict B_LB top-6 PC from A_LA, leave-one-node-out
        pooled R²) — the "normal" layer×layer alignment heatmap, on the swept cells.
RIGHT = injected next-step NEIGHBOUR MASS: fit the ridge map A_LA -> B_LB (full residual space)
        on the 16 node-means, run B and REPLACE its residual at L_B (every readout position) with
        the ridge-mapped A residual from the same walk-occurrence, and read B's downstream
        next-step neighbour mass. So: inject every A layer into every B layer, measure accuracy.

Logits-only per cell (no residual capture) -> cheap enough for a dense L_A x L_B sweep.
Single process, default Llama->Qwen (both in /workspace/hf).
Env: PRESET A_TAG(Llama) B_TAG(Qwen) GRAPH(square_grid) NWALKS(8) WLEN(300) CTXLO(100)
     ALPHA(1000) NLA(10) NLB(14) OUTDIR DEVICE
Out: <OUTDIR>/injection_<A>_to_<B>_<graph>.json + .pdf
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
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
ALPHA = float(os.environ.get("ALPHA", "1000"))
NLA = int(os.environ.get("NLA", "10"))
NLB = int(os.environ.get("NLB", "14"))
KPC = 6
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/injection")


CACHE = {"Gemma": os.environ.get("CACHE_Gemma", "/root/hf/hub")}   # Gemma lives in /root/hf; others in /workspace/hf
def cache_for(tag): return CACHE.get(tag, os.environ.get("CACHE_DEFAULT", "/workspace/hf/hub"))


def load_with_fallback(tag, cfg):
    hf, mirror = SPEC[tag]; cd = cache_for(tag)
    try:
        return M.load_model(hf, cfg, cache_dir=cd)
    except Exception:
        return M.load_model(mirror, cfg, cache_dir=cd)


def pca(B, k):
    Bc = B - B.mean(0); U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
    kk = min(k, Vt.shape[0]); return U[:, :kk] * S[:kk]


def align_loo(A, Bpc, a):
    """LOO pooled R² predicting Bpc (n x k) from A (n x d) — the cross-model probe alignment."""
    n = A.shape[0]; mu = A.mean(0); sd = A.std(0) + 1e-6; Xs = (A - mu) / sd
    ssr = 0.0; sst = 0.0
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        U, S, Vt = np.linalg.svd(Xs[idx], full_matrices=False)
        proj = Xs[k] @ Vt.T; ytr = Bpc[idx]; ymu = ytr.mean(0)
        pred = proj @ ((S / (S ** 2 + a))[:, None] * (U.T @ (ytr - ymu))) + ymu
        ssr += ((pred - Bpc[k]) ** 2).sum(); sst += ((Bpc[k] - ymu) ** 2).sum()
    return float(1 - ssr / sst) if sst > 0 else float("nan")


def ridge_map_prep(Asrc, Btgt, a):
    muA = Asrc.mean(0); sdA = Asrc.std(0) + 1e-6; Xs = (Asrc - muA) / sdA
    muB = Btgt.mean(0); Yc = Btgt - muB
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    return muA, sdA, Vt.T, (S / (S ** 2 + a))[:, None] * (U.T @ Yc), muB


def apply_map(prep, R):
    muA, sdA, P, Q, muB = prep
    return muB + (((R - muA) / sdA) @ P) @ Q


@torch.no_grad()
def capture_A(model, tok, blocks, cm, walks, dev, n, LA_layers):
    nL = cm.num_hidden_layers; grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    resid = {}; steps = {}
    try:
        for wi, wk in enumerate(walks):
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
            model(input_ids=ids); single = [t[-1] for t in spans]
            use = [s for s in range(len(nodes)) if cl[s] >= CTXLO]
            steps[wi] = use
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in use:
                    nsum[L][nodes[s]] += rows[s]
                    if L == 0: ncnt[nodes[s]] += 1
            resid[wi] = {L: grabbed[L][0][[single[s] for s in use]].float().cpu().numpy() for L in LA_layers}
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(nL)}, resid, steps


@torch.no_grad()
def capture_B_native(model, tok, blocks, cm, walks, graph, cand_t, dev, n):
    """B native: all-layer node-means (ridge targets) + native neighbour mass at checkpoints."""
    nL = cm.num_hidden_layers; CKPTS = [c for c in [150, 200, 250] if c < WLEN]; grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    acc = {C: {"m": 0.0, "t": 0} for C in CKPTS}
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
            logits = model(input_ids=ids).logits[0]; single = [t[-1] for t in spans]
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
            for C in CKPTS:
                s = C - 1
                if 0 <= s <= len(nodes) - 2:
                    p = torch.softmax(logits[spans[s + 1][0] - 1][cand_t].float(), 0).cpu().numpy()
                    acc[C]["m"] += float(p[graph.neighbors(nodes[s])].sum()); acc[C]["t"] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1); means = {L: nsum[L] / cn[:, None] for L in range(nL)}
    beh = float(np.mean([acc[C]["m"] / acc[C]["t"] for C in CKPTS if acc[C]["t"]]))
    return means, beh


@torch.no_grad()
def inj_nbr_mass(model, tok, blocks, cm, walks, graph, cand_t, dev, LB, prep, resid_A, steps_A, src_layer):
    """Inject mapped A residual at L_B readout positions; return mean next-step neighbour mass."""
    CKPTS = [c for c in [150, 200, 250] if c < WLEN]; inj = {"map": None}; handles = []
    def hook(_m, _i, out):
        m = inj["map"]
        if m:
            h = (out[0] if isinstance(out, tuple) else out); hf = h.clone()
            for p, vec in m.items(): hf[0, p, :] = vec.to(hf.dtype)
            return (hf,) + tuple(out[1:]) if isinstance(out, tuple) else hf
    handles.append(blocks[LB].register_forward_hook(hook))
    acc = {C: {"m": 0.0, "t": 0} for C in CKPTS}
    try:
        for wi, wk in enumerate(walks):
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; single = [t[-1] for t in spans]
            use = steps_A[wi]; mapped = apply_map(prep, resid_A[wi][src_layer])
            inj["map"] = {single[s]: torch.tensor(mapped[i], device=dev) for i, s in enumerate(use)}
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            logits = model(input_ids=ids).logits[0]
            for C in CKPTS:
                s = C - 1
                if 0 <= s <= len(nodes) - 2:
                    p = torch.softmax(logits[spans[s + 1][0] - 1][cand_t].float(), 0).cpu().numpy()
                    acc[C]["m"] += float(p[graph.neighbors(nodes[s])].sum()); acc[C]["t"] += 1
    finally:
        for h in handles: h.remove()
    return float(np.mean([acc[C]["m"] / acc[C]["t"] for C in CKPTS if acc[C]["t"]]))


def main():
    dev = os.environ.get("DEVICE", "cpu" if os.environ.get("PRESET") == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; walks = G.generate_walks(graph, cfg)

    print(f"[A={A_TAG}] loading", flush=True)
    mA, tA = load_with_fallback(A_TAG, cfg); cmA = mA.config; bA = M._decoder_blocks(mA); nLA = cmA.num_hidden_layers
    LA_sweep = sorted(set(int(round(x)) for x in np.linspace(0.12 * nLA, 0.95 * nLA, NLA)))
    meansA, residA, stepsA = capture_A(mA, tA, bA, cmA, walks, dev, n, LA_sweep)
    del mA, tA; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"[B={B_TAG}] loading", flush=True)
    mB, tB = load_with_fallback(B_TAG, cfg); cmB = mB.config; bB = M._decoder_blocks(mB); nLB = cmB.num_hidden_layers
    cand_t = torch.tensor([tB(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    meansB, behB_nat = capture_B_native(mB, tB, bB, cmB, walks, graph, cand_t, dev, n)
    LB_sweep = sorted(set(int(round(x)) for x in np.linspace(0.12 * nLB, 0.95 * nLB, NLB)))
    print(f"[B={B_TAG}] native beh={behB_nat:.2f} LA={LA_sweep} LB={LB_sweep}", flush=True)

    align = np.full((len(LA_sweep), len(LB_sweep)), np.nan)
    inj = np.full((len(LA_sweep), len(LB_sweep)), np.nan)
    Bpcs = {LB: pca(meansB[LB], KPC) for LB in LB_sweep}
    for ia, LA in enumerate(LA_sweep):
        for ib, LB in enumerate(LB_sweep):
            align[ia, ib] = align_loo(meansA[LA], Bpcs[LB], ALPHA)
            prep = ridge_map_prep(meansA[LA], meansB[LB], ALPHA)
            inj[ia, ib] = inj_nbr_mass(mB, tB, bB, cmB, walks, graph, cand_t, dev, LB, prep, residA, stepsA, LA)
        print(f"[{A_TAG}->{B_TAG}] LA={LA}: inj_nbr={np.round(inj[ia],2)}", flush=True)
    del mB, tB; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()

    rec = {"A": A_TAG, "B": B_TAG, "graph": GRAPH, "LA_sweep": LA_sweep, "LB_sweep": LB_sweep,
           "B_native_beh": behB_nat, "align_r2_grid": align.tolist(), "inj_nbr_mass_grid": inj.tolist()}
    prev = f"{OUTDIR}/injection_{A_TAG}_to_{B_TAG}_{GRAPH}.json"
    json.dump(rec, open(prev, "w"), indent=2)
    make_fig(rec, f"{OUTDIR}/injection_{A_TAG}_to_{B_TAG}_{GRAPH}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(rec, path):
    LA = rec["LA_sweep"]; LB = rec["LB_sweep"]
    align = np.array(rec["align_r2_grid"]); inj = np.array(rec["inj_nbr_mass_grid"])
    ext = [LB[0], LB[-1], LA[0], LA[-1]]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        im0 = ax[0].imshow(align, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=ext)
        fig.colorbar(im0, ax=ax[0], fraction=.046)
        ax[0].set_title("cross-model probe alignment R²  (predict B_LB from A_LA)", fontsize=9)
        im1 = ax[1].imshow(inj, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1, extent=ext)
        fig.colorbar(im1, ax=ax[1], fraction=.046)
        ax[1].set_title(f"injected next-step neighbour mass  (B native={rec['B_native_beh']:.2f})", fontsize=9)
        for a in ax:
            a.set_xlabel(f"{rec['B']} inject layer L_B"); a.set_ylabel(f"{rec['A']} source layer L_A")
        fig.suptitle(f"[{rec['graph']}] {rec['A']}→{rec['B']}: alignment R² (left) vs injected behaviour (right) over L_A×L_B",
                     fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
