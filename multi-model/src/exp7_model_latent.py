"""Exp7 -- Does Qwen model LLAMA's OWN latent, observing only LLAMA's emissions?

Fix over the first draft: LLAMA IS THE SOURCE. Llama autoregressively GENERATES the
walk (Llama = the HMM; its hidden state is the latent, its sampled tokens are the
emissions). Qwen observes ONLY Llama's emitted token sequence and must reconstruct
Llama's internal state from it -- the genuine observer/HMM setup, not two models
co-reading a third-party stream.

The latent is hidden even though Qwen reads the tokens: the emitted token is a lossy
SAMPLE of Llama's state, so Llama's full predictive distribution p^Llama_t and its
residual z_t carry more than any token reveals -- the observer must actually model
Llama's computation to recover them.

Scale: the walk lives on grids of growing size (16 -> 25 -> 36 hidden states).

Per grid size, decoding from QWEN's residual (held out by walk, per layer):
  - p^Llama         : Qwen reconstructs Llama's predictive/causal state       (models Llama)
  - p^ideal         : the ideal grid-walk next-node dist (uniform over nbrs)  (process only)
  - p^Llama-p^ideal : Llama's IDIOSYNCRATIC deviation from ideal              (models Llama SPECIFICALLY)
  - z_Llama (PCA)   : Llama's literal residual state
Ceiling: Llama's own residual -> p^Llama. Plus how far Llama sits from ideal.

Env: PRESET SIZES(4,5,6) NWALK(16) XCTX(40) GSTEPS(240) CTXLO(60) TEMP(1.0)
     PCADIM(80) RESID_PCA(50) TESTFRAC(0.3) RUN_DIR DEVICE
Out: <RUN_DIR>/exp7_model_latent.json + .pdf
"""
from __future__ import annotations

import os
import json
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import common as C  # noqa: E402
import graph as G   # noqa: E402
from graph import Walk  # noqa: E402
from models import resolve_token_spans  # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

SIZES = [int(x) for x in os.environ.get("SIZES", "4,5,6" if C.PRESET != "smoke" else "3").split(",")]
NWALK = int(os.environ.get("NWALK", "16" if C.PRESET != "smoke" else "5"))
XCTX = int(os.environ.get("XCTX", "40" if C.PRESET != "smoke" else "10"))
GSTEPS = int(os.environ.get("GSTEPS", "240" if C.PRESET != "smoke" else "40"))
CTXLO = int(os.environ.get("CTXLO", "60" if C.PRESET != "smoke" else "5"))
TEMP = float(os.environ.get("TEMP", "1.0"))
PCADIM = int(os.environ.get("PCADIM", "80"))
RESID_PCA = int(os.environ.get("RESID_PCA", "50"))
TESTFRAC = float(os.environ.get("TESTFRAC", "0.3"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")


def grid_graph(side, dev):
    cfg = C.make_cfg("square_grid", n_walks=max(NWALK, 8), walk_length=max(XCTX, 8), device=dev)
    cfg = replace(cfg, grid_rows=side, grid_cols=side)
    return cfg, G.build_graph(cfg)


def ideal_next(node, graph, n):
    p = np.zeros(n)
    nb = graph.neighbors(node)
    for j in nb:
        p[j] = 1.0 / len(nb)
    return p


@torch.no_grad() if torch is not None else (lambda f: f)
def capture_core(model, tok, walks, graph, n, dev, want_pred):
    """Teacher-force each walk; per occurrence (ctx>=CTXLO) return residuals at every
    layer, and (if want_pred) the model's next-NODE distribution p_t. Fixed order."""
    blocks = C.decoder_blocks(model)
    nL = C.n_layers(model)
    cand = torch.tensor([tok(" " + graph.words[j], add_special_tokens=False)["input_ids"][0]
                         for j in range(n)], device=dev)
    grabbed = {}

    def mk(L):
        def hh(_m, _i, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh

    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    Xrows = {L: [] for L in range(nL)}
    pred = []
    try:
        for wk in walks:
            ids = tok(wk.text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk)
            cl = np.arange(1, len(wk.nodes) + 1)
            keep = [s for s in range(len(wk.nodes)) if cl[s] >= CTXLO]
            grabbed.clear()
            logits = model(input_ids=ids).logits[0]
            for L in range(nL):
                Xrows[L].append(grabbed[L][0][[spans[s][-1] for s in keep]].float().cpu().numpy())
            if want_pred:
                for s in keep:
                    p = torch.softmax(logits[spans[s][-1]][cand].float(), 0).cpu().numpy()
                    pred.append(p)
    finally:
        for h in hs:
            h.remove()
    X = {L: np.concatenate(Xrows[L], 0) for L in range(nL)}
    return X, (np.array(pred) if want_pred else None)


def gen_and_capture_llama(seeds, cfg, graph, n, dev):
    """Llama GENERATES the walks (its own emissions), then we read out Llama's latent
    (residual + predictive state) by teacher-forcing Llama on what it produced."""
    model, tok = C.load_model("Llama", cfg)
    cand = C.candidate_token_ids(tok, graph, dev)
    walks = []
    for si, seed in enumerate(seeds):
        nodes, _ = C.generate_walk(model, tok, graph, cand, dev, seed.nodes[:XCTX], GSTEPS,
                                   temp=TEMP, rng=np.random.default_rng(7000 + si))
        w = C.mkwalk(nodes, graph)
        w.walk_id = si                       # mkwalk defaults to 0; give each walk a distinct id
        walks.append(w)
    XL, pL = capture_core(model, tok, walks, graph, n, dev, want_pred=True)
    C.free(model, tok)
    return walks, XL, pL


def decode_r2(X, Y, grp, pcad, testfrac, seed=0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    walks = np.unique(grp)
    rng = np.random.default_rng(seed)
    test_w = set(rng.choice(walks, max(1, int(round(testfrac * len(walks)))), replace=False))
    te = np.array([g in test_w for g in grp]); tr = ~te
    if tr.sum() < 10 or te.sum() < 5:
        return float("nan")
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    k = min(pcad, Xtr.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    pred = Ridge(alpha=10.0).fit(Ptr, Y[tr]).predict(Pte)
    sst = ((Y[te] - Y[tr].mean(0)) ** 2).sum()
    return float(1 - ((Y[te] - pred) ** 2).sum() / sst) if sst > 1e-9 else float("nan")


def peak_decode(Xdict, Y, grp, pcad, testfrac):
    r = [decode_r2(Xdict[L], Y, grp, pcad, testfrac) for L in range(len(Xdict))]
    return float(np.nanmax(r)), int(np.nanargmax(r))


def main():
    dev = C.default_device()
    os.makedirs(RUN_DIR, exist_ok=True)
    out = {"sizes": SIZES, "nwalk": NWALK, "xctx": XCTX, "gsteps": GSTEPS, "ctxlo": CTXLO, "levels": []}

    for side in SIZES:
        cfg, graph = grid_graph(side, dev)
        n = graph.n_nodes
        seeds = G.generate_walks(graph, cfg)[:NWALK]
        print(f"[exp7] grid {side}x{side} (n={n}) — Llama generates {NWALK} walks", flush=True)
        walks, XL, pL = gen_and_capture_llama(seeds, cfg, graph, n, dev)

        # occurrence-aligned ideal next-node dist, deviation, and walk groups
        pideal, grp = [], []
        for wk in walks:
            for s in range(len(wk.nodes)):
                if s + 1 >= CTXLO:
                    pideal.append(ideal_next(wk.nodes[s], graph, n)); grp.append(wk.walk_id)
        pideal = np.array(pideal); grp = np.array(grp)
        deviation = pL - pideal

        # Qwen observes Llama's emissions
        print(f"[exp7] Qwen observes Llama's emissions", flush=True)
        qmodel, qtok = C.load_model("Qwen", cfg)
        XQ, _ = capture_core(qmodel, qtok, walks, graph, n, dev, want_pred=False)
        C.free(qmodel, qtok)

        # Llama residual target (literal internal state), PCA-reduced at its most predictive layer
        from sklearn.decomposition import PCA as _PCA
        from sklearn.preprocessing import StandardScaler as _SS
        _, Lpk = peak_decode(XL, pL, grp, PCADIM, TESTFRAC)
        residY = _PCA(n_components=min(RESID_PCA, XL[Lpk].shape[0] - 1), svd_solver="randomized",
                      random_state=0).fit_transform(_SS().fit_transform(XL[Lpk]))

        r_q_llama, l1 = peak_decode(XQ, pL, grp, PCADIM, TESTFRAC)
        r_q_ideal, l2 = peak_decode(XQ, pideal, grp, PCADIM, TESTFRAC)
        r_q_dev, l3 = peak_decode(XQ, deviation, grp, PCADIM, TESTFRAC)
        r_q_resid, l4 = peak_decode(XQ, residY, grp, PCADIM, TESTFRAC)
        r_ceil, l5 = peak_decode(XL, pL, grp, PCADIM, TESTFRAC)

        llama_vs_ideal = float(1 - ((pL - pideal) ** 2).sum() / ((pL - pL.mean(0)) ** 2).sum())
        rec = {"side": side, "n": n, "n_occ": int(len(pideal)),
               "q_to_llama_pred": r_q_llama, "q_to_ideal": r_q_ideal,
               "q_to_deviation": r_q_dev, "q_to_llama_resid": r_q_resid,
               "llama_self_ceiling": r_ceil, "llama_vs_ideal_r2": llama_vs_ideal,
               "peak_layers": {"q_llama": l1, "q_ideal": l2, "q_dev": l3, "q_resid": l4, "ceil": l5}}
        out["levels"].append(rec)
        print(f"[exp7] {side}x{side}: Q→Llama_pred={r_q_llama:.3f} Q→ideal={r_q_ideal:.3f} "
              f"Q→DEVIATION={r_q_dev:.3f} Q→resid={r_q_resid:.3f} | ceiling={r_ceil:.3f} "
              f"Llama-vs-ideal R²={llama_vs_ideal:.3f}", flush=True)

    json.dump(out, open(os.path.join(RUN_DIR, "exp7_model_latent.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, "exp7_model_latent.pdf"))
    print(f"[exp7] DONE -> {RUN_DIR}/exp7_model_latent.json", flush=True)


def make_fig(out, path):
    S = [f"{lv['side']}x{lv['side']}\n(n={lv['n']})" for lv in out["levels"]]
    x = np.arange(len(S))
    def col(k):
        return [lv[k] for lv in out["levels"]]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ax[0].plot(x, col("q_to_llama_pred"), "-o", color="tab:blue", label="Qwen → Llama's predictive state")
        ax[0].plot(x, col("q_to_ideal"), "-o", color="0.5", label="Qwen → ideal walk (process only)")
        ax[0].plot(x, col("q_to_deviation"), "-o", color="tab:red", label="Qwen → Llama's DEVIATION from ideal")
        ax[0].plot(x, col("llama_self_ceiling"), "--", color="tab:blue", alpha=.5, label="ceiling: Llama → self")
        ax[0].axhline(0, color=".7", lw=.6); ax[0].set_ylim(-0.2, 1.0)
        ax[0].set_xticks(x); ax[0].set_xticklabels(S)
        ax[0].set_xlabel("hidden-state scale (grid size)"); ax[0].set_ylabel("decode R²")
        ax[0].set_title("Qwen reconstructs LLAMA's latent from Llama's emissions", fontsize=9)
        ax[0].legend(fontsize=7)
        ax[1].plot(x, col("q_to_llama_resid"), "-o", color="tab:green", label="Qwen → Llama residual (PCA)")
        ax[1].plot(x, col("llama_vs_ideal_r2"), "-o", color="k", label="Llama vs ideal (higher=less idiosyncratic)")
        ax[1].axhline(0, color=".7", lw=.6); ax[1].set_ylim(-0.2, 1.0)
        ax[1].set_xticks(x); ax[1].set_xticklabels(S)
        ax[1].set_xlabel("hidden-state scale (grid size)"); ax[1].set_ylabel("R²")
        ax[1].set_title("Literal internal state + Llama's idiosyncrasy", fontsize=9); ax[1].legend(fontsize=7)
        fig.suptitle("[scaled grid] Exp7 — Llama generates; Qwen observes Llama's emissions and reconstructs "
                     "Llama's own predictive latent (& its deviation from ideal)", fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
