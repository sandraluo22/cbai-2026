"""Exp8 -- Emotion transfer vs CONTEXT LENGTH, for Llama->Qwen AND Llama->Llama.

Revisits the emotion-transfer test (happy vs sad priming) with two changes:
  (1) MUCH LONGER walks -- the earlier runs (~120-220 steps) sat at chance, but
      Exp3 hinted the effect grows with context, so maybe a weak per-step
      emotional bias ACCUMULATES over a long random walk.
  (2) BOTH readers -- Qwen (cross-model) and Llama (within-model): does the
      shared representation of Llama-reading-itself expose emotion that the
      cross-model read misses?

Pipeline: prime Llama with a happy vs sad story, let it generate a long walk
(constrained to the 16 node words); feed the walk ONLY (no story) to a reader;
pool the reader's residual over the first `ctx` walk steps and train a linear
classifier (happy vs sad) per layer. Sweep `ctx`. Headline = decode accuracy vs
context length, one curve per reader, against a label-shuffle null.

Env: PRESET NWALK(40) XCTX(30) GSTEPS(400) TEMP(1.0) CTX_GRID(50,100,200,400)
     READERS(Qwen,Llama) PCADIM(60) KFOLD(5) NSHUF(30) RUN_DIR DEVICE
Out: <RUN_DIR>/exp8_emotion_context.json + .pdf
"""
from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import common as C  # noqa: E402
import graph as G   # noqa: E402
from models import resolve_token_spans  # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

NWALK = int(os.environ.get("NWALK", "40" if C.PRESET != "smoke" else "8"))
XCTX = int(os.environ.get("XCTX", "30" if C.PRESET != "smoke" else "8"))
GSTEPS = int(os.environ.get("GSTEPS", "400" if C.PRESET != "smoke" else "60"))
TEMP = float(os.environ.get("TEMP", "1.0"))
_CTX = "50,100,200,400" if C.PRESET != "smoke" else "15,35,60"
CTX_GRID = [int(x) for x in os.environ.get("CTX_GRID", _CTX).split(",")]
READERS = os.environ.get("READERS", "Qwen,Llama").split(",")
PCADIM = int(os.environ.get("PCADIM", "60"))
KFOLD = int(os.environ.get("KFOLD", "5"))
NSHUF = int(os.environ.get("NSHUF", "30" if C.PRESET != "smoke" else "10"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")

CONDS = {"happy": C.HAPPY_STORY, "sad": C.SAD_STORY}


def generate_walks_llama(cfg, graph, dev):
    """Prime Llama happy/sad; generate NWALK long walks per condition."""
    model, tok = C.load_model("Llama", cfg)
    cand = C.candidate_token_ids(tok, graph, dev)
    seeds = G.generate_walks(graph, cfg)
    walks, labels, beh = [], [], {c: [] for c in CONDS}
    for ci, (cond, prefix) in enumerate(CONDS.items()):
        for wi in range(NWALK):
            seed = seeds[wi % len(seeds)]
            nodes, b = C.generate_walk(model, tok, graph, cand, dev, seed.nodes[:XCTX], GSTEPS,
                                       temp=TEMP, rng=np.random.default_rng(8000 + ci * 1000 + wi),
                                       prefix=prefix)
            walks.append(C.mkwalk(nodes, graph)); labels.append(ci); beh[cond].append(b)
    C.free(model, tok)
    beho = {c: {"validity": float(np.nanmean([x["validity"] for x in beh[c]]))} for c in CONDS}
    return walks, np.array(labels), beho


@torch.no_grad() if torch is not None else (lambda f: f)
def capture_prefix_means(tag, walks, ctx_levels, cfg, dev):
    """One forward per walk (walk only, no story); per layer, pool the residual over
    the first `ctx` walk steps for each ctx in ctx_levels. -> (n_walks, nL, n_ctx, H)."""
    model, tok = C.load_model(tag, cfg)
    blocks = C.decoder_blocks(model)
    nL, H = C.n_layers(model), model.config.hidden_size
    grabbed = {}

    def mk(L):
        def hh(_m, _i, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh

    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    reps = np.zeros((len(walks), nL, len(ctx_levels), H), dtype=np.float32)
    try:
        for wi, wk in enumerate(walks):
            ids = tok(wk.text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk)
            single = [t[-1] for t in spans]
            grabbed.clear()
            model(input_ids=ids)
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()          # (n_steps, H)
                for ci, ctx in enumerate(ctx_levels):
                    m = min(ctx, rows.shape[0])
                    reps[wi, L, ci] = rows[:m].mean(0)
    finally:
        for h in hs:
            h.remove()
    C.free(model, tok)
    return reps


def classify(X, y, pcad, kfold, nshuf, seed=0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score
    import math
    n = len(y)
    pcad = max(2, min(pcad, n - math.ceil(n / kfold) - 1, X.shape[1]))
    Z = StandardScaler().fit_transform(X)
    P = PCA(n_components=pcad, svd_solver="randomized", random_state=0).fit_transform(Z)

    def acc(yy, s):
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=s)
        return accuracy_score(yy, cross_val_predict(LogisticRegression(max_iter=2000), P, yy, cv=skf))

    a = acc(y, seed)
    rng = np.random.default_rng(seed)
    nulls = [acc(rng.permutation(y), 1000 + s) for s in range(nshuf)]
    return a, float(np.mean(nulls)), float(np.std(nulls))


def main():
    dev = C.default_device()
    os.makedirs(RUN_DIR, exist_ok=True)
    cfg = C.make_cfg("square_grid", n_walks=max(NWALK, 8), walk_length=max(XCTX, 8), device=dev)
    graph, n, coords = C.build_grid(cfg)

    print(f"[exp8] Llama generates {NWALK}x2 walks (GSTEPS={GSTEPS})", flush=True)
    walks, labels, beho = generate_walks_llama(cfg, graph, dev)
    print(f"[exp8] gen validity: " + ", ".join(f"{c}={beho[c]['validity']:.2f}" for c in CONDS), flush=True)

    out = {"nwalk": NWALK, "gsteps": GSTEPS, "ctx_grid": CTX_GRID, "readers": READERS,
           "gen_validity": beho, "results": {}}
    for reader in READERS:
        print(f"[exp8] reader={reader}: capture prefix means", flush=True)
        reps = capture_prefix_means(reader, walks, CTX_GRID, cfg, dev)
        nL = reps.shape[1]
        rows = []
        for ci, ctx in enumerate(CTX_GRID):
            accs = [classify(reps[:, L, ci, :], labels, PCADIM, KFOLD, NSHUF) for L in range(nL)]
            peak = int(np.argmax([a[0] for a in accs]))
            a, nm, ns = accs[peak]
            rows.append({"ctx": ctx, "peak_layer": peak, "acc": a, "null_mean": nm, "null_std": ns})
            print(f"[exp8] {reader} ctx={ctx}: peak L{peak} acc={a:.3f} (null {nm:.3f}±{ns:.3f})", flush=True)
        out["results"][reader] = rows

    json.dump(out, open(os.path.join(RUN_DIR, "exp8_emotion_context.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, "exp8_emotion_context.pdf"))
    print(f"[exp8] DONE -> {RUN_DIR}/exp8_emotion_context.json", flush=True)


def make_fig(out, path):
    cols = {"Qwen": "tab:blue", "Llama": "tab:orange"}
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=120)
        for reader in out["readers"]:
            rows = out["results"][reader]
            ctx = [r["ctx"] for r in rows]
            ax.plot(ctx, [r["acc"] for r in rows], "-o", color=cols.get(reader), label=f"{reader} reader")
            nm = np.array([r["null_mean"] for r in rows]); ns = np.array([r["null_std"] for r in rows])
            ax.fill_between(ctx, nm - 2 * ns, nm + 2 * ns, color=cols.get(reader), alpha=.15)
        ax.axhline(0.5, color=".6", ls=":", lw=.8, label="chance")
        ax.set_ylim(0.3, 1.0); ax.set_xlabel("context length (walk steps the reader pools)")
        ax.set_ylabel("happy-vs-sad decode accuracy")
        gv = out["gen_validity"]
        ax.set_title(f"Exp8 — emotion transfer vs context length (long walks).\n"
                     f"Llama gen validity happy={gv['happy']['validity']:.2f}/sad={gv['sad']['validity']:.2f}. "
                     f"Shaded = label-shuffle null ±2σ.", fontsize=9)
        ax.legend(fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
