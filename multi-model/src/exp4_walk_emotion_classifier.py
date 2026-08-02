"""Exp4 -- Can Qwen's activations CLASSIFY the emotion Llama was primed with,
from the walk alone?

Stronger than projecting onto a fixed sadness axis (Exp2/3): we prime Llama with
a HAPPY story vs a SAD story (natural-language context, no steering), have it
generate many walks under each, let Qwen read ONLY the walks, and train a linear
classifier on Qwen's per-walk activations to separate happy- vs sad-primed walks.
Per-layer cross-validated accuracy vs a label-shuffle null answers: does the
priming emotion leave a decodable trace in the neutral-word walk that survives
the hop into a different model?

  1. Llama generates NWALK walks per condition (happy_story / sad_story prefix),
     constrained to the 16 node words. Distinct RNG per walk.
  2. Qwen reads each walk; we pool its residual over the walk's node tokens ->
     one (H,) vector per walk per layer (common.walk_reps_all_layers).
  3. Per Qwen layer: StandardScaler -> PCA -> logistic regression, stratified
     k-fold CV. Report mean accuracy + AUC, and a label-shuffle null (chance).

Requires nothing from the emotion vectors (this is a supervised decode). The
HAPPY/SAD stories are structural mirrors (same skeleton, opposite valence) so the
classifier keys on emotion, not content.

Env: PRESET GRAPH NWALK(50) XCTX(40) GSTEPS(120) TEMP(1.0) CTXLO(1) KFOLD(5)
     PCADIM(40) NSHUF(50) RUN_DIR DEVICE
Out: <RUN_DIR>/exp4_walk_emotion_classifier.json + .pdf  (+ walk_reps npz cache)
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

GRAPH = os.environ.get("GRAPH", "square_grid")
NWALK = int(os.environ.get("NWALK", "50" if C.PRESET != "smoke" else "10"))
XCTX = int(os.environ.get("XCTX", "40" if C.PRESET != "smoke" else "10"))
GSTEPS = int(os.environ.get("GSTEPS", "120" if C.PRESET != "smoke" else "30"))
TEMP = float(os.environ.get("TEMP", "1.0"))
CTXLO = int(os.environ.get("CTXLO", "1"))
KFOLD = int(os.environ.get("KFOLD", "5"))
PCADIM = int(os.environ.get("PCADIM", "40"))
NSHUF = int(os.environ.get("NSHUF", "50" if C.PRESET != "smoke" else "10"))
READER = os.environ.get("READER", "Qwen")     # model whose activations we classify (Qwen or Llama)
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")
SUF = "" if READER == "Qwen" else f"_{READER}"
NPZ = os.path.join(RUN_DIR, f"exp4_walk_reps{SUF}.npz")

CONDS = {"happy_story": C.HAPPY_STORY, "sad_story": C.SAD_STORY}


def capture():
    dev = C.default_device()
    cfg = C.make_cfg(GRAPH, n_walks=NWALK, walk_length=max(XCTX, 8), device=dev)
    graph, n, coords = C.build_grid(cfg)
    # NWALK seed prefixes (distinct start nodes cycle through the grid)
    seeds = G.generate_walks(graph, cfg)

    print(f"[exp4] loading Llama; generating {NWALK}x{len(CONDS)} walks", flush=True)
    llama, ltok = C.load_model("Llama", cfg)
    cand = C.candidate_token_ids(ltok, graph, dev)
    walks, labels, beh = [], [], {c: [] for c in CONDS}
    for ci, (cond, prefix) in enumerate(CONDS.items()):
        for wi in range(NWALK):
            seed = seeds[wi % len(seeds)]
            nodes, b = C.generate_walk(llama, ltok, graph, cand, dev, seed.nodes[:XCTX], GSTEPS,
                                       temp=TEMP, rng=np.random.default_rng(4000 + ci * 1000 + wi),
                                       prefix=prefix)
            walks.append(C.mkwalk(nodes, graph))
            labels.append(ci)
            beh[cond].append(b)
    beho = {c: {"validity": float(np.nanmean([x["validity"] for x in beh[c]])),
                "nbr_mass": float(np.nanmean([x["nbr_mass"] for x in beh[c]]))} for c in CONDS}
    print(f"[exp4] gen validity: " + ", ".join(f"{c}={beho[c]['validity']:.2f}" for c in CONDS), flush=True)
    C.free(llama, ltok)

    print(f"[exp4] loading {READER}; pooling per-walk reps (all layers)", flush=True)
    reader, rtok = C.load_model(READER, cfg)
    reps = C.walk_reps_all_layers(reader, rtok, walks, dev, ctxlo=CTXLO)   # (2*NWALK, nL, H)
    C.free(reader, rtok)

    os.makedirs(RUN_DIR, exist_ok=True)
    np.savez_compressed(NPZ, reps=reps.astype(np.float32), labels=np.array(labels),
                        conds=np.array(list(CONDS)), behaviour=json.dumps(beho))
    print(f"[exp4] cached reps -> {NPZ}  shape={reps.shape}", flush=True)


def classify():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, accuracy_score

    import math
    z = np.load(NPZ, allow_pickle=True)
    reps = z["reps"]; y = z["labels"]; conds = list(z["conds"])
    beho = json.loads(str(z["behaviour"]))
    n, nL, H = reps.shape
    # PCA dim must stay below the smallest CV training-fold size (n - largest test fold)
    train_min = n - math.ceil(n / KFOLD)
    pcad = max(2, min(PCADIM, train_min - 1, H))

    # Per layer, reduce features ONCE (StandardScaler -> randomized PCA). This is
    # UNSUPERVISED (never sees labels), so cross-validating only the logistic head
    # on the reduced features is ~100x faster with negligible leakage -- and since
    # the outcome is a null, any residual optimism only strengthens it.
    def reduce_layer(X):
        Z = StandardScaler().fit_transform(X)
        return PCA(n_components=pcad, svd_solver="randomized", random_state=0).fit_transform(Z)

    def acc_cv(P, yy, seed):
        skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=seed)
        return accuracy_score(yy, cross_val_predict(LogisticRegression(max_iter=2000), P, yy, cv=skf))

    def auc_cv(P, yy, seed):
        skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=seed)
        prob = cross_val_predict(LogisticRegression(max_iter=2000), P, yy, cv=skf, method="predict_proba")[:, 1]
        return roc_auc_score(yy, prob)

    acc, auc, null_acc_mean, null_acc_std = [], [], [], []
    rng = np.random.default_rng(0)
    for L in range(nL):
        P = reduce_layer(reps[:, L, :])
        a = acc_cv(P, y, seed=L); u = auc_cv(P, y, seed=L)   # real: accuracy + AUC
        acc.append(a); auc.append(u)
        nulls = [acc_cv(P, rng.permutation(y), seed=1000 + s) for s in range(NSHUF)]  # null: accuracy only
        null_acc_mean.append(float(np.mean(nulls))); null_acc_std.append(float(np.std(nulls)))
        print(f"[exp4] L{L:2d}: acc={a:.3f} auc={u:.3f}  null={np.mean(nulls):.3f}±{np.std(nulls):.3f}", flush=True)

    peak = int(np.argmax(acc))
    out = {"graph": GRAPH, "conds": conds, "reader": READER, "n_per_cond": int((y == 0).sum()),
           "nL": nL, "pcadim": pcad, "kfold": KFOLD, "nshuf": NSHUF,
           "behaviour": beho, "acc": acc, "auc": auc,
           "null_acc_mean": null_acc_mean, "null_acc_std": null_acc_std,
           "peak_layer": peak, "peak_acc": acc[peak], "peak_auc": auc[peak],
           "peak_null": null_acc_mean[peak]}
    json.dump(out, open(os.path.join(RUN_DIR, f"exp4_walk_emotion_classifier{SUF}.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, f"exp4_walk_emotion_classifier{SUF}.pdf"))
    print(f"[exp4] DONE peak L{peak}: acc={acc[peak]:.3f} auc={auc[peak]:.3f} "
          f"(null {null_acc_mean[peak]:.3f}) -> {RUN_DIR}/exp4_walk_emotion_classifier.json", flush=True)


def make_fig(out, path):
    rd = out.get("reader", "Qwen")
    L = list(range(out["nL"]))
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ax[0].plot(L, out["acc"], "-o", ms=3, color="tab:red", label="happy-vs-sad CV accuracy")
        nm = np.array(out["null_acc_mean"]); ns = np.array(out["null_acc_std"])
        ax[0].fill_between(L, nm - 2 * ns, nm + 2 * ns, color="gray", alpha=.3, label="label-shuffle null ±2σ")
        ax[0].axhline(0.5, color=".7", lw=.6); ax[0].set_ylim(0.3, 1.0)
        ax[0].axvline(out["peak_layer"], color="tab:red", ls=":", lw=1)
        ax[0].set_xlabel(f"{rd} layer"); ax[0].set_ylabel("accuracy")
        ax[0].set_title(f"Decode Llama's priming emotion from {rd}'s walk reps "
                        f"(peak L{out['peak_layer']}={out['peak_acc']:.2f})", fontsize=9)
        ax[0].legend(fontsize=8)
        ax[1].plot(L, out["auc"], "-o", ms=3, color="tab:blue", label="ROC AUC")
        ax[1].axhline(0.5, color=".7", lw=.6); ax[1].set_ylim(0.3, 1.0)
        ax[1].set_xlabel(f"{rd} layer"); ax[1].set_ylabel("AUC")
        ax[1].set_title("happy-vs-sad separability by layer", fontsize=9); ax[1].legend(fontsize=8)
        b = out["behaviour"]
        fig.suptitle(f"[{out['graph']}] Exp4 — Llama primed happy vs sad → walk → {rd}; linear classifier on "
                     f"{rd} activations ({out['n_per_cond']}/class, {out['kfold']}-fold, PCA{out['pcadim']}). "
                     f"Llama walk validity " + "/".join(f"{c}={b[c]['validity']:.2f}" for c in out["conds"]), fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    if not os.path.exists(NPZ):
        capture()
    classify()


if __name__ == "__main__":
    main()
