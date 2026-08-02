"""Exp6 -- One model as an HMM, the other as the observer: belief-state filtering
of a HIDDEN random walk, swept over an observability dial.

A random walk on the 4x4 grid (transition = uniform over neighbours) is the HMM.
Its emission is COARSENED: the 16 nodes are grouped into A symbols, so a token
reveals only the group, not the node -> the current node is a genuine hidden
latent the observer must FILTER from history + grid structure. A is the dial:

  A=16 -> every node its own symbol (fully observed; = Exp1, belief one-hot)
  A=1  -> one symbol for all nodes (latent invisible; the 'emotion' case)
  A in {8,4,2} -> genuine belief distributions that sharpen as the walk moves.

Both Llama and Qwen read the SAME aliased symbol stream (frozen, in-context). We
forward-filter the EXACT Bayes-optimal belief b_t = P(node_t | symbols_1..t) from
the known grid+aliasing, then per layer / per A measure:
  - node_acc   : linear (logistic) decode of the true hidden node from the residual
  - belief_r2  : ridge decode of the full belief simplex b_t (the mixed state)
  - optimal_acc: argmax b_t == true node  (model-independent observability ceiling)
  - xmodel     : correlation between Qwen's decoded belief and LLAMA's decoded
                 belief (does the observer track the OTHER model's latent, not just
                 the optimal one)

Headline = recovery vs the dial A, three curves (Qwen / Llama / Bayes-optimal).

Env: PRESET NWALK(16) WLEN(300) CTXLO(60) ALIAS(16,8,4,2,1) PCADIM(80)
     TESTFRAC(0.3) READERS(Llama,Qwen) RUN_DIR DEVICE
Out: <RUN_DIR>/exp6_latent_filtering.json + .pdf
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
from config import WORDS  # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

NWALK = int(os.environ.get("NWALK", "16" if C.PRESET != "smoke" else "6"))
WLEN = int(os.environ.get("WLEN", "300" if C.PRESET != "smoke" else "50"))
CTXLO = int(os.environ.get("CTXLO", "60" if C.PRESET != "smoke" else "5"))
PCADIM = int(os.environ.get("PCADIM", "80"))
TESTFRAC = float(os.environ.get("TESTFRAC", "0.3"))
ALIAS = [int(x) for x in os.environ.get("ALIAS", "16,8,4,2,1").split(",")]
READERS = os.environ.get("READERS", "Llama,Qwen").split(",")
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")


def transition_matrix(graph):
    n = graph.n_nodes
    T = np.zeros((n, n))
    for i in range(n):
        nb = graph.neighbors(i)
        for j in nb:
            T[i, j] = 1.0 / len(nb)
    return T


def alias_map(n, A, seed=0):
    """Fixed spatially-mixed grouping of n nodes into A symbols (equal groups when
    A | n). Returns alias_of (n,) in [0,A)."""
    perm = np.random.default_rng(seed).permutation(n)
    return perm % A


def forward_belief(sym_seq, T, alias_of, n):
    """Exact Bayes-optimal belief b_t = P(node_t | symbols_1..t), deterministic
    many-to-one emission (symbol reveals the node's group)."""
    b = np.zeros((len(sym_seq), n))
    Lmask = {s: (alias_of == s).astype(float) for s in set(sym_seq)}
    a = np.ones(n) / n * Lmask[sym_seq[0]]
    a = a / a.sum()
    b[0] = a
    for t in range(1, len(sym_seq)):
        a = (a @ T) * Lmask[sym_seq[t]]
        s = a.sum()
        a = a / s if s > 0 else np.ones(n) / n
        b[t] = a
    return b


def aliased_walk(nodes, alias_of):
    syms = [int(alias_of[j]) for j in nodes]
    words = [WORDS[s] for s in syms]           # A distinct concept words as the symbol alphabet
    return syms, words


@torch.no_grad() if torch is not None else (lambda f: f)
def capture_and_targets(model, tok, walks, alias_of, T, n, dev, ctxlo):
    """Feed each aliased walk; per occurrence (ctx>=ctxlo) collect residuals at every
    layer + the aligned belief vector, true node, and walk id."""
    blocks = C.decoder_blocks(model)
    nL, H = C.n_layers(model), model.config.hidden_size
    grabbed = {}

    def mk(L):
        def hh(_m, _i, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh

    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    Xrows = {L: [] for L in range(nL)}
    Yb, yn, grp = [], [], []
    try:
        for wi, wk in enumerate(walks):
            syms, words = aliased_walk(wk.nodes, alias_of)
            text = " ".join(words)
            b = forward_belief(syms, T, alias_of, n)
            from graph import Walk                    # aliased Walk for span resolution
            awk = Walk(walk_id=wi, nodes=wk.nodes, words=words)
            ids = tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
            spans = resolve_token_spans(tok, awk)
            cl = np.arange(1, len(wk.nodes) + 1)
            keep = [s for s in range(len(wk.nodes)) if cl[s] >= ctxlo]
            grabbed.clear()
            model(input_ids=ids)
            for L in range(nL):
                rows = grabbed[L][0][[spans[s][-1] for s in keep]].float().cpu().numpy()
                Xrows[L].append(rows)
            for s in keep:
                Yb.append(b[s]); yn.append(wk.nodes[s]); grp.append(wi)
    finally:
        for h in hs:
            h.remove()
    X = {L: np.concatenate(Xrows[L], 0) for L in range(nL)}
    return X, np.array(Yb), np.array(yn), np.array(grp)


def probe_layer(X, Yb, yn, grp, pcad, testfrac, seed=0):
    """Held-out-by-walk: logistic node accuracy + ridge belief R^2 + decoded belief
    (test rows) for cross-model comparison."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score

    walks = np.unique(grp)
    rng = np.random.default_rng(seed)
    test_w = set(rng.choice(walks, max(1, int(round(testfrac * len(walks)))), replace=False))
    te = np.array([g in test_w for g in grp]); tr = ~te
    if tr.sum() < 10 or te.sum() < 5:
        return None
    sc = StandardScaler().fit(X[tr])
    Xs_tr, Xs_te = sc.transform(X[tr]), sc.transform(X[te])
    k = min(pcad, Xs_tr.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(Xs_tr)
    Ptr, Pte = pca.transform(Xs_tr), pca.transform(Xs_te)
    # node accuracy
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ptr, yn[tr])
    nacc = accuracy_score(yn[te], clf.predict(Pte))
    # belief R^2 (ridge, multi-output)
    rg = Ridge(alpha=10.0).fit(Ptr, Yb[tr])
    bpred = rg.predict(Pte)
    sst = ((Yb[te] - Yb[tr].mean(0)) ** 2).sum()
    # when the latent is (near-)fully hidden the belief is ~constant -> sst~0 and
    # R^2 is meaningless; report nan rather than a blown-up value.
    br2 = float(1 - ((Yb[te] - bpred) ** 2).sum() / sst) if sst > 1e-6 else float("nan")
    return {"node_acc": float(nacc), "belief_r2": br2, "test_mask": te, "bpred": bpred}


def run_model(tag, walks, alias_of, T, n, dev, pcad):
    cfg = C.make_cfg("square_grid", n_walks=NWALK, walk_length=WLEN, device=dev)
    print(f"[exp6] loading {tag}", flush=True)
    model, tok = C.load_model(tag, cfg)
    X, Yb, yn, grp = capture_and_targets(model, tok, walks, alias_of, T, n, dev, CTXLO)
    C.free(model, tok)
    nL = len(X)
    per = [probe_layer(X[L], Yb, yn, grp, pcad, TESTFRAC) for L in range(nL)]
    nacc = [p["node_acc"] if p else np.nan for p in per]
    br2 = [p["belief_r2"] if p else np.nan for p in per]
    peak = int(np.nanargmax(nacc))
    return {"node_acc": nacc, "belief_r2": br2, "peak_layer": peak,
            "peak_node_acc": nacc[peak], "peak_belief_r2": br2[peak],
            "_peak_bpred": per[peak]["bpred"], "_peak_testmask": per[peak]["test_mask"]}


def main():
    dev = C.default_device()
    os.makedirs(RUN_DIR, exist_ok=True)
    cfg = C.make_cfg("square_grid", n_walks=NWALK, walk_length=WLEN, device=dev)
    graph, n, coords = C.build_grid(cfg)
    walks = G.generate_walks(graph, cfg)               # true hidden process (uniform walk)
    T = transition_matrix(graph)

    out = {"graph": "square_grid", "n_nodes": n, "nwalk": NWALK, "wlen": WLEN,
           "ctxlo": CTXLO, "alias_levels": ALIAS, "readers": READERS, "levels": []}
    for A in ALIAS:
        alias_of = alias_map(n, A, seed=0)
        # optimal-filter ceiling (model-independent), on ctx>=CTXLO occurrences
        opt_hits, opt_tot = 0, 0
        for wk in walks:
            syms, _ = aliased_walk(wk.nodes, alias_of)
            b = forward_belief(syms, T, alias_of, n)
            for s in range(len(wk.nodes)):
                if s + 1 >= CTXLO:
                    opt_hits += int(np.argmax(b[s]) == wk.nodes[s]); opt_tot += 1
        opt_acc = opt_hits / max(opt_tot, 1)
        rec = {"A": A, "group_size": n // A, "optimal_node_acc": opt_acc, "models": {}}
        beliefs = {}
        for tag in READERS:
            r = run_model(tag, walks, alias_of, T, n, dev, PCADIM)
            beliefs[tag] = (r.pop("_peak_bpred"), r.pop("_peak_testmask"))
            rec["models"][tag] = r
            print(f"[exp6] A={A} {tag}: peak L{r['peak_layer']} node_acc={r['peak_node_acc']:.3f} "
                  f"belief_r2={r['peak_belief_r2']:.3f}  (optimal_acc={opt_acc:.3f})", flush=True)
        # cross-model belief agreement (align on the intersection of test occurrences)
        if len(READERS) == 2:
            (b1, m1), (b2, m2) = beliefs[READERS[0]], beliefs[READERS[1]]
            both = m1 & m2
            i1 = both[m1]; i2 = both[m2]
            if both.sum() > 5:
                a, b = b1[i1].ravel(), b2[i2].ravel()
                rec["xmodel_belief_corr"] = float(np.corrcoef(a, b)[0, 1])
                print(f"[exp6] A={A} xmodel belief corr({READERS[0]},{READERS[1]})="
                      f"{rec['xmodel_belief_corr']:.3f}", flush=True)
        out["levels"].append(rec)

    json.dump(out, open(os.path.join(RUN_DIR, "exp6_latent_filtering.json"), "w"), indent=2)
    make_fig(out, os.path.join(RUN_DIR, "exp6_latent_filtering.pdf"))
    print(f"[exp6] DONE -> {RUN_DIR}/exp6_latent_filtering.json", flush=True)


def make_fig(out, path):
    A = [lv["A"] for lv in out["levels"]]
    x = np.arange(len(A))
    readers = out["readers"]
    cols = {"Llama": "tab:orange", "Qwen": "tab:blue"}
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        ax[0].plot(x, [lv["optimal_node_acc"] for lv in out["levels"]], "-o", color="k", label="Bayes-optimal filter")
        for tag in readers:
            ax[0].plot(x, [lv["models"][tag]["peak_node_acc"] for lv in out["levels"]], "-o",
                       color=cols.get(tag, None), label=tag)
        ax[0].axhline(1.0 / out["n_nodes"], color=".7", ls=":", lw=.8, label="chance (1/16)")
        ax[0].set_xticks(x); ax[0].set_xticklabels([f"A={a}\n(g={out['n_nodes']//a})" for a in A])
        ax[0].set_ylim(0, 1.02); ax[0].set_xlabel("observability dial (symbols A)")
        ax[0].set_ylabel("hidden-node decode accuracy")
        ax[0].set_title("Recover Llama's hidden node from the aliased stream", fontsize=9)
        ax[0].legend(fontsize=8); ax[0].invert_xaxis()
        for tag in readers:
            ax[1].plot(x, [lv["models"][tag]["peak_belief_r2"] for lv in out["levels"]], "-o",
                       color=cols.get(tag, None), label=tag)
        ax[1].axhline(0, color=".7", lw=.6); ax[1].set_xticks(x); ax[1].set_xticklabels([f"A={a}" for a in A])
        ax[1].set_ylim(-0.2, 1.0); ax[1].set_xlabel("observability dial (symbols A)")
        ax[1].set_ylabel("belief-simplex decode R²")
        ax[1].set_title("Does the residual encode the belief (mixed state)?", fontsize=9)
        ax[1].legend(fontsize=8); ax[1].invert_xaxis()
        xc = [lv.get("xmodel_belief_corr", np.nan) for lv in out["levels"]]
        ax[2].plot(x, xc, "-o", color="tab:purple")
        ax[2].axhline(0, color=".7", lw=.6); ax[2].set_xticks(x); ax[2].set_xticklabels([f"A={a}" for a in A])
        ax[2].set_ylim(-0.2, 1.0); ax[2].set_xlabel("observability dial (symbols A)")
        ax[2].set_ylabel("corr(Qwen belief, Llama belief)")
        ax[2].set_title("Does Qwen track the SAME latent as Llama?", fontsize=9); ax[2].invert_xaxis()
        fig.suptitle(f"[square_grid] Exp6 — hidden random walk (HMM), observers filter the latent. "
                     f"A=16 fully observed (=Exp1) → A=1 invisible (=emotion).", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
