"""Generalized offline analysis for a dueling-context pair run (run_pair.py + capture_fresh.py).

Reads the graph names from the npz (ctx_names). Produces, per RUN_OUT dir:
  pair_r2.png         R^2 from A-coords vs B-coords by layer, per window, both contexts
  pair_pca.png        PCA maps at layer 26, base vs joint windows
  pair_candidates.png candidate-structure R^2 for the late window (+ fresh control)
  pair_procrustes.png shape-convergence curves (incl. fresh control)
  pair_behavior.png   prob. mass each generator puts on A-nbrs vs B-nbrs over time
  pair_summary.json   all headline numbers + trap/occupancy stats
NO RSA anywhere.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RUN_OUT", os.path.join(HERE, "runs", "out_pair"))
WINS = ("base", "joint_early", "joint_mid", "joint_late")


def norm_laplacian_modes(W):
    W = W.astype(float)
    d = W.sum(1)
    di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(len(W)) - di[:, None] * W * di[None, :]
    w, U = np.linalg.eigh(L)
    return w, U


def r2_from_feats(Hc, F):
    F = np.atleast_2d(np.asarray(F, float).T).T
    Fc = F - F.mean(0)
    Fc = Fc / np.maximum(Fc.std(0), 1e-12)
    B, *_ = np.linalg.lstsq(Fc, Hc, rcond=None)
    resid = Hc - Fc @ B
    return float(1.0 - (resid ** 2).sum() / np.maximum((Hc ** 2).sum(), 1e-12))


def unit_shape(X):
    X = X - X.mean(0)
    return X / np.maximum(np.linalg.norm(X), 1e-12)


def procrustes_sim(A, B):
    A, B = unit_shape(A), unit_shape(B)
    Ua, Sa, _ = np.linalg.svd(A, full_matrices=False)
    Ub, Sb, _ = np.linalg.svd(B, full_matrices=False)
    M = (Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :]
    return float(np.linalg.svd(M, compute_uv=False).sum())


def main():
    z = np.load(os.path.join(OUT, "nodemeans_dueling.npz"), allow_pickle=False)
    zf = np.load(os.path.join(OUT, "nodemeans_fresh.npz"), allow_pickle=False)
    log = json.load(open(os.path.join(OUT, "gen_log.json")))
    GA, GB = [str(x) for x in z["ctx_names"]]
    nL = int(z["n_layers"][0])
    words = [str(w) for w in z["words"]]
    n = len(words)
    Aa = z[f"adjacency_{GA}"].astype(bool)
    Ab = z[f"adjacency_{GB}"].astype(bool)
    ca, cb = z[f"coords_{GA}"], z[f"coords_{GB}"]
    P, T = log["npairs"], log["tgen"]
    joint = np.array([[log["steps"][f"pair{p}"][t]["node"] for t in range(T)]
                      for p in range(P)])

    A_union = Aa | Ab
    shared = Aa & Ab
    a_only = Aa & ~Ab
    b_only = Ab & ~Aa
    non_edge = ~A_union & ~np.eye(n, dtype=bool)
    print(f"pair {GA}+{GB}: shared={shared.sum()//2} {GA}-only={a_only.sum()//2} "
          f"{GB}-only={b_only.sum()//2} non={non_edge.sum()//2}")

    C = np.zeros((n, n))
    for p in range(P):
        for t in range(max(300, 1), T):
            C[joint[p, t - 1], joint[p, t]] += 1
    W_emp = C + C.T
    np.fill_diagonal(W_emp, 0)
    mass = W_emp.sum()
    stream_mass = {"shared": float(W_emp[shared].sum() / mass),
                   f"{GA}_only": float(W_emp[a_only].sum() / mass),
                   f"{GB}_only": float(W_emp[b_only].sum() / mass),
                   "non_edge": float(W_emp[non_edge].sum() / mass)}
    print("late-stream transition mass:", {k: round(v, 3) for k, v in stream_mass.items()})
    occ = np.array([(joint[:, 300:] == i).sum() for i in range(n)], float)
    cnt = {}
    tot = 0
    for p in range(P):
        for t in range(301, T):
            tot += 1
            k = tuple(sorted((joint[p, t - 1], joint[p, t])))
            cnt[k] = cnt.get(k, 0) + 1
    top_pairs = [(words[a], words[b], round(c / tot, 3))
                 for (a, b), c in sorted(cnt.items(), key=lambda kv: -kv[1])[:3]]
    print("top oscillating pairs:", top_pairs)

    _, Uu = norm_laplacian_modes(A_union.astype(float))
    _, Ue = norm_laplacian_modes(W_emp)

    def H(key, L):
        Hm = (zf if key.startswith("fresh") else z)[f"{key}_layer_{L}"].astype(np.float64)
        return Hm - Hm.mean(0)

    # ---- fig 1: coordinate-regression R^2 by layer ------------------------
    res = {c: {win: {"r2_a": [], "r2_b": []} for win in WINS} for c in (GA, GB)}
    for c in (GA, GB):
        for win in WINS:
            for L in range(nL):
                Hc = H(f"{c}_{win}", L)
                res[c][win]["r2_a"].append(r2_from_feats(Hc, ca))
                res[c][win]["r2_b"].append(r2_from_feats(Hc, cb))
    colors = {"base": "0.55", "joint_early": "#8ecae6", "joint_mid": "#219ebc",
              "joint_late": "#023047"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, c in zip(axes, (GA, GB)):
        for win in WINS:
            ax.plot(res[c][win]["r2_a"], color=colors[win], ls="-")
            ax.plot(res[c][win]["r2_b"], color=colors[win], ls="--")
        ax.set_title(f"{c.upper()}-primed context")
        ax.set_xlabel("layer"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"R$^2$ of node-means from 2D coords")
    axes[0].plot([], [], color="k", ls="-", label=f"{GA} coords")
    axes[0].plot([], [], color="k", ls="--", label=f"{GB} coords")
    for win in WINS:
        axes[0].plot([], [], color=colors[win], label=win)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{GA}+{GB}: which geometry explains the representation? "
                 f"(solid={GA}, dashed={GB})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"pair_r2.{ext}"), dpi=160)
    plt.close(fig)

    # ---- fig 2: PCA maps ---------------------------------------------------
    Lshow = min(26, nL - 1)
    hue = plt.cm.hsv(np.arange(n) / n)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    for i, c in enumerate((GA, GB)):
        for j, win in enumerate(WINS):
            ax = axes[i, j]
            Hc = H(f"{c}_{win}", Lshow)
            U, s, _ = np.linalg.svd(Hc, full_matrices=False)
            P2 = U[:, :2] * s[:2]
            for a in range(n):
                for b in range(a + 1, n):
                    if shared[a, b]:
                        ax.plot(P2[[a, b], 0], P2[[a, b], 1], color="0.35", lw=1.1, zorder=1)
                    elif a_only[a, b]:
                        ax.plot(P2[[a, b], 0], P2[[a, b], 1], color="#219ebc", lw=0.8,
                                alpha=0.8, zorder=1)
                    elif b_only[a, b]:
                        ax.plot(P2[[a, b], 0], P2[[a, b], 1], color="crimson", lw=0.8,
                                alpha=0.8, zorder=2)
            ax.scatter(P2[:, 0], P2[:, 1], c=hue, s=60, zorder=3, edgecolors="k", lw=0.4)
            for a in range(n):
                ax.annotate(words[a], P2[a], fontsize=6, xytext=(2, 2),
                            textcoords="offset points")
            ax.set_title(f"{c}-primed | {win}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{GA}+{GB} PCA layer {Lshow} (dark = shared, blue = {GA}-only, "
                 f"red = {GB}-only edges; hue = node index)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"pair_pca.{ext}"), dpi=160)
    plt.close(fig)

    # ---- fig 3: candidate structures for the late window -------------------
    FEATS = {f"{GA} coords": ca, f"{GB} coords": cb, "union modes 1-2": Uu[:, 1:3],
             "union modes 1-4": Uu[:, 1:5], "empirical modes 1-2": Ue[:, 1:3],
             "log-occupancy (1D)": np.log(np.maximum(occ, 1.0))[:, None]}
    keys = [(f"{GA}_joint_late", f"{GA.upper()}-primed late"),
            (f"{GB}_joint_late", f"{GB.upper()}-primed late"),
            ("fresh_late", "FRESH late")]
    r2c = {k: {f: [r2_from_feats(H(k, L), F) for L in range(nL)] for f, F in FEATS.items()}
           for k, _ in keys}
    fcol = {f"{GA} coords": "#219ebc", f"{GB} coords": "crimson",
            "union modes 1-2": "#fb8500", "union modes 1-4": "#8338ec",
            "empirical modes 1-2": "#2a9d8f", "log-occupancy (1D)": "0.5"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    for ax, (k, title) in zip(axes, keys):
        for f in FEATS:
            ax.plot(r2c[k][f], color=fcol[f], label=f, ls=":" if "occupancy" in f else "-")
        ax.set_title(title); ax.set_xlabel("layer"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"R$^2$"); axes[0].legend(fontsize=7)
    fig.suptitle(f"{GA}+{GB}: candidate structures for the converged geometry")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"pair_candidates.{ext}"), dpi=160)
    plt.close(fig)

    # ---- fig 4: Procrustes convergence -------------------------------------
    pairs = [(f"{GA}_base vs {GB}_base", f"{GA}_base", f"{GB}_base", "0.55", "-"),
             ("late vs late", f"{GA}_joint_late", f"{GB}_joint_late", "#023047", "-"),
             (f"{GA}_late vs FRESH", f"{GA}_joint_late", "fresh_late", "#fb8500", "--"),
             (f"{GB}_late vs FRESH", f"{GB}_joint_late", "fresh_late", "#2a9d8f", "--"),
             (f"{GA}: base vs late", f"{GA}_base", f"{GA}_joint_late", "#8ecae6", ":"),
             (f"{GB}: base vs late", f"{GB}_base", f"{GB}_joint_late", "crimson", ":")]
    sims = {}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for label, a, b, col, ls in pairs:
        ys = [procrustes_sim(H(a, L), H(b, L)) for L in range(nL)]
        sims[label] = ys
        ax.plot(ys, color=col, ls=ls, label=label)
    ax.set_xlabel("layer"); ax.set_ylabel("Procrustes shape similarity")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{GA}+{GB}: shape convergence")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"pair_procrustes.{ext}"), dpi=160)
    plt.close(fig)

    # ---- fig 5: behavior ----------------------------------------------------
    series = {(w, k): [] for w in (GA, GB) for k in ("a", "b")}
    ts = {GA: [], GB: []}
    for t in range(T):
        who = GB if t % 2 == 0 else GA
        va = np.mean([log["steps"][f"pair{p}"][t][f"p_{GA}_nbrs"] for p in range(P)])
        vb = np.mean([log["steps"][f"pair{p}"][t][f"p_{GB}_nbrs"] for p in range(P)])
        series[(who, "a")].append(va); series[(who, "b")].append(vb)
        ts[who].append(t)

    def smooth(x, k=15):
        x = np.asarray(x, float)
        return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) >= k else x

    deg_a, deg_b = Aa.sum() / n, Ab.sum() / n
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, who in zip(axes, (GA, GB)):
        for k, col, lab in (("a", "#219ebc", f"mass on {GA.upper()}-nbrs"),
                            ("b", "crimson", f"mass on {GB.upper()}-nbrs")):
            y = smooth(series[(who, k)])
            ax.plot(ts[who][: len(y)], y, color=col, label=lab)
        ax.axhline(deg_a / (n - 1), color="#219ebc", ls=":", lw=1)
        ax.axhline(deg_b / (n - 1), color="crimson", ls=":", lw=1)
        ax.set_title(f"{who.upper()}-primed context's predictions")
        ax.set_xlabel("joint generation step"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("prob. mass on neighbours of prev node")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{GA}+{GB}: behavioral coupling (dotted = chance)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"pair_behavior.{ext}"), dpi=160)
    plt.close(fig)

    summary = {
        "pair": [GA, GB], "layer_shown": Lshow,
        "edge_counts": {"shared": int(shared.sum() // 2), f"{GA}_only": int(a_only.sum() // 2),
                        f"{GB}_only": int(b_only.sum() // 2)},
        "late_stream_edge_mass": stream_mass,
        "top_oscillating_pairs": top_pairs,
        "occupancy_late_per_node": {words[i]: int(occ[i]) for i in range(n)},
        "headline_layer": {c: {win: {"r2_a": res[c][win]["r2_a"][Lshow],
                                     "r2_b": res[c][win]["r2_b"][Lshow]} for win in WINS}
                           for c in (GA, GB)},
        "r2_deep_mean_24_31": {k: {f: float(np.mean(v[24:])) for f, v in r2c[k].items()}
                               for k in r2c},
        "procrustes_deep_mean_24_31": {lab: float(np.mean(v[24:])) for lab, v in sims.items()},
    }
    json.dump(summary, open(os.path.join(OUT, "pair_summary.json"), "w"), indent=2)
    print(json.dumps({k: summary[k] for k in
                      ("headline_layer", "r2_deep_mean_24_31", "procrustes_deep_mean_24_31")},
                     indent=2))
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
