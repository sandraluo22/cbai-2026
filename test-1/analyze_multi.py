"""Analysis for an N-way dueling run (run_multi.py + capture_fresh.py). NO RSA.

Per RUN_OUT: stream composition, per-context coordinate R^2 (base -> late), candidate
structures for the late window (each graph's coords, N-way-union Laplacian modes,
empirical stream modes, log-occupancy), Fourier energy spectra, parity probes
(i%2 cycle parity + grid checkerboard), and all-pairs Procrustes convergence incl. the
fresh control.

Out: multi_r2.png, multi_pca.png, multi_procrustes.png, multi_summary.json
"""
from __future__ import annotations
import json, os
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RUN_OUT", os.path.join(HERE, "runs", "out_tri"))
WINS = ("base", "joint_early", "joint_mid", "joint_late")


def norm_lap_modes(W):
    W = W.astype(float)
    d = W.sum(1)
    di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(len(W)) - di[:, None] * W * di[None, :]
    return np.linalg.eigh(L)


def r2_from_feats(Hc, F):
    F = np.atleast_2d(np.asarray(F, float).T).T
    Fc = F - F.mean(0)
    Fc = Fc / np.maximum(Fc.std(0), 1e-12)
    B, *_ = np.linalg.lstsq(Fc, Hc, rcond=None)
    return float(1.0 - ((Hc - Fc @ B) ** 2).sum() / max((Hc ** 2).sum(), 1e-12))


def unit_shape(X):
    X = X - X.mean(0)
    return X / max(np.linalg.norm(X), 1e-12)


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
    NAMES = [str(x) for x in z["ctx_names"]]
    nL = int(z["n_layers"][0])
    words = [str(w) for w in z["words"]]
    n = len(words)
    adjs = {nm: z[f"adjacency_{nm}"].astype(bool) for nm in NAMES}
    coords = {nm: z[f"coords_{nm}"] for nm in NAMES}
    P, T = log["npairs"], log["tgen"]
    joint = np.array([[log["steps"][f"pair{p}"][t]["node"] for t in range(T)]
                      for p in range(P)])

    A_union = np.zeros((n, n), bool)
    for nm in NAMES:
        A_union |= adjs[nm]
    non_edge = ~A_union & ~np.eye(n, dtype=bool)

    # stream composition (late)
    C = np.zeros((n, n))
    for p in range(P):
        for t in range(301, T):
            C[joint[p, t - 1], joint[p, t]] += 1
    W_emp = C + C.T
    np.fill_diagonal(W_emp, 0)
    mass = W_emp.sum()
    stream = {nm: float(W_emp[adjs[nm]].sum() / mass) for nm in NAMES}
    stream["union"] = float(W_emp[A_union].sum() / mass)
    stream["non_edge"] = float(W_emp[non_edge].sum() / mass)
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
    print("late-stream validity mass (classes overlap):",
          {k: round(v, 3) for k, v in stream.items()})
    print("top oscillating pairs:", top_pairs)

    wu, Uu = norm_lap_modes(A_union.astype(float))
    _, Ue = norm_lap_modes(W_emp)

    def H(key, L):
        Hm = (zf if key.startswith("fresh") else z)[f"{key}_layer_{L}"].astype(np.float64)
        return Hm - Hm.mean(0)

    # ---- coordinate R^2 curves --------------------------------------------
    res = {c: {win: {nm: [] for nm in NAMES} for win in WINS} for c in NAMES}
    for c in NAMES:
        for win in WINS:
            for L in range(nL):
                Hc = H(f"{c}_{win}", L)
                for nm in NAMES:
                    res[c][win][nm].append(r2_from_feats(Hc, coords[nm]))
    wincol = {"base": "0.55", "joint_early": "#8ecae6", "joint_mid": "#219ebc",
              "joint_late": "#023047"}
    ls = {NAMES[0]: "-", NAMES[1]: "--", NAMES[2]: ":"} if len(NAMES) == 3 else \
         {nm: s for nm, s in zip(NAMES, ("-", "--", ":", "-."))}
    fig, axes = plt.subplots(1, len(NAMES), figsize=(5.4 * len(NAMES), 4.2), sharey=True)
    for ax, c in zip(axes, NAMES):
        for win in ("base", "joint_late"):
            for nm in NAMES:
                ax.plot(res[c][win][nm], color=wincol[win], ls=ls[nm])
        ax.set_title(f"{c.upper()}-primed context")
        ax.set_xlabel("layer"); ax.grid(alpha=0.3)
    for nm in NAMES:
        axes[0].plot([], [], color="k", ls=ls[nm], label=f"{nm} coords")
    for win in ("base", "joint_late"):
        axes[0].plot([], [], color=wincol[win], label=win)
    axes[0].set_ylabel(r"R$^2$"); axes[0].legend(fontsize=7)
    fig.suptitle("+".join(NAMES) + ": coordinate R2, base vs joint_late")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"multi_r2.{ext}"), dpi=160)
    plt.close(fig)

    # ---- candidate structures for late windows -----------------------------
    FEATS = {f"{nm} coords": coords[nm] for nm in NAMES}
    FEATS.update({"union modes 1-2": Uu[:, 1:3], "union modes 1-4": Uu[:, 1:5],
                  "union modes 1-6": Uu[:, 1:7], "empirical modes 1-2": Ue[:, 1:3],
                  "log-occupancy (1D)": np.log(np.maximum(occ, 1.0))[:, None]})
    late_keys = [f"{nm}_joint_late" for nm in NAMES] + ["fresh_late"]
    r2c = {k: {f: float(np.mean([r2_from_feats(H(k, L), F) for L in range(24, nL)]))
               for f, F in FEATS.items()} for k in late_keys}

    # ---- Fourier + parity probes -------------------------------------------
    i16 = np.arange(n)
    def fbasis(f):
        B = [np.cos(2 * np.pi * f * i16 / n)] + \
            ([np.sin(2 * np.pi * f * i16 / n)] if f < n // 2 else [])
        B = np.stack(B, 1); B -= B.mean(0)
        Q, _ = np.linalg.qr(B); return Q
    def deepE(key, Q):
        return float(np.mean([np.linalg.norm(Q.T @ H(key, L)) ** 2 /
                              (H(key, L) ** 2).sum() for L in range(24, nL)]))
    fourier = {key: {f"f{f}": deepE(key, fbasis(f)) for f in range(1, 9)}
               for key in [f"{nm}_base" for nm in NAMES] + late_keys}
    par_cyc = ((i16 % 2) * 2.0 - 1.0)
    par_cyc = (par_cyc - par_cyc.mean())[:, None] / np.linalg.norm(par_cyc - par_cyc.mean())
    cg = coords.get("grid")
    parity = {}
    for key in [f"{nm}_base" for nm in NAMES] + late_keys:
        parity[key] = {"cycle_parity_i%2": deepE(key, par_cyc)}
        if cg is not None:
            pg = ((cg[:, 0] + cg[:, 1]) % 2 * 2 - 1)
            pg = (pg - pg.mean())[:, None] / np.linalg.norm(pg - pg.mean())
            parity[key]["grid_checkerboard"] = deepE(key, pg)

    # ---- Procrustes: all context pairs + fresh -----------------------------
    sims = {}
    for a, b in combinations(NAMES, 2):
        sims[f"{a}_base vs {b}_base"] = float(np.mean(
            [procrustes_sim(H(f"{a}_base", L), H(f"{b}_base", L)) for L in range(24, nL)]))
        sims[f"{a}_late vs {b}_late"] = float(np.mean(
            [procrustes_sim(H(f"{a}_joint_late", L), H(f"{b}_joint_late", L))
             for L in range(24, nL)]))
    for nm in NAMES:
        sims[f"{nm}_late vs FRESH"] = float(np.mean(
            [procrustes_sim(H(f"{nm}_joint_late", L), H("fresh_late", L))
             for L in range(24, nL)]))
        sims[f"{nm}: base vs late"] = float(np.mean(
            [procrustes_sim(H(f"{nm}_base", L), H(f"{nm}_joint_late", L))
             for L in range(24, nL)]))

    # ---- PCA figure ----------------------------------------------------------
    Lshow = min(26, nL - 1)
    hue = plt.cm.hsv(np.arange(n) / n)
    rows_fig = NAMES + ["fresh"]
    fig, axes = plt.subplots(len(rows_fig), 2, figsize=(9, 4.2 * len(rows_fig)))
    for i, c in enumerate(rows_fig):
        for j, win in enumerate(("base", "joint_late") if c != "fresh"
                                else ("early", "late")):
            ax = axes[i, j]
            key = f"{c}_{win}" if c != "fresh" else f"fresh_{win}"
            Hc = H(key, Lshow)
            U, s, _ = np.linalg.svd(Hc, full_matrices=False)
            P2 = U[:, :2] * s[:2]
            for a in range(n):
                for b in range(a + 1, n):
                    if A_union[a, b]:
                        nedges = sum(adjs[nm][a, b] for nm in NAMES)
                        ax.plot(P2[[a, b], 0], P2[[a, b], 1],
                                color="0.3" if nedges > 1 else "0.8",
                                lw=1.0 if nedges > 1 else 0.7, zorder=1)
            ax.scatter(P2[:, 0], P2[:, 1], c=hue, s=55, zorder=3, edgecolors="k", lw=0.4)
            for a in range(n):
                ax.annotate(words[a], P2[a], fontsize=6, xytext=(2, 2),
                            textcoords="offset points")
            ax.set_title(f"{c} | {win}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("+".join(NAMES) + f" PCA layer {Lshow} (dark = multi-graph edges)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"multi_pca.{ext}"), dpi=160)
    plt.close(fig)

    summary = {"graphs": NAMES, "stream_mass": stream, "top_oscillating_pairs": top_pairs,
               "occupancy": {words[i]: int(occ[i]) for i in range(n)},
               "headline_layer26": {c: {win: {nm: res[c][win][nm][Lshow] for nm in NAMES}
                                        for win in ("base", "joint_late")} for c in NAMES},
               "r2_deep_mean_late": r2c, "fourier_deep_mean": fourier,
               "parity_deep_mean": parity, "procrustes_deep_mean": sims,
               "union_low_modes_lambda": [round(float(x), 3) for x in wu[:7]]}
    json.dump(summary, open(os.path.join(OUT, "multi_summary.json"), "w"), indent=2)
    print(json.dumps({k: summary[k] for k in ("headline_layer26", "r2_deep_mean_late",
                                              "procrustes_deep_mean")}, indent=2))
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
