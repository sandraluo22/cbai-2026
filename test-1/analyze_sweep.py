"""Aggregate the degree-controlled 4-regular sweep into outcome curves. NO RSA.

Per cell outcomes:
  P_A_win / win_margin : per-pair late-stream mass on A-only minus B-only edges
                         (generator-token stream)
  T_consensus          : first step where the 21-step rolling mean of the JS divergence
                         between the agents' predictives stays below 0.05
  js_final             : mean JS over the last 50 steps
  rep_dist_late        : 1 - Procrustes(A_late, B_late), deep layers (24..31)
  fit_A/fit_B/fit_U/fit_E : deep-mean R^2 of each side's late node-means on spectral
                         embeddings (modes 1-4) of graph A / B / union / empirical
                         late stream graph (averaged over both contexts)

Figures: figs/sweep_axes.png (disagreement / prior strength / evidence quality),
         figs/sweep_scatter.png (outcomes vs edit & spectral distance, all comparable
         cells), and a printed table for the secondary families.
Out: runs/sweep_summary.json
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, "runs", "sweep")
SPEC = json.load(open(os.path.join(HERE, "runs", "sweep_spec.json")))
N = 16


def lap_modes(A, k=4):
    A = A.astype(float)
    d = np.maximum(A.sum(1), 1e-12)
    L = np.eye(N) - (d ** -0.5)[:, None] * A * (d ** -0.5)[None, :]
    lam, U = np.linalg.eigh(L)
    return U[:, 1:k + 1]


def r2(Hc, F):
    Fc = F - F.mean(0)
    Fc = Fc / np.maximum(Fc.std(0), 1e-12)
    B, *_ = np.linalg.lstsq(Fc, Hc, rcond=None)
    return float(1 - ((Hc - Fc @ B) ** 2).sum() / max((Hc ** 2).sum(), 1e-12))


def unit(X):
    X = X - X.mean(0)
    return X / max(np.linalg.norm(X), 1e-12)


def psim(Aa, Bb):
    Aa, Bb = unit(Aa), unit(Bb)
    Ua, Sa, _ = np.linalg.svd(Aa, full_matrices=False)
    Ub, Sb, _ = np.linalg.svd(Bb, full_matrices=False)
    return float(np.linalg.svd((Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :],
                               compute_uv=False).sum())


def outcomes(cellname):
    d = os.path.join(SW, cellname)
    z = np.load(os.path.join(d, "nodemeans.npz"))
    mt = json.load(open(os.path.join(d, "metrics.json")))
    cell = mt["cell"]
    T = cell["tgen"]
    nL = int(z["n_layers"][0])
    adjA = z["adjacency_A"].astype(bool)
    adjB = z["adjacency_B"].astype(bool)
    a_only = adjA & ~adjB
    b_only = adjB & ~adjA

    gen = np.array(mt["gen_nodes"])
    margins = []
    W_emp = np.zeros((N, N))
    for p in range(gen.shape[0]):
        am = bm = tot = 0
        for t in range(T // 2 + 1, T):
            a, b = gen[p, t - 1], gen[p, t]
            W_emp[a, b] += 1
            W_emp[b, a] += 1
            tot += 1
            am += a_only[a, b]
            bm += b_only[a, b]
        margins.append((am - bm) / max(tot, 1))
    margins = np.array(margins)

    jsr = np.array(mt["js"])
    k = 21
    roll = np.convolve(jsr, np.ones(k) / k, "valid")
    below = np.where(roll < 0.05)[0]
    T_cons = int(below[0]) if len(below) else T
    js_final = float(jsr[-50:].mean())

    def H(si, w, L):
        Hm = z[f"{si}_{w}_layer_{L}"].astype(np.float64)
        return Hm - Hm.mean(0)

    deep = range(24, nL)
    rep_dist_late = 1 - float(np.mean([psim(H("A", "joint_late", L),
                                            H("B", "joint_late", L)) for L in deep]))
    rep_dist_base = 1 - float(np.mean([psim(H("A", "base", L), H("B", "base", L))
                                       for L in deep]))
    FE = {"A": lap_modes(adjA), "B": lap_modes(adjB),
          "U": lap_modes((adjA | adjB).astype(float)), "E": lap_modes(W_emp + 1e-9)}
    fit = {f: float(np.mean([r2(H(si, "joint_late", L), F)
                             for si in ("A", "B") for L in deep]))
           for f, F in FE.items()}
    fitAB_by_side = {si: {f: float(np.mean([r2(H(si, "joint_late", L), FE[f])
                                            for L in deep])) for f in ("A", "B")}
                     for si in ("A", "B")}
    return {"cell": cell, "P_A_win": float((margins > 0).mean()),
            "win_margin": float(margins.mean()), "T_consensus": T_cons,
            "js_final": js_final, "js0": float(jsr[:20].mean()),
            "rep_dist_late": rep_dist_late, "rep_dist_base": rep_dist_base,
            "fit": fit, "fit_by_side": fitAB_by_side}


def main():
    res = {}
    for cell in SPEC["cells"]:
        nm = cell["name"]
        if os.path.isdir(os.path.join(SW, nm)):
            try:
                res[nm] = outcomes(nm)
            except Exception as e:
                print(f"skip {nm}: {e}")
    json.dump(res, open(os.path.join(HERE, "runs", "sweep_summary.json"), "w"), indent=1)
    print(f"{len(res)} cells analyzed")

    def g(nm, key):
        r = res[nm]
        return r[key] if not isinstance(key, tuple) else r[key[0]][key[1]]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    # panel 1: disagreement axis
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    anames = [f"alpha{a}" for a in alphas]
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(alphas, [g(n, "rep_dist_base") for n in anames], "o--", color="0.6",
            label="rep. distance (base)")
    ax.plot(alphas, [g(n, "rep_dist_late") for n in anames], "o-", color="#0e7c86",
            label="rep. distance (late)")
    ax.plot(alphas, [g(n, "js_final") for n in anames], "s-", color="#fb8500",
            label="JS divergence (final)")
    ax2.plot(alphas, [g(n, "T_consensus") for n in anames], "^-", color="crimson",
             label="T_consensus")
    ax.set_xlabel(r"prior disagreement $\alpha$ (torus $\to$ C16(1,3))")
    ax.set_ylabel("distance / divergence")
    ax2.set_ylabel("T_consensus (steps)", color="crimson")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("disagreement axis")
    # panel 2: prior strength
    ax = axes[1]
    sym = [("prior_100v100", 100), ("prior_300v300", 300), ("alpha1.0", 600),
           ("prior_1000v1000", 1000)]
    asym = [("prior_1000v100", 100), ("prior_1000v300", 300),
            ("prior_1000v1000", 1000)]
    ax.plot([x for _, x in sym], [g(n, "rep_dist_late") for n, _ in sym], "o-",
            color="#0e7c86", label="rep. dist late (symmetric ctx)")
    ax.plot([x for _, x in asym], [g(n, "win_margin") for n, _ in asym], "s-",
            color="#023047", label="win margin, A=1000 vs B=ctx")
    ax.axhline(0, color="0.7", lw=1)
    ax.set_xlabel("context tokens (B side for asymmetric)")
    ax.set_ylabel("rep. distance  /  win margin (A-only $-$ B-only edge mass)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("prior strength")
    # panel 3: evidence quality
    ax = axes[2]
    ev = [("ev_temp0.7", "T=0.7"), ("alpha1.0", "T=1,k4,q1"), ("ev_temp1.3", "T=1.3"),
          ("ev_free", "free"), ("ev_q0.5", "q=0.5"), ("ev_q0.25", "q=0.25")]
    xs = np.arange(len(ev))
    ax.bar(xs - 0.2, [g(n, "rep_dist_late") for n, _ in ev], 0.4,
           color="#0e7c86", label="rep. distance late")
    ax.bar(xs + 0.2, [g(n, "js_final") for n, _ in ev], 0.4,
           color="#fb8500", label="JS final")
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in ev], fontsize=8)
    ax.set_xlabel("sampling / communication condition")
    ax.set_ylabel("rep. distance late  /  final JS divergence")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    ax.set_title("evidence quality (torus vs C16(1,3))")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"sweep_axes.{ext}"), dpi=160)
    plt.close(fig)

    # scatter: outcomes vs continuous pair distance (all default-knob cells)
    comp = [c["name"] for c in SPEC["cells"]
            if c["ctx_a"] == 600 and c["ctx_b"] == 600 and c["temp"] == 1.0
            and c["topk"] == 4 and c["qcomm"] == 1.0 and c["name"] in res]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, xk, xl in ((axes[0], "edit", "edge edit distance"),
                       (axes[1], "spectral", "spectral distance")):
        xs = [res[n]["cell"]["metrics"][xk] for n in comp]
        for yk, col, lab in (("rep_dist_late", "#0e7c86", "rep. dist late"),
                             ("js_final", "#fb8500", "JS final"),):
            ax.scatter(xs, [g(n, yk) for n in comp], color=col, label=lab)
        for n, x in zip(comp, xs):
            ax.annotate(n.replace("fam_", "").replace("alpha", "a"),
                        (x, g(n, "rep_dist_late")), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel(xl + " between $G_A$ and $G_B$")
        ax.set_ylabel("outcome (rep. distance late / final JS)")
        ax.set_title(f"convergence vs {xl}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[2]
    lam2 = SPEC["lambda2"]
    xs = [lam2[res[n]["cell"]["gb"]] for n in comp]
    ax.scatter(xs, [g(n, "win_margin") for n in comp], color="#023047")
    for n, x in zip(comp, xs):
        ax.annotate(n.replace("fam_", ""), (x, g(n, "win_margin")), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="0.7", lw=1)
    ax.set_xlabel(r"$\lambda_2$ of B's graph (mixing speed)")
    ax.set_ylabel("win margin (A-only minus B-only late edge mass)")
    ax.grid(alpha=0.3)
    ax.set_title("does mixing speed set prior strength?")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"sweep_scatter.{ext}"), dpi=160)
    plt.close(fig)

    print(f"{'cell':16s} {'PAwin':>5s} {'marg':>6s} {'Tcons':>5s} {'JSfin':>6s} "
          f"{'repD':>6s} {'fitA':>5s} {'fitB':>5s} {'fitU':>5s} {'fitE':>5s}")
    for n in [c["name"] for c in SPEC["cells"] if c["name"] in res]:
        r = res[n]
        print(f"{n:16s} {r['P_A_win']:5.2f} {r['win_margin']:6.3f} "
              f"{r['T_consensus']:5d} {r['js_final']:6.3f} {r['rep_dist_late']:6.3f} "
              f"{r['fit']['A']:5.2f} {r['fit']['B']:5.2f} {r['fit']['U']:5.2f} "
              f"{r['fit']['E']:5.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
