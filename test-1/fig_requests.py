"""Requested figures:
 1+2  figs/fig_q_curves.png       JS convergence curves per q + q vs steps-to-threshold
 3    figs/fig_conv_shapes.png    convergence trajectories in the (R2_grid, R2_ring) plane
 4    figs/fig_model_viz.png      full visualization of the Dirichlet-Markov and
                                  hierarchical/mixture surrogates (posterior heatmaps)
 5    figs/fig_interleave_detect.png  how latent-source (interleaving) was detected
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bayes_hier import Learner, load_run   # noqa: E402

SM = 21


def smooth(x, k=SM):
    return np.convolve(np.asarray(x, float), np.ones(k) / k, "valid")


def jsr(cell):
    return np.array(json.load(open(os.path.join(HERE, "runs", "sweep", cell,
                                                "metrics.json")))["js"])


# ---------------- fig 1+2 ------------------------------------------------------
def fig_q():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    series = [("alpha1.0", "q = 1 (full coupling)", "#0e7c86", "-"),
              ("ev_q0.5", "q = 0.5", "#fb8500", "-"),
              ("ev_q0.25", "q = 0.25", "#c22f4d", "-"),
              ("ctrl_free", "q = 0 (isolated)", "0.35", "-"),
              ("ctrl_ow_A2B", "one-way A->B", "#8338ec", "--"),
              ("ctrl_ow_B2A", "one-way B->A", "#2a9d8f", "--")]
    for cell, lab, col, ls in series:
        y = smooth(jsr(cell))
        ax.plot(np.arange(len(y)) + SM // 2, y, color=col, ls=ls, label=lab)
    ax.axhline(0.05, color="0.7", ls=":", lw=1)
    ax.annotate("consensus threshold 0.05", (5, 0.055), fontsize=7.5, color="0.4")
    ax.set_xlabel("joint generation step")
    ax.set_ylabel("JS divergence between the two agents' predictives")
    ax.set_title("(1) convergence curves by communication probability q")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    qs = [0.0, 0.25, 0.5, 1.0]
    cells = ["ctrl_free", "ev_q0.25", "ev_q0.5", "alpha1.0"]
    T = len(jsr("alpha1.0"))
    for th, col in ((0.15, "#8ecae6"), (0.10, "#219ebc"), (0.05, "#023047")):
        ys = []
        for c in cells:
            roll = smooth(jsr(c))
            below = np.where(roll < th)[0]
            ys.append(int(below[0]) if len(below) else np.nan)
        ax.plot(qs, ys, "o-", color=col, label=f"threshold JS < {th}")
        for q, y in zip(qs, ys):
            if np.isnan(y):
                ax.scatter([q], [T], marker="x", s=70, color=col)
    ax.annotate("x = never reached\n(within 400 steps)", (0.03, T - 60), fontsize=8,
                color="0.4")
    ax.set_xlabel("communication probability q")
    ax.set_ylabel("steps to reach threshold")
    ax.set_title("(2) q vs. steps to convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"fig_q_curves.{ext}"), dpi=160)
    plt.close(fig)


# ---------------- fig 3 --------------------------------------------------------
def unitF(X):
    X = X - X.mean(0)
    return X


def r2(Hc, F):
    Fc = F - F.mean(0)
    Fc = Fc / np.maximum(Fc.std(0), 1e-12)
    B, *_ = np.linalg.lstsq(Fc, Hc, rcond=None)
    return float(1 - ((Hc - Fc @ B) ** 2).sum() / max((Hc ** 2).sum(), 1e-12))


def traj(path, ctxs, legacy=True):
    z = np.load(os.path.join(HERE, path, "nodemeans_dueling.npz"), allow_pickle=False)
    cg = z["coords_grid"] if legacy else z[f"coords_{ctxs[0]}"]
    if legacy:
        cg, cr = z["coords_grid"], z["coords_ring"]
        nls = {c: int(z["n_layers"][0]) for c in ctxs}
    else:
        gks = {c: c.split("-")[1] for c in ctxs}
        cg = z[f"adjacency_{ctxs[0]}"]  # placeholder replaced below
        # cross-model runs store coords per ctx; grid/ring coords identical across
        cg = z[[k for k in z.files if k.startswith("coords_") and "grid" in k][0]]
        cr = z[[k for k in z.files if k.startswith("coords_") and "ring" in k][0]]
        nls = {c: int(z[f"nlayers_{c}"][0]) for c in ctxs}
    out = {}
    for c in ctxs:
        nl = nls[c]
        deep = range(int(0.75 * nl), nl)
        pts = []
        for w in ("base", "joint_early", "joint_mid", "joint_late"):
            Hs = [z[f"{c}_{w}_layer_{L}"].astype(np.float64) for L in deep]
            Hs = [h - h.mean(0) for h in Hs]
            pts.append((np.mean([r2(h, cg) for h in Hs]),
                        np.mean([r2(h, cr) for h in Hs])))
        out[c] = np.array(pts)
    return out


def fig_shapes():
    RUNS = [
        ("runs/out_topk4", ["grid", "ring"], True, "top-k 4/4 (grid wins)", "#023047"),
        ("runs/out_k2_fix", ["grid", "ring"], True, "2/2 clean vocab (draw)", "#0e7c86"),
        ("runs/out", ["grid", "ring"], True, "T=1 free (degenerate)", "0.5"),
        ("runs/out_x_QG", ["Qwen-grid", "Gemma-ring"], False,
         "Qwen+Gemma (ring wins)", "#c22f4d"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for path, ctxs, legacy, label, col in RUNS:
        try:
            tr = traj(path, ctxs, legacy)
        except Exception as e:
            print("skip", path, e)
            continue
        for c, mk in zip(ctxs, ("o", "s")):
            P = tr[c]
            ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=1.2, alpha=0.8)
            ax.scatter(P[0, 0], P[0, 1], marker=mk, s=46, facecolors="none",
                       edgecolors=col)
            ax.scatter(P[-1, 0], P[-1, 1], marker=mk, s=60, color=col)
            ax.annotate("", xy=P[-1], xytext=P[-2],
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        ax.plot([], [], color=col, label=label)
    lim = ax.get_xlim()[1]
    ax.plot([0, 0.55], [0, 0.55], ":", color="0.75", lw=1)
    ax.annotate("balanced (union) line", (0.38, 0.40), fontsize=8, color="0.5",
                rotation=42)
    ax.set_xlabel(r"R$^2$ from GRID coordinates (deep layers)")
    ax.set_ylabel(r"R$^2$ from RING coordinates (deep layers)")
    ax.set_title("(3) convergence trajectories: base $\\to$ late in the "
                 "(grid-ness, ring-ness) plane\ncircles = grid-primed agent, "
                 "squares = ring-primed agent; open = base, filled = late")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"fig_conv_shapes.{ext}"), dpi=160)
    plt.close(fig)


# ---------------- fig 4 --------------------------------------------------------
def run_learner(kind, run, ctxname, snap_at):
    L = Learner(kind)
    pref = run["prefixes"][ctxname][0]
    for a, b in zip(pref, pref[1:]):
        L.prefix_step(a, b)
    snaps = {"prefix": (L.C.copy(), L.C0.copy(), L.C1.copy())}
    seq = list(run["joint"][0])
    for t in range(run["T"]):
        x1 = seq[t - 1] if t >= 1 else pref[-1]
        x2 = seq[t - 2] if t >= 2 else pref[-1]
        L.joint_step(t, seq[t], x1, x2)
        if t + 1 == snap_at:
            snaps["late"] = (L.C.copy(), L.C0.copy(), L.C1.copy())
    return snaps


def fig_modelviz():
    coup = load_run("runs/out_k2_fix", 1000)
    real = load_run("runs/out_ctrl_real", 1300)
    sA = run_learner("merged", coup, "grid", coup["T"])
    sB = run_learner("merged", coup, "ring", coup["T"])
    sH = run_learner("lag2", real, "ring", real["T"])
    sM = run_learner("merged", real, "ring", real["T"])

    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.2))

    def show(ax, M, title, sym=True, log=True):
        W = M + M.T if sym else M
        ax.imshow(np.log1p(W / max(W.max(), 1e-9) * 50) if log else W, cmap="magma")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("to node j", fontsize=7)
        ax.set_ylabel("from node i", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    show(axes[0, 0], coup["A_g"].astype(float) * 0 + coup["A_g"], "ground truth: GRID adjacency")
    show(axes[0, 1], sA["prefix"][0], "Dirichlet-Markov, grid-primed:\nposterior counts at end of PREFIX")
    show(axes[0, 2], sA["late"][0], "grid-primed: counts after 600\nJOINT steps (coupled stream)")
    show(axes[0, 3], sB["late"][0], "ring-primed: counts after 600\nJOINT steps (same stream)")
    show(axes[0, 4], coup["A_r"].astype(float), "ground truth: RING adjacency")
    axes[0, 2].annotate("", xy=(1.12, 0.5), xycoords="axes fraction",
                        xytext=(0.95, 0.5),
                        arrowprops=dict(arrowstyle="<->", color="0.3"))

    show(axes[1, 0], real["A_r"].astype(float), "ground truth: RING adjacency")
    show(axes[1, 1], sH["late"][1], "HIERARCHICAL learner on the real\ninterleaved stream: phase-0 matrix $C_0$")
    show(axes[1, 2], sH["late"][2], "phase-1 matrix $C_1$")
    show(axes[1, 3], sM["late"][0], "MERGED learner, same stream:\nsingle matrix C (cross-bigram blur)")
    show(axes[1, 4], real["A_g"].astype(float), "ground truth: GRID adjacency")
    fig.suptitle("(4) the two descriptive models, visualized as their learned transition posteriors (log-scaled counts)\n"
                 "top: coupled stream -> the two Dirichlet-Markov posteriors become identical (convergence)   "
                 "bottom: interleaved stream -> the hierarchical learner's $C_0$/$C_1$ recover ring & grid; "
                 "LLM $\\approx$ 0.64 hierarchical + 0.36 merged", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"fig_model_viz.{ext}"), dpi=160)
    plt.close(fig)


# ---------------- fig 5 --------------------------------------------------------
def fig_interleave():
    hier = json.load(open(os.path.join(HERE, "runs", "bayes_hier.json")))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    toks = [("R$_1$", "#c22f4d"), ("G$_1$", "#2f6f9f"), ("R$_2$", "#c22f4d"),
            ("G$_2$", "#2f6f9f"), ("R$_3$", "#c22f4d"), ("G$_3$", "#2f6f9f"),
            ("?", "0.3")]
    for i, (t, c) in enumerate(toks):
        ax.add_patch(plt.Circle((1 + 1.3 * i, 3.4), 0.42, color=c, alpha=0.85))
        ax.text(1 + 1.3 * i, 3.4, t, ha="center", va="center", color="w",
                fontsize=11, weight="bold")
    ax.annotate("lag-1 hypothesis:\npredict from G$_3$'s neighbours\n(junk: different walk)",
                xy=(8.8, 3.0), xytext=(6.6, 1.3), fontsize=8.5, color="#2f6f9f",
                arrowprops=dict(arrowstyle="->", color="#2f6f9f"))
    ax.annotate("lag-2 hypothesis:\npredict a RING-neighbour of R$_3$\n(the true generator)",
                xy=(8.8, 3.8), xytext=(1.2, 4.5), fontsize=8.5, color="#c22f4d",
                arrowprops=dict(arrowstyle="->", color="#c22f4d",
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_title("(5) detecting interleaving: the stream is two independent walks;\n"
                 "only a learner that has DE-INTERLEAVED them can use R$_3$", fontsize=10)
    ax = axes[1]
    KEYS = ["even_ring_lag2", "even_ring_lag1", "even_grid_lag1"]
    labels = ["ring-nbrs of R$_3$\n(lag-2, correct)", "ring-nbrs of G$_3$\n(lag-1)",
              "grid-nbrs of G$_3$\n(lag-1)"]
    xs = np.arange(3)
    for i, (nm, col) in enumerate([("LLM", "#111111"), ("merged", "#999999"),
                                   ("lag2", "#0e7c86")]):
        ax.bar(xs + (i - 1) * 0.25, [hier["real_masses"][nm][k] for k in KEYS], 0.23,
               color=col, label={"lag2": "hierarchical"}.get(nm, nm))
    ax.axhline(2 / 16, color="crimson", ls=":", lw=1, label="chance (2/16)")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("where the predictive mass is measured")
    ax.set_ylabel("mean predictive mass before R-turns")
    ax.set_title("the measurement: LLM sits between merged\nand hierarchical "
                 f"(mixture $\\lambda$ = {hier['lambda_hat']:.2f})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"fig_interleave_detect.{ext}"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_q()
    fig_shapes()
    fig_modelviz()
    fig_interleave()
    print("DONE -> figs/fig_q_curves, fig_conv_shapes, fig_model_viz, fig_interleave_detect")
