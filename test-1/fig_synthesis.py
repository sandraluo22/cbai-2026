"""Synthesis figures across the whole project.

  figs/fig_kernel.png       measured ICL update kernel (w_row, w_col) vs exponential
                            surrogates + successor-slot attention profile
  figs/fig_convergence_overview.png
                            base -> late cross-agent representational distance for
                            every multi-context run (dumbbell chart, grouped)
  figs/fig_bandwidth.png    consensus vs shared-evidence fraction (4-regular family)
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def unit(X):
    X = X - X.mean(0)
    return X / max(np.linalg.norm(X), 1e-12)


def psim(A, B):
    A, B = unit(A), unit(B)
    Ua, Sa, _ = np.linalg.svd(A, full_matrices=False)
    Ub, Sb, _ = np.linalg.svd(B, full_matrices=False)
    return float(np.linalg.svd((Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :],
                               compute_uv=False).sum())


# ---------------- fig 1: kernel ------------------------------------------------
def fig_kernel():
    k = json.load(open(os.path.join(HERE, "runs", "out_probe", "update_kernel.json")))
    c = np.array(k["bin_centers"])
    c[-1] = 500
    wr, wc = np.array(k["w_row"]), np.array(k["w_col"])
    prof = np.array(k["top6_head_age_profile"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    ax.plot(c, wr, "o-", color="#0e7c86", label="matched-row weight $w_{row}$ (a$\\to$j evidence)")
    ax.plot(c, wc, "s-", color="#fb8500", label="mismatched-row leakage $w_{col}$ (c$\\to$j)")
    for gam, col in ((0.96, "0.55"), (0.997, "0.75")):
        ax.plot(c, wr[0] * gam ** (c - c[0]), ":", color=col,
                label=f"exponential $\\gamma={gam}$ (scaled)")
    ax.set_xscale("log")
    ax.set_xlabel("age of the observation (words before current position)")
    ax.set_ylabel("log-odds weight per observation")
    ax.set_title("measured in-context update kernel (Llama-3.1-8B, grid walks)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(c, prof, "o-", color="#8338ec", label="top-6 successor-slot heads")
    ax.set_xscale("log")
    ax.set_xlabel("age of the matching occurrence (words)")
    ax.set_ylabel("attention mass on successor slots")
    ax.set_title("mechanism: induction-head attention by evidence age")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"fig_kernel.{ext}"), dpi=160)
    plt.close(fig)


# ---------------- fig 2: convergence overview ---------------------------------
def run_dists(path, kind):
    z = np.load(os.path.join(HERE, path,
                             "nodemeans.npz" if kind == "sweep" else
                             "nodemeans_dueling.npz"), allow_pickle=False)
    if kind == "legacy":
        names, nls = ["grid", "ring"], None
    elif kind == "sweep":
        names, nls = ["A", "B"], None
    else:
        names = [str(x) for x in z["ctx_names"]]
        nls = {nm: int(z[f"nlayers_{nm}"][0]) for nm in names} \
            if f"nlayers_{names[0]}" in z.files else None
    nL = int(z["n_layers"][0]) if "n_layers" in z.files else 0

    def H(nm, w, L):
        Hm = z[f"{nm}_{w}_layer_{L}"].astype(np.float64)
        return Hm - Hm.mean(0)

    def dist(a, b, w):
        if nls and (nls[a] != nls[b]):
            fr = np.linspace(0.75, 0.97, 8)
            return 1 - float(np.mean([psim(H(a, w, int(round(f * (nls[a] - 1)))),
                                           H(b, w, int(round(f * (nls[b] - 1)))))
                                      for f in fr]))
        n0 = nls[a] if nls else nL
        return 1 - float(np.mean([psim(H(a, w, L), H(b, w, L))
                                  for L in range(int(0.75 * n0), n0)]))

    pairs = [(names[i], names[j]) for i in range(len(names))
             for j in range(i + 1, len(names)) if not names[i].startswith("fresh")
             and not names[j].startswith("fresh")]
    b = np.mean([dist(a, c, "base") for a, c in pairs])
    l = np.mean([dist(a, c, "joint_late") for a, c in pairs])
    return float(b), float(l)


def fig_overview():
    ROWS = [
        ("grid+ring (Llama)", [
            ("runs/out", "T=1 free", "legacy"),
            ("runs/out_topk4", "top-k 4/4", "legacy"),
            ("runs/out_k2", "top-k 2/2", "legacy"),
            ("runs/out_k2_fix", "2/2 + clean vocab", "legacy")]),
        ("other graph pairs", [
            ("runs/out_hexgrid", "hex+grid", "ctx"),
            ("runs/out_hexring", "hex+ring", "ctx"),
            ("runs/out_prismring", "prism+ring", "ctx"),
            ("runs/out_ringring3", "ring+ring3 (0 shared)", "ctx"),
            ("runs/out_antigrid", "antiprism+grid", "ctx")]),
        ("N-way & cross-model", [
            ("runs/out_tri", "3-way ring+grid+ring3", "ctx"),
            ("runs/out_x_LQ", "Llama+Qwen", "ctx"),
            ("runs/out_x_LG", "Llama+Gemma", "ctx"),
            ("runs/out_x_QG", "Qwen+Gemma", "ctx"),
            ("runs/out_x_LQG", "3-way cross-model", "ctx")]),
        ("4-regular sweep (selection)", [
            ("runs/sweep/alpha0.25", "alpha=0.25", "sweep"),
            ("runs/sweep/alpha1.0", "alpha=1.0 coupled", "sweep"),
            ("runs/sweep/ev_q0.5", "q=0.5", "sweep"),
            ("runs/sweep/ev_q0.25", "q=0.25", "sweep"),
            ("runs/sweep/ctrl_real", "real steps", "sweep"),
            ("runs/sweep/ctrl_ow_A2B", "one-way A->B", "sweep"),
            ("runs/sweep/ctrl_free", "isolated (q=0)", "sweep")]),
    ]
    rows, groups = [], []
    for gname, items in ROWS:
        for path, label, kind in items:
            try:
                b, l = run_dists(path, kind)
                rows.append((label, b, l))
                groups.append(gname)
            except Exception as e:
                print(f"skip {path}: {e}")
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 2))
    ys = np.arange(len(rows))[::-1]
    gcols = {g: c for g, c in zip(dict.fromkeys(groups),
                                  ("#0e7c86", "#c22f4d", "#8338ec", "#fb8500"))}
    for y, (label, b, l), g in zip(ys, rows, groups):
        ax.plot([b, l], [y, y], "-", color="0.75", lw=1.6, zorder=1)
        ax.scatter([b], [y], color="0.55", s=42, zorder=2)
        ax.scatter([l], [y], color=gcols[g], s=52, zorder=3)
        ax.annotate("", xy=(l, y), xytext=(b, y),
                    arrowprops=dict(arrowstyle="->", color="0.6", lw=1))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("cross-agent representational distance (1 $-$ Procrustes sim, deep layers)")
    ax.set_title("Convergence across every experiment: base (gray) $\\to$ late (colored)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g)
               for g, c in gcols.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="0.55", label="base window"))
    ax.legend(handles=handles, fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    ax.axvline(0, color="0.85", lw=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"fig_convergence_overview.{ext}"), dpi=160)
    plt.close(fig)


# ---------------- fig 3: bandwidth master curve --------------------------------
def fig_bandwidth():
    sw = json.load(open(os.path.join(HERE, "runs", "sweep_summary.json")))
    def oc(cell):
        p = os.path.join(HERE, "runs", "sweep", cell, "outcome.json")
        if os.path.isfile(p):
            return json.load(open(p))
        return sw[cell]
    pts = [  # (shared-evidence fraction q_eff, cell, marker, label)
        (0.0, "ctrl_free", "o", "isolated (q=0)"),
        (0.25, "ev_q0.25", "o", "q=0.25"),
        (0.5, "ev_q0.5", "o", "q=0.5"),
        (1.0, "alpha1.0", "o", "coupled (q=1)"),
        (1.0, "ctrl_real", "*", "real steps (q=1, exogenous)"),
        (0.5, "ctrl_ow_A2B", "^", "one-way A->B"),
        (0.5, "ctrl_ow_B2A", "v", "one-way B->A"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    line = [(q, oc(c)["rep_dist_late"]) for q, c, m, _ in pts if m == "o"]
    line.sort()
    ax.plot([x for x, _ in line], [y for _, y in line], "-", color="#0e7c86", lw=1.5)
    for q, cell, m, label in pts:
        r = oc(cell)
        ax.scatter([q], [r["rep_dist_late"]], marker=m, s=110, color="#0e7c86",
                   edgecolors="k", linewidths=0.5, zorder=3)
        jf = r.get("js_final")
        txt = label + (f"\nJS={jf:.2f}" if jf is not None and cell != "ctrl_real" else "")
        ax.annotate(txt, (q, r["rep_dist_late"]), fontsize=7.5,
                    xytext=(6, 5), textcoords="offset points")
    base = oc("alpha1.0")["rep_dist_base"]
    ax.axhline(base, color="0.6", ls="--", lw=1)
    ax.annotate("initial prior distance", (0.02, base), fontsize=8, color="0.4",
                xytext=(0, 4), textcoords="offset points")
    ax.set_xlabel("shared-evidence fraction (communication probability q)")
    ax.set_ylabel("final representational distance (1 $-$ Procrustes)")
    ax.set_title("Consensus is bandwidth-limited (torus vs C16(1,3), all else fixed)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"fig_bandwidth.{ext}"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_kernel()
    fig_overview()
    fig_bandwidth()
    print("DONE -> figs/fig_kernel, fig_convergence_overview, fig_bandwidth")
