"""What SPECIFIC hybrid shape do the coupled contexts converge on?

Candidates tested against the joint_late node-means (per layer, per context):
  grid      : 2D grid coords (x, y)
  ring      : 2D ring coords (cos, sin)
  union     : modes 1-2 of the normalized Laplacian of the UNION graph (grid OR ring edges)
  empirical : modes 1-2 of the normalized Laplacian of the SYMMETRIZED BIGRAM graph actually
              realized by the jointly-generated stream in the matching window
  log-occ   : 1D log occupancy (token-frequency geometry control)

Plus:
  * edge-class mean distances in representation space (shared / grid-only / ring-only /
    non-edges) -- a basis-free readout of which links the geometry keeps short,
  * Procrustes shape similarity between the two contexts (base vs late) and against the
    fresh-context control (blank context fed ONLY the joint stream),
  * occupancy counts (is the "shape" partly just frequency?).

Out: out/hybrid_r2.png, out/hybrid_edges.png, out/hybrid_procrustes.png,
     out/hybrid_layouts.png (+ pdf), out/hybrid_summary.json
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RUN_OUT", os.path.join(HERE, "runs", "out"))
CTXS = ("grid", "ring")


def norm_laplacian_modes(W):
    W = W.astype(float)
    d = W.sum(1)
    di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(len(W)) - di[:, None] * W * di[None, :]
    w, U = np.linalg.eigh(L)
    return w, U


def r2_from_feats(Hc, F):
    F = np.atleast_2d(F.T).T
    Fc = F - F.mean(0)
    Fc = Fc / np.maximum(Fc.std(0), 1e-12)
    B, *_ = np.linalg.lstsq(Fc, Hc, rcond=None)
    resid = Hc - Fc @ B
    return float(1.0 - (resid ** 2).sum() / np.maximum((Hc ** 2).sum(), 1e-12))


def unit_shape(X):
    X = X - X.mean(0)
    return X / np.maximum(np.linalg.norm(X), 1e-12)


def procrustes_sim(A, B):
    """max_R tr(R B^T A) for orthogonal R, on centered unit-Frobenius shapes; in [0, 1].
    Computed in the 16-dim row space: sv(A^T B) = sv(Sa (Ua^T Ub) Sb) from thin SVDs."""
    A, B = unit_shape(A), unit_shape(B)
    Ua, Sa, _ = np.linalg.svd(A, full_matrices=False)
    Ub, Sb, _ = np.linalg.svd(B, full_matrices=False)
    M = (Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :]
    return float(np.linalg.svd(M, compute_uv=False).sum())


def procrustes_align(A, B):
    """Rotate 2D config B onto A (both centered/unit-scaled)."""
    A2, B2 = unit_shape(A), unit_shape(B)
    U, _, Vt = np.linalg.svd(B2.T @ A2)
    return A2, B2 @ U @ Vt


def main():
    z = np.load(os.path.join(OUT, "nodemeans_dueling.npz"), allow_pickle=False)
    zf = np.load(os.path.join(OUT, "nodemeans_fresh.npz"), allow_pickle=False)
    log = json.load(open(os.path.join(OUT, "gen_log.json")))
    nL = int(z["n_layers"][0])
    words = [str(w) for w in z["words"]]
    n = len(words)
    A_grid = z["adjacency_grid"].astype(bool)
    A_ring = z["adjacency_ring"].astype(bool)
    cg, cr = z["coords_grid"], z["coords_ring"]
    P, T = log["npairs"], log["tgen"]
    joint = np.array([[log["steps"][f"pair{p}"][t]["node"] for t in range(T)]
                      for p in range(P)])

    A_union = A_grid | A_ring
    shared = A_grid & A_ring
    grid_only = A_grid & ~A_ring
    ring_only = A_ring & ~A_grid
    non_edge = ~A_union & ~np.eye(n, dtype=bool)
    print(f"edges: shared={shared.sum()//2} grid-only={grid_only.sum()//2} "
          f"ring-only={ring_only.sum()//2} non={non_edge.sum()//2}")

    # empirical symmetrized bigram graph of the joint stream, late window (matches joint_late)
    def bigram(lo, hi):
        C = np.zeros((n, n))
        for p in range(P):
            for t in range(max(lo, 1), hi):
                C[joint[p, t - 1], joint[p, t]] += 1
        W = C + C.T
        np.fill_diagonal(W, 0)
        return W

    W_emp = bigram(300, T)
    occ_late = np.array([(joint[:, 300:] == i).sum() for i in range(n)], float)

    wg, Ug = norm_laplacian_modes(A_grid.astype(float))
    wr, Ur = norm_laplacian_modes(A_ring.astype(float))
    wu, Uu = norm_laplacian_modes(A_union.astype(float))
    we, Ue = norm_laplacian_modes(W_emp)

    # how walk-like is the late stream w.r.t. each graph? edge mass fractions
    mass = W_emp.sum()
    print(f"late-stream transition mass: shared={W_emp[shared].sum()/mass:.2f} "
          f"grid-only={W_emp[grid_only].sum()/mass:.2f} "
          f"ring-only={W_emp[ring_only].sum()/mass:.2f} "
          f"non-edge={W_emp[non_edge].sum()/mass:.2f}")

    def H(key, L):
        Hm = (z if not key.startswith("fresh") else zf)[f"{key}_layer_{L}"].astype(np.float64)
        return Hm - Hm.mean(0)

    FEATS = {"grid coords": cg, "ring coords": cr, "union modes 1-2": Uu[:, 1:3],
             "empirical modes 1-2": Ue[:, 1:3],
             "log-occupancy (1D)": np.log(np.maximum(occ_late, 1.0))[:, None]}

    # ---- fig 1: candidate R^2 by layer, joint_late (and fresh control) ----
    keys = [("grid_joint_late", "GRID-primed, joint_late"),
            ("ring_joint_late", "RING-primed, joint_late"),
            ("fresh_late", "FRESH context, late")]
    r2 = {k: {f: [] for f in FEATS} for k, _ in keys}
    for k, _ in keys:
        for L in range(nL):
            Hc = H(k, L)
            for f, F in FEATS.items():
                r2[k][f].append(r2_from_feats(Hc, np.asarray(F, float)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    fcol = {"grid coords": "#023047", "ring coords": "crimson", "union modes 1-2": "#fb8500",
            "empirical modes 1-2": "#2a9d8f", "log-occupancy (1D)": "0.5"}
    for ax, (k, title) in zip(axes, keys):
        for f in FEATS:
            ax.plot(r2[k][f], color=fcol[f], label=f,
                    ls=":" if "occupancy" in f else "-")
        ax.set_title(title); ax.set_xlabel("layer"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"R$^2$ of node-means")
    axes[0].legend(fontsize=7)
    fig.suptitle("Which candidate structure explains the converged geometry?")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"hybrid_r2.{ext}"), dpi=160)
    plt.close(fig)

    # ---- fig 2: edge-class mean distances (base vs late, both contexts) ---
    def edge_dist(Hc, mask):
        Hs = unit_shape(Hc)
        D = np.sqrt(np.maximum(((Hs[:, None, :] - Hs[None, :, :]) ** 2).sum(-1), 0))
        return float(D[mask].mean())

    classes = [("shared (grid & ring)", shared, "#6a040f"),
               ("grid-only", grid_only, "#023047"),
               ("ring-only (wrap)", ring_only, "crimson"),
               ("non-edge", non_edge, "0.6")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for i, c in enumerate(CTXS):
        for j, win in enumerate(("base", "joint_late")):
            ax = axes[i, j]
            for name, mask, col in classes:
                ys = [edge_dist(H(f"{c}_{win}", L), mask) for L in range(nL)]
                ax.plot(ys, color=col, label=name)
            ax.set_title(f"{c}-primed | {win}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == 1: ax.set_xlabel("layer")
            if j == 0: ax.set_ylabel("mean pair distance (unit-shape)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Edge-class distances: which links does the geometry keep short?")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"hybrid_edges.{ext}"), dpi=160)
    plt.close(fig)

    # ---- fig 3: Procrustes shape similarities across layers ---------------
    pairs = [("grid_base vs ring_base", "grid_base", "ring_base", "0.55", "-"),
             ("grid_late vs ring_late", "grid_joint_late", "ring_joint_late", "#023047", "-"),
             ("grid_late vs FRESH", "grid_joint_late", "fresh_late", "#fb8500", "--"),
             ("ring_late vs FRESH", "ring_joint_late", "fresh_late", "#2a9d8f", "--"),
             ("grid: base vs late", "grid_base", "grid_joint_late", "#8ecae6", ":"),
             ("ring: base vs late", "ring_base", "ring_joint_late", "crimson", ":")]
    key_map = {"grid_base": "grid_base", "ring_base": "ring_base",
               "grid_joint_late": "grid_joint_late", "ring_joint_late": "ring_joint_late",
               "fresh_late": "fresh_late"}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sims = {}
    for label, a, b, col, ls in pairs:
        ys = [procrustes_sim(H(key_map[a], L), H(key_map[b], L)) for L in range(nL)]
        sims[label] = ys
        ax.plot(ys, color=col, ls=ls, label=label)
    ax.set_xlabel("layer"); ax.set_ylabel("Procrustes shape similarity")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Shape convergence: the two contexts vs each other and vs the fresh control")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"hybrid_procrustes.{ext}"), dpi=160)
    plt.close(fig)

    # ---- fig 4: layouts at deep layer, aligned, vs candidate embeddings ---
    Lshow = min(26, nL - 1)
    hue = plt.cm.hsv(np.arange(n) / n)

    def draw(ax, P2, title):
        for a in range(n):
            for b in range(a + 1, n):
                if shared[a, b]:
                    ax.plot(P2[[a, b], 0], P2[[a, b], 1], color="0.35", lw=1.2, zorder=1)
                elif grid_only[a, b]:
                    ax.plot(P2[[a, b], 0], P2[[a, b], 1], color="0.75", lw=1.0, zorder=1)
                elif ring_only[a, b]:
                    ax.plot(P2[[a, b], 0], P2[[a, b], 1], color="crimson", lw=1.0,
                            alpha=0.8, zorder=2)
        ax.scatter(P2[:, 0], P2[:, 1], c=hue, s=70, zorder=3, edgecolors="k", lw=0.4)
        for a in range(n):
            ax.annotate(words[a], P2[a], fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

    def pc2(Hc):
        U, s, _ = np.linalg.svd(Hc, full_matrices=False)
        return U[:, :2] * s[:2]

    Hg = pc2(H("grid_joint_late", Lshow))
    Hr = pc2(H("ring_joint_late", Lshow))
    Hf = pc2(H("fresh_late", Lshow))
    ref, Hr_al = procrustes_align(Hg, Hr)
    _, Hf_al = procrustes_align(Hg, Hf)
    _, Uu_al = procrustes_align(Hg, Uu[:, 1:3])
    _, Ue_al = procrustes_align(Hg, Ue[:, 1:3])
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
    draw(axes[0], ref, f"GRID-primed joint_late (layer {Lshow}, PC1-2)")
    draw(axes[1], Hr_al, "RING-primed joint_late (aligned)")
    draw(axes[2], Hf_al, "FRESH context, late (aligned)")
    draw(axes[3], Uu_al, "UNION graph Laplacian modes 1-2 (aligned)")
    draw(axes[4], Ue_al, "EMPIRICAL stream graph modes 1-2 (aligned)")
    fig.suptitle("Converged shape vs candidates (dark = shared edges, gray = grid-only, "
                 "red = ring-only wrap edges; hue = ring position)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"hybrid_layouts.{ext}"), dpi=160)
    plt.close(fig)

    # ---- summary -----------------------------------------------------------
    summary = {
        "layer_shown": Lshow,
        "occupancy_late_per_node": {words[i]: int(occ_late[i]) for i in range(n)},
        "late_stream_edge_mass": {
            "shared": float(W_emp[shared].sum() / mass),
            "grid_only": float(W_emp[grid_only].sum() / mass),
            "ring_only": float(W_emp[ring_only].sum() / mass),
            "non_edge": float(W_emp[non_edge].sum() / mass)},
        "r2_at_layer": {k: {f: r2[k][f][Lshow] for f in FEATS} for k, _ in keys},
        "r2_deep_mean_24_31": {k: {f: float(np.mean(r2[k][f][24:])) for f in FEATS}
                               for k, _ in keys},
        "procrustes_at_layer": {lab: sims[lab][Lshow] for lab in sims},
        "procrustes_deep_mean_24_31": {lab: float(np.mean(sims[lab][24:])) for lab in sims},
    }
    json.dump(summary, open(os.path.join(OUT, "hybrid_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
