"""Ground-truth belief-state geometry (Mixed-State Presentation) for a hidden-state
HMM -- the object the LessWrong/comp-mech 'belief state geometry' work probes.

Unlike our fully-observed graph walks (whose belief is a trivial one-hot on the
current node -> simplex VERTICES, no fractal), a *hidden*-state HMM has belief
states that are genuine mixtures; iterating the per-symbol Bayesian belief-update
(an IFS of contractions) gives a FRACTAL in the state simplex.

Here we use the symmetric 3-state / 3-symbol Mess3 process. We sample sequences,
forward-filter the exact belief, and scatter the beliefs in the 2-simplex.

-> runs/belief_geometry/mess3_groundtruth_fractal.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mess3(x=0.05, alpha=0.85):
    """Symmetric 3-state, 3-symbol HMM. Returns per-symbol transition-emission
    matrices T[s] with T[s][i,j] = P(emit s, next state j | state i).
    Emission peaked on the state's 'own' symbol (alpha); transitions sticky (x)."""
    n = 3
    A = np.full((n, n), x); np.fill_diagonal(A, 1 - 2 * x)          # transition i->j
    b = (1 - alpha) / 2
    E = np.full((n, n), b); np.fill_diagonal(E, alpha)             # P(symbol s | arriving state j)
    # emit-on-arrival: T[s][i,j] = A[i,j] * E[j,s]
    T = np.stack([A * E[:, s][None, :] for s in range(n)])         # (s, i, j)
    return T


def stationary(T):
    A = T.sum(0)                                                   # marginal transition
    vals, vecs = np.linalg.eig(A.T)
    p = np.real(vecs[:, np.argmin(np.abs(vals - 1))]); return p / p.sum()


def sample_beliefs(T, n_seq=4000, length=60, seed=0):
    rng = np.random.default_rng(seed)
    n = T.shape[1]
    A = T.sum(0)
    p0 = stationary(T)
    pts = []
    for _ in range(n_seq):
        s_state = rng.choice(n, p=p0)
        b = p0.copy()
        for _ in range(length):
            # emit from current state, transition
            j = rng.choice(n, p=A[s_state])
            sym = rng.choice(n, p=(T[:, s_state, j] / A[s_state, j]))
            s_state = j
            # belief update on observed symbol
            b = b @ T[sym]; b = b / b.sum()
            pts.append((b.copy(), sym))
    return pts


def to_2d(b):
    # barycentric -> 2D equilateral triangle
    v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    return b @ v


def main():
    os.makedirs("runs/belief_geometry", exist_ok=True)
    T = mess3(x=0.05, alpha=0.85)
    pts = sample_beliefs(T)
    XY = np.array([to_2d(b) for b, _ in pts])
    sym = np.array([s for _, s in pts])

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, c, ttl in [(ax[0], "0.25", "Mess3 ground-truth MSP (belief simplex)"),
                      (ax[1], None, "colored by last emitted symbol")]:
        tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
        a.plot(tri[:, 0], tri[:, 1], color="0.7", lw=1)
        if c:
            a.scatter(XY[:, 0], XY[:, 1], s=1, c=c, alpha=0.15, linewidths=0, rasterized=True)
        else:
            a.scatter(XY[:, 0], XY[:, 1], s=1, c=plt.cm.tab10(sym), alpha=0.2, linewidths=0, rasterized=True)
        a.set_title(ttl, fontsize=10); a.set_aspect("equal"); a.axis("off")
    fig.suptitle("Belief-state geometry: fractal Mixed-State Presentation of a HIDDEN-state HMM\n"
                 "(our fully-observed graph walks instead give the 3 trivial simplex vertices)")
    fig.tight_layout()
    out = "runs/belief_geometry/mess3_groundtruth_fractal.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out, "| belief points:", len(pts))


if __name__ == "__main__":
    main()
