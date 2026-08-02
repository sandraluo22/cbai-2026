"""Square-grid graph, random walks, and a leave-one-node-out coordinate probe.

Self-contained (mirrors the cross-model project's graph.py + coord_decode.py so
games/ has no cross-repo import dependency). Used by the random-walk ping-pong
game: the seed walk lives on this grid, and the probe maps a residual-stream
vector to a predicted (row, col) grid coordinate.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np

# Fixed, semantically-unrelated concept words (same set/order as cross-model).
WORDS = [
    "apple", "bird", "sand", "math", "chair", "river", "music", "glass",
    "cloud", "knife", "paper", "tiger", "plant", "stone", "bread", "clock",
]


class Grid:
    def __init__(self, rows: int = 4, cols: int = 4):
        self.rows, self.cols = rows, cols
        self.n = rows * cols
        self.words = WORDS[: self.n]
        self.coords = [(i // cols, i % cols) for i in range(self.n)]
        self.adj: List[List[int]] = []
        for r in range(rows):
            for c in range(cols):
                nb = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        nb.append(rr * cols + cc)
                self.adj.append(sorted(nb))

    def neighbors(self, node: int) -> List[int]:
        return self.adj[node]

    def word_to_node(self):
        return {w: i for i, w in enumerate(self.words)}

    def random_walk(self, length: int, start: int = 0, seed: int = 0) -> List[int]:
        rng = np.random.default_rng(seed)
        nodes = [start]
        cur = start
        for _ in range(length - 1):
            cur = int(rng.choice(self.adj[cur]))
            nodes.append(cur)
        return nodes

    def coord_array(self) -> np.ndarray:
        return np.array(self.coords, dtype=float)   # [n, 2] = (row, col)


def _ridge_fit(X, Y, alpha):
    """Closed-form ridge on standardized X. Returns predict(x)->y."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    coef = Vt.T @ ((S / (S ** 2 + alpha))[:, None] * (U.T @ (Y - Y.mean(0))))
    ymu = Y.mean(0)

    def predict(x):
        xs = (np.atleast_2d(x) - mu) / sd
        return xs @ coef + ymu
    return predict, (mu, sd, coef, ymu)


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


class CoordProbe:
    """Linear map: residual vector -> (row, col). Fit on per-node mean activations.

    `loo_r2()` reports leave-one-node-out R^2 per axis (real geometric structure,
    no node-identity leakage); `fit_full()` fits on all node means for projecting
    arbitrary occurrences during free-running relay.
    """
    ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]

    def __init__(self, grid: Grid):
        self.grid = grid
        self.coords = grid.coord_array()
        self._predict = None
        self.alpha = None

    def loo_r2(self, node_means: np.ndarray) -> Tuple[float, float]:
        n = node_means.shape[0]
        best = (-9.0, -9.0, None)
        for a in self.ALPHAS:
            pred = np.zeros((n, 2))
            for k in range(n):
                idx = [i for i in range(n) if i != k]
                p, _ = _ridge_fit(node_means[idx], self.coords[idx], a)
                pred[k] = p(node_means[k])
            rr, rc = _r2(self.coords[:, 0], pred[:, 0]), _r2(self.coords[:, 1], pred[:, 1])
            if rr + rc > best[0] + best[1]:
                best = (rr, rc, a)
        return best[0], best[1]

    def fit_full(self, node_means: np.ndarray, alpha: float | None = None):
        if alpha is None:
            # choose the alpha that maximizes LOO fit, then refit on all nodes
            best_a, best_s = self.ALPHAS[0], -9.0
            for a in self.ALPHAS:
                rr, rc = self._loo_at(node_means, a)
                if rr + rc > best_s:
                    best_s, best_a = rr + rc, a
            alpha = best_a
        self.alpha = alpha
        self._predict, _ = _ridge_fit(node_means, self.coords, alpha)
        return self

    def _loo_at(self, node_means, a):
        n = node_means.shape[0]
        pred = np.zeros((n, 2))
        for k in range(n):
            idx = [i for i in range(n) if i != k]
            p, _ = _ridge_fit(node_means[idx], self.coords[idx], a)
            pred[k] = p(node_means[k])
        return _r2(self.coords[:, 0], pred[:, 0]), _r2(self.coords[:, 1], pred[:, 1])

    def project(self, x) -> np.ndarray:
        """Map a residual vector (or [N,d] batch) to predicted (row,col)."""
        assert self._predict is not None, "call fit_full() first"
        return self._predict(x)
