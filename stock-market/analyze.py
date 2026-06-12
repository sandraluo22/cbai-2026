"""Analysis: (a) behavioral revealed social-weight vs the rational target;
(b) per-layer linear probes for the private-implied estimate, social-implied
estimate, and combined expected reward (R² by layer); (c) patching localization.
Cross-check where the social signal is *decodable* (probe) vs where it is
*causal* (patching).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Out-of-sample ridge R² (shared probe primitive)                              #
# --------------------------------------------------------------------------- #
def ridge_r2_oos(X, y, ridge=1.0, train_frac=0.7, n_splits=5, seed=0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, d = X.shape
    if n < 6:
        return float("nan")
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_splits):
        idx = rng.permutation(n); k = max(2, int(train_frac * n))
        tr, te = idx[:k], idx[k:]
        Xa = np.concatenate([X[tr], np.ones((len(tr), 1))], 1)
        W = np.linalg.solve(Xa.T @ Xa + ridge * np.eye(d + 1), Xa.T @ y[tr])
        pred = np.concatenate([X[te], np.ones((len(te), 1))], 1) @ W
        sse = np.sum((y[te] - pred) ** 2); sst = np.sum((y[te] - y[te].mean()) ** 2)
        scores.append(1 - sse / sst if sst > 0 else 0.0)
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# (a) revealed social weight                                                   #
# --------------------------------------------------------------------------- #
def revealed_social_weight(rows) -> dict:
    """Standardized linear-probability regression of 'chose target' on the target's
    private- and social-implied estimates. lambda_hat = |b_social|/(|b_social|+|b_private|),
    grouped by (lambda, w, tau), compared to the mean rational effective social weight.
    """
    groups = defaultdict(list)
    for r in rows:
        if r["parsed"]["company"] is None:
            continue
        tgt = r["target"]
        groups[(r["lmbda"], r["w"], r["tau"])].append((
            r["private_implied"][tgt], r["social_implied"][tgt],
            1.0 if r["parsed"]["company"] == tgt else 0.0,
            r["rational_eff_social_weight"]))
    out = {}
    for key, vals in groups.items():
        a = np.array(vals)
        if len(a) < 6:
            continue
        Xp, Xs, y, ratw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        # standardize predictors
        def z(v): s = v.std(); return (v - v.mean()) / s if s > 1e-9 else v * 0
        X = np.vstack([z(Xp), z(Xs), np.ones_like(y)]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        bp, bs = abs(beta[0]), abs(beta[1])
        lam_hat = bs / (bp + bs) if (bp + bs) > 1e-9 else float("nan")
        out[str(key)] = {"lambda": key[0], "w": key[1], "tau": key[2],
                         "lambda_hat": lam_hat, "rational_eff_weight": float(ratw.mean()),
                         "n": len(a)}
    return out


# --------------------------------------------------------------------------- #
# (b) per-layer probes                                                         #
# --------------------------------------------------------------------------- #
def layer_probes(rows, results_dir: Path) -> dict:
    """R² by layer for decoding the target's private-implied, social-implied, and
    expected-reward values from the marker activation."""
    A, yp, ys, yr = [], [], [], []
    for r in rows:
        acts = np.load(r["acts_path"])            # (L+1, hidden)
        A.append(acts)
        tgt = r["target"]
        yp.append(r["private_implied"][tgt]); ys.append(r["social_implied"][tgt])
        yr.append(r["E_reward"][tgt])
    A = np.stack(A)                               # (N, L+1, hidden)
    n_layers = A.shape[1]
    res = {"private_implied": [], "social_implied": [], "E_reward": []}
    for L in range(n_layers):
        X = A[:, L, :]
        res["private_implied"].append(ridge_r2_oos(X, yp))
        res["social_implied"].append(ridge_r2_oos(X, ys))
        res["E_reward"].append(ridge_r2_oos(X, yr))
    return res


# --------------------------------------------------------------------------- #
# (c) patching localization                                                    #
# --------------------------------------------------------------------------- #
def patching_localization(patch_rows) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for r in patch_rows:
        if r.get("skipped") or "layer" not in r:
            continue
        by[r["position_set"]][r["layer"]].append(1.0 if r["flipped_to_low"] else 0.0)
    out = {}
    for pset, layers in by.items():
        ls = sorted(layers)
        out[pset] = {"layers": ls, "flip_rate": [float(np.mean(layers[l])) for l in ls]}
    return out


# --------------------------------------------------------------------------- #
def plots(probes, patch_loc, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if probes:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for k in probes:
            ax.plot(probes[k], "-o", ms=3, label=k)
        ax.set(title="Per-layer probe R² (where each signal is decodable)",
               xlabel="layer", ylabel="out-of-sample R²"); ax.legend(); ax.axhline(0, color="grey", lw=.5)
        fig.tight_layout(); fig.savefig(out_dir / "layer_probes.png", dpi=140); plt.close(fig)
    if patch_loc:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for pset, d in patch_loc.items():
            ax.plot(d["layers"], d["flip_rate"], "-o", ms=3, label=pset)
        ax.set(title="Patching localization (where social is causal)",
               xlabel="layer", ylabel="choice flip-to-donor rate"); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "patching_localization.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp")
    args = ap.parse_args()
    d = Path(args.out)
    rows = [json.loads(l) for l in (d / "trials.jsonl").read_text().splitlines()]
    rw = revealed_social_weight(rows)
    probes = layer_probes(rows, d)
    patch_rows = []
    if (d / "patching.jsonl").exists():
        patch_rows = [json.loads(l) for l in (d / "patching.jsonl").read_text().splitlines()]
    ploc = patching_localization(patch_rows)
    plots(probes, ploc, d / "plots")
    (d / "analysis.json").write_text(json.dumps(
        {"revealed_weight": rw, "layer_probes": probes, "patching_localization": ploc}, indent=2))
    print("revealed social weight (lambda_hat vs rational) by (lambda,w,tau):")
    for k, v in rw.items():
        print(f"  {k}: lambda_hat={v['lambda_hat']:.2f}  rational={v['rational_eff_weight']:.2f}  n={v['n']}")
    print(f"\nwrote {d}/analysis.json + plots/")


if __name__ == "__main__":
    main()
