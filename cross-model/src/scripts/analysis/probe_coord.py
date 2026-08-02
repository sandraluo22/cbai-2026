"""Per-layer coordinate-decoding probe: ridge regression from per-occurrence residual activations to
the node's (x,y) grid coords, held-out R^2 per layer. Reads a per-occurrence npz (layer_* + meta_node
+ meta_context_length). Env: NPZ TAG OUTDIR CTXLO(100) ALPHA(1e3)
Out: <OUTDIR>/probe_coord_<TAG>.json  ({rel_depth, r2, layers})
"""
import os, json
import numpy as np

NPZ = os.environ["NPZ"]; TAG = os.environ.get("TAG", "model"); OUTDIR = os.environ.get("OUTDIR", ".")
CTXLO = int(os.environ.get("CTXLO", "100")); ALPHA = float(os.environ.get("ALPHA", "1e3"))
coords = np.array([[i // 4, i % 4] for i in range(16)], float)


def ridge_r2(X, Y, alpha, rng):
    n = len(X); idx = rng.permutation(n); tr, te = idx[:n // 2], idx[n // 2:]
    Xtr, Xte = X[tr], X[te]; Ytr, Yte = Y[tr], Y[te]
    mu = Xtr.mean(0); Xtr = Xtr - mu; Xte = Xte - mu
    d = Xtr.shape[1]; W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ (Ytr - Ytr.mean(0)))
    pred = Xte @ W + Ytr.mean(0)
    ss = ((Yte - Yte.mean(0)) ** 2).sum(); return float(1 - ((Yte - pred) ** 2).sum() / ss)


def main():
    os.makedirs(OUTDIR, exist_ok=True); z = np.load(NPZ)
    node = z["meta_node"]; mask = z["meta_context_length"] >= CTXLO
    nd = node[mask]; Y = coords[nd]
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
    rng = np.random.default_rng(0); r2 = []
    for L in layers:
        X = z[f"layer_{L}"].astype(np.float64)[mask]
        r2.append(ridge_r2(X, Y, ALPHA, rng))
    out = {"tag": TAG, "layers": layers, "rel_depth": list(np.linspace(0, 1, len(layers))), "r2": r2}
    json.dump(out, open(f"{OUTDIR}/probe_coord_{TAG}.json", "w"), indent=2)
    print(f"[{TAG}] peak coord-probe R2={max(r2):.3f} @ L{layers[int(np.argmax(r2))]}  DONE")


if __name__ == "__main__":
    main()
