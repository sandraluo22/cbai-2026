"""Does the grid move INTO the top PCs as in-context length grows? From per-occurrence acts, bin
occurrences by context length; per bin compute node-means and grid RSA via (a) raw top-2 PCs and
(b) supervised best-2D (top-6 PCs -> coords). If raw-PC RSA rises toward best-2D with context, that
reconciles our low-PCA-RSA with the paper's grid-in-PCA (the reorganization completes with context).

Env: NPZ LAYER(auto) NBINS(8) OUTDIR TAG
"""
import os, json
import numpy as np

NPZ = os.environ["NPZ"]; TAG = os.environ.get("TAG", "Llama")
NBINS = int(os.environ.get("NBINS", "8")); OUTDIR = os.environ.get("OUTDIR", ".")
coords = np.array([[i // 4, i % 4] for i in range(16)], float); Gc = coords - coords.mean(0)
GD = np.abs(coords[:, None] - coords[None]).sum(-1)[np.triu_indices(16, 1)]


def rdm(H): iu = np.triu_indices(16, 1); return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]
def sp(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
def nodemeans(X, node, mask):
    return np.stack([X[mask & (node == k)].mean(0) for k in range(16)])
def rsas(H):
    Hc = H - H.mean(0); U, S, Vh = np.linalg.svd(Hc, full_matrices=False)
    pc2 = sp(rdm(Hc @ Vh[:2].T), GD)
    Z = U[:, :6] * S[:6]; W = np.linalg.lstsq(Z, Gc, rcond=None)[0]; b2 = sp(rdm(Z @ W), GD)
    return pc2, b2, sp(rdm(Hc), GD)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    z = np.load(NPZ); node = z["meta_node"]; ctx = z["meta_context_length"]
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
    # pick the layer with max full-dim grid RSA over ALL occurrences
    best_L, best = layers[0], -1
    for L in layers:
        H = nodemeans(z[f"layer_{L}"].astype(np.float64), node, ctx >= ctx.min())
        r = sp(rdm(H - H.mean(0)), GD)
        if r > best: best, best_L = r, L
    print(f"[{TAG}] peak full-RSA layer = L{best_L} (RSA {best:.2f}); ctx range [{ctx.min()},{ctx.max()}]", flush=True)
    X = z[f"layer_{best_L}"].astype(np.float64)
    edges = np.linspace(ctx.min(), ctx.max() + 1, NBINS + 1).astype(int)
    out = {"tag": TAG, "layer": best_L, "bins": [], "pc2_rsa": [], "best2d_rsa": [], "full_rsa": [], "n": []}
    for i in range(NBINS):
        lo, hi = edges[i], edges[i + 1]; mask = (ctx >= lo) & (ctx < hi)
        if mask.sum() < 200 or (node[mask].max() < 15): continue
        H = nodemeans(X, node, mask); pc2, b2, full = rsas(H)
        out["bins"].append(int((lo + hi) // 2)); out["pc2_rsa"].append(pc2)
        out["best2d_rsa"].append(b2); out["full_rsa"].append(full); out["n"].append(int(mask.sum()))
        print(f"  ctx~{(lo+hi)//2:5d}: raw-PC2 RSA={pc2:.2f}  best-2D RSA={b2:.2f}  full={full:.2f}  (n={mask.sum()})", flush=True)
    json.dump(out, open(f"{OUTDIR}/pca_context_sweep_{TAG}.json", "w"), indent=2)
    print(f"DONE -> {OUTDIR}/pca_context_sweep_{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
