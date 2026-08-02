"""Eigenmode power spectrum vs in-context length (grid), from per-occurrence acts. Bin by context,
compute node-means, project onto unnormalized-Laplacian eigenmodes -> power fraction per mode.
Shows whether parity/coord power strengthens at long context. Env: NPZ TAG OUTDIR NBINS
"""
import os, json
import numpy as np

NPZ = os.environ["NPZ"]; TAG = os.environ.get("TAG", "Llama"); OUTDIR = os.environ.get("OUTDIR", ".")
NBINS = int(os.environ.get("NBINS", "6"))
coords = np.array([[i // 4, i % 4] for i in range(16)], float)
GD = np.abs(coords[:, None] - coords[None]).sum(-1)[np.triu_indices(16, 1)]
A = np.zeros((16, 16))
for i in range(16):
    r, c = i // 4, i % 4
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < 4 and 0 <= cc < 4: A[i, rr * 4 + cc] = 1
w, V = np.linalg.eigh(np.diag(A.sum(1)) - A)                      # unnormalized eigenmodes


def nm(X, node, mask): return np.stack([X[mask & (node == k)].mean(0) for k in range(16)])
def spec(H):
    Hc = H - H.mean(0); c = V.T @ Hc; p = (c ** 2).sum(1); p[0] = 0; return p / (p.sum() + 1e-12)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    z = np.load(NPZ); node = z["meta_node"]; ctx = z["meta_context_length"]
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
    def rdm(H): iu = np.triu_indices(16, 1); return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]
    def sp(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    bestL, best = layers[0], -1
    for L in layers:
        H = nm(z[f"layer_{L}"].astype(np.float64), node, ctx >= ctx.min())
        r = sp(rdm(H - H.mean(0)), GD)
        if r > best: best, bestL = r, L
    X = z[f"layer_{bestL}"].astype(np.float64)
    edges = np.linspace(ctx.min(), ctx.max() + 1, NBINS + 1).astype(int)
    out = {"tag": TAG, "layer": bestL, "eigenvalues": [float(x) for x in w[1:]], "bins": [], "spectra": [],
           "parity_pow": [], "coord_pow": []}
    for i in range(NBINS):
        lo, hi = edges[i], edges[i + 1]; mask = (ctx >= lo) & (ctx < hi)
        if mask.sum() < 200 or node[mask].max() < 15: continue
        p = spec(nm(X, node, mask))
        out["bins"].append(int((lo + hi) // 2)); out["spectra"].append([float(x) for x in p[1:]])
        out["parity_pow"].append(float(p[15])); out["coord_pow"].append(float(p[1] + p[2]))
        print(f"[{TAG}] ctx~{(lo+hi)//2:5d}: parity={p[15]:.3f} coord={p[1]+p[2]:.3f}", flush=True)
    json.dump(out, open(f"{OUTDIR}/eigmode_context_{TAG}.json", "w"), indent=2)
    print(f"DONE (L{bestL}) -> {OUTDIR}/eigmode_context_{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
