"""Export per-LAYER top-3 PCA of the node representation for the interactive 3-D viewer.

For every layer: PCA (top-3) of the centred node-means; project BOTH the 16 node-means and a
subsample of per-occurrence points onto those same 3 PCs (so node-mean and per-occ share axes and
sit side by side). PC sign is aligned to the previous layer to keep scrubbing smooth. Colour key =
graph 2-colouring (parity). Emits a compact JSON that the HTML artifact embeds.

Reads a per-occurrence acts_sub npz (layer_* + meta_node + meta_context_length) for the cloud, or a
node-mean npz (no cloud). CPU-only.
Env: PEROCC | NODEMEAN, TAG, GRAPH, ROWS(4) COLS(4), CTXLO(100), NPTS(300), OUTJSON
"""
from __future__ import annotations
import os, json
import numpy as np

PEROCC = os.environ.get("PEROCC", "runs/v2/square_grid/Llama_acts_sub.npz")
NODEMEAN = os.environ.get("NODEMEAN", "")
TAG = os.environ.get("TAG", "Llama"); GRAPH = os.environ.get("GRAPH", "square_grid")
ROWS = int(os.environ.get("ROWS", "4")); COLS = int(os.environ.get("COLS", "4"))
CTXLO = int(os.environ.get("CTXLO", "100")); NPTS = int(os.environ.get("NPTS", "300"))
OUTJSON = os.environ.get("OUTJSON", "runs/axes/2_geometry/axis_geometry/pca3d_Llama_square_grid.json")


def grid_adjacency(rows, cols):
    n = rows * cols; A = np.zeros((n, n))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols: A[i, rr * cols + cc] = 1
    return A


def two_colour(A):
    n = A.shape[0]; col = np.zeros(n)
    for s in range(n):
        if col[s] != 0: continue
        col[s] = 1; st = [s]
        while st:
            u = st.pop()
            for v in np.where(A[u] > 0)[0]:
                if col[v] == 0: col[v] = -col[u]; st.append(v)
    return col.astype(int)


def main():
    os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
    if PEROCC and os.path.exists(PEROCC):
        npz = np.load(PEROCC, allow_pickle=True); has_cloud = True
        node = np.asarray(npz["meta_node"]); ctx = np.asarray(npz["meta_context_length"])
        n = ROWS * COLS; A = grid_adjacency(ROWS, COLS)
        keep = np.where(ctx >= CTXLO)[0]
        rng = np.random.default_rng(0)
        sub = keep if len(keep) <= NPTS else rng.choice(keep, NPTS, replace=False)
    else:
        npz = np.load(NODEMEAN, allow_pickle=True); has_cloud = False
        A = np.array(npz["adjacency"], float); n = A.shape[0]; node = None; sub = None
    nL = sum(1 for k in npz.files if k.startswith("layer_"))
    parity = two_colour(A).tolist()

    layers = []; prevV = None
    for L in range(nL):
        H = npz[f"layer_{L}"].astype(np.float64)
        if has_cloud:
            means = np.stack([H[(node == j) & (ctx >= CTXLO)].mean(0) for j in range(n)])
        else:
            means = H
        mu = means.mean(0); Hc = means - mu
        U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
        V3 = Vt[:3].T                                   # d x 3
        # sign-align to previous layer via node-mean projection correlation
        nm = Hc @ V3
        if prevV is not None:
            for a in range(3):
                if float((nm[:, a] * prevV[:, a]).sum()) < 0:
                    V3[:, a] *= -1; nm[:, a] *= -1
        prevV = nm.copy()
        evr = (S[:3] ** 2 / (S ** 2).sum()).tolist()
        entry = {"layer": L, "evr": [round(x, 3) for x in evr],
                 "nodemean": [[round(float(v), 3) for v in row] for row in nm]}
        if has_cloud:
            cl = (H[sub] - mu) @ V3
            entry["cloud"] = [[round(float(v), 2) for v in row] for row in cl]
        layers.append(entry)

    out = {"tag": TAG, "graph": GRAPH, "n": n, "nL": nL, "parity": parity,
           "has_cloud": has_cloud, "cloud_node": ([int(x) for x in node[sub]] if has_cloud else None),
           "layers": layers}
    json.dump(out, open(OUTJSON, "w"))
    print(f"[{TAG}/{GRAPH}] exported {nL} layers, cloud={has_cloud} ({len(sub) if sub is not None else 0} pts) -> {OUTJSON} "
          f"({os.path.getsize(OUTJSON)//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
