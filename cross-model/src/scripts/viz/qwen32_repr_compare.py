"""Representation comparison: Qwen-32B vs the 8B models (Llama-8B, Qwen-8B) on the grid, from node-means.
(a) best-2D grid RSA vs relative depth; (b) full-dim grid RSA vs relative depth;
(c) graph-Fourier eigenmode power spectrum at each model's peak layer; (d) cross-model RSA at matched
relative depth (Qwen32 vs each 8B); + best-2D layout of Qwen-32B at its peak.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "runs/axes/1_decomposition/divider_basis"
MODELS = {"Llama-8B": "Llama", "Qwen-8B": "Qwen", "Qwen-32B": "Qwen32"}
COL = {"Llama-8B": "#1D4ED8", "Qwen-8B": "#059669", "Qwen-32B": "#C2410C"}
coords = np.array([[i // 4, i % 4] for i in range(16)], float); Gc = coords - coords.mean(0)
GD = np.abs(coords[:, None] - coords[None]).sum(-1)[np.triu_indices(16, 1)]
# normalized Laplacian eigenmodes (grid)
A = np.zeros((16, 16))
for i in range(16):
    r, c = i // 4, i % 4
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < 4 and 0 <= cc < 4: A[i, rr * 4 + cc] = 1
dg = A.sum(1); di = 1 / np.sqrt(dg); U = np.linalg.eigh(np.eye(16) - di[:, None] * A * di[None, :])[1]


def rdm(H): iu = np.triu_indices(16, 1); return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]
def sp(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def load(tag):
    z = np.load(f"{DIR}/nodemeans_{tag}_square_grid.npz")
    nL = sum(k.startswith("layer_") for k in z.files)
    return [z[f"layer_{L}"].astype(np.float64) for L in range(nL)]


def best2d(H):
    Hc = H - H.mean(0); Uu, S, Vh = np.linalg.svd(Hc, full_matrices=False)
    Z = Uu[:, :6] * S[:6]; W = np.linalg.lstsq(Z, Gc, rcond=None)[0]; return Z @ W


def curves(Hs):
    full, b2 = [], []
    for H in Hs:
        Hc = H - H.mean(0)
        full.append(sp(rdm(Hc), GD)); b2.append(sp(rdm(best2d(H)), GD))
    return np.array(full), np.array(b2)


data = {name: load(tag) for name, tag in MODELS.items()}
rd = {name: np.linspace(0, 1, len(Hs)) for name, Hs in data.items()}
crv = {name: curves(Hs) for name, Hs in data.items()}
peak = {name: int(np.argmax(crv[name][1])) for name in data}          # best-2d peak layer

fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(2, 3)
# (a) best-2d RSA vs rel depth
axa = fig.add_subplot(gs[0, 0])
for name in data:
    axa.plot(rd[name], crv[name][1], "-", color=COL[name], lw=2, label=f"{name} (peak L{peak[name]}={crv[name][1][peak[name]]:.2f})")
axa.set_title("Grid best-2D RSA vs relative depth"); axa.set_xlabel("relative depth"); axa.set_ylabel("RSA")
axa.set_ylim(0, 1); axa.legend(fontsize=8, frameon=False); axa.grid(color="#EEE")
# (b) full-dim RSA
axb = fig.add_subplot(gs[0, 1])
for name in data: axb.plot(rd[name], crv[name][0], "-", color=COL[name], lw=2, label=name)
axb.set_title("Grid full-dim RSA vs relative depth"); axb.set_xlabel("relative depth"); axb.set_ylabel("RSA")
axb.set_ylim(0, 1); axb.legend(fontsize=8, frameon=False); axb.grid(color="#EEE")
# (c) graph-Fourier eigenmode power at peak
axc = fig.add_subplot(gs[0, 2]); k = np.arange(1, 16); nm = len(data); bw = 0.8 / nm
for i, name in enumerate(data):
    H = data[name][peak[name]]; Hc = H - H.mean(0); tot = (Hc ** 2).sum()
    P = np.array([((Hc.T @ U[:, kk]) ** 2).sum() / tot for kk in k])
    axc.bar(k + (i - (nm - 1) / 2) * bw, P, bw, color=COL[name], label=name)
axc.set_title("Graph-Fourier: eigenmode power at peak layer"); axc.set_xlabel("eigenmode (low→high freq)")
axc.set_ylabel("power frac"); axc.set_xticks(k[::2]); axc.legend(fontsize=8, frameon=False)
# (d) cross-model RSA at matched relative depth
axd = fig.add_subplot(gs[1, 0]); grid_d = np.linspace(0, 1, 21)
def rdm_at(Hs, d): return rdm((lambda H: H - H.mean(0))(Hs[int(round(d * (len(Hs) - 1)))]))
for other in ["Llama-8B", "Qwen-8B"]:
    xr = [sp(rdm_at(data["Qwen-32B"], d), rdm_at(data[other], d)) for d in grid_d]
    axd.plot(grid_d, xr, "-", lw=2, label=f"Qwen-32B vs {other}", color=COL[other])
axd.set_title("Cross-model RSA at matched relative depth"); axd.set_xlabel("relative depth")
axd.set_ylabel("RDM Spearman"); axd.set_ylim(0, 1); axd.legend(fontsize=8, frameon=False); axd.grid(color="#EEE")
# (e) best-2d layout Qwen-32B at peak
axe = fig.add_subplot(gs[1, 1:]); B = best2d(data["Qwen-32B"][peak["Qwen-32B"]])
for j in range(2):
    if np.corrcoef(B[:, j], Gc[:, j])[0, 1] < 0: B[:, j] = -B[:, j]
edges = [(i, j) for i in range(16) for dr, dc in [(0, 1), (1, 0)] for j in [i + (dc + 4 * dr)]
         if (i % 4 + dc < 4 and i // 4 + dr < 4)]
for a, b in edges: axe.plot([B[a, 0], B[b, 0]], [B[a, 1], B[b, 1]], color="0.8", zorder=1)
axe.scatter(B[:, 0], B[:, 1], c=np.arange(16), cmap="tab20", s=120, zorder=2, edgecolors="white")
axe.set_title(f"Qwen-32B best-2D grid layout @ L{peak['Qwen-32B']} (RSA {crv['Qwen-32B'][1][peak['Qwen-32B']]:.2f})")
axe.set_xticks([]); axe.set_yticks([]); axe.set_aspect("equal")

fig.suptitle("Qwen-32B vs 8B — in-context grid representation (Qwen-32B is post-trained; 8B are base)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs("runs/axes/1_decomposition/qwen32", exist_ok=True)
for ext in ("pdf", "png"): fig.savefig(f"runs/axes/1_decomposition/qwen32/qwen32_repr_compare.{ext}", dpi=140, bbox_inches="tight")
json.dump({name: {"peak": peak[name], "best2d_peak_rsa": float(crv[name][1][peak[name]]),
                  "full_peak_rsa": float(crv[name][0].max())} for name in data},
          open("runs/axes/1_decomposition/qwen32/qwen32_repr_compare.json", "w"), indent=2)
print("wrote qwen32_repr_compare.png ; peaks:", {n: (peak[n], round(float(crv[n][1][peak[n]]), 2)) for n in data})
