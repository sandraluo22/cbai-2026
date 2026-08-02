"""Correlate the DAS-learned parity subspace with the graph eigenmodes -- does the causally-defined
parity direction (found in head L14H26's 128-dim OUTPUT space) correspond to the analytic parity
eigenmode (m15 = alt×alt), or to something else (e.g. the product mode m14 = parity×fold that exp 2
found more causally important)?

Bridge from head-output space to node space: project each node's mean head output onto the DAS
direction -> a 16-value node pattern a_i = (z̄_node_i - mean)·v. Correlate a with every eigenmode u_k.
We do this for (i) the DAS r=1 learned direction and (ii) the raw prototype axis z̄(+)-z̄(-) as a
reference. Strong corr with m15 => causal and analytic pictures agree; strong corr with m14/mix => the
head's causal parity handle is the conjunctive mode, a dissociation.

Env: MODEL(Llama) HEAD(L14H26) DIR
Reads das_parity_<model>_<head>.json (needs znode/eigU/subspace fields).
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/4_circuits/das")
MODEL = os.environ.get("MODEL", "Llama"); HEAD = os.environ.get("HEAD", "L14H26")
VAR = os.environ.get("VAR", "parity")                               # parity | row | col
IDX = {1: "coord", 2: "coord", 3: "coord", 4: "fold", 5: "fold", 6: "product", 7: "parity×coord",
       8: "product", 9: "parity×coord", 10: "parity×coord", 11: "parity×coord", 12: "fold",
       13: "parity×fold", 14: "parity×fold", 15: "parity"}
CMAP = {"parity": "#C2410C", "parity×fold": "#EA580C", "parity×coord": "#F59E0B",
        "coord": "#1D4ED8", "fold": "#6B7280", "product": "#9CA3AF"}

ckpt = f"{DIR}/das_{VAR}_{MODEL}_{HEAD}.npz"                         # compact reusable checkpoint (no GPU/model needed)
if os.path.exists(ckpt):
    z = np.load(ckpt)
    znode = z["znode"]; U = z["eigU"]; w = z["eigw"]; das_v = z["R_1"][0]; proto = z["proto_delta"]
else:
    d = json.load(open(f"{DIR}/das_{VAR}_{MODEL}_{HEAD}.json"))
    znode = np.array(d["znode"]); U = np.array(d["eigU"]); w = np.array(d["eigw"])
    das_v = np.array(d["results"]["1"]["subspace"])[0]; proto = np.array(d["proto_delta"])
n = znode.shape[0]; Zc = znode - znode.mean(0)                       # center across nodes
das_v = das_v / (np.linalg.norm(das_v) + 1e-12)                     # DAS r=1 direction (unit, 128-d)
proto = proto / (np.linalg.norm(proto) + 1e-12)

def node_pattern(v): return Zc @ v                                   # [n] head-output-along-v per node
def corrs(a):
    a = a - a.mean()
    return np.array([abs(np.corrcoef(a, U[:, k] - U[:, k].mean())[0, 1]) for k in range(n)])

a_das = node_pattern(das_v); a_pro = node_pattern(proto)
c_das = corrs(a_das); c_pro = corrs(a_pro)
align_vp = float(abs(das_v @ proto))                                 # DAS dir vs raw prototype axis (in 128-d)

ks = list(range(1, n))
top = int(1 + np.argmax(c_das[1:]))
print(f"[{MODEL}/{HEAD}] DAS r=1 dir vs prototype axis (128-d): |cos|={align_vp:.2f}")
print(f"  DAS node-pattern best eigenmode: m{top} ({IDX.get(top,'?')})  |corr|={c_das[top]:.2f}")
print("  DAS |corr| by mode:", {f"m{k}": round(float(c_das[k]), 2) for k in ks if c_das[k] > 0.2})

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
cols = [CMAP.get(IDX.get(k, ""), "#9CA3AF") for k in ks]
ax[0].bar([k - 0.2 for k in ks], c_das[1:], 0.4, color=cols, label="DAS r=1 direction")
ax[0].bar([k + 0.2 for k in ks], c_pro[1:], 0.4, color="#D1D5DB", label="raw prototype axis")
ax[0].set_xticks(ks); ax[0].set_xticklabels([f"m{k}" for k in ks], fontsize=7)
ax[0].set_xlabel("eigenmode"); ax[0].set_ylabel("|corr( DAS node-pattern , eigenmode )|")
ax[0].set_title(f"which eigenmode is the causal {VAR} direction? (best = m{top} {IDX.get(top,'')})", fontsize=9)
ax[0].legend(fontsize=8, frameon=False); ax[0].spines[["top", "right"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CMAP.values()]
ax[0].legend(handles + [plt.Rectangle((0, 0), 1, 1, color="#D1D5DB")], list(CMAP) + ["prototype axis"],
             fontsize=6.5, frameon=False, ncol=2)

# the DAS node pattern on the 4x4 grid, next to the best-match eigenmode
coords = np.array([[i // 4, i % 4] for i in range(n)]) if n == 16 else None
def gridimg(a, axx, ti):
    if coords is None: axx.axis("off"); return
    g = np.full((4, 4), np.nan)
    for i, (r, c) in enumerate(coords): g[r, c] = a[i]
    vmax = np.nanmax(np.abs(a)); axx.imshow(g, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axx.set_title(ti, fontsize=9); axx.set_xticks([]); axx.set_yticks([])
sub = fig.add_gridspec(1, 2, left=0.56, right=0.98, top=0.82, bottom=0.18, wspace=0.3)
gridimg(a_das, fig.add_subplot(sub[0]), f"DAS r=1 node pattern")
gridimg(U[:, top], fig.add_subplot(sub[1]), f"eigenmode m{top} ({IDX.get(top,'')})")
ax[1].axis("off")

fig.suptitle(f"DAS {VAR} direction ↔ eigenmodes ({MODEL}, {HEAD}): "
             f"DAS·prototype |cos|={align_vp:.2f}, best mode m{top} |corr|={c_das[top]:.2f}", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = {"variable": VAR, "best_mode": top, "best_label": IDX.get(top, "?"), "best_corr": float(c_das[top]),
       "das_vs_prototype_cos": align_vp, "corr_by_mode": {f"m{k}": float(c_das[k]) for k in ks}}
json.dump(out, open(f"{DIR}/das_eigenmode_corr_{VAR}_{MODEL}_{HEAD}.json", "w"), indent=2)
for ext in ("pdf",):
    p = f"{DIR}/das_eigenmode_corr_{VAR}_{MODEL}_{HEAD}.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight"); print("wrote", p)
