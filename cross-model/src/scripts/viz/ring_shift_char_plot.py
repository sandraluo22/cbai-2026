"""Characterize the shift of the shared ring circuit across sizes n=3..16. Three panels:
 (a) per-step rotation: principal angle between consecutive-size directions, with a c/n^p fit -> ~1/n law
     (the fundamental of an n-cycle has angular frequency 2*pi/n, so the axis reorients ~1/n per size step);
 (b) gap-decay: mean subspace alignment vs size gap -> smooth Toeplitz band, alignment ~ geometric in gap;
 (c) MDS trajectory: the 14 directions embedded via classical MDS on principal-angle distance -> a curved
     2-D arc that ROTATES FAST among small rings and CONVERGES to a large-ring limit (10-16 crowd together).
Reads ring_shift_dirs_<model>.npz.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as mcm

DIR = os.environ.get("DIR", "runs/axes/5_cyclic"); MODEL = os.environ.get("MODEL", "Llama"); L = int(os.environ.get("LAYER", "30"))
z = np.load(f"{DIR}/ring_shift_dirs_{MODEL}.npz")
D = z["dirs"]; R = list(z["rings"]); nR = len(R)

def align(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False); return float((s ** 2).mean())
def ang(A, B):
    return np.degrees(np.arccos(np.sqrt(max(0.0, min(1.0, align(A, B))))))

Dist = np.array([[ang(D[i, L], D[j, L]) for j in range(nR)] for i in range(nR)])
Al = np.array([[align(D[i, L], D[j, L]) for j in range(nR)] for i in range(nR)])

# classical MDS on angular distance
Dsq = Dist ** 2; J = np.eye(nR) - np.ones((nR, nR)) / nR; B = -0.5 * J @ Dsq @ J
w, V = np.linalg.eigh(B); idx = np.argsort(w)[::-1]; w = np.clip(w[idx], 0, None); V = V[:, idx]
emb = V[:, :2] * np.sqrt(w[:2]); share2 = w[:2].sum() / w.sum() * 100

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
cols = mcm.viridis(np.linspace(0, 1, nR))

# (a) per-step rotation + 1/n fit
steps = np.array([Dist[i, i + 1] for i in range(nR - 1)]); mids = np.array([(R[i] + R[i + 1]) / 2 for i in range(nR - 1)])
lp = np.polyfit(np.log(mids), np.log(steps), 1)
xf = np.linspace(mids.min(), mids.max(), 100)
ax[0].plot(mids, steps, "o-", color="#1D4ED8", lw=1.5, ms=6)
ax[0].plot(xf, np.exp(np.polyval(lp, np.log(xf))), "--", color="#C2410C", lw=1.5, label=f"fit  $\\theta\\propto n^{{{lp[0]:.2f}}}$")
ax[0].set_xlabel("ring size (midpoint of step n$\\to$n+1)"); ax[0].set_ylabel("principal angle between\nconsecutive-size directions (deg)")
ax[0].set_title("(a) rotation rate of the shared axis\nfast for small rings, freezes for large", fontsize=10)
ax[0].legend(frameon=False, fontsize=10); ax[0].spines[["top", "right"]].set_visible(False)

# (b) gap decay
gaps = range(1, nR); gm = [np.mean([Al[i, i + g] for i in range(nR - g)]) for g in gaps]
ax[1].plot(list(gaps), gm, "o-", color="#059669", lw=1.5, ms=6)
ax[1].set_xlabel("size gap  |n - m|"); ax[1].set_ylabel("mean direction alignment (cos$^2$)")
ax[1].set_title("(b) alignment decays smoothly with size gap\n(the Toeplitz sharing band)", fontsize=10)
ax[1].axhline(0, color="k", lw=.5); ax[1].spines[["top", "right"]].set_visible(False)

# (c) MDS trajectory
ax[2].plot(emb[:, 0], emb[:, 1], "-", color="#9CA3AF", lw=1.2, zorder=1)
sc = ax[2].scatter(emb[:, 0], emb[:, 1], c=R, cmap="viridis", s=90, zorder=2, edgecolor="k", lw=.5)
for i in range(nR):
    ax[2].annotate(str(R[i]), (emb[i, 0], emb[i, 1]), fontsize=7.5, ha="center", va="center", color="white", fontweight="bold")
ax[2].set_xlabel("MDS-1"); ax[2].set_ylabel("MDS-2")
ax[2].set_title(f"(c) the shift is a curved arc that converges\n(2-D captures {share2:.0f}%; large rings crowd together)", fontsize=10)
ax[2].spines[["top", "right"]].set_visible(False); ax[2].set_aspect("equal", "datalim")
fig.colorbar(sc, ax=ax[2], fraction=0.046, pad=0.04, label="ring size")

fig.suptitle(f"How the shared ring circuit shifts with size ({MODEL}, layer {L}): a ~1/n rotation converging to a large-ring limit", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = f"{DIR}/ring_shift_char_{MODEL}.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
