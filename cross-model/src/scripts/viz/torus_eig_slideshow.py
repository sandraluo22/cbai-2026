"""Slideshows of torus eigenmodes RECOVERED FROM THE MODEL (node's projection onto the model's mode-k
representational direction), over the full (w,L) grid.
  FAMILY=w : one slide per WIDTH w; each slide shows the top modes for EVERY length L at that w.
  FAMILY=L : one slide per LENGTH L; each slide shows the top modes for EVERY width w at that L.
Each cell is the mode drawn on the flattened w×L torus (width on y, length on x), labelled by its Fourier
frequency (a=length, b=width), eigenvalue and firing power — so you can watch the frequency content shift
as the varied dimension grows. Reads torus_walk_<model>.json.

Env: MODEL(Llama) FAMILY(w|L) KMODES(4) OUTDIR
Out: <OUTDIR>/torus_eig_slideshow_<model>_<family>.pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DIR = os.environ.get("DIR", "runs/axes/7_torus"); MODEL = os.environ.get("MODEL", "Llama")
FAMILY = os.environ.get("FAMILY", "w"); K = int(os.environ.get("KMODES", "4"))
C = json.load(open(f"{DIR}/torus_walk_{MODEL}.json"))["combos"]

ws = sorted(set(c["w"] for c in C.values())); Ls = sorted(set(c["L"] for c in C.values()))
fixvals, varyvals = (ws, Ls) if FAMILY == "w" else (Ls, ws)
fixname, varyname = ("w", "L") if FAMILY == "w" else ("L", "w")


def key_for(fx, vary):
    w, L = (fx, vary) if FAMILY == "w" else (vary, fx)
    return f"w{w}_L{L}"


out = f"{DIR}/torus_eig_slideshow_{MODEL}_{FAMILY}.pdf"
with PdfPages(out) as pdf:
    for fx in fixvals:
        rows = [(v, key_for(fx, v)) for v in varyvals if key_for(fx, v) in C]
        if not rows: continue
        fig, axes = plt.subplots(len(rows), K + 1, figsize=(2.2 * (K + 1), 1.9 * len(rows)), squeeze=False)
        for r, (v, k) in enumerate(rows):
            c = C[k]; w, L, Lst = c["w"], c["L"], c["best_layer"]
            proj = np.array(c["eig_proj"][Lst])                    # [NTOP, n] model projections at best layer
            for m in range(K):
                axm = axes[r][m]
                if m < proj.shape[0]:
                    g = proj[m].reshape(L, w); vmax = np.abs(g).max() + 1e-9
                    axm.imshow(g.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                    a, b = c["eig_freq"][m]
                    axm.set_title(f"(a{a},b{b}) p{c['eig_power'][m]:.2f}", fontsize=7)
                axm.set_xticks([]); axm.set_yticks([])
                if m == 0: axm.set_ylabel(f"{varyname}={v}\n{c['n']}n", fontsize=8)
            # firing spectrum in last column
            p = np.array(c["eig_power_by_mode"]); axes[r][K].bar(range(1, len(p)), p[1:], color="#7C3AED")
            axes[r][K].set_xticks([]); axes[r][K].set_yticks([]); axes[r][K].set_title("spectrum", fontsize=7)
            axes[r][K].spines[["top", "right"]].set_visible(False)
        for m in range(K): axes[0][m].annotate(f"mode {m+1}", (0.5, 1.35), xycoords="axes fraction", ha="center", fontsize=8, color="#6B7280")
        fig.suptitle(f"Model-recovered torus eigenmodes — {fixname} = {fx} (varying {varyname}); "
                     f"columns = top-{K} firing modes, drawn on the w×L torus", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)
print("wrote", out, f"({len(fixvals)} slides)")
