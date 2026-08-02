"""Experiment 4 summary + circuit cross-check. Three complementary causal maps come out of the head
interchange (inject source-graph head-output into the grid run), each answering a different question:

  grid-disruption  = drop in grid-neighbour mass when head h is overwritten (source-agnostic) -> which
                     heads are NECESSARY for grid prediction. Cross-checks the ablation circuit.
  src-redirection  = gain in SOURCE-neighbour mass -> which heads MOVE behaviour toward the source
                     structure (movers / readers).
  parity-specific  = antiprism-disruption - ring-disruption -> heads whose role is parity-SPECIFIC
                     (vs coord-specific), isolating the parity builders from the coord builders.

We compare grid-disruption to the ablation-derived parity/coord circuit (head_eig_sweep 'damage') by
rank-overlap (prec@k) and correlation restricted to the top circuit heads (global Pearson is diluted by
~1000 near-zero heads and is the wrong statistic).

Env: MODEL(Llama) DIR SWEEPDIR OUTDIR
Reads interchange_<model>_{antiprism,ring}.json + head_eig_sweep_<model>_grid.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

MODEL = os.environ.get("MODEL", "Llama")
DIR = os.environ.get("DIR", "runs/axes/4_circuits/interchange")
SWEEPDIR = os.environ.get("SWEEPDIR", "runs/axes/4_circuits/head_eig_sweep")
OUTDIR = os.environ.get("OUTDIR", DIR)

ap = json.load(open(f"{DIR}/interchange_{MODEL}_antiprism.json"))
rg = json.load(open(f"{DIR}/interchange_{MODEL}_ring.json"))
sw = json.load(open(f"{SWEEPDIR}/head_eig_sweep_{MODEL}_grid.json"))
nH = sw["nH"]; dmg = np.array(sw["damage"]); eig = np.array(sw["eigenvalues"])
par_w = dmg[int(np.argmax(eig))]; lo = np.argsort(eig)[:2]; coord_w = dmg[lo[0]] + dmg[lo[1]]

ap_disrupt = -np.array(ap["d_grid_nbr"]); rg_disrupt = -np.array(rg["d_grid_nbr"])
ap_redir = np.array(ap["d_src_nbr"]); rg_redir = np.array(rg["d_src_nbr"])
par_specific = ap_disrupt - rg_disrupt                        # parity- vs coord-specific
nL = ap_disrupt.shape[0]

def hn(i): return f"L{i // nH}H{i % nH}"
def topk(m, k=10): return [hn(i) for i in np.argsort(m, axis=None)[::-1][:k]]
def prec_at_k(a, b, k=15):
    A = set(np.argsort(a, axis=None)[::-1][:k]); B = set(np.argsort(b, axis=None)[::-1][:k]); return len(A & B) / k
def corr_top(m, w, k=50):
    idx = np.argsort(w, axis=None)[::-1][:k]; return float(np.corrcoef(m.flatten()[idx], w.flatten()[idx])[0, 1])

stats = {
    "antiprism_disrupt_vs_parity": {"prec@15": prec_at_k(ap_disrupt, par_w), "corr_top50": corr_top(ap_disrupt, par_w)},
    "ring_disrupt_vs_coord": {"prec@15": prec_at_k(rg_disrupt, coord_w), "corr_top50": corr_top(rg_disrupt, coord_w)},
    "parity_specific_top": topk(par_specific, 8), "coord_specific_top": topk(-par_specific, 8),
    "src_redirection_top(movers)": topk(ap_redir, 8),
}
json.dump(stats, open(f"{OUTDIR}/interchange_summary_{MODEL}.json", "w"), indent=2)
for k, v in stats.items(): print(k, "=", v)

# ---- figure ----
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
def heat(a, m, ti, lab=False):
    v = max(1e-6, float(np.nanpercentile(np.abs(m), 99)))
    im = a.imshow(m, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    a.set_xlabel("head"); a.set_ylabel("layer"); a.set_title(ti, fontsize=9); fig.colorbar(im, ax=a, fraction=.046)
    if lab:
        for i in np.argsort(m, axis=None)[::-1][:5]:
            a.annotate(hn(i), (i % nH, i // nH), fontsize=6.5, color="k", ha="center")

heat(ax[0, 0], ap_disrupt, "antiprism → grid-disruption (parity arm necessity)")
heat(ax[0, 1], rg_disrupt, "ring → grid-disruption (coord arm necessity)")
heat(ax[0, 2], par_specific, "parity-specific = antiprism − ring disruption", lab=True)
heat(ax[1, 0], par_w, "parity circuit (ablation damage, top-eig mode)")
heat(ax[1, 1], coord_w, "coord circuit (ablation damage, 2 lowest modes)")

axs = ax[1, 2]
axs.scatter(par_w.flatten(), ap_disrupt.flatten(), s=8, alpha=.4, color="#C2410C", label="parity: disrupt vs circuit")
axs.scatter(coord_w.flatten(), rg_disrupt.flatten(), s=8, alpha=.4, color="#1D4ED8", label="coord: disrupt vs circuit")
axs.set_xlabel("ablation circuit-write (damage)"); axs.set_ylabel("interchange grid-disruption")
axs.set_title(f"disrupt vs circuit\nparity r={stats['antiprism_disrupt_vs_parity']['corr_top50']:.2f}, "
              f"coord r={stats['ring_disrupt_vs_coord']['corr_top50']:.2f} (top-50)", fontsize=9)
axs.legend(fontsize=7, frameon=False); axs.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Exp 4 — causal head interchange vs the parity/coord circuit ({MODEL}). "
             f"Parity-specific top: {', '.join(stats['parity_specific_top'][:4])}", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("pdf",):
    out = f"{OUTDIR}/interchange_summary_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
