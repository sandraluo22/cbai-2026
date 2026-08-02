"""Experiment 6 -- decompose the distributed parity circuit across layers. The parity computation is not
one head: exp 4 showed it spread over depth. Here we lay four per-head (layer x head) maps on a common
depth axis and aggregate per layer to expose the builder -> mover -> reader flow:

  builders (parity-mode build, head_eig_sweep 'damage', top-eig mode)   -- write parity into the residual
  movers   (induction/QK task score)                                    -- carry it to the readout token
  readers  (direct-logit attribution, head_attribution 'head_attr')     -- turn it into output logits
  causal   (exp-4 antiprism-minus-ring grid-disruption = parity-specific interchange)

For each role we take the per-layer sum of the positive part, normalise to its peak, and report the
layer centre-of-mass. If builders peak earliest, movers in the middle, readers latest -- and the causal
parity-specific interchange tracks the movers/readers -- the distributed circuit has a clean depth order.

Env: MODEL(Llama) DIR SWEEPDIR INDJSON DLAJSON OUTDIR
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

MODEL = os.environ.get("MODEL", "Llama")
DIR = os.environ.get("DIR", "runs/axes/4_circuits/interchange")
SWEEPDIR = os.environ.get("SWEEPDIR", "runs/axes/4_circuits/head_eig_sweep")
INDJSON = os.environ.get("INDJSON", "runs/induction-head/1_circuits/induction_heads/induction.json")
DLAJSON = os.environ.get("DLAJSON", "runs/induction-head/1_circuits/attribution/head_attribution_square_grid.json")
OUTDIR = os.environ.get("OUTDIR", DIR)

sw = json.load(open(f"{SWEEPDIR}/head_eig_sweep_{MODEL}_grid.json"))
dmg = np.array(sw["damage"]); eig = np.array(sw["eigenvalues"]); nL, nH = sw["nL"], sw["nH"]
builders = dmg[int(np.argmax(eig))]                                          # parity-mode build
movers = np.array(json.load(open(INDJSON))["models"][MODEL]["task"])          # induction/QK
readers = np.array(json.load(open(DLAJSON))["models"][MODEL]["head_attr"])    # DLA
ap = json.load(open(f"{DIR}/interchange_{MODEL}_antiprism.json"))
rg = json.load(open(f"{DIR}/interchange_{MODEL}_ring.json"))
causal = (-np.array(ap["d_grid_nbr"])) - (-np.array(rg["d_grid_nbr"]))         # parity-specific disruption
redir = np.array(ap["d_src_nbr"])                                             # redirection (movers, causal)

ROLES = [("builders (parity write)", builders, "#C2410C"),
         ("movers (induction QK)", movers, "#7C3AED"),
         ("readers (DLA)", readers, "#059669"),
         ("causal parity-specific (exp4)", causal, "#111827"),
         ("causal redirection (exp4)", redir, "#9CA3AF")]

def profile(m):
    p = np.clip(m, 0, None).sum(1)                                           # per-layer positive mass
    return p / (p.max() + 1e-12)
def com(p):                                                                  # layer centre-of-mass
    return float((np.arange(len(p)) * p).sum() / (p.sum() + 1e-12))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
layers = np.arange(nL)
coms = {}
for name, m, c in ROLES:
    p = profile(m); coms[name] = com(p)
    ax[0].plot(layers, p, "-", color=c, lw=2, label=f"{name}  (CoM L{coms[name]:.0f})")
ax[0].set_xlabel("layer"); ax[0].set_ylabel("per-layer positive mass (norm)")
ax[0].set_title(f"{MODEL}: parity circuit decomposed by depth", fontsize=10)
ax[0].legend(fontsize=7.5, frameon=False); ax[0].grid(axis="y", color="#EEE", lw=.6)
ax[0].spines[["top", "right"]].set_visible(False)

# centre-of-mass ordering bar
names = [r[0] for r in ROLES]; cs = [coms[n] for n in names]; colors = [r[2] for r in ROLES]
ax[1].barh(range(len(names)), cs, color=colors)
ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=8); ax[1].invert_yaxis()
ax[1].set_xlabel("layer centre-of-mass"); ax[1].set_title("depth ordering of roles", fontsize=10)
for i, v in enumerate(cs): ax[1].text(v + 0.2, i, f"L{v:.1f}", va="center", fontsize=8)
ax[1].spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Exp 6 — the distributed parity circuit: builder → mover → reader across depth ({MODEL})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
json.dump({"centre_of_mass": coms}, open(f"{OUTDIR}/interchange_decompose_{MODEL}.json", "w"), indent=2)
for ext in ("pdf",):
    out = f"{OUTDIR}/interchange_decompose_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
print("centre-of-mass:", {k: round(v, 1) for k, v in coms.items()})
