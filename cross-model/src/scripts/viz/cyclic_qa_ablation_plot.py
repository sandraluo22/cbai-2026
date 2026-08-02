"""Plot the cyclic-arithmetic ablation: does the model do modular arithmetic on real learned cycles
(months=12, days=7) THROUGH the cyclic-position eigenmode? Left: arithmetic accuracy under baseline vs
ablating the fundamental (circular-position) eigenmode, a high-frequency mode, and a random direction of
equal rank. Right: represented power per cycle eigenmode (the fundamental should dominate).
Reads cyclic_qa_ablation_<model>_{months,days}.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/5_cyclic"); MODEL = os.environ.get("MODEL", "Llama")
CYCLES = os.environ.get("CYCLES", "months,days").split(",")
COND = ["baseline", "fundamental(pos)", "high_freq", "random"]
CCOL = {"baseline": "#111827", "fundamental(pos)": "#C2410C", "high_freq": "#7C3AED", "random": "#9CA3AF"}

data = {c: json.load(open(f"{DIR}/cyclic_qa_ablation_{MODEL}_{c}.json")) for c in CYCLES}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

x = np.arange(len(CYCLES)); w = 0.2
for j, cond in enumerate(COND):
    accs = [data[c]["baseline"] if cond == "baseline" else data[c]["ablate"][cond]["accuracy"] for c in CYCLES]
    ax[0].bar(x + (j - 1.5) * w, accs, w, color=CCOL[cond], label=cond)
    for xi, a in zip(x + (j - 1.5) * w, accs): ax[0].text(xi, a + 0.01, f"{a:.2f}", ha="center", fontsize=7)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{c}\n({data[c]['n']}-cycle)" for c in CYCLES])
ax[0].set_ylabel("modular-arithmetic accuracy"); ax[0].set_ylim(0, 1.12)
ax[0].axhline(0, color="k", lw=0.5); ax[0].set_title("ablating the cyclic-position eigenmode breaks the arithmetic", fontsize=10)
ax[0].legend(fontsize=8, frameon=False, ncol=2); ax[0].spines[["top", "right"]].set_visible(False)

for c in CYCLES:
    p = np.array(data[c]["power_by_mode"]); ks = np.arange(1, len(p))
    ax[1].plot(ks, p[1:], "o-", lw=2, label=f"{c} (prominent m{data[c]['prominent_mode']})")
ax[1].set_xlabel("cycle eigenmode (Fourier, k→)"); ax[1].set_ylabel("represented power fraction")
ax[1].set_title("fundamental (m1/m2) dominates the representation", fontsize=10)
ax[1].legend(fontsize=8, frameon=False); ax[1].grid(color="#EEE", lw=.6); ax[1].spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Cyclic-arithmetic ablation ({MODEL}): the model computes 'N months/days after X' through the "
             f"circular-position eigenmode", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf",):
    out = f"{DIR}/cyclic_qa_ablation_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
