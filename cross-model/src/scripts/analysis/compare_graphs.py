"""Compare the three models across ring / hex / days-of-week graphs."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["Llama", "Gemma", "Qwen"]
C = {"Llama": "tab:blue", "Gemma": "tab:orange", "Qwen": "tab:green"}


def load(g, m):
    return json.load(open(f"runs/v1/{g}/{m}_analysis.json"))


fig, ax = plt.subplots(1, 3, figsize=(17, 5))

# (1,2) ring + hex emergence
for col, g in enumerate(["ring", "hex"]):
    for m in MODELS:
        d = load(g, m)
        rows = d["emergence"]["rows"]
        ax[col].plot([r["ctx"] for r in rows], [r["rsa"] for r in rows],
                     "-o", ms=3, color=C[m], label=m)
    ax[col].set_xscale("log"); ax[col].axhline(0, color=".85", lw=.6)
    ax[col].set_xlabel("context length"); ax[col].set_ylabel("grid RSA (Nw=50)")
    ax[col].set_title(f"({col+1}) {g}: in-context emergence"); ax[col].legend()
    ax[col].set_ylim(-0.2, 0.9)

# (3) days: context vs semantic
x = np.arange(len(MODELS)); w = 0.35
ctx = [max(load("days", m)["grid_rsa"].values()) for m in MODELS]
sem = [max(load("days", m)["semantic_rsa"].values()) for m in MODELS]
ax[2].bar(x - w/2, ctx, w, label="context ring (in-context)", color="tab:purple")
ax[2].bar(x + w/2, sem, w, label="natural weekday order (semantic)", color="tab:gray")
ax[2].set_xticks(x); ax[2].set_xticklabels(MODELS)
ax[2].set_ylabel("peak RSA"); ax[2].set_title("(3) days-of-week: semantic conflict")
ax[2].legend(); ax[2].axhline(0, color=".85", lw=.6)

fig.suptitle("Three models across topologies: ring, hex, days-of-week (same walks)")
fig.tight_layout()
fig.savefig("runs/v1/overview/graph_comparison.png", dpi=140)
print("wrote runs/v1/overview/graph_comparison.png")
