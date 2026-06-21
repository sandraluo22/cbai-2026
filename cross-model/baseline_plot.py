"""Plot model rule-following accuracy (neighbour mass) atop the paper's
memorization baseline p_seen, and examine whether the baseline is informative
across configs (which differ in node count n).

Behavioural data only exists for the SQUARE GRID (Llama/Gemma/Qwen). For the
other configs we show the p_seen baseline shape (it depends only on n).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from config import get_config
import graph as G


def p_seen1(l, n):
    return 1 - ((n - 1) / n) ** l


def p_seen2(l, n):
    return p_seen1(l, n) - l * (1 / n) * ((n - 1) / n) ** (l - 1)


# ---- square-grid behavioural data (the only config with logits) ----
acc = json.load(open("runs/accuracy/accuracy.json"))
la = json.load(open("runs/llama/llama_accuracy.json"))
ctxs = [r["ctx"] for r in acc["gemma"]["by_context"]]
mass = {"Llama": [r["neighbor_mass"] for r in la["llama"]["by_context"]],
        "Gemma": [r["neighbor_mass"] for r in acc["gemma"]["by_context"]],
        "Qwen":  [r["neighbor_mass"] for r in acc["qwen"]["by_context"]]}

# in-context counter (empirical memorization) on the square grid
cfg = replace(get_config("gemma_qwen"), n_walks=40)
graph = G.build_grid_graph(cfg)
walks = G.generate_walks(graph, cfg)
bins = [(1, 15), (15, 50), (50, 150), (150, 500), (500, 1001)]
recs = []
for wk in walks:
    counts = np.zeros((16, 16))
    for s in range(len(wk.nodes) - 1):
        cur = wk.nodes[s]
        c = counts[cur]
        recs.append((s + 1, 1.0 if c.sum() > 0 else len(graph.neighbors(cur)) / 16))
        counts[cur, wk.nodes[s + 1]] += 1
recs = np.array(recs)
counter = [recs[(recs[:, 0] >= lo) & (recs[:, 0] < hi)][:, 1].mean() for lo, hi in bins]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# (1) square grid: models + counter atop p_seen (n=16)
C = {"Llama": "tab:blue", "Gemma": "tab:orange", "Qwen": "tab:green"}
for m in ["Llama", "Gemma", "Qwen"]:
    ax[0].plot(ctxs, mass[m], "-o", ms=4, color=C[m], label=m)
ax[0].plot(ctxs, counter, "-s", ms=4, color="black", label="in-context counter (empirical)")
ax[0].plot(ctxs, [p_seen1(l, 16) for l in ctxs], "--", color="0.4", label="p_seen1 (paper)")
ax[0].plot(ctxs, [p_seen2(l, 16) for l in ctxs], ":", color="0.4", label="p_seen2 (paper)")
ax[0].set_xscale("log"); ax[0].set_xlabel("context length"); ax[0].set_ylabel("neighbour mass / p_seen")
ax[0].set_title("Square grid (n=16): models vs memorization baseline"); ax[0].legend(fontsize=8)

# (2) baseline shape vs node count -> saturation
L = np.arange(1, 1001)
for n, lab in [(16, "n=16 (grid/ring/hex)"), (7, "n=7 (days)"), (25, "n=25 (5x5)")]:
    ax[1].plot(L, p_seen1(L, n), label=f"p_seen1, {lab}")
for c in [10, 30, 100, 300, 1000]:
    ax[1].axvline(c, color="0.9", lw=.6)
ax[1].set_xscale("log"); ax[1].set_xlabel("context length"); ax[1].set_ylabel("p_seen1")
ax[1].set_title("Baseline saturates fast for small n (our regime)"); ax[1].legend(fontsize=8)

fig.suptitle("Memorization baseline (p_seen): models atop it, and when it is informative")
fig.tight_layout()
fig.savefig("runs/baseline_vs_models.png", dpi=140)

print("square grid p_seen1(n=16):", [round(p_seen1(l, 16), 3) for l in ctxs])
print("days     p_seen1(n=7) :", [round(p_seen1(l, 7), 3) for l in ctxs])
print("wrote runs/baseline_vs_models.png")
