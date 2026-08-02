"""Per-round trace for the qwen32 run: theta, Source A (noisy), Source B (accurate),
and the model's estimate, for the anchor condition gap2.0_moderate_lean.

Top row: one representative game (seed 0), one panel per company — the literal four
values the model saw/produced each round. Bottom: pooled MAE per source + the model,
averaged over all companies and games, so the trust migration A->B is visible in aggregate.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/scoped32_qwen32")
COND = "gap2.0_moderate_lean"
SEED = 0

rows = [json.loads(l) for l in (OUT / "rounds.jsonl").read_text().splitlines()]
rows = [r for r in rows if r["condition"] == COND]
T = max(r["t"] for r in rows) + 1
M = max(r["company"] for r in rows) + 1

# --- one game (seed 0): per-company series ---
g = {(r["t"], r["company"]): r for r in rows if r["seed"] == SEED}

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, M, height_ratios=[1.0, 0.9], hspace=0.32, wspace=0.22)

styles = dict(theta=dict(color="black", lw=2.4, label="θ (truth)"),
              a=dict(color="#d6604d", lw=1.6, ls="--", marker="o", ms=3,
                     label="Source A (reputable, noisy)"),
              b=dict(color="#4393c3", lw=1.6, ls="--", marker="s", ms=3,
                     label="Source B (new, accurate)"),
              m=dict(color="#1a9850", lw=2.0, marker="D", ms=3.5,
                     label="Model estimate"))

for c in range(M):
    ax = fig.add_subplot(gs[0, c])
    ts = np.arange(1, T + 1)
    theta = np.array([g[(t, c)]["theta"] for t in range(T)])
    a = np.array([g[(t, c)]["a"] for t in range(T)])
    b = np.array([g[(t, c)]["b"] for t in range(T)])
    m = np.array([g[(t, c)]["model_est"] for t in range(T)])
    ax.plot(ts, theta, **styles["theta"])
    ax.plot(ts, a, **styles["a"])
    ax.plot(ts, b, **styles["b"])
    ax.plot(ts, m, **styles["m"])
    ax.set_title(f"Company {c + 1} (seed {SEED})")
    ax.set_xlabel("round")
    if c == 0:
        ax.set_ylabel("value")
        ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

# --- pooled: mean |error| per source + model, over all companies & games ---
err = defaultdict(lambda: defaultdict(list))   # err[t]['a'|'b'|'m'] -> list of abs errs
for r in rows:
    if np.isnan(r["model_est"]):
        continue
    err[r["t"]]["a"].append(abs(r["a"] - r["theta"]))
    err[r["t"]]["b"].append(abs(r["b"] - r["theta"]))
    err[r["t"]]["m"].append(abs(r["model_est"] - r["theta"]))

ts = np.arange(1, T + 1)
mae = {k: np.array([np.mean(err[t][k]) for t in range(T)]) for k in ("a", "b", "m")}

ax = fig.add_subplot(gs[1, :])
ax.plot(ts, mae["a"], color="#d6604d", lw=2, ls="--", marker="o", ms=3,
        label="Source A |error| (noisy)")
ax.plot(ts, mae["b"], color="#4393c3", lw=2, ls="--", marker="s", ms=3,
        label="Source B |error| (accurate)")
ax.plot(ts, mae["m"], color="#1a9850", lw=2.4, marker="D", ms=4,
        label="Model |error|")
ax.set_title(f"Pooled mean absolute error per round  —  {COND}  "
             f"(all {M} companies × {len(set(r['seed'] for r in rows))} games)")
ax.set_xlabel("round")
ax.set_ylabel("mean |estimate − θ|")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle("Qwen3-32B — sources vs. truth vs. model estimate  (gap=2.0, moderate rep, lean info, T=32)",
             fontsize=13, y=0.98)
dest = OUT / "sources_trace.png"
fig.savefig(dest, dpi=130, bbox_inches="tight")
print(f"wrote {dest}")
