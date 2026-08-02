"""Single-panel raw-value trace for the qwen32 run: θ, Source A, Source B, and the
model estimate per round, for the anchor condition gap2.0_moderate_lean, one game
(seed 0), averaged over the M companies (one line per series).

Pooling raw values across games is uninformative (θ~N(mu, theta_scale) fresh each round,
all sources ~unbiased -> every line collapses to ~mu). Within one game θ varies, so the
four series are meaningfully comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/scoped32_qwen32")
COND = "gap2.0_moderate_lean"
SEED = 0

rows = [json.loads(l) for l in (OUT / "rounds.jsonl").read_text().splitlines()]
rows = [r for r in rows if r["condition"] == COND and r["seed"] == SEED]
T = max(r["t"] for r in rows) + 1

def series(key):
    out = np.full(T, np.nan)
    for t in range(T):
        vals = [r[key] for r in rows if r["t"] == t]
        out[t] = np.nanmean(vals)
    return out

ts = np.arange(1, T + 1)
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.plot(ts, series("theta"), color="black", lw=2.6, label="θ (truth)")
ax.plot(ts, series("a"), color="#d6604d", lw=1.8, ls="--", marker="o", ms=4,
        label="Source A (reputable, noisy)")
ax.plot(ts, series("b"), color="#4393c3", lw=1.8, ls="--", marker="s", ms=4,
        label="Source B (new, accurate)")
ax.plot(ts, series("model_est"), color="#1a9850", lw=2.2, marker="D", ms=4.5,
        label="Model estimate")
ax.set_xlabel("round")
ax.set_ylabel("value (mean over the 3 companies, seed 0)")
ax.set_title("Qwen3-32B — A vs. θ vs. B vs. model  (gap=2.0, moderate rep, lean info, seed 0)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
dest = OUT / "sources_values.png"
fig.savefig(dest, dpi=130, bbox_inches="tight")
print(f"wrote {dest}")
