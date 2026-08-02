"""Signed-error trace for the qwen32 run, anchor condition gap2.0_moderate_lean:
model−θ, A−θ, B−θ per round, one game (seed 0), averaged over the M companies.

This is the raw-value plot with θ subtracted: it removes θ's large round-to-round swings
and shows only the errors. (Pooling signed errors across games is uninformative — all
three are ~unbiased and average to ≈0; within one game the realized errors are visible.)
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
rows = [r for r in rows if r["condition"] == COND and r["seed"] == SEED
        and not np.isnan(r["model_est"])]
T = max(r["t"] for r in rows) + 1

def err_series(key):
    out = np.full(T, np.nan)
    for t in range(T):
        out[t] = np.mean([r[key] - r["theta"] for r in rows if r["t"] == t])
    return out

ts = np.arange(1, T + 1)
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.axhline(0.0, color="black", lw=1.0)
ax.plot(ts, err_series("a"), color="#d6604d", lw=1.8, ls="--", marker="o", ms=4,
        label="A − θ  (reputable, noisy)")
ax.plot(ts, err_series("b"), color="#4393c3", lw=1.8, ls="--", marker="s", ms=4,
        label="B − θ  (new, accurate)")
ax.plot(ts, err_series("model_est"), color="#1a9850", lw=2.2, marker="D", ms=4.5,
        label="model − θ  (estimate)")
ax.set_xlabel("round")
ax.set_ylabel("signed error  (estimate − θ),  mean over 3 companies, seed 0")
ax.set_title("Qwen3-32B — signed error: model−θ vs A−θ vs B−θ  (gap=2.0, moderate rep, lean info, seed 0)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
dest = OUT / "errors_trace.png"
fig.savefig(dest, dpi=130, bbox_inches="tight")
print(f"wrote {dest}")
