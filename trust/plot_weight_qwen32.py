"""Implied per-round weight on the accurate source B for the qwen32 run, anchor condition.

The MAE panel compresses the high-weight region (error ~ sqrt(variance) is flat near the
optimum), so equal-weighting looks like B-following. This plots the weight DIRECTLY:
per observation, w = (model - a) / (b - a) is the convex weight on B (linear in the weight,
full resolution across [0,1]). We show the per-round median + IQR band over all
companies x games (restricted to |b-a| >= 8 so the denominator is well-conditioned), with
the regression lambda_hat overlaid and reference lines for pure-A / equal / oracle / pure-B.
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
MIN_SPREAD = 8.0          # only use obs where |b-a| is large enough to read a weight

rows = [json.loads(l) for l in (OUT / "rounds.jsonl").read_text().splitlines()]
rows = [r for r in rows if r["condition"] == COND and not np.isnan(r["model_est"])]
T = max(r["t"] for r in rows) + 1

per_round = defaultdict(list)
for r in rows:
    d = r["b"] - r["a"]
    if abs(d) >= MIN_SPREAD:
        per_round[r["t"]].append((r["model_est"] - r["a"]) / d)

ts = np.arange(1, T + 1)
med = np.array([np.median(per_round[t]) if per_round[t] else np.nan for t in range(T)])
q25 = np.array([np.percentile(per_round[t], 25) if per_round[t] else np.nan for t in range(T)])
q75 = np.array([np.percentile(per_round[t], 75) if per_round[t] else np.nan for t in range(T)])

# regression lambda_hat (the analysis estimator) for comparison
res = json.load(open(OUT / "analysis.json"))
c = res["conditions"][COND]
lam = np.array(c["lambda_hat"], float)
oracle = c["oracle"]

fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.fill_between(ts, q25, q75, color="#1a9850", alpha=0.15, label="IQR of per-obs weight")
ax.plot(ts, med, "-D", ms=4.5, color="#1a9850", label="median implied weight on B  (model−a)/(b−a)")
ax.plot(ts, lam, "--o", ms=3.5, color="#dd8452", label="regression λ̂ₜ (analysis estimator)")
ax.axhline(1.0, color="#4393c3", ls=":", lw=1.2, label="pure B (weight 1)")
ax.axhline(oracle, color="#55a868", ls="--", lw=1.2, label=f"oracle / accuracy-justified ({oracle:.2f})")
ax.axhline(0.5, color="gray", ls="-", lw=0.8, label="equal blend (0.5)")
ax.axhline(0.0, color="#d6604d", ls=":", lw=1.2, label="pure A (weight 0)")
ax.set_xlabel("round")
ax.set_ylabel("weight on the ACCURATE source B")
ax.set_ylim(-0.15, 1.15)
ax.set_title("Qwen3-32B — implied weight on B per round  (gap=2.0, moderate rep, lean info)\n"
             "linear, full-resolution view — replaces the compressive |error| panel")
ax.legend(loc="center right", fontsize=8.5)
ax.grid(alpha=0.3)
dest = OUT / "weight_trace.png"
fig.savefig(dest, dpi=130, bbox_inches="tight")
print(f"wrote {dest}")
