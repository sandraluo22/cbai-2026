"""World-clustered percentile bootstrap and paired permutation tests.

Agents within a world are never treated as independent: resampling draws
whole worlds (retaining all agents, rounds, branches, and seeds), and paired
tests operate on world-level paired differences.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..seeds import rng as make_rng


@dataclass
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_worlds: int
    n_resamples: int
    p_value: float | None = None
    standardized_effect: float | None = None


def cluster_bootstrap(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_resamples: int = 1000,
    world_col: str = "world_id",
    seed_parts: tuple = ("bootstrap",),
) -> BootstrapResult:
    worlds = sorted(df[world_col].unique())
    est = float(statistic(df))
    if len(worlds) < 2:
        return BootstrapResult(est, est, est, len(worlds), 0)
    r = make_rng(*seed_parts)
    groups = {w: g for w, g in df.groupby(world_col)}
    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        chosen = r.choice(worlds, size=len(worlds), replace=True)
        sample = pd.concat([groups[w] for w in chosen], ignore_index=True)
        stats[b] = statistic(sample)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return BootstrapResult(est, float(lo), float(hi), len(worlds), n_resamples)


def paired_world_test(
    values_by_world: pd.Series,
    n_resamples: int = 1000,
    seed_parts: tuple = ("paired",),
) -> BootstrapResult:
    """Paired analysis of world-level effects: bootstrap CI, sign-flip p, d.

    ``values_by_world``: one paired difference per world (already aggregated
    within world).
    """
    x = values_by_world.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    est = float(np.mean(x)) if n else float("nan")
    if n < 2:
        return BootstrapResult(est, est, est, n, 0, p_value=None, standardized_effect=None)
    r = make_rng(*seed_parts)
    idx = np.arange(n)
    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        boots[b] = x[r.choice(idx, size=n, replace=True)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided sign-flip permutation p-value
    flips = np.empty(n_resamples)
    for b in range(n_resamples):
        signs = r.choice([-1.0, 1.0], size=n)
        flips[b] = (x * signs).mean()
    p = float((np.sum(np.abs(flips) >= abs(est)) + 1) / (n_resamples + 1))
    sd = float(np.std(x, ddof=1))
    d = est / sd if sd > 0 else float("nan")
    return BootstrapResult(est, float(lo), float(hi), n, n_resamples, p_value=p, standardized_effect=d)
