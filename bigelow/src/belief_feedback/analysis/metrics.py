"""Episode-level outcomes and causal branch-effect decomposition."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..agents.protocol import EpisodeResult, graph_distance
from ..config import Config, id_fields
from ..world.oracle import network_oracle
from ..world.schema import UPSTREAM, World


def majority_threshold(n: int) -> int:
    return int(math.ceil(0.625 * n))  # 5 of 8


def strong_threshold(n: int) -> int:
    return int(math.ceil(0.875 * n))  # 7 of 8


def episode_metrics(cfg: Config, world: World, res: EpisodeResult, replicate_seed: int) -> dict[str, Any]:
    b = res.beliefs  # [T+1, n]
    n = b.shape[1]
    truth_up = world.true_hypothesis == UPSTREAM
    maj_k = majority_threshold(n)
    strong_k = strong_threshold(n)

    def majority_side(row: np.ndarray) -> int:
        n_up = int((row > 0).sum())
        if n_up >= maj_k:
            return 1
        if (n - n_up) >= maj_k:
            return -1
        return 0

    sides = [majority_side(b[t]) for t in range(b.shape[0])]
    t_first = next((t for t, s in enumerate(sides) if s != 0), -1)
    t_stable = -1
    for t in range(b.shape[0]):
        s = sides[t]
        if s != 0 and all(x == s for x in sides[t:]):
            t_stable = t
            break
    final = b[-1]
    n_up = int((final > 0).sum())
    final_side = majority_side(final)
    probs = 1.0 / (1.0 + np.exp(-final))
    truth01 = 1.0 if truth_up else 0.0
    parsed = list(res.parsed.values())
    n_msgs = max(len(parsed), 1)
    cited_events: list[str] = []
    for pm in parsed:
        for rid in pm.valid_citations:
            cited_events.append(world.report(rid).event_id)
    strong = (
        (n_up >= strong_k or (n - n_up) >= strong_k) and abs(float(final.mean())) >= 1.0
    )
    return {
        **id_fields(cfg),
        "run_id": res.run_id,
        "world_id": world.world_id,
        "split": world.split,
        "condition": res.branch,
        "branch": res.branch,
        "branch_parent": None,
        "replicate_seed": replicate_seed,
        "agent_id": -1,
        "round": b.shape[0] - 1,
        "seed": 0,
        "true_hypothesis": world.true_hypothesis,
        "network_oracle_log_odds": network_oracle(world).oracle_log_odds,
        "final_mean_ell": float(final.mean()),
        "final_variance": float(final.var()),
        "fraction_upstream": n_up / n,
        "majority_decision": final_side,
        "majority_accuracy": float(final_side == (1 if truth_up else -1)),
        "strong_consensus": bool(strong),
        "time_to_first_majority": t_first,
        "time_to_stable_majority": t_stable,
        "final_disagreement": float(final.std()),
        "final_brier": float(np.mean((probs - truth01) ** 2)),
        "final_calibration_error": float(abs(probs.mean() - truth01)),
        "unique_evidence_transmitted": len(set(cited_events)),
        "repeated_evidence_transmitted": len(cited_events) - len(set(cited_events)),
        "hallucinated_citation_rate": sum(len(p.hallucinated_report_ids) for p in parsed) / n_msgs,
        "malformed_rate": sum(0 if p.format_valid else 1 for p in parsed) / n_msgs,
    }


# ---- causal decomposition (Part 9) ---------------------------------------

DECOMP_PAIRS = {
    "one_hop_effect": ("{s}_one_hop", "baseline"),
    "forward_cascade_effect": ("{s}_no_return", "{s}_one_hop"),
    "reciprocal_feedback_effect": ("{s}_impulse", "{s}_no_return"),
    "total_closed_loop_effect": ("{s}_impulse", "baseline"),
    "text_mediated_effect": ("{s}_impulse", "{s}_full_text_clamp"),
}


def compute_branch_effects(cfg: Config, belief_states: pd.DataFrame, worlds: dict[str, World]) -> pd.DataFrame:
    """Paired per-(world, seed, agent, round) branch differences."""
    df = belief_states[belief_states["split"] == "endogenous_test"]
    key = ["world_id", "replicate_seed", "agent_id", "round"]
    piv = df.pivot_table(index=key, columns="branch", values="semantic_log_odds").reset_index()
    rows: list[dict[str, Any]] = []
    topo = cfg.network.topology
    for _, r in piv.iterrows():
        world = worlds[r["world_id"]]
        dist = graph_distance(topo, world.n_agents, 0, int(r["agent_id"]))
        oracle0 = network_oracle(world).oracle_log_odds
        stratum = "neg" if oracle0 < -1 else ("pos" if oracle0 > 1 else "neutral")
        for sign in ("positive", "negative"):
            for name, (a_t, b_t) in DECOMP_PAIRS.items():
                a = a_t.format(s=sign)
                b = b_t.format(s=sign)
                a = "fixed_replay_" + sign if a == f"{sign}_fixed_replay" else a
                if a not in piv.columns or b not in piv.columns:
                    continue
                if pd.isna(r.get(a)) or pd.isna(r.get(b)):
                    continue
                rows.append(
                    {
                        **id_fields(cfg),
                        "world_id": r["world_id"],
                        "replicate_seed": r["replicate_seed"],
                        "agent_id": int(r["agent_id"]),
                        "round": int(r["round"]),
                        "seed": 0,
                        "condition": name,
                        "branch": a,
                        "branch_parent": b,
                        "sign": sign,
                        "graph_distance": dist,
                        "evidence_stratum": stratum,
                        "effect": float(r[a] - r[b]),
                    }
                )
    return pd.DataFrame(rows)


def amplification_ratios(cfg: Config, belief_states: pd.DataFrame, eps: float = 1e-3) -> pd.DataFrame:
    """Sum_i |delta ell_i,T| / |delta ell_source at intervention round|."""
    df = belief_states[belief_states["split"] == "endogenous_test"]
    key = ["world_id", "replicate_seed"]
    rows = []
    excluded = 0
    for (wid, rep), grp in df.groupby(key):
        piv = grp.pivot_table(index=["agent_id", "round"], columns="branch", values="semantic_log_odds")
        for sign in ("positive", "negative"):
            cond = f"{sign}_impulse"
            if cond not in piv.columns or "baseline" not in piv.columns:
                continue
            diff = (piv[cond] - piv["baseline"]).unstack(level="round")
            t_final = diff.columns.max()
            src = abs(diff.loc[0, 1]) if (0 in diff.index and 1 in diff.columns) else np.nan
            if not np.isfinite(src) or src < eps:
                excluded += 1
                continue
            total = float(diff[t_final].abs().sum())
            rows.append(
                {
                    **id_fields(cfg),
                    "world_id": wid,
                    "replicate_seed": rep,
                    "agent_id": -1,
                    "round": int(t_final),
                    "seed": 0,
                    "condition": cond,
                    "branch": cond,
                    "branch_parent": "baseline",
                    "source_effect": float(src),
                    "total_final_effect": total,
                    "amplification_ratio": total / float(src),
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["excluded_near_zero"] = excluded
    return out
