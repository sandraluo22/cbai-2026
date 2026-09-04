"""Feature extraction shared by the emission and receiver model fits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..world.schema import World


def emission_citation_table(trials: pd.DataFrame, worlds: dict[str, World]) -> pd.DataFrame:
    """Event-level citation rows: for every accessible event, was it cited?"""
    rows = []
    for _, t in trials.iterrows():
        world = worlds[t["world_id"]]
        accessible = [rid for rid in str(t["accessible_report_ids"]).split(",") if rid]
        private = set(str(t["private_report_ids"]).split(","))
        cited_events = set(str(t["cited_event_ids"]).split(",")) - {""}
        by_event: dict[str, list[str]] = {}
        for rid in accessible:
            by_event.setdefault(world.report(rid).event_id, []).append(rid)
        for eid, rids in by_event.items():
            ev = world.event(eid)
            rows.append(
                {
                    "trial_id": t["trial_id"],
                    "usage": t["usage"],
                    "world_id": t["world_id"],
                    "event_id": eid,
                    "cited": int(eid in cited_events),
                    "event_llr": ev.llr,
                    "abs_event_llr": abs(ev.llr),
                    "alignment_with_belief": float(np.sign(ev.llr) == np.sign(t["ell_pre"]))
                    if t["ell_pre"] != 0
                    else 0.5,
                    "is_private": float(any(r in private for r in rids)),
                    "first_seen_round": 0 if any(r in private for r in rids) else 1,
                    "n_prior_mentions": len(rids) - 1,
                    "previously_cited": 0.0,
                    "source_family": ev.family,
                    "ell_pre": t["ell_pre"],
                }
            )
    return pd.DataFrame(rows)


RECEIVER_FEATURES = [
    "ell_pre",
    "m",
    "n_messages",
    "signed_message_count",
    "unique_new_llr",
    "repeated_llr_if_naively_counted",
    "repeated_report_count",
    "mean_public_stance",
    "confidence_weighted_stance",
    "agreement_with_prior",
    "cumulative_unique_event_count",
    "context_age",
]


def receiver_design(df: pd.DataFrame) -> pd.DataFrame:
    x = df[RECEIVER_FEATURES].copy().astype(float)
    x["conflict_magnitude"] = (np.sign(x["unique_new_llr"]) != np.sign(x["ell_pre"])).astype(
        float
    ) * x["unique_new_llr"].abs()
    x["prior_x_incoming"] = x["ell_pre"] * x["confidence_weighted_stance"]
    return x
