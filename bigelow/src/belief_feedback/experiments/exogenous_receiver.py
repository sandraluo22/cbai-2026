"""Exogenous receiver calibration for F (Part 6B).

Balanced fractional-factorial receiver trials: measured pre-belief,
controlled incoming memos (count, unique LLR, repeated-source mentions,
confidence, provenance, role attribution), optional receiver steering, and
measured post-belief.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from ..agents.prompts import PROBE_CHOICES, probe_messages, system_prompt
from ..agents.roles import role_for_agent
from ..agents.transcript import Transcript
from ..config import Config, id_fields
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models import make_backend
from ..models.base import Backend
from ..seeds import rng as make_rng
from ..world.generator import load_worlds, worlds_in_split
from ..world.oracle import oracle_for_reports
from ..world.schema import World
from .common import delta_from_meta, load_steering, outputs_exist, save_df
from .stimuli import CONF_LEVELS, make_stimulus_memo

log = get_logger(__name__)

PRE_BINS = [-3.0, -1.0, 0.0, 1.0, 3.0]
LLR_BINS = [-3.0, -1.0, 0.0, 1.0, 3.0]
REPEAT_MENTIONS = [0, 1, 2, 4]
MSG_COUNTS = [1, 2, 4]
STEER_FRACS = [-1.0, 0.0, 1.0]  # multiples of delta = 0.5*m_max
CONFS = ["low", "medium", "high"]

STIMULUS_CLASSES = [
    "independent_corroboration",
    "repeated_source",
    "exact_repetition",
    "conflicting_independent",
    "confident_weak",
    "uncertain_strong",
    "majority_weak_vs_minority_strong",
    "aligned_with_prior",
    "opposing_prior",
    "role_attribution",
]


def _subset_near(world: World, target: float, r: np.random.Generator, max_reports: int = 3) -> list[str]:
    """Greedy subset of reports whose oracle log odds approach a target."""
    rids = [rep.report_id for rep in world.reports if not rep.is_secondary]
    chosen: list[str] = []
    current = 0.0
    for _ in range(max_reports):
        best, best_gap = None, abs(current - target)
        for rid in rids:
            if rid in chosen:
                continue
            cand = oracle_for_reports(world, chosen + [rid]).oracle_log_odds
            if abs(cand - target) < best_gap - 1e-9:
                best, best_gap = rid, abs(cand - target)
        if best is None:
            break
        chosen.append(best)
        current = oracle_for_reports(world, chosen).oracle_log_odds
    return chosen


def _events_near(world: World, exclude: set[str], target: float, n: int) -> list[str]:
    """Pick up to n distinct primary reports approximating a target sum LLR."""
    pool = [rep for rep in world.reports if not rep.is_secondary and rep.report_id not in exclude]
    pool.sort(key=lambda rep: abs(world.event(rep.event_id).llr - target / max(n, 1)))
    seen_events: set[str] = set()
    out = []
    for rep in pool:
        if rep.event_id in seen_events:
            continue
        out.append(rep.report_id)
        seen_events.add(rep.event_id)
        if len(out) >= n:
            break
    return out


def _receiver_trial(
    cfg: Config,
    backend: Backend,
    world: World,
    trial_id: str,
    usage: str,
    cell: dict[str, Any],
    steer_ctx,
    delta: float,
    r: np.random.Generator,
) -> dict[str, Any]:
    backend.register_world(world)
    agent_id = 0
    role = role_for_agent(agent_id)
    priv = _subset_near(world, cell["pre_bin"], r)
    docs = "\n\n---\n\n".join(world.report(rid).text for rid in priv) or "(no private records)"
    tr = Transcript(
        system=system_prompt(world, agent_id, role),
        private_records=f"Your private records:\n\n{docs}",
    )
    ctx = tr.context_messages()
    steering = None
    if cell["steer_frac"] != 0.0 and steer_ctx is not None:
        steering = steer_ctx.spec(cell["steer_frac"] * delta)
    pre = backend.score_choices(probe_messages(ctx), PROBE_CHOICES, steering=steering)
    ell_pre = world.visible_to_semantic(pre.logps[0] - pre.logps[1])

    # ---- construct incoming messages -------------------------------------
    n_msgs = cell["n_msgs"]
    stance = 0 if cell["llr_bin"] == 0 else (1 if cell["llr_bin"] > 0 else -1)
    if cell["stim_class"] == "aligned_with_prior" and ell_pre != 0:
        stance = 1 if ell_pre > 0 else -1
    if cell["stim_class"] == "opposing_prior" and ell_pre != 0:
        stance = -1 if ell_pre > 0 else 1
    exclude = set(priv)
    unique_rids = _events_near(world, exclude, cell["llr_bin"], max(1, n_msgs))
    repeat_rid = unique_rids[0] if unique_rids else None

    memos: list[tuple[int, str]] = []  # (sender, text)
    n_repeat_mentions = cell["repeat_mentions"]
    exact_text: str | None = None
    for k in range(n_msgs):
        conf = cell["confidence"]
        if cell["stim_class"] == "majority_weak_vs_minority_strong":
            conf = "low" if k < n_msgs - 1 else "high"
        if n_repeat_mentions > 0 and repeat_rid is not None and k < n_repeat_mentions:
            cited = [repeat_rid]
        else:
            cited = [unique_rids[k % len(unique_rids)]] if unique_rids else []
        variant = 0 if cell["stim_class"] == "exact_repetition" else k
        text = make_stimulus_memo(
            world,
            stance_semantic=stance,
            confidence_level=conf,
            cited_report_ids=cited,
            variant=variant,
            exact_copy_of=exact_text if cell["stim_class"] == "exact_repetition" and exact_text else None,
        )
        if cell["stim_class"] == "exact_repetition" and exact_text is None:
            exact_text = text
        sender = 1 + (k % (world.n_agents - 1))
        if cell["stim_class"] == "role_attribution":
            sender = 1 + ((k + int(r.integers(0, world.n_agents - 1))) % (world.n_agents - 1))
        memos.append((sender, text))

    for sender, text in memos:
        tr.add_received(1, sender, role_for_agent(sender), text)
    ctx_post = tr.context_messages()
    post = backend.score_choices(probe_messages(ctx_post), PROBE_CHOICES, steering=steering)
    ell_post = world.visible_to_semantic(post.logps[0] - post.logps[1])

    # ---- normative features ----------------------------------------------
    cited_all = [rid for _, text in memos for rid in [c for c in text.splitlines()[2].split(": ")[-1].split(", ") if c.startswith("R-")]]
    cited_events = [world.report(rid).event_id for rid in cited_all if any(rep.report_id == rid for rep in world.reports)]
    priv_events = {world.report(rid).event_id for rid in priv}
    new_events = [e for e in dict.fromkeys(cited_events) if e not in priv_events]
    unique_new_llr = sum(world.event(e).llr for e in new_events)
    naive_llr = sum(world.event(e).llr for e in cited_events if e not in priv_events)
    repeated_report_count = len(cited_events) - len(set(cited_events))
    conf_val = CONF_LEVELS[cell["confidence"]] / 100.0
    signed_count = stance * n_msgs

    return {
        **id_fields(cfg),
        "trial_id": trial_id,
        "usage": usage,
        "world_id": world.world_id,
        "split": world.split,
        "condition": "exogenous_receiver",
        "branch": "exogenous",
        "branch_parent": None,
        "replicate_seed": 0,
        "agent_id": agent_id,
        "role": role,
        "round": 1,
        "seed": 0,
        "stimulus_class": cell["stim_class"],
        "pre_bin": cell["pre_bin"],
        "llr_bin": cell["llr_bin"],
        "repeat_mentions_nominal": cell["repeat_mentions"],
        "steer_frac": cell["steer_frac"],
        "confidence_level": cell["confidence"],
        "ell_pre": ell_pre,
        "ell_post": ell_post,
        "delta_ell": ell_post - ell_pre,
        "m": cell["steer_frac"] * delta,
        "n_messages": n_msgs,
        "signed_message_count": signed_count,
        "unique_new_llr": unique_new_llr,
        "repeated_llr_if_naively_counted": naive_llr,
        "repeated_report_count": repeated_report_count,
        "mean_public_stance": float(stance),
        "confidence_weighted_stance": float(stance) * conf_val,
        "agreement_with_prior": float(np.sign(ell_pre) == np.sign(stance)) if stance != 0 else 0.5,
        "cumulative_unique_event_count": len(priv_events) + len(new_events),
        "context_age": 1,
        "sender_roles": ",".join(role_for_agent(s) for s, _ in memos),
    }


def run(cfg: Config) -> pd.DataFrame:
    out = cfg.paths.runs / "exogenous_receiver_trials.parquet"
    if outputs_exist([out]):
        log.info("receiver trials exist; skipping")
        return pd.read_parquet(out)
    started = now_iso()
    backend = make_backend(cfg)
    worlds = load_worlds(cfg)
    steer_ctx, meta = load_steering(cfg)
    delta = delta_from_meta(meta)

    cells = [
        {"pre_bin": p, "llr_bin": v, "repeat_mentions": rep, "n_msgs": nm, "steer_frac": s, "confidence": c, "stim_class": sc}
        for (p, v, rep, nm, s, c), sc in zip(
            product(PRE_BINS, LLR_BINS, REPEAT_MENTIONS, MSG_COUNTS, STEER_FRACS, CONFS),
            _cycle_classes(),
        )
        if rep <= nm  # can't mention one source more often than there are messages, except 0
    ]
    plan = [
        ("train", "exogenous_train", cfg.exogenous.receiver_train),
        ("validation", "exogenous_validation", cfg.exogenous.receiver_validation),
        ("test", "exogenous_test", cfg.exogenous.receiver_test),
    ]
    rows = []
    for usage, split, n in plan:
        split_worlds = worlds_in_split(worlds, split)
        for i in range(n):
            cell = cells[(i * 7919) % len(cells)]  # balanced coverage via coprime stride
            world = split_worlds[i % len(split_worlds)]
            r = make_rng("receiver_trial", usage, i)
            rows.append(
                _receiver_trial(cfg, backend, world, f"rc_{usage}_{i:05d}", usage, cell, steer_ctx, delta, r)
            )
    df = pd.DataFrame(rows)
    save_df(df, out)
    write_manifest(
        cfg, "exogenous_receiver", started=started, artifact_paths=[str(out)], completed_jobs=len(df)
    )
    return df


def _cycle_classes():
    while True:
        yield from STIMULUS_CLASSES
