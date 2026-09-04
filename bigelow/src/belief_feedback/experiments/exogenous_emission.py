"""Exogenous emission calibration for G (Part 6A).

Controlled single-agent contexts with 0-4 private reports and 0-3 rounds of
prerecorded neighbor memos; the agent's memo is generated and its features
extracted for fitting the emission model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..agents.memo_parser import parse_memo
from ..agents.prompts import MEMO_REQUEST, PROBE_CHOICES, probe_messages, system_prompt
from ..agents.roles import role_for_agent
from ..agents.transcript import Transcript
from ..config import Config, id_fields
from ..logging_utils import get_logger, now_iso, write_manifest
from ..models import make_backend
from ..models.base import Backend
from ..seeds import derive_seed
from ..seeds import rng as make_rng
from ..world.generator import load_worlds, worlds_in_split
from ..world.oracle import oracle_for_reports
from ..world.schema import World
from .common import outputs_exist, save_df
from .stimuli import make_stimulus_memo

log = get_logger(__name__)


def _emission_trial(
    cfg: Config, backend: Backend, world: World, trial_id: str, usage: str, r: np.random.Generator
) -> dict[str, Any]:
    backend.register_world(world)
    agent_id = int(r.integers(0, world.n_agents))
    role = role_for_agent(agent_id)
    all_rids = [rep.report_id for rep in world.reports]
    n_priv = int(r.integers(0, 5))
    priv = list(r.choice(all_rids, size=min(n_priv, len(all_rids)), replace=False))
    docs = "\n\n---\n\n".join(world.report(rid).text for rid in priv) or "(no private records)"

    tr = Transcript(
        system=system_prompt(world, agent_id, role),
        private_records=f"Your private records:\n\n{docs}",
    )
    # prerecorded neighbor memos: 0-3 rounds, mixture of new/repeated/conflicting
    n_hist_rounds = int(r.integers(0, 4))
    accessible = set(priv)
    n_received_reports = 0
    for rr in range(1, n_hist_rounds + 1):
        n_msgs = int(r.integers(1, 3))
        for _ in range(n_msgs):
            mode = r.choice(["new", "repeat", "conflict"])
            if mode == "repeat" and accessible:
                cited = [str(r.choice(sorted(accessible)))]
            else:
                fresh = [x for x in all_rids if x not in accessible]
                cited = [str(r.choice(fresh))] if fresh else []
            stance_pool = [-1, 1]
            stance = int(r.choice(stance_pool))
            if mode == "conflict" and priv:
                own = oracle_for_reports(world, priv).oracle_log_odds
                stance = -1 if own > 0 else 1
            memo = make_stimulus_memo(
                world,
                stance_semantic=stance,
                confidence_level=str(r.choice(["low", "medium", "high"])),
                cited_report_ids=cited,
                variant=int(r.integers(0, 4)),
            )
            sender = int((agent_id + 1 + r.integers(0, world.n_agents - 1)) % world.n_agents)
            tr.add_received(rr, sender, role_for_agent(sender), memo)
            accessible.update(cited)
            n_received_reports += len(cited)

    ctx = tr.context_messages()
    score = backend.score_choices(probe_messages(ctx), PROBE_CHOICES)
    ell_pre = world.visible_to_semantic(score.logps[0] - score.logps[1])

    seed = derive_seed("emission", trial_id)
    gen = backend.generate(ctx + [{"role": "user", "content": MEMO_REQUEST}], seed=seed)
    pm = parse_memo(gen.text, world, accessible_report_ids=accessible)
    cited_events = sorted({world.report(rid).event_id for rid in pm.valid_citations})
    accessible_events = {world.report(rid).event_id for rid in accessible}
    total_cited_llr = sum(world.event(eid).llr for eid in cited_events)
    priv_events = {world.report(rid).event_id for rid in priv}
    body_stance_words = ("ALPHA" in pm.memo_body, "BETA" in pm.memo_body)
    consistent = (
        pm.parsed_assessment in ("ALPHA", "BETA")
        and body_stance_words[0 if pm.parsed_assessment == "ALPHA" else 1]
    ) or pm.parsed_assessment == "UNRESOLVED"

    oracle = oracle_for_reports(world, accessible)
    return {
        **id_fields(cfg),
        "trial_id": trial_id,
        "usage": usage,
        "world_id": world.world_id,
        "split": world.split,
        "condition": "exogenous_emission",
        "branch": "exogenous",
        "branch_parent": None,
        "replicate_seed": 0,
        "agent_id": agent_id,
        "role": role,
        "round": n_hist_rounds,
        "seed": seed,
        "n_private_reports": len(priv),
        "n_history_rounds": n_hist_rounds,
        "n_received_reports": n_received_reports,
        "ell_pre": ell_pre,
        "oracle_log_odds": oracle.oracle_log_odds,
        "accessible_event_count": len(accessible_events),
        "public_stance": pm.semantic_stance(world),
        "parsed_confidence": pm.parsed_confidence,
        "format_valid": pm.format_valid,
        "cited_report_ids": ",".join(pm.cited_ids),
        "cited_event_ids": ",".join(cited_events),
        "n_cited": len(pm.valid_citations),
        "total_cited_llr": total_cited_llr,
        "n_new_events_cited": len([e for e in cited_events if e not in priv_events]),
        "n_repeated_events_cited": len([e for e in cited_events if e in priv_events]),
        "n_private_cited": len([rid for rid in pm.valid_citations if rid in priv]),
        "n_received_cited": len([rid for rid in pm.valid_citations if rid not in priv]),
        "invalid_citation_count": len(pm.invalid_citations) + len(pm.hallucinated_report_ids),
        "message_length": pm.word_count,
        "header_body_consistent": consistent,
        "raw_text": gen.text,
        # per-event citation table is derived in analysis.message_features
        "accessible_report_ids": ",".join(sorted(accessible)),
        "private_report_ids": ",".join(priv),
    }


def run(cfg: Config) -> pd.DataFrame:
    out = cfg.paths.runs / "exogenous_emission_trials.parquet"
    if outputs_exist([out]):
        log.info("emission trials exist; skipping")
        return pd.read_parquet(out)
    started = now_iso()
    backend = make_backend(cfg)
    worlds = load_worlds(cfg)
    plan = [
        ("train", "exogenous_train", cfg.exogenous.emission_train),
        ("validation", "exogenous_validation", cfg.exogenous.emission_validation),
        ("test", "exogenous_test", cfg.exogenous.emission_test),
    ]
    rows = []
    for usage, split, n in plan:
        split_worlds = worlds_in_split(worlds, split)
        for i in range(n):
            world = split_worlds[i % len(split_worlds)]
            r = make_rng("emission_trial", usage, i)
            rows.append(_emission_trial(cfg, backend, world, f"em_{usage}_{i:05d}", usage, r))
    df = pd.DataFrame(rows)
    save_df(df, out)
    write_manifest(
        cfg, "exogenous_emission", started=started, artifact_paths=[str(out)], completed_jobs=len(df)
    )
    return df
