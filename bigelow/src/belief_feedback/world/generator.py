"""World generation: latent events, reports, assignments, and splits.

Ordinary worlds follow the symmetric generative model with rejection
sampling on evidence-balance constraints. Recycling worlds come in matched
independent/recycled pairs differing only in provenance. Phase-boundary
worlds are evidence-stratified to hit target network-level oracle log odds.

Truth and the ALPHA/BETA mapping are counterbalanced deterministically by
world index within each split (exact balance up to one world).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..config import TEST_SPLITS, Config
from ..logging_utils import get_logger, now_iso, write_manifest
from ..seeds import derive_seed
from ..seeds import rng as make_rng
from .documents import draw_lineage, render_report
from .oracle import agent_initial_oracle, network_oracle
from .schema import LOCAL, UPSTREAM, Event, Report, World
from .templates import FAMILY_TEMPLATES, FIRST_NAMES, HELD_OUT_VARIANT, LAST_NAMES

log = get_logger(__name__)

FAMILIES = list(FAMILY_TEMPLATES)
TRAIN_VARIANTS = [0, 1, 2, 3]


def _pick_variant(split: str, r: np.random.Generator) -> int:
    if split in TEST_SPLITS:
        return HELD_OUT_VARIANT
    return int(r.choice(TRAIN_VARIANTS))


def _agent_names(world_id: str) -> dict[int, str]:
    r = make_rng("agent_names", world_id)
    names = {}
    used: set[str] = set()
    for i in range(12):
        while True:
            n = f"{r.choice(FIRST_NAMES)} {r.choice(LAST_NAMES)}"
            if n not in used:
                used.add(n)
                names[i] = n
                break
    return names


def _make_event(world_id: str, idx: int, family: str, orientation: int, r: np.random.Generator) -> Event:
    return Event(
        event_id=f"E-{world_id}-{idx:03d}",
        world_id=world_id,
        family=family,
        reliability=FAMILY_TEMPLATES[family]["reliability"],
        orientation=orientation,
        lineage=draw_lineage(r),
    )


def _sample_orientation(truth: str, reliability: float, r: np.random.Generator) -> int:
    """P(s=+1 | UPSTREAM) = r ; P(s=-1 | LOCAL) = r (symmetric)."""
    aligned = r.random() < reliability
    truth_sign = 1 if truth == UPSTREAM else -1
    return truth_sign if aligned else -truth_sign


def build_ordinary_world(
    cfg: Config,
    world_id: str,
    split: str,
    idx: int,
    *,
    n_agents: int | None = None,
) -> World:
    """Build one ordinary world with rejection sampling on balance constraints."""
    wc = cfg.worlds
    n_agents = n_agents or cfg.network.n_agents
    n_reports = n_agents * wc.reports_per_agent
    n_secondary = n_agents // 2
    n_events = n_reports - n_secondary
    truth = UPSTREAM if idx % 2 == 0 else LOCAL
    alpha_is_upstream = idx % 4 in (0, 3)  # balanced for any even n, crossed with truth
    min_uncertain = min(wc.min_uncertain_agents, n_agents // 2)

    for attempt in range(10_000):
        r = make_rng("world", world_id, attempt)
        fams = list(r.choice(FAMILIES, size=n_events, replace=n_events > len(FAMILIES)))
        events = []
        for i, fam in enumerate(fams):
            orientation = _sample_orientation(truth, FAMILY_TEMPLATES[fam]["reliability"], r)
            events.append(_make_event(world_id, i, str(fam), orientation, r))

        world = World(
            world_id=world_id,
            split=split,
            true_hypothesis=truth,
            alpha_is_upstream=alpha_is_upstream,
            n_agents=n_agents,
            events=events,
            agent_names=_agent_names(world_id),
        )
        _attach_reports(world, split, n_secondary, r)
        if not _assign_reports(world, r):
            continue
        if _passes_constraints(world, wc, min_uncertain):
            return world
    raise RuntimeError(f"Rejection sampling failed for {world_id}")


def _attach_reports(world: World, split: str, n_secondary: int, r: np.random.Generator) -> None:
    reports: list[Report] = []
    for i, ev in enumerate(world.events):
        rid = f"R-{world.world_id}-{i:03d}"
        reports.append(
            render_report(rid, ev, template_variant=_pick_variant(split, r), is_secondary=False)
        )
    sec_events = r.choice(len(world.events), size=min(n_secondary, len(world.events)), replace=False)
    for j, ev_idx in enumerate(sec_events):
        ev = world.events[int(ev_idx)]
        primary = reports[int(ev_idx)]
        rid = f"R-{world.world_id}-S{j:02d}"
        variant = _pick_variant(split, r)
        wrapper = HELD_OUT_VARIANT if split in TEST_SPLITS else int(r.choice(TRAIN_VARIANTS))
        reports.append(
            render_report(
                rid,
                ev,
                template_variant=variant,
                is_secondary=True,
                source_report_id=primary.report_id,
                wrapper_variant=wrapper,
            )
        )
    world.reports = reports


def _assign_reports(world: World, r: np.random.Generator) -> bool:
    """Assign reports to agents; no agent may hold two reports of one event."""
    per_agent = len(world.reports) // world.n_agents
    order = list(r.permutation(len(world.reports)))
    assignments: dict[int, list[str]] = {a: [] for a in range(world.n_agents)}
    events_held: dict[int, set[str]] = {a: set() for a in range(world.n_agents)}
    for ri in order:
        rep = world.reports[int(ri)]
        placed = False
        agent_order = list(r.permutation(world.n_agents))
        agent_order.sort(key=lambda a: len(assignments[a]))
        for a in agent_order:
            if len(assignments[a]) < per_agent and rep.event_id not in events_held[a]:
                assignments[a].append(rep.report_id)
                events_held[a].add(rep.event_id)
                placed = True
                break
        if not placed:
            return False
    world.assignments = assignments
    return True


def _passes_constraints(world: World, wc: Any, min_uncertain: int) -> bool:
    net = network_oracle(world)
    if abs(net.oracle_log_odds) > wc.max_network_abs_log_odds:
        return False
    total_abs = sum(abs(e.llr) for e in world.events) or 1e-9
    n_uncertain = 0
    for a in range(world.n_agents):
        res = agent_initial_oracle(world, a)
        if abs(res.oracle_log_odds) > wc.max_agent_abs_log_odds:
            return False
        agent_abs = sum(
            abs(world.event(eid).llr) for eid in res.event_ids_seen
        )
        if agent_abs / total_abs > wc.max_agent_evidence_share:
            return False
        if abs(res.oracle_log_odds) < 1.5:
            n_uncertain += 1
    return n_uncertain >= min_uncertain


# --- recycling pairs -------------------------------------------------------


def build_recycling_pair(cfg: Config, pair_idx: int) -> list[World]:
    """Matched independent/recycled worlds differing only in provenance.

    Three focal agents each receive one focal report of the same orientation.
    Independent: three distinct events with distinct lineages, same family.
    Recycled: one event; one primary and two secondary reports sharing the
    event id and visible lineage.
    """
    split = "recycling_test"
    n_agents = cfg.network.n_agents
    focal_orientation = 1 if pair_idx % 2 == 0 else -1
    truth = UPSTREAM if pair_idx % 2 == 0 else LOCAL
    # both worlds in a pair share the mapping; balance across pairs
    alpha_is_upstream = pair_idx % 2 == 1
    focal_family = "plc_cycle_log"  # mid-reliability family, both orientations available
    worlds = []
    for role in ("independent", "recycled"):
        world_id = f"w_recycling_{pair_idx:04d}_{role}"
        r = make_rng("recycling", pair_idx, role)
        events: list[Event] = []
        reports: list[Report] = []
        n_focal_agents = min(3, n_agents)
        if role == "independent":
            for i in range(n_focal_agents):
                ev = _make_event(world_id, i, focal_family, focal_orientation, r)
                events.append(ev)
                reports.append(
                    render_report(
                        f"R-{world_id}-F{i:02d}",
                        ev,
                        template_variant=HELD_OUT_VARIANT,
                        is_secondary=False,
                    )
                )
        else:
            ev = _make_event(world_id, 0, focal_family, focal_orientation, r)
            events.append(ev)
            primary = render_report(
                f"R-{world_id}-F00", ev, template_variant=HELD_OUT_VARIANT, is_secondary=False
            )
            reports.append(primary)
            for i in range(1, n_focal_agents):
                reports.append(
                    render_report(
                        f"R-{world_id}-F{i:02d}",
                        ev,
                        template_variant=HELD_OUT_VARIANT,
                        is_secondary=True,
                        source_report_id=primary.report_id,
                        wrapper_variant=HELD_OUT_VARIANT,
                    )
                )
        # filler: one weak event per remaining agent slot, plus one per focal agent
        filler_needed = n_agents * cfg.worlds.reports_per_agent - len(reports)
        for i in range(filler_needed):
            fam = str(r.choice(["witness_interview", "shipping_environment_log", "operator_handover_note"]))
            orientation = _sample_orientation(truth, FAMILY_TEMPLATES[fam]["reliability"], r)
            ev = _make_event(world_id, 100 + i, fam, orientation, r)
            events.append(ev)
            reports.append(
                render_report(
                    f"R-{world_id}-X{i:02d}", ev, template_variant=HELD_OUT_VARIANT, is_secondary=False
                )
            )
        world = World(
            world_id=world_id,
            split=split,
            true_hypothesis=truth,
            alpha_is_upstream=alpha_is_upstream,
            n_agents=n_agents,
            events=events,
            reports=reports,
            agent_names=_agent_names(world_id),
            tags={"recycling_role": role, "pair_id": str(pair_idx)},
        )
        # deterministic assignment: focal agents 0..2 get focal report first
        assignments: dict[int, list[str]] = {a: [] for a in range(n_agents)}
        fillers = [rep for rep in reports if rep.report_id.split("-")[-1].startswith("X")]
        focals = [rep for rep in reports if rep.report_id.split("-")[-1].startswith("F")]
        for i, rep in enumerate(focals):
            assignments[i].append(rep.report_id)
        fi = 0
        for a in range(n_agents):
            while len(assignments[a]) < cfg.worlds.reports_per_agent and fi < len(fillers):
                assignments[a].append(fillers[fi].report_id)
                fi += 1
        world.assignments = assignments
        worlds.append(world)
    return worlds


# --- phase-boundary worlds -------------------------------------------------


def build_phase_world(cfg: Config, bin_center: float, idx: int, global_idx: int) -> World:
    """Evidence-stratified world targeting a network oracle log-odds bin."""
    split = "phase_boundary_test"
    world_id = f"w_phase_{bin_center:+.0f}_{idx:04d}"
    truth = UPSTREAM if global_idx % 2 == 0 else LOCAL
    alpha_is_upstream = global_idx % 4 in (0, 3)
    n_agents = cfg.network.n_agents
    n_reports = n_agents * cfg.worlds.reports_per_agent
    n_secondary = n_agents // 2
    n_events = n_reports - n_secondary
    for attempt in range(10_000):
        r = make_rng("phase_world", world_id, attempt)
        fams = list(r.choice(FAMILIES, size=n_events, replace=n_events > len(FAMILIES)))
        events = []
        for i, fam in enumerate(fams):
            orientation = _sample_orientation(truth, FAMILY_TEMPLATES[fam]["reliability"], r)
            events.append(_make_event(world_id, i, str(fam), orientation, r))
        # greedy flips toward the target total oracle log odds
        for _ in range(4 * n_events):
            total = sum(e.llr for e in events)
            if abs(total - bin_center) <= 0.75:
                break
            need_up = total < bin_center
            candidates = [e for e in events if (e.orientation < 0) == need_up]
            if not candidates:
                break
            ev = candidates[int(r.integers(0, len(candidates)))]
            ev.orientation *= -1
        world = World(
            world_id=world_id,
            split=split,
            true_hypothesis=truth,
            alpha_is_upstream=alpha_is_upstream,
            n_agents=n_agents,
            events=events,
            agent_names=_agent_names(world_id),
            tags={"phase_bin": f"{bin_center:+.1f}"},
        )
        _attach_reports(world, split, n_secondary, r)
        if not _assign_reports(world, r):
            continue
        if abs(network_oracle(world).oracle_log_odds - bin_center) <= 1.0:
            return world
    raise RuntimeError(f"Phase-world sampling failed for {world_id}")


# --- top-level generation and persistence ---------------------------------


def generate_all_worlds(cfg: Config) -> dict[str, World]:
    started = now_iso()
    worlds: dict[str, World] = {}
    for split, n in cfg.worlds.splits.items():
        if split == "recycling_test":
            for pair in range(max(n // 2, 0)):
                for w in build_recycling_pair(cfg, pair):
                    worlds[w.world_id] = w
            continue
        if split == "phase_boundary_test":
            continue  # handled below via bins
        for i in range(n):
            wid = f"w_{split}_{i:04d}"
            worlds[wid] = build_ordinary_world(cfg, wid, split, i)
    phase_counter = 0
    for b in cfg.worlds.phase_bins:
        for i in range(cfg.worlds.phase_worlds_per_cell):
            w = build_phase_world(cfg, b, i, phase_counter)
            phase_counter += 1
            worlds[w.world_id] = w
    save_worlds(cfg, worlds)
    write_manifest(
        cfg,
        "generate_worlds",
        started=started,
        artifact_paths=[str(cfg.paths.data)],
        completed_jobs=len(worlds),
        extra={"dataset_hash": dataset_hash(worlds)},
    )
    log.info("generated %d worlds", len(worlds))
    return worlds


def dataset_hash(worlds: dict[str, World]) -> str:
    parts = sorted(f"{w.world_id}:{len(w.reports)}:{w.true_hypothesis}" for w in worlds.values())
    return f"{derive_seed('dataset', *parts):08x}"


def save_worlds(cfg: Config, worlds: dict[str, World]) -> None:
    data_dir = cfg.paths.data
    with open(data_dir / "worlds_full.jsonl", "w") as f:
        for w in worlds.values():
            f.write(json.dumps(w.model_dump(mode="json")) + "\n")

    world_rows, event_rows, report_rows, assign_rows = [], [], [], []
    splits: dict[str, list[str]] = {}
    for w in worlds.values():
        splits.setdefault(w.split, []).append(w.world_id)
        net = network_oracle(w)
        world_rows.append(
            {
                "world_id": w.world_id,
                "split": w.split,
                "true_hypothesis": w.true_hypothesis,
                "alpha_is_upstream": w.alpha_is_upstream,
                "n_agents": w.n_agents,
                "n_events": len(w.events),
                "n_reports": len(w.reports),
                "network_oracle_log_odds": net.oracle_log_odds,
                "tags": json.dumps(w.tags),
            }
        )
        for e in w.events:
            event_rows.append(
                {
                    "world_id": w.world_id,
                    "event_id": e.event_id,
                    "family": e.family,
                    "reliability": e.reliability,
                    "orientation": e.orientation,
                    "llr": e.llr,
                    **{f"lineage_{k}": v for k, v in e.lineage.items()},
                }
            )
        doc_dir = cfg.paths.rendered_documents / w.world_id
        doc_dir.mkdir(exist_ok=True)
        for rep in w.reports:
            report_rows.append(
                {
                    "world_id": w.world_id,
                    "report_id": rep.report_id,
                    "event_id": rep.event_id,
                    "family": rep.family,
                    "orientation": rep.orientation,
                    "is_secondary": rep.is_secondary,
                    "source_report_id": rep.source_report_id,
                    "template_variant": rep.template_variant,
                    "word_count": rep.word_count,
                }
            )
            (doc_dir / f"{rep.report_id}.txt").write_text(rep.text)
        for agent_id, rids in w.assignments.items():
            for order, rid in enumerate(rids):
                assign_rows.append(
                    {
                        "world_id": w.world_id,
                        "agent_id": agent_id,
                        "report_id": rid,
                        "order": order,
                    }
                )

    pd.DataFrame(world_rows).to_parquet(data_dir / "worlds.parquet", index=False)
    pd.DataFrame(event_rows).to_parquet(data_dir / "events.parquet", index=False)
    pd.DataFrame(report_rows).to_parquet(data_dir / "reports.parquet", index=False)
    pd.DataFrame(assign_rows).to_parquet(data_dir / "agent_assignments.parquet", index=False)
    (data_dir / "splits.json").write_text(json.dumps(splits, indent=2))


def load_worlds(cfg: Config) -> dict[str, World]:
    path = cfg.paths.data / "worlds_full.jsonl"
    worlds: dict[str, World] = {}
    with open(path) as f:
        for line in f:
            w = World.model_validate(json.loads(line))
            worlds[w.world_id] = w
    return worlds


def worlds_in_split(worlds: dict[str, World], split: str) -> list[World]:
    return sorted((w for w in worlds.values() if w.split == split), key=lambda w: w.world_id)
