"""Dataset validation: leakage, balance, constraints, and word counts."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from ..config import TEST_SPLITS, Config
from ..logging_utils import get_logger
from .documents import MAX_WORDS, MIN_WORDS
from .generator import load_worlds
from .oracle import agent_initial_oracle, network_oracle
from .schema import UPSTREAM, World
from .templates import HELD_OUT_VARIANT

log = get_logger(__name__)

FORBIDDEN_PATTERNS = [
    r"UPSTREAM_CONTAMINATION",
    r"LOCAL_CALIBRATION_DRIFT",
    r"\bE-w_",  # hidden event ids
    r"reliability\s*[:=]?\s*0\.\d",
    r"log[- ]likelihood",
    r"\bLLR\b",
]


def validate_dataset(cfg: Config) -> dict[str, Any]:
    worlds = load_worlds(cfg)
    checks: dict[str, Any] = {}
    problems: list[str] = []

    # 1. Word counts and forbidden content
    bad_words, leaks = [], []
    for w in worlds.values():
        for rep in w.reports:
            if not (MIN_WORDS <= rep.word_count <= MAX_WORDS):
                bad_words.append((rep.report_id, rep.word_count))
            for pat in FORBIDDEN_PATTERNS:
                if re.search(pat, rep.text):
                    leaks.append((rep.report_id, pat))
    checks["word_count_violations"] = bad_words
    checks["hidden_info_leaks"] = leaks
    if bad_words:
        problems.append(f"{len(bad_words)} documents outside {MIN_WORDS}-{MAX_WORDS} words")
    if leaks:
        problems.append(f"{len(leaks)} documents leak hidden information")

    # 2. Split disjointness
    split_of: dict[str, str] = {w.world_id: w.split for w in worlds.values()}
    checks["n_worlds"] = len(split_of)
    checks["splits"] = dict(Counter(split_of.values()))

    # 3. Balance per split (truth and label mapping, within one world)
    balance: dict[str, Any] = {}
    for split in {w.split for w in worlds.values()}:
        ws = [w for w in worlds.values() if w.split == split]
        if split == "recycling_test":
            # worlds within a pair intentionally share truth and mapping;
            # counterbalancing is enforced across pairs
            by_pair: dict[str, World] = {}
            for w in ws:
                by_pair.setdefault(w.tags.get("pair_id", w.world_id), w)
            ws = list(by_pair.values())
        n_up = sum(w.true_hypothesis == UPSTREAM for w in ws)
        n_alpha_up = sum(w.alpha_is_upstream for w in ws)
        balance[split] = {"n": len(ws), "truth_upstream": n_up, "alpha_is_upstream": n_alpha_up}
        if abs(2 * n_up - len(ws)) > 1:
            problems.append(f"truth imbalance in {split}: {n_up}/{len(ws)}")
        if abs(2 * n_alpha_up - len(ws)) > 1:
            problems.append(f"label-mapping imbalance in {split}: {n_alpha_up}/{len(ws)}")
    checks["balance"] = balance

    # 4. Template holdout: test splits use only held-out variants; train never do
    variant_violations = []
    for w in worlds.values():
        for rep in w.reports:
            if w.split in TEST_SPLITS and rep.template_variant != HELD_OUT_VARIANT:
                variant_violations.append((w.world_id, rep.report_id, "train-variant-in-test"))
            if w.split not in TEST_SPLITS and rep.template_variant == HELD_OUT_VARIANT:
                variant_violations.append((w.world_id, rep.report_id, "heldout-variant-in-train"))
    checks["template_holdout_violations"] = variant_violations
    if variant_violations:
        problems.append(f"{len(variant_violations)} template-holdout violations")

    # 5. No agent holds two reports of one event
    dup = []
    for w in worlds.values():
        for a, rids in w.assignments.items():
            evs = [w.report(rid).event_id for rid in rids]
            if len(evs) != len(set(evs)):
                dup.append((w.world_id, a))
    checks["duplicate_event_assignments"] = dup
    if dup:
        problems.append(f"{len(dup)} agents hold duplicate-event reports")

    # 6. Ordinary-world constraints
    constraint_fails = []
    for w in worlds.values():
        if w.tags.get("phase_bin") or w.tags.get("recycling_role"):
            continue
        net = network_oracle(w)
        if abs(net.oracle_log_odds) > cfg.worlds.max_network_abs_log_odds + 1e-9:
            constraint_fails.append((w.world_id, "network", net.oracle_log_odds))
        for a in range(w.n_agents):
            res = agent_initial_oracle(w, a)
            if abs(res.oracle_log_odds) > cfg.worlds.max_agent_abs_log_odds + 1e-9:
                constraint_fails.append((w.world_id, f"agent{a}", res.oracle_log_odds))
    checks["constraint_failures"] = constraint_fails
    if constraint_fails:
        problems.append(f"{len(constraint_fails)} evidence-constraint failures")

    checks["passed"] = not problems
    checks["problems"] = problems
    out = cfg.paths.data / "data_validation_report.json"
    out.write_text(json.dumps(checks, indent=2, default=str))
    log.info("validation %s: %s", "PASSED" if checks["passed"] else "FAILED", problems or "ok")
    return checks


def world_summary(world: World) -> dict[str, Any]:
    net = network_oracle(world)
    return {
        "world_id": world.world_id,
        "split": world.split,
        "network_oracle_log_odds": net.oracle_log_odds,
        "unique_events": net.unique_event_count,
    }
