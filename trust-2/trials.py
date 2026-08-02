"""Turn condition `TrialSpec`s into concrete trial conversations + prompts.

A trial is: a chronological verification log of each source's early claims
(checked against an in-context authoritative record), followed by a single novel,
unverifiable claim on which the sources disagree. The model is asked which source
to trust.

Controls baked in here
-----------------------
* Source names are sampled per trial and prompt order is shuffled (counterbalanced),
  so trust cannot attach to a name or to prompt position.
* Every source draws its verifiable claims from the same synthetic item
  distribution (same topic family / difficulty); only reliability varies.
* The final contested entity is novel — never appears in the early claims — so the
  answer cannot be retrieved from context.
* The early-claim log is interleaved round-by-round, so "errors early vs late"
  (order/recovery conditions) appears as a genuine temporal sequence.

No torch here — pure data + string assembly, importable by tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from conditions import SourcePlan, TrialSpec, build_specs
from sources import NAME_POOL, Item, make_item, perturb


# --------------------------------------------------------------------------- #
@dataclass
class Claim:
    """One verifiable claim by a source, with its in-context verification result."""

    source_key: str
    source_name: str
    round_idx: int
    item: Item
    asserted_value: int
    correct: bool


@dataclass
class SourceView:
    """A source as concretised in a trial (plan + assigned name/position)."""

    key: str
    name: str
    label: Optional[str]
    position: int                         # index in the (shuffled) prompt order
    demonstrated_accuracy: Optional[float]
    n_claims: int
    n_correct: int
    error_magnitude: str
    final_value: int                      # its assertion on the contested item


@dataclass
class Trial:
    trial_id: str
    condition: str
    seed: int
    sources: list[SourceView]
    claims: list[Claim]                   # chronological (interleaved by round)
    final_entity: str
    final_prop: str
    final_unit: str
    params: dict = field(default_factory=dict)

    # ----- serialisation / identity ------------------------------------- #
    def trial_hash(self) -> str:
        """Stable content hash (independent of model) — used as a cache key seed."""
        payload = {
            "condition": self.condition,
            "final": [self.final_entity, self.final_prop, self.final_unit],
            "sources": [
                {"key": s.key, "name": s.name, "label": s.label, "pos": s.position,
                 "final": s.final_value} for s in self.sources
            ],
            "claims": [
                {"src": c.source_name, "r": c.round_idx, "ent": c.item.entity,
                 "prop": c.item.prop, "true": c.item.true_value,
                 "said": c.asserted_value, "ok": c.correct}
                for c in self.claims
            ],
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def record(self) -> dict:
        """Everything the analysis needs about this trial (sans the response)."""
        return {
            "trial_id": self.trial_id,
            "condition": self.condition,
            "seed": self.seed,
            "trial_hash": self.trial_hash(),
            "final_entity": self.final_entity,
            "final_prop": self.final_prop,
            "final_unit": self.final_unit,
            "params": self.params,
            "sources": [
                {"key": s.key, "name": s.name, "label": s.label,
                 "position": s.position,
                 "demonstrated_accuracy": s.demonstrated_accuracy,
                 "n_claims": s.n_claims, "n_correct": s.n_correct,
                 "error_magnitude": s.error_magnitude,
                 "final_value": s.final_value}
                for s in self.sources
            ],
        }


# --------------------------------------------------------------------------- #
# concretisation
# --------------------------------------------------------------------------- #
def _concretise(spec: TrialSpec, trial_id: str, seed: int,
                rng: np.random.Generator) -> Trial:
    plans = spec.plans
    n = len(plans)

    # Distinct names; shuffled prompt order (counterbalance position).
    names = list(rng.choice(NAME_POOL, size=n, replace=False))
    order = list(rng.permutation(n))      # order[p] = plan index shown at position p

    used_entities: set[str] = set()
    claims: list[Claim] = []

    # Build each source's early claims (own items, shared difficulty distribution).
    per_source_claims: dict[int, list[Claim]] = {pi: [] for pi in range(n)}
    for pi, plan in enumerate(plans):
        for r, ok in enumerate(plan.correctness):
            item = make_item(rng, used_entities)
            asserted = item.true_value if ok else perturb(
                item.true_value, plan.error_magnitude, rng)
            per_source_claims[pi].append(
                Claim(source_key=plan.key, source_name=names[pi], round_idx=r,
                      item=item, asserted_value=asserted, correct=ok))

    # Interleave chronologically by round, iterating sources in prompt order so the
    # timeline reads naturally. Sources with fewer claims simply drop out of later
    # rounds.
    max_rounds = max((p.n_claims for p in plans), default=0)
    for r in range(max_rounds):
        for p in order:
            cs = per_source_claims[p]
            if r < len(cs):
                claims.append(cs[r])

    # Novel contested item: fresh entity, distinct value per source.
    final_item = make_item(rng, used_entities)
    centre = int(rng.integers(100, 900))
    offsets = list(rng.choice(range(-200, 201), size=n, replace=False))
    final_values = {}
    for pi in range(n):
        v = centre + int(offsets[pi])
        final_values[pi] = max(1, v)

    # Assemble SourceView list (in plan order; position from the shuffle).
    pos_of_plan = {plan_idx: p for p, plan_idx in enumerate(order)}
    sources = []
    for pi, plan in enumerate(plans):
        sources.append(SourceView(
            key=plan.key, name=names[pi], label=plan.label,
            position=pos_of_plan[pi],
            demonstrated_accuracy=plan.demonstrated_accuracy,
            n_claims=plan.n_claims, n_correct=plan.n_correct,
            error_magnitude=plan.error_magnitude,
            final_value=final_values[pi]))

    return Trial(
        trial_id=trial_id, condition=spec.condition, seed=seed,
        sources=sources, claims=claims,
        final_entity=final_item.entity, final_prop=final_item.prop,
        final_unit=final_item.unit, params=dict(spec.params))


def generate_trials(condition: str, n_trials: int, seed: int,
                    n_claims: int = 10) -> list[Trial]:
    """Deterministic for a given (condition, n_trials, seed)."""
    rng = np.random.default_rng(seed)
    specs = build_specs(condition, n_trials, rng, n_claims=n_claims)
    trials = []
    for i, spec in enumerate(specs):
        tid = f"{condition}-{seed}-{i:04d}"
        trials.append(_concretise(spec, tid, seed, rng))
    return trials


# --------------------------------------------------------------------------- #
# prompt rendering
# --------------------------------------------------------------------------- #
SYSTEM = (
    "You are assessing competing technical claims. You will see a verification log "
    "in which several sources made earlier measurements that were each checked "
    "against an authoritative master record. Then the sources disagree on one final "
    "measurement that cannot be checked against any record. Judge which source's "
    "final value to believe and report your judgment as a probability distribution "
    "over the sources."
)


def _source_descriptor(s: SourceView) -> str:
    if s.label:
        return f"{s.name} ({s.label})"
    return s.name


def render_prompt(trial: Trial) -> tuple[str, str]:
    """Return (system, user) strings for one trial."""
    by_name = {s.name: s for s in trial.sources}
    ordered = sorted(trial.sources, key=lambda s: s.position)

    lines: list[str] = []
    lines.append("SOURCES")
    for s in ordered:
        lines.append(f"  - {_source_descriptor(s)}")
    lines.append("")

    if trial.claims:
        lines.append("VERIFICATION LOG (chronological; each claim checked against the "
                     "authoritative master record)")
        for idx, c in enumerate(trial.claims, 1):
            s = by_name[c.source_name]
            verdict = "CORRECT" if c.correct else "INCORRECT"
            lines.append(
                f"  {idx:>3}. {_source_descriptor(s)} reported the {c.item.prop} of "
                f"{c.item.entity} as {c.asserted_value} {c.item.unit}. "
                f"Master record: {c.item.true_value} {c.item.unit}. -> {verdict}.")
    else:
        lines.append("VERIFICATION LOG")
        lines.append("  (no prior measurements on record for these sources)")
    lines.append("")

    lines.append("FINAL CONTESTED CLAIM (cannot be verified against any record)")
    lines.append(f"  Question: what is the {trial.final_prop} of {trial.final_entity}?")
    lines.append("  The sources report different values:")
    for s in ordered:
        lines.append(f"    - {_source_descriptor(s)}: {s.final_value} {trial.final_unit}")
    lines.append("")

    names = [s.name for s in ordered]
    example = ", ".join(f'"{nm}": {round(1/len(names), 2)}' for nm in names)
    lines.append("TASK")
    lines.append(
        "Decide how much to trust each source's final value. Respond with ONLY a "
        "JSON object, no other text, of exactly this form:")
    lines.append(
        '  {"trust": {' + example + '}, "confidence": 0.0, '
        '"justification": "one short sentence"}')
    lines.append(
        "The values under \"trust\" are probabilities in [0,1] that must sum to 1 "
        "(one per source named above). \"confidence\" is your overall confidence in "
        "[0,1]. \"justification\" briefly states your reasoning.")

    return SYSTEM, "\n".join(lines)
