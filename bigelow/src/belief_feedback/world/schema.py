"""Data schema for latent worlds, evidence events, and rendered reports.

Semantic convention used everywhere in the codebase:

* Semantic log odds ``ell = log P(UPSTREAM_CONTAMINATION) - log P(LOCAL_CALIBRATION_DRIFT)``.
* Event orientation ``s_e = +1`` supports UPSTREAM_CONTAMINATION, ``-1``
  supports LOCAL_CALIBRATION_DRIFT.
* Visible labels ALPHA/BETA map onto the two semantic hypotheses with a
  per-world counterbalanced assignment (``alpha_is_upstream``).
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

UPSTREAM = "UPSTREAM_CONTAMINATION"
LOCAL = "LOCAL_CALIBRATION_DRIFT"

HYPOTHESIS_DESCRIPTIONS = {
    UPSTREAM: (
        "A supplier-origin electrolyte contamination was present before the "
        "material entered the assembly line."
    ),
    LOCAL: (
        "A fill-calibration fault at one assembly station caused an incorrect "
        "electrolyte volume during assembly."
    ),
}


class Event(BaseModel):
    """A latent evidence event; the unit of independent information."""

    event_id: str
    world_id: str
    family: str
    reliability: float
    orientation: int  # +1 supports UPSTREAM, -1 supports LOCAL
    lineage: dict[str, str] = Field(default_factory=dict)

    @property
    def llr(self) -> float:
        """Normative semantic log-likelihood ratio of this unique event."""
        return self.orientation * math.log(self.reliability / (1.0 - self.reliability))


class Report(BaseModel):
    """A rendered natural-language document derived from one latent event."""

    report_id: str
    world_id: str
    event_id: str
    family: str
    orientation: int
    is_secondary: bool = False
    source_report_id: str | None = None  # for secondary reports
    template_variant: int = 0
    author: str = ""
    author_unit: str = ""
    date: str = ""
    title: str = ""
    lineage_visible: dict[str, str] = Field(default_factory=dict)
    text: str = ""
    word_count: int = 0


class World(BaseModel):
    """One static hidden world: truth, events, reports, and assignments."""

    world_id: str
    split: str
    true_hypothesis: str  # UPSTREAM or LOCAL
    alpha_is_upstream: bool
    n_agents: int
    events: list[Event] = Field(default_factory=list)
    reports: list[Report] = Field(default_factory=list)
    # agent_id -> ordered list of report_ids privately assigned
    assignments: dict[int, list[str]] = Field(default_factory=dict)
    agent_names: dict[int, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)  # e.g. phase-bin metadata

    # ---- label mapping helpers -------------------------------------------
    def semantic_to_visible(self, ell_semantic: float) -> float:
        """Convert semantic log odds to visible (ALPHA-vs-BETA) log odds."""
        return ell_semantic if self.alpha_is_upstream else -ell_semantic

    def visible_to_semantic(self, ell_visible: float) -> float:
        return ell_visible if self.alpha_is_upstream else -ell_visible

    def visible_label(self, hypothesis: str) -> str:
        if hypothesis == UPSTREAM:
            return "ALPHA" if self.alpha_is_upstream else "BETA"
        return "BETA" if self.alpha_is_upstream else "ALPHA"

    def semantic_hypothesis(self, visible_label: str) -> str:
        if visible_label == "ALPHA":
            return UPSTREAM if self.alpha_is_upstream else LOCAL
        return LOCAL if self.alpha_is_upstream else UPSTREAM

    @property
    def alpha_description(self) -> str:
        return HYPOTHESIS_DESCRIPTIONS[UPSTREAM if self.alpha_is_upstream else LOCAL]

    @property
    def beta_description(self) -> str:
        return HYPOTHESIS_DESCRIPTIONS[LOCAL if self.alpha_is_upstream else UPSTREAM]

    # ---- lookup helpers ---------------------------------------------------
    def report(self, report_id: str) -> Report:
        for r in self.reports:
            if r.report_id == report_id:
                return r
        raise KeyError(report_id)

    def event(self, event_id: str) -> Event:
        for e in self.events:
            if e.event_id == event_id:
                return e
        raise KeyError(event_id)

    def event_of_report(self, report_id: str) -> Event:
        return self.event(self.report(report_id).event_id)
