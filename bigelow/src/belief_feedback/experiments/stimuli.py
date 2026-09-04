"""Controlled stimulus memos for exogenous calibration trials.

Stimulus memos use the exact public memo format of live agents, with natural
prose assembled from the cited documents' own template-rendered findings.
"""

from __future__ import annotations

import numpy as np

from ..seeds import rng as make_rng
from ..world.schema import World

CONF_LEVELS = {"low": 30, "medium": 60, "high": 90}

_STANCE_PHRASES = {
    1: "the balance of evidence points to hypothesis {label}",
    0: "the evidence does not yet separate the two hypotheses",
    -1: "the balance of evidence points to hypothesis {label}",
}

_HEDGES = {
    "low": "I hold this view tentatively and it could easily change.",
    "medium": "I hold this view with moderate conviction.",
    "high": "I consider this well supported by the material I reviewed.",
}


def _finding_snippet(world: World, rid: str, n_words: int = 30) -> str:
    text = world.report(rid).text
    marker = "Findings: "
    idx = text.find(marker)
    snippet = text[idx + len(marker) :] if idx >= 0 else text
    return " ".join(snippet.split()[:n_words])


def make_stimulus_memo(
    world: World,
    *,
    stance_semantic: int,
    confidence_level: str,
    cited_report_ids: list[str],
    variant: int = 0,
    exact_copy_of: str | None = None,
) -> str:
    """Render one controlled incoming memo in the live public format."""
    if exact_copy_of is not None:
        return exact_copy_of
    r = make_rng("stimulus", world.world_id, stance_semantic, confidence_level, *cited_report_ids, variant)
    conf = CONF_LEVELS[confidence_level] + int(r.integers(-4, 5))
    if stance_semantic == 0:
        visible = "UNRESOLVED"
    else:
        hyp = "UPSTREAM_CONTAMINATION" if stance_semantic > 0 else "LOCAL_CALIBRATION_DRIFT"
        visible = world.visible_label(hyp)
    stance_txt = _STANCE_PHRASES[stance_semantic].format(label=visible)
    snippets = [
        f"Report {rid} notes that { _finding_snippet(world, rid, 22) }"
        for rid in cited_report_ids[:2]
    ]
    body_bits = [
        f"Having reviewed the records available on my side, my reading is that {stance_txt}.",
        *snippets,
        _HEDGES[confidence_level],
        "I would welcome any independent measurement that bears on this question from your side of the network.",
    ]
    body = " ".join(body_bits)
    words = body.split()
    if len(words) > 130:
        body = " ".join(words[:128]) + "."
    cited = ", ".join(cited_report_ids) if cited_report_ids else "none"
    return (
        f"Current assessment: {visible}\n"
        f"Confidence: {int(np.clip(conf, 0, 100))}\n"
        f"Evidence cited: {cited}\n"
        f"Memo: {body}\n"
        f"Request to team: Please report any evidence that would confirm or contradict this reading."
    )
