"""Document rendering: word counts, hidden-information hygiene (Part 24: 3)."""

from __future__ import annotations

import re

from belief_feedback.world.documents import MAX_WORDS, MIN_WORDS
from belief_feedback.world.validation import FORBIDDEN_PATTERNS


def test_word_counts(world):
    for rep in world.reports:
        assert MIN_WORDS <= rep.word_count <= MAX_WORDS, rep.report_id


def test_no_hidden_information(world):
    for rep in world.reports:
        for pat in FORBIDDEN_PATTERNS:
            assert not re.search(pat, rep.text), f"{rep.report_id} leaks {pat}"
        # numeric reliabilities and event ids never appear
        assert rep.event_id not in rep.text
        assert str(world.event(rep.event_id).reliability) not in rep.text


def test_required_fields(world):
    for rep in world.reports:
        assert rep.text.startswith(f"Report {rep.report_id}")
        assert "Author:" in rep.text
        assert "Date:" in rep.text
        assert "References:" in rep.text
        assert "Findings:" in rep.text
        assert "Limitations:" in rep.text


def test_secondary_reports_share_lineage(world):
    secondaries = [r for r in world.reports if r.is_secondary]
    assert secondaries, "world composition must include secondary reports"
    for sec in secondaries:
        primary = world.report(sec.source_report_id)
        assert sec.event_id == primary.event_id
        assert sec.report_id != primary.report_id
        assert sec.lineage_visible == primary.lineage_visible
        assert primary.report_id in sec.text  # visible reference to the original
        assert sec.text != primary.text
