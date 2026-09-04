"""Memo parsing, citation classification, malformed retention (Part 24: 13)."""

from __future__ import annotations

from belief_feedback.agents.memo_parser import parse_memo, repeated_4gram_rate
from belief_feedback.agents.protocol import BranchSpec, run_episode
from belief_feedback.models.mock_backend import MockBackend

VALID = (
    "Current assessment: ALPHA\n"
    "Confidence: 72\n"
    "Evidence cited: {rid1}, {rid2}\n"
    "Memo: The chromatography and the drum retest line up in the same direction.\n"
    "Request to team: Please verify the drum retest independently."
)


def test_parse_valid_memo(world):
    rid1 = world.reports[0].report_id
    rid2 = world.reports[1].report_id
    accessible = {rid1}
    pm = parse_memo(VALID.format(rid1=rid1, rid2=rid2), world, accessible)
    assert pm.format_valid
    assert pm.parsed_assessment == "ALPHA"
    assert pm.parsed_confidence == 72
    assert pm.valid_citations == [rid1]
    assert pm.invalid_citations == [rid2]  # exists in world, not accessible to author
    assert pm.hallucinated_report_ids == []
    assert pm.request.startswith("Please verify")


def test_hallucinated_citation_detected(world):
    text = VALID.format(rid1=world.reports[0].report_id, rid2="R-w_phantom_0001-000")
    pm = parse_memo(text, world, {world.reports[0].report_id})
    assert pm.hallucinated_report_ids == ["R-w_phantom_0001-000"]


def test_malformed_memo_is_parsed_not_rejected():
    pm = parse_memo("I really cannot commit to a structured answer here.")
    assert not pm.format_valid
    assert pm.parsed_assessment is None
    assert pm.parsed_confidence is None
    assert pm.raw_text  # the raw text is always retained


def test_confidence_out_of_range_invalid():
    pm = parse_memo("Current assessment: BETA\nConfidence: 350\nMemo: x")
    assert pm.parsed_confidence is None


def test_invalid_memos_remain_in_dataset(cfg, test_world, steer_ctx):
    """With forced malformed generation, rows are kept and flagged."""
    be = MockBackend(cfg, p_malformed=1.0)
    res = run_episode(cfg, be, test_world, 11, BranchSpec(name="baseline"), steer_ctx)
    assert res.message_rows  # nothing was regenerated or dropped
    assert all(not r["format_valid"] for r in res.message_rows)
    assert all(r["raw_text"] for r in res.message_rows)


def test_repeated_4gram_rate():
    assert repeated_4gram_rate("a b c d e f g h") == 0.0
    assert repeated_4gram_rate("x y z w x y z w x y z w") > 0.3
