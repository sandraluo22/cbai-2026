"""Unit tests for the robust JSON parsing + justification coding (no model / API)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import parse_response, references_track_record

NAMES = ["Avery", "Blake"]


def test_plain_json():
    out = parse_response(
        '{"trust": {"Avery": 0.8, "Blake": 0.2}, "confidence": 0.7, '
        '"justification": "Avery was accurate."}', NAMES)
    assert out is not None
    assert abs(out["trust"]["Avery"] - 0.8) < 1e-9
    assert out["confidence"] == 0.7


def test_code_fence_stripped():
    text = '```json\n{"trust": {"Avery": 1, "Blake": 0}, "confidence": 0.9, ' \
           '"justification": "x"}\n```'
    out = parse_response(text, NAMES)
    assert out is not None
    assert out["trust"]["Avery"] == 1.0
    assert out["trust"]["Blake"] == 0.0


def test_embedded_json_salvaged():
    text = 'Sure! Here is my answer:\n{"trust": {"Avery": 0.6, "Blake": 0.4}, ' \
           '"confidence": 0.5, "justification": "y"} Hope that helps.'
    out = parse_response(text, NAMES)
    assert out is not None
    assert abs(out["trust"]["Avery"] - 0.6) < 1e-9


def test_probabilities_normalised():
    out = parse_response('{"trust": {"Avery": 6, "Blake": 2}}', NAMES)
    assert out is not None
    assert abs(out["trust"]["Avery"] - 0.75) < 1e-9
    assert abs(out["trust"]["Blake"] - 0.25) < 1e-9
    assert abs(sum(out["trust"].values()) - 1.0) < 1e-9


def test_probabilities_key_alias():
    out = parse_response('{"probabilities": {"Avery": 0.5, "Blake": 0.5}}', NAMES)
    assert out is not None
    assert abs(out["trust"]["Avery"] - 0.5) < 1e-9


def test_missing_source_filled_zero():
    out = parse_response('{"trust": {"Avery": 1.0}}', NAMES)
    assert out is not None
    assert out["trust"]["Blake"] == 0.0


def test_line_scrape_fallback():
    text = "I trust Avery: 0.9 and Blake: 0.1 based on the log."
    out = parse_response(text, NAMES)
    assert out is not None
    assert out["trust"]["Avery"] > out["trust"]["Blake"]


def test_unparsable_returns_none():
    assert parse_response("no numbers, no json here", NAMES) is None
    assert parse_response("", NAMES) is None


def test_garbage_confidence_tolerated():
    out = parse_response(
        '{"trust": {"Avery": 0.5, "Blake": 0.5}, "confidence": "high"}', NAMES)
    assert out is not None
    assert out["confidence"] is None


def test_track_record_coding():
    assert references_track_record("Avery had a better track record.")
    assert references_track_record("Blake was more accurate in the log.")
    assert references_track_record("Avery made fewer errors earlier.")
    assert not references_track_record("I picked the peer-reviewed one.")
    assert not references_track_record("")
