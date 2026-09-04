"""Parse public memos into structured features.

Malformed generations are never regenerated; whatever parses is recorded and
``format_valid`` marks failures. Citation classification:

* valid: report id exists in the world and is accessible to the author.
* invalid: id exists in the world but was not accessible to the author.
* hallucinated: id-shaped string that matches no report in the world.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..world.schema import World

ASSESS_RE = re.compile(r"^Current assessment:\s*(ALPHA|BETA|UNRESOLVED)\s*$", re.IGNORECASE | re.MULTILINE)
CONF_RE = re.compile(r"^Confidence:\s*(\d{1,3})\s*$", re.MULTILINE)
CITED_RE = re.compile(r"^Evidence cited:\s*(.*)$", re.MULTILINE)
MEMO_RE = re.compile(r"^Memo:\s*(.*?)(?=^Request to team:|\Z)", re.MULTILINE | re.DOTALL)
REQUEST_RE = re.compile(r"^Request to team:\s*(.*)$", re.MULTILINE)
RID_RE = re.compile(r"R-[A-Za-z0-9_+.]+-[A-Za-z0-9]+")


@dataclass
class ParsedMemo:
    raw_text: str
    format_valid: bool
    parsed_assessment: str | None  # visible label or UNRESOLVED
    parsed_confidence: int | None
    cited_ids: list[str] = field(default_factory=list)
    valid_citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    hallucinated_report_ids: list[str] = field(default_factory=list)
    memo_body: str = ""
    request: str = ""
    word_count: int = 0

    def semantic_stance(self, world: World) -> int:
        """-1 / 0 / +1 in semantic (upstream-vs-local) terms."""
        if self.parsed_assessment is None or self.parsed_assessment == "UNRESOLVED":
            return 0
        hyp = world.semantic_hypothesis(self.parsed_assessment)
        return 1 if hyp == "UPSTREAM_CONTAMINATION" else -1


def parse_memo(
    text: str,
    world: World | None = None,
    accessible_report_ids: set[str] | None = None,
) -> ParsedMemo:
    assessment = None
    m = ASSESS_RE.search(text)
    if m:
        assessment = m.group(1).upper()
    confidence = None
    m = CONF_RE.search(text)
    if m:
        c = int(m.group(1))
        confidence = c if 0 <= c <= 100 else None
    cited_raw = ""
    m = CITED_RE.search(text)
    if m:
        cited_raw = m.group(1)
    body = ""
    m = MEMO_RE.search(text)
    if m:
        body = m.group(1).strip()
    request = ""
    m = REQUEST_RE.search(text)
    if m:
        request = m.group(1).strip()

    cited_ids = RID_RE.findall(cited_raw) if cited_raw else RID_RE.findall(text)
    cited_ids = list(dict.fromkeys(cited_ids))  # dedupe, keep order

    valid, invalid, hallucinated = [], [], []
    if world is not None:
        known = {r.report_id for r in world.reports}
        accessible = accessible_report_ids if accessible_report_ids is not None else known
        for cid in cited_ids:
            if cid not in known:
                hallucinated.append(cid)
            elif cid in accessible:
                valid.append(cid)
            else:
                invalid.append(cid)

    format_valid = assessment is not None and confidence is not None and bool(body)
    return ParsedMemo(
        raw_text=text,
        format_valid=format_valid,
        parsed_assessment=assessment,
        parsed_confidence=confidence,
        cited_ids=cited_ids,
        valid_citations=valid,
        invalid_citations=invalid,
        hallucinated_report_ids=hallucinated,
        memo_body=body,
        request=request,
        word_count=len(text.split()),
    )


def repeated_4gram_rate(text: str) -> float:
    """Fraction of 4-grams that are repeats (coherence diagnostic)."""
    toks = text.lower().split()
    if len(toks) < 5:
        return 0.0
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - len(set(grams)) / len(grams)
