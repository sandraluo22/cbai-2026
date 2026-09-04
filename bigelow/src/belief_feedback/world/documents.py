"""Render latent events into natural-language incident documents.

Every rendered document contains a report id, title, author, date, visible
source lineage, prose findings, and a limitations sentence — and never the
hidden truth, event ids, reliabilities, or likelihoods.
"""

from __future__ import annotations

import numpy as np

from ..seeds import rng as make_rng
from .schema import Event, Report
from .templates import (
    CONTEXT_SENTENCES,
    FAMILY_TEMPLATES,
    FIRST_NAMES,
    LAST_NAMES,
    LIMITATION_SENTENCES,
    SECONDARY_WRAPPERS,
    SITES,
)

MIN_WORDS, MAX_WORDS = 80, 170

_PAD_SENTENCES = [
    "Copies of the underlying raw records are retained in the investigation file and are available on request.",
    "All measurements referenced above were performed using the standard procedures qualified for this material.",
    "The responsible unit will issue an updated record if any of the referenced items are re-examined.",
    "Personnel involved in this work have been listed in the investigation roster for follow-up questions.",
    "Where applicable, instrument identifiers and procedure revisions are recorded in the source system.",
]


def draw_lineage(r: np.random.Generator) -> dict[str, str]:
    """Draw a fresh visible lineage bundle for a latent event."""
    return {
        "sample_id": f"S-{r.integers(1000, 9999)}",
        "lot_id": f"L-{r.integers(10000, 99999)}",
        "station_id": f"ST-{r.integers(10, 99)}",
        "ticket_id": f"T-{r.integers(100000, 999999)}",
        "run_id": f"RUN-{r.integers(1000, 9999)}",
    }


def _fill_values(lineage: dict[str, str], r: np.random.Generator) -> dict[str, str]:
    vals = dict(lineage)
    vals["ppm"] = str(int(r.integers(12, 240)))
    vals["pct"] = f"{r.uniform(2.0, 4.0):.1f}"
    vals["minutes"] = str(int(r.integers(15, 31)))
    vals["n_units"] = str(int(r.integers(18, 96)))
    vals["site"] = str(r.choice(SITES))
    return vals


def _date(r: np.random.Generator) -> str:
    month = int(r.integers(1, 13))
    day = int(r.integers(1, 28))
    return f"2025-{month:02d}-{day:02d}"


def _author(r: np.random.Generator) -> str:
    return f"{r.choice(FIRST_NAMES)} {r.choice(LAST_NAMES)}"


def lineage_phrase(lineage: dict[str, str]) -> str:
    return f"sample {lineage['sample_id']} (lot {lineage['lot_id']}, station {lineage['station_id']})"


def render_report(
    report_id: str,
    event: Event,
    *,
    template_variant: int,
    is_secondary: bool = False,
    source_report_id: str | None = None,
    wrapper_variant: int = 0,
) -> Report:
    """Render one report deterministically from its identifying tuple."""
    fam = FAMILY_TEMPLATES[event.family]
    orient_key = "up" if event.orientation > 0 else "local"
    r = make_rng("doc", report_id, event.event_id, template_variant, is_secondary)

    vals = _fill_values(event.lineage, r)
    finding = fam[orient_key][template_variant].format(**vals)
    title = str(r.choice(fam["titles"]))
    author = _author(r)
    unit = fam["unit"]
    date = _date(r)
    context = CONTEXT_SENTENCES[int(r.integers(0, len(CONTEXT_SENTENCES)))].format(**vals)
    limitation = LIMITATION_SENTENCES[int(r.integers(0, len(LIMITATION_SENTENCES)))]

    lines = [
        f"Report {report_id} — {title}",
        f"Author: {author} ({unit})",
        f"Date: {date}",
        (
            f"References: sample {event.lineage['sample_id']}; lot {event.lineage['lot_id']}; "
            f"station {event.lineage['station_id']}; ticket {event.lineage['ticket_id']}"
        ),
        "",
    ]
    body = [context]
    if is_secondary and source_report_id is not None:
        wrapper = SECONDARY_WRAPPERS[wrapper_variant].format(
            orig_report_id=source_report_id, lineage_phrase=lineage_phrase(event.lineage)
        )
        body.append(wrapper)
    body.append(f"Findings: {finding}")
    body.append(f"Limitations: {limitation}")

    text = "\n".join(lines) + "\n" + " ".join(body)
    pad_i = 0
    while len(text.split()) < MIN_WORDS and pad_i < len(_PAD_SENTENCES):
        text += " " + _PAD_SENTENCES[(int(r.integers(0, 5)) + pad_i) % len(_PAD_SENTENCES)]
        pad_i += 1

    return Report(
        report_id=report_id,
        world_id=event.world_id,
        event_id=event.event_id,
        family=event.family,
        orientation=event.orientation,
        is_secondary=is_secondary,
        source_report_id=source_report_id,
        template_variant=template_variant,
        author=author,
        author_unit=unit,
        date=date,
        title=title,
        lineage_visible=dict(event.lineage),
        text=text,
        word_count=len(text.split()),
    )
