"""Synthetic sources and the verifiable-claim material they make claims about.

A *source* has a name, an optional surface-status label ("peer-reviewed" vs.
"anonymous forum"), and a configurable accuracy rate on verifiable claims. The
accuracy is realised by `conditions.py`/`trials.py` as a concrete per-claim
correctness plan; this module just defines the data model plus the synthetic
"facts" a claim is about.

Everything here is content/data only — no model backend, no torch — so it can be
imported by the unit tests and the analysis without a GPU.

Design notes
------------
* Topics are fully synthetic ("the resonance index of compound QX-417 is 612 Hz")
  so the model cannot retrieve a remembered answer — reliability is the only thing
  that varies, and the final contested item is a *novel* entity not seen earlier.
* A wrong claim is the true value perturbed by a magnitude that is configurable
  (trivial / normal / large) — this is what the cost-of-error-asymmetry condition
  manipulates while holding error *frequency* fixed.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Source data model
# --------------------------------------------------------------------------- #
# Surface-status labels. The hypothesis is that demonstrated track record should
# dominate these priors; condition 1 crosses label against accuracy to test it.
PEER_LABEL = "peer-reviewed lab"
FORUM_LABEL = "anonymous forum poster"

# Pool of neutral, position-free names. Names are sampled (and shuffled) per trial
# so trust cannot attach to a fixed name or to "Source A" / prompt position.
NAME_POOL = [
    "Avery", "Blake", "Casey", "Devon", "Ellis", "Finley", "Gray", "Harlow",
    "Indra", "Jordan", "Kerry", "Logan", "Marlow", "Niko", "Oakley", "Perry",
    "Quinn", "Reese", "Sage", "Tatum", "Uma", "Vesper", "Wren", "Yael",
]


@dataclass
class Source:
    """A synthetic source.

    Attributes
    ----------
    name : str
        Display name (sampled per trial; carries no positional meaning).
    status_label : Optional[str]
        Surface-status label (e.g. "peer-reviewed lab"). ``None`` means no
        affiliation is shown — used to isolate track record from labels.
    accuracy : Optional[float]
        Target accuracy on verifiable claims in [0, 1]. ``None`` means the source
        makes *no* verifiable claims (no track record) — the label-only baseline.
    """

    name: str
    status_label: Optional[str]
    accuracy: Optional[float]


# --------------------------------------------------------------------------- #
# Synthetic claim material
# --------------------------------------------------------------------------- #
PROPERTIES = [
    ("resonance index", "Hz"),
    ("binding coefficient", "kJ/mol"),
    ("thermal drift", "mK/s"),
    ("phase offset", "deg"),
    ("decay constant", "1/s"),
    ("yield factor", "%"),
    ("torsion modulus", "GPa"),
    ("spectral width", "nm"),
    ("flux density", "mWb"),
    ("attenuation", "dB"),
]


@dataclass
class Item:
    """A single verifiable fact: a property of a synthetic entity has a true value."""

    entity: str          # e.g. "compound QX-417"
    prop: str            # e.g. "resonance index"
    unit: str            # e.g. "Hz"
    true_value: int

    def record_line(self) -> str:
        return f"the {self.prop} of {self.entity} is {self.true_value} {self.unit}"


def _entity_code(rng: np.random.Generator) -> str:
    letters = "".join(rng.choice(list(string.ascii_uppercase), size=2))
    digits = int(rng.integers(100, 1000))
    return f"compound {letters}-{digits}"


def make_item(rng: np.random.Generator, used_entities: set[str]) -> Item:
    """Draw a fresh synthetic item with a unique entity code."""
    while True:
        entity = _entity_code(rng)
        if entity not in used_entities:
            used_entities.add(entity)
            break
    prop, unit = PROPERTIES[int(rng.integers(len(PROPERTIES)))]
    true_value = int(rng.integers(20, 980))
    return Item(entity=entity, prop=prop, unit=unit, true_value=true_value)


def perturb(true_value: int, magnitude: str, rng: np.random.Generator) -> int:
    """Return a wrong value: the true value shifted by a configurable magnitude.

    magnitude ∈ {"trivial", "normal", "large"} controls how far off a wrong claim
    is — the lever for the cost-of-error-asymmetry condition (same error *rate*,
    different error *size*).
    """
    if magnitude == "trivial":
        frac = rng.uniform(0.005, 0.02)
        delta = max(1, round(true_value * frac))
    elif magnitude == "large":
        frac = rng.uniform(0.5, 1.0)
        delta = max(10, round(true_value * frac))
    else:  # normal
        frac = rng.uniform(0.1, 0.3)
        delta = max(3, round(true_value * frac))
    sign = 1 if rng.random() < 0.5 else -1
    wrong = true_value + sign * delta
    if wrong <= 0:
        wrong = true_value + delta  # keep values positive/plausible
    if wrong == true_value:
        wrong = true_value + max(1, delta)
    return int(wrong)
