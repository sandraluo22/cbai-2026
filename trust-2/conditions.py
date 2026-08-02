"""The five experimental conditions, expressed as per-source *plans*.

A condition is a recipe that, for each trial, produces a list of `SourcePlan`s.
A plan fixes everything reliability-related about a source — its surface label,
its per-claim correctness sequence, and the magnitude of its errors — but NOT its
name, prompt position, or the concrete numbers, which `trials.py` randomises so
that nothing leaks through name/position.

`demonstrated_accuracy` is read straight off the correctness sequence, so the
analysis always compares trust against the accuracy the model actually *saw*.

Conditions
----------
1. labels    — track record vs surface labels: 90% / 30% / no-record sources
               crossed with peer-reviewed vs anonymous-forum labels.
2. order     — same accuracy, errors clustered early vs late.
3. recovery  — improving vs degrading over the sequence (same total accuracy).
4. dose      — 90% vs 30% sources, sweeping #verifiable claims ∈ {2,5,10,20}.
5. cost      — same error rate, large vs trivial errors.
baseline     — label-only control: no track record, measures the label prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sources import PEER_LABEL, FORUM_LABEL


# --------------------------------------------------------------------------- #
@dataclass
class SourcePlan:
    """Everything reliability-related about one source in one trial."""

    key: str                              # stable analysis id, e.g. "high_acc"
    label: Optional[str]                  # surface-status label or None
    correctness: list[bool] = field(default_factory=list)  # per early claim
    error_magnitude: str = "normal"       # "trivial" | "normal" | "large"

    @property
    def n_claims(self) -> int:
        return len(self.correctness)

    @property
    def n_correct(self) -> int:
        return int(sum(self.correctness))

    @property
    def demonstrated_accuracy(self) -> Optional[float]:
        if not self.correctness:
            return None
        return self.n_correct / self.n_claims


@dataclass
class TrialSpec:
    """A condition's recipe for one trial."""

    condition: str
    plans: list[SourcePlan]
    params: dict = field(default_factory=dict)   # condition-specific metadata


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _correctness(n: int, accuracy: float, rng: np.random.Generator) -> list[bool]:
    """A shuffled correctness vector with exactly round(accuracy*n) hits."""
    n_correct = int(round(accuracy * n))
    seq = [True] * n_correct + [False] * (n - n_correct)
    rng.shuffle(seq)
    return seq


# --------------------------------------------------------------------------- #
# condition builders: each yields a TrialSpec given the trial index + rng
# --------------------------------------------------------------------------- #
def _baseline(i: int, rng: np.random.Generator, n_claims: int) -> TrialSpec:
    # Label-only: two sources, no verifiable claims → measures the pure label prior.
    return TrialSpec(
        condition="baseline",
        plans=[
            SourcePlan(key="peer_label", label=PEER_LABEL, correctness=[]),
            SourcePlan(key="forum_label", label=FORUM_LABEL, correctness=[]),
        ],
        params={},
    )


def _labels(i: int, rng: np.random.Generator, n_claims: int) -> TrialSpec:
    # Cross accuracy with label. "aligned": high-accuracy source also wears the
    # high-status (peer-reviewed) label. "crossed": high-accuracy source wears the
    # low-status (forum) label, so following the label means trusting the worse
    # source. Counterbalanced across trials.
    crossed = (i % 2 == 0)
    hi = SourcePlan(key="high_acc",
                    label=FORUM_LABEL if crossed else PEER_LABEL,
                    correctness=_correctness(n_claims, 0.9, rng))
    lo = SourcePlan(key="low_acc",
                    label=PEER_LABEL if crossed else FORUM_LABEL,
                    correctness=_correctness(n_claims, 0.3, rng))
    return TrialSpec("labels", [hi, lo], params={"crossed": crossed})


def _order(i: int, rng: np.random.Generator, n_claims: int) -> TrialSpec:
    # Same accuracy (0.6), but one clusters its errors EARLY, the other LATE.
    acc = 0.6
    n_err = n_claims - int(round(acc * n_claims))
    early = [False] * n_err + [True] * (n_claims - n_err)          # errors first
    late = [True] * (n_claims - n_err) + [False] * n_err           # errors last
    return TrialSpec(
        "order",
        [SourcePlan(key="errors_early", label=None, correctness=early),
         SourcePlan(key="errors_late", label=None, correctness=late)],
        params={"accuracy": acc},
    )


def _recovery(i: int, rng: np.random.Generator, n_claims: int) -> TrialSpec:
    # Same total accuracy (~0.5), but one IMPROVES (bad→good) and one DEGRADES.
    half = n_claims // 2
    improving = [False] * half + [True] * (n_claims - half)        # bad then good
    degrading = [True] * half + [False] * (n_claims - half)        # good then bad
    return TrialSpec(
        "recovery",
        [SourcePlan(key="improving", label=None, correctness=improving),
         SourcePlan(key="degrading", label=None, correctness=degrading)],
        params={},
    )


def _cost(i: int, rng: np.random.Generator, n_claims: int) -> TrialSpec:
    # Same error RATE (0.6 accuracy), but one source's errors are LARGE and the
    # other's are TRIVIAL. Tests whether trust tracks error magnitude, not just count.
    acc = 0.6
    c_large = _correctness(n_claims, acc, rng)
    c_triv = _correctness(n_claims, acc, rng)
    return TrialSpec(
        "cost",
        [SourcePlan(key="large_errors", label=None, correctness=c_large,
                    error_magnitude="large"),
         SourcePlan(key="trivial_errors", label=None, correctness=c_triv,
                    error_magnitude="trivial")],
        params={"accuracy": acc},
    )


DOSE_LEVELS = [2, 5, 10, 20]


def _dose(i: int, rng: np.random.Generator, n_claims: int) -> TrialSpec:
    # 90% vs 30% sources, but #verifiable claims sweeps {2,5,10,20}. Trust
    # *discrimination* should grow with evidence. n_claims arg is ignored here.
    n = DOSE_LEVELS[i % len(DOSE_LEVELS)]
    hi = SourcePlan(key="high_acc", label=None, correctness=_correctness(n, 0.9, rng))
    lo = SourcePlan(key="low_acc", label=None, correctness=_correctness(n, 0.3, rng))
    return TrialSpec("dose", [hi, lo], params={"n_claims": n})


# Registry. Each builder has signature (i, rng, n_claims) -> TrialSpec.
CONDITIONS = {
    "baseline": _baseline,
    "labels": _labels,
    "order": _order,
    "recovery": _recovery,
    "cost": _cost,
    "dose": _dose,
}


def build_specs(condition: str, n_trials: int, rng: np.random.Generator,
                n_claims: int = 10) -> list[TrialSpec]:
    """Build `n_trials` specs for a single named condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choices: {list(CONDITIONS)}")
    builder = CONDITIONS[condition]
    return [builder(i, rng, n_claims) for i in range(n_trials)]
