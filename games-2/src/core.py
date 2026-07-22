"""Core engine for the mutual-theory-of-mind games.

Abstract, bounded, frozen-by-construction reference agents (a Rational-Speech-Acts
recursion). Everything the two games need lives here:

  * Board            : N abstract concept-IDs, a small clue vocabulary of size C,
                       and a fixed (seeded) clue<->item association matrix -- the
                       shared "code". C < N FORCES coordination (no clue names one
                       item); C >= N is the easy regime.
  * ListenerAgent    : maintains a BELIEF (distribution over the board simplex)
                       about which items are in the speaker's hidden target set,
                       updated from clues + correct/incorrect feedback.
                       level 1 = literal (decode via association);
                       level 2 = pragmatic (interpret a clue RELATIVE to what it
                       already knows -- "what would A bother to say, given what
                       I've shown I understand").
  * SpeakerAgent     : privately holds the target set (its latent z) and emits a
                       (clue, count). level 1 ignores the listener; level 2 ADAPTS
                       the clue to the listener's demonstrated belief (skip what B
                       already has, reinforce what B is missing) -- the symmetric
                       mutual-modeling knob.

Design invariants demanded by the spec:
  - CHANNEL is bounded: clues live in {0..C-1}; a "swap" is c -> c', enumerable.
  - MEASUREMENT is bounded: every read-out is a distribution over the N board
    items (a fixed simplex), regardless of the clue channel.
  - Agents are FROZEN: no learning; the counterfactual fork holds all state fixed
    so KL isolates the causal effect of an intervention, not drift or weight change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# small numerics
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-300, None)


def normalize(p, eps=1e-12):
    p = np.clip(np.asarray(p, float), 0, None)
    s = p.sum()
    return p / s if s > eps else np.ones_like(p) / len(p)


def kl(p, q, eps=1e-12):
    """KL(p || q) over a discrete simplex, in nats."""
    p = normalize(p) + eps
    q = normalize(q) + eps
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def entropy(p, eps=1e-12):
    p = normalize(p) + eps
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------
@dataclass
class Board:
    n_items: int = 16
    n_clues: int = 3               # C < N forces coordination; C >= N is the easy regime
    n_targets: int = 4
    seed: int = 0
    assoc: np.ndarray = field(default=None, repr=False)   # (C, N) >= 0, the shared code

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        if self.assoc is None:
            # log-normal affinities; in the easy regime (C>=N) bias toward a near
            # one-to-one code so a single clue can name an item.
            base = np.exp(rng.normal(0, 1.0, size=(self.n_clues, self.n_items)))
            if self.n_clues >= self.n_items:
                base += 3.0 * np.eye(self.n_clues, self.n_items)
            self.assoc = base
        self.L0 = self.assoc / self.assoc.sum(axis=1, keepdims=True)   # literal listener P(i|c)
        self.logL0 = np.log(np.clip(self.L0, 1e-12, None))

    def sample_targets(self, seed) -> List[int]:
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(self.n_items, size=self.n_targets, replace=False).tolist())


# ---------------------------------------------------------------------------
# Speaker model S1 (level-1 pragmatic speaker over a target set)
# ---------------------------------------------------------------------------
def speaker_utility(board: Board, targets) -> np.ndarray:
    """U(c) = mean_{i in targets} log L0[c,i] : how well clue c points the literal
    listener at the target set. Returns (C,)."""
    if len(targets) == 0:
        return np.zeros(board.n_clues)
    return board.logL0[:, list(targets)].mean(axis=1)


def speaker_dist(board: Board, targets, alpha) -> np.ndarray:
    """S1(c | targets) ∝ exp(alpha * U(c)). alpha = rationality (1/temperature)."""
    return softmax(alpha * speaker_utility(board, targets))


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------
@dataclass
class ListenerAgent:
    board: Board
    level: int = 2                 # 1 = literal decode, 2 = pragmatic (models the speaker)
    beta: float = 2.0              # evidence gain per clue
    alpha: float = 3.0             # speaker rationality assumed by an L2 listener
    prior: float = None

    def __post_init__(self):
        N = self.board.n_items
        p0 = self.prior if self.prior is not None else self.board.n_targets / N
        self.logodds = np.full(N, np.log(p0 / (1 - p0)))
        self.known: set = set()     # confirmed targets (correct guesses)
        self.dead: set = set()      # confirmed non-targets (wrong guesses)
        self.guessed: set = set()

    # ---- evidence that clue c points at item i, per level ----
    def _evidence(self, c: int) -> np.ndarray:
        N = self.board.n_items
        if self.level <= 1:
            e = self.board.logL0[c].copy()                      # literal association
        else:
            # pragmatic: marginal boost to speaker's preference for c if i were a
            # target ON TOP of what's already known -> conditions on B's own state.
            K = list(self.known)
            baseU = speaker_utility(self.board, K) if K else np.zeros(self.board.n_clues)
            base = softmax(self.alpha * baseU)[c]
            e = np.zeros(N)
            for i in range(N):
                Ui = speaker_utility(self.board, K + [i])
                e[i] = np.log(softmax(self.alpha * Ui)[c] + 1e-12) - np.log(base + 1e-12)
        return e

    def belief(self) -> np.ndarray:
        """Full board posterior p_i = P(item i in T). known->1, dead->0."""
        p = 1.0 / (1.0 + np.exp(-self.logodds))
        for i in self.known:
            p[i] = 1.0
        for i in self.dead:
            p[i] = 0.0
        return p

    def guess_dist(self) -> np.ndarray:
        """Bounded read-out: a distribution over the board simplex (unguessed items)."""
        p = self.belief().copy()
        for i in self.guessed:
            p[i] = 0.0
        return normalize(p)

    def update(self, c: int, count: int = 1):
        e = self._evidence(c)
        active = [i for i in range(self.board.n_items) if i not in self.known and i not in self.dead]
        if active:
            e = e - e[active].mean()
            for i in active:
                self.logodds[i] += self.beta * e[i]
        return self

    def observe(self, guess: int, correct: bool):
        self.guessed.add(guess)
        if correct:
            self.known.add(guess); self.logodds[guess] = 20.0
        else:
            self.dead.add(guess); self.logodds[guess] = -20.0

    def pick_guess(self) -> int:
        d = self.guess_dist()
        return int(np.argmax(d))

    def copy(self) -> "ListenerAgent":
        import copy as _c
        L = ListenerAgent(self.board, self.level, self.beta, self.alpha, self.prior)
        L.logodds = self.logodds.copy()
        L.known = set(self.known); L.dead = set(self.dead); L.guessed = set(self.guessed)
        return L


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------
@dataclass
class SpeakerAgent:
    board: Board
    targets: List[int]             # the hidden latent z
    level: int = 2                 # 1 = ignore listener, 2 = adapt clue to listener belief
    alpha: float = 3.0             # rationality / (1/temperature) for clue choice
    tau: float = 1.0               # extra sampling temperature ("make it non-obvious")

    def __post_init__(self):
        self.remaining = list(self.targets)

    def remaining_targets(self, listener: Optional[ListenerAgent]) -> List[int]:
        return [t for t in self.remaining if t not in (listener.known if listener else set())]

    def clue_dist(self, listener: Optional[ListenerAgent] = None) -> np.ndarray:
        """Distribution over clue tokens. L1: informativeness about the remaining
        targets, ignoring the listener. L2: expected GAIN in the listener's target
        mass -> adapts to the listener's demonstrated belief."""
        if self.level <= 1 or listener is None:
            tgt = list(self.remaining)
            return speaker_dist(self.board, tgt, self.alpha / max(self.tau, 1e-6))
        # L2: for each clue, simulate the listener's update and score the gain on
        # the targets the listener is still MISSING.
        miss = [t for t in self.remaining if t not in listener.known]
        if not miss:
            return np.ones(self.board.n_clues) / self.board.n_clues
        before = listener.belief()
        gains = np.zeros(self.board.n_clues)
        for c in range(self.board.n_clues):
            probe = listener.copy(); probe.update(c)
            after = probe.belief()
            gains[c] = float((after - before)[miss].sum())
        return softmax(self.alpha / max(self.tau, 1e-6) * gains)

    def clue(self, listener: Optional[ListenerAgent] = None, rng=None) -> Tuple[int, int]:
        d = self.clue_dist(listener)
        c = int(rng.choice(len(d), p=d)) if rng is not None else int(np.argmax(d))
        miss = [t for t in self.remaining if not listener or t not in listener.known]
        count = max(1, len(miss))
        return c, count

    def observe(self, guess: int, correct: bool):
        if correct and guess in self.remaining:
            self.remaining.remove(guess)     # consumed: latent shifts, moving target


# ---------------------------------------------------------------------------
# recovery metrics (need ground truth T)
# ---------------------------------------------------------------------------
def target_mass(belief: np.ndarray, targets) -> float:
    """Fraction of belief mass on the true targets (belief normalized over board)."""
    b = normalize(belief)
    return float(b[list(targets)].sum())


def recovery_f1(belief: np.ndarray, targets, k: int) -> float:
    pred = set(np.argsort(belief)[::-1][:k].tolist())
    tgt = set(targets)
    if not pred or not tgt:
        return 0.0
    tp = len(pred & tgt)
    prec = tp / len(pred); rec = tp / len(tgt)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
