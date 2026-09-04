"""Deterministic mock backend.

A synthetic agent whose belief is a configurable function of the text it can
see, so that every pipeline operation — probes, memo generation, steering,
activation collection, branching, and malformed-output handling — can be
exercised without a GPU.

Mechanics (all deterministic given text, seed, and steering):

* Evidence: the belief sums the semantic LLR of every latent event whose
  registered report ids appear in the visible text, with a partial
  double-count of repeated reports of one event (factor ``1 + kappa*(n-1)``)
  — the mock deliberately recycles evidence. The provenance-aware
  instruction marker reduces ``kappa``.
* Messages: each parsed "Current assessment" header contributes its
  confidence-weighted semantic stance.
* Steering: adding ``m * d`` at layer ``l`` shifts belief by
  ``m * (d . e0) * layer_beta[l]``, where ``e0`` is the mock's latent belief
  direction; extreme shifts degrade formatting coherence (so magnitude
  calibration has a real ceiling to find).
* Activations: component 0 of the residual stream carries the belief with a
  layer-specific scale; other components are reproducible noise.
"""

from __future__ import annotations

import math
import re
import time

import numpy as np

from ..config import Config
from ..seeds import rng as make_rng
from ..world.schema import World
from .base import Backend
from .generation import GenerationResult, Message, ScoreResult, prompt_hash
from .steering import SteeringSpec

PROVENANCE_MARKER = "not independent corroboration"
ASSESS_RE = re.compile(r"Current assessment:\s*(ALPHA|BETA|UNRESOLVED)", re.IGNORECASE)
CONF_RE = re.compile(r"Confidence:\s*(\d{1,3})")

LAYER_BETA = [0.10, 0.50, 0.90, 0.40]  # behavioral steering gain per layer
LAYER_SCALE = [0.30, 0.70, 1.00, 0.60]  # belief readout scale per layer


class MockBackend(Backend):
    """See module docstring."""

    def __init__(
        self,
        cfg: Config,
        *,
        kappa: float = 0.55,
        kappa_provenance: float = 0.15,
        msg_weight: float = 0.35,
        squash: float = 3.2,
        p_malformed: float = 0.03,
    ) -> None:
        self.cfg = cfg
        self.model_id = cfg.model.model_id
        self.n_layers = 4
        self.hidden_size = 16
        self.kappa = kappa
        self.kappa_provenance = kappa_provenance
        self.msg_weight = msg_weight
        self.squash = squash
        self.p_malformed = p_malformed
        # report_id -> (world_id, event_id, semantic llr)
        self._reports: dict[str, tuple[str, str, float]] = {}
        self._alpha_is_upstream: dict[str, bool] = {}

    # ------------------------------------------------------------------
    def register_world(self, world: World) -> None:
        self._alpha_is_upstream[world.world_id] = world.alpha_is_upstream
        for rep in world.reports:
            ev = world.event(rep.event_id)
            self._reports[rep.report_id] = (world.world_id, ev.event_id, ev.llr)

    # ------------------------------------------------------------------
    def _visible_rids(self, text: str) -> list[str]:
        return [rid for rid in self._reports if rid in text]

    def _world_of_text(self, text: str) -> str | None:
        counts: dict[str, int] = {}
        for rid in self._visible_rids(text):
            counts[self._reports[rid][0]] = counts.get(self._reports[rid][0], 0) + 1
        return max(counts, key=lambda k: counts[k]) if counts else None

    def _steer_contribution(self, steering: SteeringSpec | None, current_belief: float) -> float:
        if steering is None or not steering.active:
            return 0.0
        e0 = np.zeros(self.hidden_size)
        e0[0] = 1.0
        beta = LAYER_BETA[steering.layer % self.n_layers]
        if steering.mode == "add":
            return float(steering.magnitude * (steering.vector @ e0) * beta)
        # project_set: pull belief toward the target projection value
        scale = LAYER_SCALE[steering.layer % self.n_layers]
        current_proj = current_belief * scale
        return float((steering.project_value - current_proj) / max(scale, 1e-6) * 0.8)

    def belief_semantic(
        self, messages: list[Message], steering: SteeringSpec | None = None
    ) -> float:
        """The mock's latent semantic log odds for a context."""
        text = "\n".join(m["content"] for m in messages)
        kappa = self.kappa_provenance if PROVENANCE_MARKER in text else self.kappa
        world_id = self._world_of_text(text)
        by_event: dict[str, list[float]] = {}
        for rid in self._visible_rids(text):
            _, eid, llr = self._reports[rid]
            by_event.setdefault(eid, []).append(llr)
        evidence = sum(llrs[0] * (1.0 + kappa * (len(llrs) - 1)) for llrs in by_event.values())

        msg_part = 0.0
        if world_id is not None:
            alpha_up = self._alpha_is_upstream[world_id]
            stances = ASSESS_RE.findall(text)
            confs = [int(c) for c in CONF_RE.findall(text)]
            for i, s in enumerate(stances):
                s = s.upper()
                if s == "UNRESOLVED":
                    continue
                sign_vis = 1.0 if s == "ALPHA" else -1.0
                sign_sem = sign_vis if alpha_up else -sign_vis
                conf = confs[i] / 100.0 if i < len(confs) else 0.5
                msg_part += self.msg_weight * sign_sem * conf

        raw = evidence + msg_part
        raw += self._steer_contribution(steering, raw)
        noise = make_rng("mock_belief", prompt_hash(messages)).normal(0.0, 0.15)
        raw += float(noise)
        return float(self.squash * math.tanh(raw / self.squash))

    # ------------------------------------------------------------------
    def score_choices(
        self,
        messages: list[Message],
        choices: list[str],
        steering: SteeringSpec | None = None,
    ) -> ScoreResult:
        text = "\n".join(m["content"] for m in messages)
        world_id = self._world_of_text(text)
        alpha_up = self._alpha_is_upstream.get(world_id, True) if world_id else True
        b_sem = self.belief_semantic(messages, steering)
        b_vis = b_sem if alpha_up else -b_sem
        logps, norms, counts = [], [], []
        for choice in choices:
            n_tok = max(1, len(choice.split()))
            label = choice.strip().upper()
            if label == "ALPHA":
                lp = -math.log1p(math.exp(-b_vis)) if b_vis > -30 else b_vis
            elif label == "BETA":
                lp = -math.log1p(math.exp(b_vis)) if b_vis < 30 else -b_vis
            else:  # arbitrary completion: length-penalized neutral score
                lp = -1.5 * n_tok
            logps.append(float(lp))
            norms.append(float(lp) / n_tok)
            counts.append(n_tok)
        return ScoreResult(logps=logps, logps_normalized=norms, token_counts=counts)

    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        seed: int,
        steering: SteeringSpec | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> GenerationResult:
        t0 = time.time()
        temperature = self.cfg.model.temperature if temperature is None else temperature
        text = "\n".join(m["content"] for m in messages)
        world_id = self._world_of_text(text)
        alpha_up = self._alpha_is_upstream.get(world_id, True) if world_id else True
        r = make_rng("mock_gen", seed)
        b_sem = self.belief_semantic(messages, steering)
        b_sem += float(r.normal(0.0, 0.25 * temperature))
        b_vis = b_sem if alpha_up else -b_sem

        steer_abs = abs(self._steer_contribution(steering, b_sem))
        p_bad = self.p_malformed + max(0.0, steer_abs - 3.0) * 0.4
        malformed = bool(r.random() < p_bad)

        rids = self._visible_rids(text)
        aligned = [rid for rid in rids if self._reports[rid][2] * b_sem >= 0]
        pool = aligned if aligned else rids
        n_cite = int(min(len(pool), 1 + r.integers(0, 3)))
        cites = list(r.choice(pool, size=n_cite, replace=False)) if n_cite else []
        if r.random() < 0.03:
            cites.append(f"R-w_phantom_{int(r.integers(0, 999)):04d}-000")

        if abs(b_vis) < 0.35:
            assessment = "UNRESOLVED"
        else:
            assessment = "ALPHA" if b_vis > 0 else "BETA"
        confidence = int(np.clip(25 + 22 * abs(b_vis) + r.normal(0, 4), 3, 97))

        body = _memo_body(assessment, confidence, cites, b_vis, r)
        if steer_abs > 2.5:  # repetition creeps in near the coherence ceiling
            body += " The evidence is consistent. The evidence is consistent."
        if malformed:
            out_text = _malformed_text(r, body)
        else:
            out_text = (
                f"Current assessment: {assessment}\n"
                f"Confidence: {confidence}\n"
                f"Evidence cited: {', '.join(cites) if cites else 'none'}\n"
                f"Memo: {body}\n"
                f"Request to team: {_request_line(r)}"
            )
        return GenerationResult(
            text=out_text,
            seed=seed,
            prompt_hash=prompt_hash(messages),
            context_token_count=len(text.split()),
            generation_token_count=len(out_text.split()),
            wall_time=time.time() - t0,
            peak_gpu_memory=0.0,
        )

    # ------------------------------------------------------------------
    def get_activations(
        self,
        messages: list[Message],
        layers: list[int] | None = None,
        steering: SteeringSpec | None = None,
    ) -> np.ndarray:
        b_sem = self.belief_semantic(messages, steering)
        r = make_rng("mock_act", prompt_hash(messages))
        acts = r.normal(0.0, 0.30, size=(self.n_layers, self.hidden_size))
        for line in range(self.n_layers):
            acts[line, 0] = LAYER_SCALE[line] * b_sem + r.normal(0.0, 0.05)
        if steering is not None and steering.active and steering.mode == "add":
            acts[steering.layer % self.n_layers] += steering.magnitude * steering.vector
        if layers is not None:
            acts = acts[layers]
        return acts


def _memo_body(assessment: str, confidence: int, cites: list[str], b_vis: float, r) -> str:
    lean = {
        "ALPHA": "the evidence currently favors hypothesis ALPHA",
        "BETA": "the evidence currently favors hypothesis BETA",
        "UNRESOLVED": "the evidence does not yet separate the two hypotheses",
    }[assessment]
    cite_phrase = ", ".join(cites[:3]) if cites else "the records reviewed so far"
    openers = [
        "After reviewing the material available to me,",
        "Based on my records and the messages received,",
        "Taking the current file together,",
        "On the balance of what I can see,",
    ]
    middles = [
        f"my reading is that {lean}. The strongest items in my view are {cite_phrase}, which point in a consistent direction when weighed against the remaining records.",
        f"I conclude that {lean}. I rest this mainly on {cite_phrase}; the other documents I hold neither confirm nor contradict this reading in a decisive way.",
        f"my working position is that {lean}. In particular {cite_phrase} carry the most weight for me, and I have tried to discount weaker or ambiguous entries accordingly.",
    ]
    closers = [
        "I will revise this position if incoming messages surface materially new observations or contradict the sources I have cited here.",
        "I remain open to revising this assessment as neighboring specialists report their own findings in later rounds.",
        "Further corroboration from an independent line of evidence would raise my confidence; contradiction would lower it.",
    ]
    return f"{r.choice(openers)} {r.choice(middles)} {r.choice(closers)}"


def _request_line(r) -> str:
    reqs = [
        "Please share any measurement that bears directly on the fill-volume question.",
        "Can anyone confirm whether their sources are independent of the ones I cited?",
        "I would value a check of the supplier-lot records from a second source.",
        "Please flag any station-level telemetry I may not have seen.",
    ]
    return str(r.choice(reqs))


def _malformed_text(r, body: str) -> str:
    modes = [
        lambda: body,  # header entirely missing
        lambda: f"Assessment: maybe ALPHA?? {body[:120]}",  # wrong header format
        lambda: "I think we need more data before I can commit to a structured answer. " + body[:80],
        lambda: f"Current assessment: BOTH\nConfidence: high\nMemo: {body[:100]}",
    ]
    return str(modes[int(r.integers(0, len(modes)))]())
