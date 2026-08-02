"""Experiment configuration for the negotiation / opponent-latent experiment.

Two instances of the SAME open-weight chat model play repeated negotiation
(split 100 points per round; B proposes, A accepts or counters). B carries a
hidden continuous latent alpha in [0, 1] ("greediness"):

  - tier 1 : alpha is a number in B's system prompt (latent is in-context)
  - tier 2 : alpha is a steering-vector coefficient added to B's residual
             stream (latent lives in B's activations, not its prompt)

After every round we cache A's residual stream (last token, ALL layers) and
later train ridge probes to predict alpha from A's activations, per layer per
turn. Controls: transcript-shadow observer, text-only baselines, verbalized
guesses. Upgrades: causal steering of A along the probe direction, and the
same-space test cos(v, w) between B's greed direction and A's estimate-of-B
direction (enabled by tier 2 + same weights).

A config is a frozen dataclass. Presets:
  - DEFAULT : Llama-3.1-8B-Instruct, tier 1, 400 episodes (24GB card, bf16)
  - TIER2   : same but alpha applied as a steering coefficient
  - GEMMA   : Gemma-2-9b-it variant of DEFAULT
  - SMOKE   : tiny CPU preset (stub model) to exercise the plumbing end-to-end
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple
import json
import os


@dataclass(frozen=True)
class Config:
    name: str = "default"

    # ---- reproducibility -------------------------------------------------
    seed: int = 0

    # ---- model (ONE model; both agents + observer share weights) ----------
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"
    use_stub_model: bool = False       # SMOKE only: no chat template, tiny

    # ---- game ------------------------------------------------------------
    pie: int = 100                     # points to split each round
    n_rounds: int = 10                 # rounds per episode (fixed; no early end)
    n_episodes: int = 400
    # Sampling temperature for ALL in-game generations. Kept >= 0.7 on purpose:
    # alpha must shift the DISTRIBUTION of B's offers, not determine them. If
    # alpha were deducible from any single offer the transcript would trivially
    # contain it and A's internals would have nothing to add. Noise creates the
    # inference problem -- A has to accumulate evidence about B over rounds.
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 60           # per in-game utterance (short by design)

    # ---- latent ----------------------------------------------------------
    tier: int = 1                      # 1 = alpha in B's system prompt; 2 = steering
    # alpha ~ U(alpha_lo, alpha_hi), one draw per episode
    alpha_lo: float = 0.0
    alpha_hi: float = 1.0

    # ---- tier-2 steering (B) ----------------------------------------------
    # Greed direction v is extracted from contrastive prompt pairs (mean diff of
    # last-token residuals, unit norm, per layer). During B's forward passes we
    # add  steer_coef(alpha) * v  to the residual stream at steer_layers
    # (decoder-block indices). With steer_center=True (default),
    # steer_coef(alpha) = steer_scale * (2*alpha - 1) in [-scale, +scale]:
    # negative pushes generous, positive greedy -- this doubles the behavioral
    # range vs the one-sided alpha*scale mapping, because the unsteered model
    # already negotiates greedily (opens ~70/30). Calibrated on Llama-3.1-8B:
    # coherent in [-3, +2.5]; coef -2 -> B opens ~45/55, 0 -> ~70/30,
    # +2 -> ~90/10. Recalibrate per model:
    # `python src/steering.py --preset tier2 --calibrate`.
    steer_layers: Tuple[int, ...] = (12, 14, 16, 18, 20)
    steer_scale: float = 2.0
    steer_center: bool = True
    # Round-level jitter: each round B's effective coefficient is
    #   clip(steer_coef(alpha) + N(0, steer_noise_sd), +/- steer_coef_max).
    # Without it (tier2 v0) the dose-response is so clean that B's FIRST offer
    # reveals alpha (tf-idf R2 = 0.90 at turn 1) and the inference problem
    # vanishes -- the exact single-offer-deducibility failure the design
    # warns about. Jitter makes alpha shift the DISTRIBUTION of offers;
    # A must average over rounds to recover it. steer_coef_max keeps the
    # jittered coefficient inside the calibrated coherence band [-3, +3].
    steer_noise_sd: float = 1.2
    steer_coef_max: float = 3.0
    n_contrast_pairs: int = 24         # prompt pairs for direction extraction

    def steer_coef(self, alpha: float) -> float:
        if self.steer_center:
            return self.steer_scale * (2.0 * alpha - 1.0)
        return self.steer_scale * alpha

    # ---- capture -----------------------------------------------------------
    # After each round (once A has acted) we run one forward pass over A's full
    # context and keep hidden_states[k][last token] for ALL k in 0..n_layers
    # (index 0 = embeddings, k = output of block k-1). fp16, sharded npz.
    shard_size: int = 50               # episodes per activation shard

    # ---- verbalized-guess control ------------------------------------------
    # At these rounds, fork A's context (side branch; never appended back) and
    # ask A to estimate B's greediness 0-100. (0,) disables.
    verbalize_rounds: Tuple[int, ...] = (2, 4, 6, 8, 10)

    # ---- probes ------------------------------------------------------------
    ridge_alphas: Tuple[float, ...] = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
    # Separate grid for the LOW-DIMENSIONAL text baselines (8 behavioral
    # features / sparse tf-idf): the activation-scale penalties above crush an
    # 8-feature regression into a constant (v1 lesson: baseline underfit).
    baseline_ridge_alphas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1e0, 1e1,
                                                1e2, 1e3)
    test_frac: float = 0.25            # held out BY EPISODE

    # ---- causal step -------------------------------------------------------
    causal_alpha: float = 0.5          # fixed mid-alpha opponent
    # Dose-response along the probe direction w. v1/v2 used +/-8, which is
    # outside the coherence band found in steering calibration (|coef| <~ 3)
    # and showed degradation signatures (non-monotone demand shifts).
    causal_gammas: Tuple[float, ...] = (-3.0, -1.5, 0.0, 1.5, 3.0)
    causal_layer: int = -1             # hidden_states index; -1 = best probe layer
    causal_episodes: int = 40          # per gamma

    # ---- io ----------------------------------------------------------------
    out_dir: str = "runs"

    def run_dir(self) -> str:
        d = os.path.join(self.out_dir, self.name)
        os.makedirs(d, exist_ok=True)
        return d

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


DEFAULT = Config(name="default")

TIER2 = Config(name="tier2", tier=2)

GEMMA = Config(
    name="gemma",
    model_name="google/gemma-2-9b-it",
    # Gemma-2-9b-it has 42 blocks; steer a mid band.
    steer_layers=(16, 19, 22, 25, 28),
)

GEMMA_TIER2 = Config(
    name="gemma_tier2",
    model_name="google/gemma-2-9b-it",
    tier=2,
    steer_layers=(16, 19, 22, 25, 28),
)

# Tiny end-to-end test: a small public model with no chat template (a plain
#"System:/User:/Assistant:" fallback formatter is used) on CPU. The stub can't
# follow the offer format, so almost every move takes the scripted fallback
# path -- which is fine: SMOKE tests the plumbing (game loop -> capture ->
# shards -> probes -> plots), not the science. Fallback B-offers are drawn from
# an alpha-dependent noisy distribution precisely so the probes have a real
# signal to find and the sanity check is meaningful.
SMOKE = Config(
    name="smoke",
    model_name="distilgpt2",
    use_stub_model=True,
    dtype="float32",
    device="cpu",
    n_rounds=4,
    n_episodes=24,
    shard_size=8,
    max_new_tokens=24,
    steer_layers=(2, 3),
    steer_scale=2.0,
    n_contrast_pairs=4,
    verbalize_rounds=(2, 4),
    causal_gammas=(-2.0, 0.0, 2.0),
    causal_episodes=4,
    test_frac=0.34,
)

PRESETS = {
    "default": DEFAULT,
    "tier2": TIER2,
    "gemma": GEMMA,
    "gemma_tier2": GEMMA_TIER2,
    "smoke": SMOKE,
}


def get_config(name: str) -> Config:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    cfg = PRESETS[name]
    # NEGOTIATION_MODEL overrides the checkpoint without touching presets --
    # e.g. an ungated mirror of the same weights when the canonical repo is
    # gated and no HF token is available:
    #   NEGOTIATION_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct
    # Same weights -> same run_dir; results stay comparable across the
    # canonical repo and its mirrors.
    override = os.environ.get("NEGOTIATION_MODEL")
    if override:
        from dataclasses import replace
        cfg = replace(cfg, model_name=override)
    return cfg
