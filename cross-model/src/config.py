"""Experiment configuration for cross-model alignment of in-context graph
representations.

A config is a frozen dataclass. Two presets are provided:
  - DEFAULT : the real run (H200, full-size models, ~1000-token walks)
  - SMOKE   : a tiny CPU-friendly preset for testing the plumbing end-to-end
              without GPUs or gated model downloads (uses `MODEL_STUB`).

Everything that affects results is here and is seeded. `run.py` selects a
preset by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import json


# ---------------------------------------------------------------------------
# Fixed concept-word vocabulary.
#
# Common, *semantically unrelated* words so there is no competing pretrained
# prior tying nodes together (this is the "plain random walk" condition of the
# paper, NOT the semantic-conflict / days-of-week condition). The list is long
# enough to support grids larger than 4x4; assignment is index-order and fixed,
# so node i always gets WORDS[i] in BOTH models.
# ---------------------------------------------------------------------------
WORDS: List[str] = [
    "apple", "bird", "sand", "math", "chair", "river", "music", "glass",
    "cloud", "knife", "paper", "tiger", "plant", "stone", "bread", "clock",
    "wheel", "ocean", "flame", "brush", "coin", "lemon", "horse", "table",
    "ghost", "robot", "candle", "pencil", "garden", "planet", "button",
    "mirror", "jacket", "rocket", "pillow", "anchor",
]

# Semantic-conflict condition (days-of-week): the nodes are the 7 weekdays, which
# carry a STRONG pretrained cyclic order. We arrange them on the 7-ring in a
# PERMUTED order (each ring step = +3 days) so the in-context ring CONFLICTS with
# the natural weekday cycle -- the paper's test of context overriding semantics.
DAYS: List[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                   "Saturday", "Sunday"]
DAYS_PERMUTED: List[str] = [DAYS[(3 * i) % 7] for i in range(7)]   # Mon,Thu,Sun,...


@dataclass(frozen=True)
class Config:
    name: str = "default"

    # ---- reproducibility -------------------------------------------------
    seed: int = 0

    # ---- graph -----------------------------------------------------------
    graph_type: str = "grid"           # "grid" | "ring" | "hex"
    word_set: str = "concepts"         # "concepts" (unrelated) | "days" (semantic conflict)
    grid_rows: int = 4
    grid_cols: int = 4                 # grid: n_nodes = rows * cols
    ring_size: int = 16                # ring: n_nodes = ring_size
    hex_rows: int = 4
    hex_cols: int = 4                  # hex: n_nodes = hex_rows * hex_cols

    # ---- walk generation -------------------------------------------------
    # We generate walks of `walk_length` *node steps*. Each step emits one
    # concept word -> one occurrence. "Context length" of an occurrence is the
    # number of nodes emitted up to and including it (its 1-based step), which
    # is identical across models because both run on the SAME word sequence.
    # That shared word-step axis is what makes the matched/mismatched-context
    # control (align.py) comparable despite tokenizer differences.
    walk_length: int = 1000            # node steps per walk
    n_walks: int = 200                 # each starts at a (cycled) distinct node
    # Word-step checkpoints at which alignment is evaluated. Roughly the paper's
    # ~10/30/100/300/1000 *token* lengths (1 word ~ 1 token for these vocab).
    context_checkpoints: Tuple[int, ...] = (10, 30, 100, 300, 1000)
    # Relative half-width of the bin around each checkpoint when selecting
    # occurrences for per-context-length evaluation (0.2 -> [0.8C, 1.2C]).
    checkpoint_window: float = 0.2

    # ---- models ----------------------------------------------------------
    model_a: str = "meta-llama/Llama-3.1-8B"
    model_b: str = "google/gemma-2-9b"
    dtype: str = "bfloat16"            # bf16 weights; activations cached as fp16
    device: str = "cuda"
    # Layers whose post-block residual stream we capture (forward hooks).
    # Capture a band around the alignment layer so reproduce.py / layer sweeps
    # can reuse the same cache without re-inference. Indices are 0-based decoder
    # blocks. The paper uses ~layer 26 / last for Llama.
    capture_layers_a: Tuple[int, ...] = (16, 20, 24, 26, 28, 31)
    capture_layers_b: Tuple[int, ...] = (16, 20, 24, 28, 32, 41)
    # The single layer pair used for the headline alignment analysis.
    align_layer_a: int = 26
    align_layer_b: int = 32

    # Tokenizer-alignment rule for multi-subword concept words.
    # CONFIRMED WITH USER: "last" -> use the concept's final subword token.
    subword_rule: str = "last"        # one of: "last", "first", "mean"

    # ---- alignment / metrics --------------------------------------------
    pca_k: int = 100                   # shared subspace dim for Procrustes (b)
    ridge_alpha: float = 1e3           # regularization for full-space ridge (a)
    test_frac: float = 0.25            # held-out fraction, split BY walk_id
    wellposed_ratio: float = 10.0      # warn if n_samples < ratio * n_params

    # ---- capture scheduling ---------------------------------------------
    # When True AND the device has >= parallel_min_gb total VRAM, load BOTH
    # models at once and run each walk through both in a single shared pass
    # (co-resident). Otherwise fall back to sequential load/free + disk cache.
    # On the H200 (141GB) both 8-9B bf16 models (~16+18.5GB) fit with headroom.
    parallel_models: bool = False
    parallel_min_gb: float = 50.0
    # When True, delete each captured model's HF weight cache before loading the
    # next one. Needed when disk/quota can't hold both models at once (forces the
    # sequential path and trades re-downloadability for space).
    free_cache_between: bool = False

    # ---- io --------------------------------------------------------------
    out_dir: str = "runs"
    # Internal: set True only by SMOKE to swap in a tiny public stub model.
    use_stub_model: bool = False

    @property
    def n_nodes(self) -> int:
        if self.graph_type == "ring":
            return self.ring_size
        if self.graph_type == "hex":
            return self.hex_rows * self.hex_cols
        return self.grid_rows * self.grid_cols

    def words(self) -> List[str]:
        if self.word_set == "days":
            assert self.n_nodes == 7, "days condition uses the 7-node ring"
            return list(DAYS_PERMUTED)
        assert self.n_nodes <= len(WORDS), (
            f"need {self.n_nodes} words but only {len(WORDS)} defined"
        )
        return WORDS[: self.n_nodes]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# Real run: H200 (141GB). Both 8-9B models fit in bf16 simultaneously, but we
# still load/run them SEQUENTIALLY and cache activations to disk so alignment
# re-runs offline without re-inference (and so peak memory stays well bounded).
DEFAULT = Config(name="default", parallel_models=True)

# Tiny end-to-end test. Uses a small public model for BOTH "models" so the
# pipeline (graph -> capture -> align) is exercised on CPU in seconds. Hidden
# sizes are then equal, but the code path (rectangular map, PCA Procrustes,
# CKA, well-posedness guard) is identical.
SMOKE = Config(
    name="smoke",
    grid_rows=3,
    grid_cols=3,
    walk_length=60,
    n_walks=20,
    context_checkpoints=(5, 15, 40),
    model_a="distilgpt2",
    model_b="distilgpt2",
    dtype="float32",
    device="cpu",
    capture_layers_a=(3,),
    capture_layers_b=(3,),
    align_layer_a=3,
    align_layer_b=3,
    pca_k=10,
    ridge_alpha=1.0,
    use_stub_model=True,
)

# Gemma-2-9b (base) vs Qwen3-8B-Base. Used while Llama-3.1-8B access is pending.
# Both are base (non-instruct) models, so neither carries post-training priors.
# Hidden sizes still differ (3584 vs 4096) -> rectangular map exercised as in
# the faithful run. Deep layers chosen at ~0.77 relative depth in each:
#   Gemma: 42 layers -> 32 ;  Qwen3: 36 layers -> 28.
GEMMA_QWEN = Config(
    name="gemma_qwen",
    # Sequential + cache-cleanup: the network volume's quota can't hold both
    # fp32 Gemma (~35GB) and Qwen at once, so we capture one at a time and free
    # each model's weights before downloading the next. Activations go to a
    # local (out_dir) path because the volume is full once Gemma is present.
    parallel_models=False,
    free_cache_between=True,
    out_dir="/root/cmrun",
    model_a="google/gemma-2-9b",
    model_b="Qwen/Qwen3-8B-Base",
    capture_layers_a=(24, 28, 32, 36, 40, 41),   # Gemma-2-9b (0..41)
    capture_layers_b=(20, 24, 28, 32, 34, 35),   # Qwen3-8B-Base (0..35)
    align_layer_a=32,
    align_layer_b=28,
)

PRESETS = {"default": DEFAULT, "smoke": SMOKE, "gemma_qwen": GEMMA_QWEN}


def get_config(name: str) -> Config:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return PRESETS[name]
