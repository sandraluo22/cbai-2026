"""Typed configuration: pydantic models + YAML load + dotted CLI override.

A single YAML describes one *experiment*; ``sweep`` fields hold lists that
``sweep.py`` expands into a Cartesian product of concrete :class:`RunConfig`s.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class Arm(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class CouplingMode(str, Enum):
    PURE_QSG = "pure_qsg"      # Mode 1: simplex-only QSG, observation seeds x_i(0)
    TWO_LAYER = "two_layer"    # Mode 3: persistent private obs + public QSG label exchange
    REASONING_EXCHANGE = "reasoning_exchange"  # listener reads speaker's FULL reasoning, not a label


class ReadoutCanonical(str, Enum):
    FIRST_TOKEN = "first_token"
    LENGTH_NORM = "length_norm"


# --------------------------------------------------------------------------- #
# Sub-configs                                                                  #
# --------------------------------------------------------------------------- #
class ModelConfig(BaseModel):
    text_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    vision_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    vision_model_fallback: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    # Tiny model for CPU/MPS smoke tests (no GPU). Used only when smoke_test=True.
    smoke_text_model: str = "sshleifer/tiny-gpt2"


class ObservationConfig(BaseModel):
    """How the hidden object's partial information is handed to each agent."""

    # TEXT arm
    feature_subset_size: int = 3        # how many ground-truth features each agent sees
    distractor_features: int = 0        # optional misleading features per agent
    # IMAGE arm
    patch_grid: tuple[int, int] = (3, 3)
    patch_overlap: float = 0.0
    source_image_dir: Optional[str] = None

    @field_validator("patch_grid", mode="before")
    @classmethod
    def _coerce_grid(cls, v: Any) -> Any:
        if isinstance(v, list):
            return tuple(v)
        return v


class ReadoutConfig(BaseModel):
    canonical: ReadoutCanonical = ReadoutCanonical.LENGTH_NORM
    do_hard_readout: bool = True
    anchor_template: str = (
        "{private}\n\nBased on the clues, the object is most likely one of "
        "{candidate_list}.\nAnswer:"
    )
    hard_template: str = (
        "{private}\n\n"
        "Estimate how likely the hidden object is each candidate. Reply with ONLY a "
        "JSON object mapping EVERY candidate to a probability in [0,1] (summing to "
        "about 1). Candidates: {candidate_list}.\n"
        'Example: {{"elephant": 0.5, "snake": 0.2, "tree": 0.1, "wall": 0.1, '
        '"rope": 0.04, "fan": 0.03, "spear": 0.03}}\n'
        "JSON:"
    )
    # Speaker's free-text reasoning prompt (reasoning_exchange mode).
    reasoning_template: str = (
        "You are one of several blind observers, each sensing a different property of "
        "the SAME single hidden object. {own}\n"
        "The object is one of: {candidate_list}.\n"
        "{lean}In 2-3 sentences, reason about what the object could be, then end with "
        "a line 'My guess: <object>'.\nReasoning:"
    )
    reasoning_max_tokens: int = 160

    def template_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.anchor_template.encode())
        h.update(self.hard_template.encode())
        return h.hexdigest()[:16]


class ActivationConfig(BaseModel):
    capture: bool = True
    layers: str = "all"                 # "all" or comma-separated indices
    dtype: str = "float16"
    max_disk_gb: float = 50.0           # refuse-with-warning cap for the whole sweep
    store_format: str = "memmap"        # "memmap" | "npz"


class QSGConfig(BaseModel):
    n: int = 8                          # population size N
    alpha: float = 0.3                  # adaptation rate
    m: float = 1.0                      # bandwidth: 1=Hard, k=Top-k, inf=Soft
    rounds: int = 30
    interactions_per_round: Optional[int] = None  # default N inside engine
    topology: str = "well_mixed"        # hook for graph topology later
    consensus_threshold: float = 0.95   # U threshold for consensus time

    @field_validator("m", mode="before")
    @classmethod
    def _coerce_m(cls, v: Any) -> Any:
        if isinstance(v, str) and v.lower() in ("inf", "infinity", "soft"):
            return float("inf")
        return v


# --------------------------------------------------------------------------- #
# Single run                                                                   #
# --------------------------------------------------------------------------- #
class RunConfig(BaseModel):
    arm: Arm = Arm.TEXT
    coupling_mode: CouplingMode = CouplingMode.TWO_LAYER
    seed: int = 0
    object_name: Optional[str] = None   # None -> drawn from ontology by seed
    neutral_ablation: bool = False      # strip the private observation (drift null)
    smoke_test: bool = False            # use tiny model on CPU/MPS

    model: ModelConfig = Field(default_factory=ModelConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    readout: ReadoutConfig = Field(default_factory=ReadoutConfig)
    activations: ActivationConfig = Field(default_factory=ActivationConfig)
    qsg: QSGConfig = Field(default_factory=QSGConfig)

    ontology_path: str = "configs/ontology.yaml"
    output_root: str = "results"

    def run_id(self) -> str:
        key = (
            f"{self.arm.value}_{self.coupling_mode.value}_N{self.qsg.n}"
            f"_a{self.qsg.alpha}_m{self.qsg.m}_seed{self.seed}"
            f"{'_ablate' if self.neutral_ablation else ''}"
        )
        return key.replace(".", "p").replace("inf", "Soft")


# --------------------------------------------------------------------------- #
# Sweep                                                                        #
# --------------------------------------------------------------------------- #
class SweepConfig(BaseModel):
    """Lists expanded into a Cartesian product of :class:`RunConfig`s."""

    arms: list[Arm] = [Arm.TEXT]
    coupling_modes: list[CouplingMode] = [CouplingMode.PURE_QSG, CouplingMode.TWO_LAYER]
    n_values: list[int] = [8]
    alpha_values: list[float] = [0.3]
    m_values: list[float] = [1.0]
    seeds: list[int] = [0]
    neutral_ablation_values: list[bool] = [False]

    base: RunConfig = Field(default_factory=RunConfig)

    @field_validator("m_values", mode="before")
    @classmethod
    def _coerce_ms(cls, v: Any) -> Any:
        out = []
        for item in v:
            if isinstance(item, str) and item.lower() in ("inf", "infinity", "soft"):
                out.append(float("inf"))
            else:
                out.append(float(item))
        return out

    def expand(self) -> list[RunConfig]:
        runs: list[RunConfig] = []
        for arm in self.arms:
            for mode in self.coupling_modes:
                for n in self.n_values:
                    for alpha in self.alpha_values:
                        for m in self.m_values:
                            for ablate in self.neutral_ablation_values:
                                for seed in self.seeds:
                                    rc = self.base.model_copy(deep=True)
                                    rc.arm = arm
                                    rc.coupling_mode = mode
                                    rc.seed = seed
                                    rc.neutral_ablation = ablate
                                    rc.qsg.n = n
                                    rc.qsg.alpha = alpha
                                    rc.qsg.m = m
                                    runs.append(rc)
        return runs


# --------------------------------------------------------------------------- #
# Loading + CLI override                                                       #
# --------------------------------------------------------------------------- #
def _apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` dotted overrides (values parsed as YAML scalars)."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        val = yaml.safe_load(raw)
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return data


def load_sweep_config(path: str | Path, overrides: Optional[list[str]] = None) -> SweepConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        data = _apply_overrides(data, overrides)
    return SweepConfig.model_validate(data)


def load_run_config(path: str | Path, overrides: Optional[list[str]] = None) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        data = _apply_overrides(data, overrides)
    return RunConfig.model_validate(data)


def dump_config(cfg: BaseModel, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(json.loads(cfg.model_dump_json()), sort_keys=False))
