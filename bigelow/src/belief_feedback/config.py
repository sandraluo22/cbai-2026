"""Configuration models and loading.

A single YAML file fully specifies a run size (smoke / pilot / full / ...).
The configuration hash stamps every result row so any artifact can be traced
back to the exact configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .paths import REPO_ROOT, ArtifactPaths

SPLITS = [
    "steering_train",
    "steering_validation",
    "exogenous_train",
    "exogenous_validation",
    "exogenous_test",
    "endogenous_test",
    "recycling_test",
    "hysteresis_test",
    "phase_boundary_test",
    "robustness_test",
]

TEST_SPLITS = {
    "exogenous_test",
    "endogenous_test",
    "recycling_test",
    "hysteresis_test",
    "phase_boundary_test",
    "robustness_test",
}

ENDOGENOUS_CONDITIONS = [
    "baseline",
    "positive_impulse",
    "negative_impulse",
    "positive_persistent",
    "negative_persistent",
    "positive_one_hop",
    "negative_one_hop",
    "positive_no_return",
    "negative_no_return",
    "positive_full_text_clamp",
    "negative_full_text_clamp",
    "fixed_replay_positive",
    "fixed_replay_negative",
]


class ModelConfig(BaseModel):
    backend: Literal["mock", "hf"] = "mock"
    model_id: str = "mock/deterministic-agent"
    tokenizer_id: str | None = None
    revision: str = "main"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    load_in_4bit: bool = False
    device: str = "auto"
    batch_size: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 220
    # Deliberate fallback substitution must be explicit and lands in manifests.
    fallback_model_id: str | None = None

    @property
    def resolved_tokenizer_id(self) -> str:
        return self.tokenizer_id or self.model_id

    @property
    def slug(self) -> str:
        return self.model_id.replace("/", "__")


class SteeringConfig(BaseModel):
    layer: int | None = None  # None -> selected by the layer scan
    scope: Literal["final_token_and_generation", "all_tokens"] = "final_token_and_generation"
    layer_scan_magnitudes: list[float] = Field(default_factory=lambda: [-1.0, 0.0, 1.0])
    magnitude_scan: list[float] = Field(
        default_factory=lambda: [round(-4.0 + 0.5 * i, 1) for i in range(17)]
    )
    caa_train_pairs: int = 16
    caa_val_pairs: int = 8


class NetworkConfig(BaseModel):
    n_agents: int = 8
    rounds: int = 6
    hysteresis_rounds: int = 8
    topology: Literal["ring", "star", "complete"] = "ring"
    replicate_seeds: list[int] = Field(default_factory=lambda: [11])


class WorldsConfig(BaseModel):
    splits: dict[str, int] = Field(default_factory=dict)
    reports_per_agent: int = 2
    phase_bins: list[float] = Field(default_factory=lambda: [-4.0, -2.0, 0.0, 2.0, 4.0])
    phase_worlds_per_cell: int = 4
    # rejection-sampling constraints for ordinary worlds
    max_network_abs_log_odds: float = 5.0
    max_agent_abs_log_odds: float = 3.5
    max_agent_evidence_share: float = 0.35
    min_uncertain_agents: int = 4


class ExogenousConfig(BaseModel):
    emission_train: int = 32
    emission_validation: int = 16
    emission_test: int = 16
    receiver_train: int = 64
    receiver_validation: int = 32
    receiver_test: int = 32


class AnalysisConfig(BaseModel):
    rollout_samples: int = 20
    bootstrap_samples: int = 100
    jacobian_worlds: int = 2
    # None: intervene at every round (full config); k: rounds 1..k only
    jacobian_rounds: int | None = None
    mechanistic_worlds: int = 2
    robustness_worlds: int = 2
    phase_steering_fracs: list[float] = Field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])


class Config(BaseModel):
    name: str = "smoke"
    label: str = ""  # e.g. "MOCK SMOKE TEST" stamped on all figures
    quantized: bool = False
    model: ModelConfig = Field(default_factory=ModelConfig)
    steering: SteeringConfig = Field(default_factory=SteeringConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    worlds: WorldsConfig = Field(default_factory=WorldsConfig)
    exogenous: ExogenousConfig = Field(default_factory=ExogenousConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    endogenous_conditions: list[str] = Field(default_factory=lambda: list(ENDOGENOUS_CONDITIONS))

    @property
    def paths(self) -> ArtifactPaths:
        return ArtifactPaths(config_name=self.name)

    @property
    def is_mock(self) -> bool:
        return self.model.backend == "mock"

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def id_fields(cfg: Config) -> dict[str, str]:
    """Identifying fields stamped on every result row."""
    return {
        "config_hash": cfg.config_hash(),
        "git_commit": git_commit(),
        "model_id": cfg.model.model_id,
        "tokenizer_id": cfg.model.resolved_tokenizer_id,
    }
