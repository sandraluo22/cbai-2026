"""Shared experiment utilities: idempotent stage guards and steering IO."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..agents.protocol import SteeringContext
from ..config import Config
from ..logging_utils import get_logger

log = get_logger(__name__)


def outputs_exist(paths: list[Path]) -> bool:
    """True when every output exists and is nonempty (stage may be skipped)."""
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def save_df(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def steering_paths(cfg: Config) -> tuple[Path, Path]:
    d = cfg.paths.vectors / cfg.model.slug
    d.mkdir(parents=True, exist_ok=True)
    return d / "steering_vector.safetensors", d / "steering_metadata.json"


def save_steering(cfg: Config, vector: np.ndarray, metadata: dict) -> None:
    vec_path, meta_path = steering_paths(cfg)
    from safetensors.numpy import save_file

    save_file({"steering_vector": vector.astype(np.float32)}, str(vec_path))
    metadata = dict(metadata)
    metadata["vector_hash"] = f"{abs(hash(vector.tobytes())):016x}"
    meta_path.write_text(json.dumps(metadata, indent=2, default=float))


def load_steering(cfg: Config) -> tuple[SteeringContext, dict]:
    vec_path, meta_path = steering_paths(cfg)
    from safetensors.numpy import load_file

    vector = load_file(str(vec_path))["steering_vector"].astype(np.float64)
    meta = json.loads(meta_path.read_text())
    ctx = SteeringContext(vector=vector, layer=int(meta["layer"]), scope=cfg.steering.scope)
    return ctx, meta


def delta_from_meta(meta: dict) -> float:
    """Finite-difference / impulse magnitude: 0.5 * m_max."""
    return 0.5 * float(meta["m_max"])
