"""Logging and per-run manifests."""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config, git_commit


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "torch", "transformers"]:
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "?")
        except Exception:
            versions[mod] = "not-installed"
    return versions


def _gpu_info() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
            }
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return {"cuda": None, "device": "mps"}
    except Exception:
        pass
    return {"cuda": None, "device": "cpu"}


def write_manifest(
    cfg: Config,
    stage: str,
    *,
    started: str,
    artifact_paths: list[str],
    completed_jobs: int = 0,
    failed_jobs: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one manifest per (config, stage) run."""
    manifest = {
        "stage": stage,
        "config_name": cfg.name,
        "config_hash": cfg.config_hash(),
        "resolved_config": cfg.model_dump(mode="json"),
        "git_commit": git_commit(),
        "package_versions": _package_versions(),
        "gpu": _gpu_info(),
        "model_id": cfg.model.model_id,
        "model_revision": cfg.model.revision,
        "tokenizer_id": cfg.model.resolved_tokenizer_id,
        "quantization_mode": "4bit" if cfg.model.load_in_4bit else cfg.model.dtype,
        "started": started,
        "ended": now_iso(),
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "artifact_paths": artifact_paths,
    }
    if extra:
        manifest.update(extra)
    out = cfg.paths.manifests / f"{stage}.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    return out


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
