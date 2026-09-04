"""Resume idempotency and hash stability (Part 24: 14, 15)."""

from __future__ import annotations

import shutil

import pandas as pd

from belief_feedback.config import Config, load_config
from belief_feedback.experiments import network_runner
from belief_feedback.paths import REPO_ROOT
from belief_feedback.world.generator import generate_all_worlds


def _tiny_cfg() -> Config:
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.name = "test_resume"
    cfg.worlds.splits = {
        "steering_train": 2,
        "steering_validation": 2,
        "exogenous_train": 2,
        "exogenous_validation": 2,
        "exogenous_test": 2,
        "endogenous_test": 2,
        "recycling_test": 2,
        "hysteresis_test": 2,
        "phase_boundary_test": 0,
        "robustness_test": 2,
    }
    cfg.worlds.phase_worlds_per_cell = 0
    cfg.endogenous_conditions = ["baseline", "positive_impulse"]
    return cfg


def test_config_hash_stable():
    a = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    b = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    assert a.config_hash() == b.config_hash()
    c = a.model_copy(deep=True)
    c.network.rounds += 1
    assert c.config_hash() != a.config_hash()


def test_dataset_hash_stable():
    from belief_feedback.world.generator import build_ordinary_world, dataset_hash

    cfg = _tiny_cfg()
    w1 = {w.world_id: w for w in [build_ordinary_world(cfg, "w_h_0", "exogenous_train", 0)]}
    w2 = {w.world_id: w for w in [build_ordinary_world(cfg, "w_h_0", "exogenous_train", 0)]}
    assert dataset_hash(w1) == dataset_hash(w2)


def test_network_resume_does_not_duplicate_rows(tmp_path, monkeypatch):
    cfg = _tiny_cfg()
    for sub in ("data", "runs", "vectors", "manifests", "activations", "models"):
        shutil.rmtree(REPO_ROOT / "artifacts" / sub / cfg.name, ignore_errors=True)
    generate_all_worlds(cfg)
    from belief_feedback.experiments import steering_calibration, steering_dataset

    steering_dataset.run(cfg)
    steering_calibration.run(cfg)
    network_runner.run(cfg)
    df1 = pd.read_parquet(cfg.paths.runs / "belief_states.parquet")

    # simulate an interrupted rerun: final outputs missing, parts intact
    (cfg.paths.runs / "belief_states.parquet").unlink()
    network_runner.run(cfg)
    df2 = pd.read_parquet(cfg.paths.runs / "belief_states.parquet")
    assert len(df1) == len(df2)
    key = ["world_id", "replicate_seed", "branch", "agent_id", "round"]
    assert not df2.duplicated(subset=key).any()

    # full rerun with outputs present is a no-op
    network_runner.run(cfg)
    df3 = pd.read_parquet(cfg.paths.runs / "belief_states.parquet")
    assert len(df3) == len(df1)
    for sub in ("data", "runs", "vectors", "manifests", "activations", "models"):
        shutil.rmtree(REPO_ROOT / "artifacts" / sub / cfg.name, ignore_errors=True)
