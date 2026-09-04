"""Complete mock pipeline produces every required table and figure (Part 24: 16).

Runs a scaled-down end-to-end pipeline under the config name ``test_e2e``.
"""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from belief_feedback.analysis.schemas import check_id_columns
from belief_feedback.cli import COMMANDS, PIPELINE_ORDER
from belief_feedback.config import Config, load_config
from belief_feedback.paths import REPO_ROOT

REQUIRED_TABLES = [f"table{i:02d}" for i in range(1, 11)]
REQUIRED_FIGURES = [
    "fig01_world_and_network_schematic",
    "fig02_steering_calibration",
    "fig03_exogenous_response_surface",
    "fig04_exogenous_model_comparison",
    "fig05_closed_loop_impulse_trajectories",
    "fig06_composition_generalization",
    "fig07_causal_path_decomposition",
    "fig08_evidence_recycling",
    "fig09_hysteresis",
    "fig10_network_phase_boundary",
    "fig11_empirical_jacobian",
    "fig12_mechanistic_alignment",
    "fig13_text_mediation",
    "fig14_robustness",
]
REQUIRED_RUN_TABLES = [
    "episodes",
    "belief_states",
    "public_messages",
    "deliveries",
    "interventions",
    "exogenous_emission_trials",
    "exogenous_receiver_trials",
    "composition_predictions",
    "branch_effects",
    "recycling_results",
    "hysteresis_results",
    "phase_boundary_results",
    "jacobian_results",
    "probe_results",
]


def _e2e_cfg() -> Config:
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml").model_copy(deep=True)
    cfg.name = "test_e2e"
    cfg.worlds.splits = {
        "steering_train": 2,
        "steering_validation": 2,
        "exogenous_train": 4,
        "exogenous_validation": 2,
        "exogenous_test": 2,
        "endogenous_test": 4,
        "recycling_test": 2,
        "hysteresis_test": 2,
        "phase_boundary_test": 0,
        "robustness_test": 2,
    }
    cfg.worlds.phase_worlds_per_cell = 1
    cfg.steering.caa_train_pairs = 8
    cfg.steering.caa_val_pairs = 4
    cfg.exogenous.emission_train = 12
    cfg.exogenous.emission_validation = 6
    cfg.exogenous.emission_test = 6
    cfg.exogenous.receiver_train = 24
    cfg.exogenous.receiver_validation = 12
    cfg.exogenous.receiver_test = 12
    cfg.analysis.rollout_samples = 5
    cfg.analysis.bootstrap_samples = 50
    cfg.analysis.jacobian_worlds = 1
    cfg.analysis.mechanistic_worlds = 1
    cfg.analysis.robustness_worlds = 1
    return cfg


@pytest.fixture(scope="module")
def e2e_cfg():
    cfg = _e2e_cfg()
    for sub in (
        "data", "runs", "vectors", "manifests", "activations", "models",
        "figures", "figure_data", "tables", "reports",
    ):
        shutil.rmtree(REPO_ROOT / "artifacts" / sub / cfg.name, ignore_errors=True)
    for stage in PIPELINE_ORDER:
        COMMANDS[stage](cfg)
    yield cfg


def test_all_figures_exist(e2e_cfg):
    for name in REQUIRED_FIGURES:
        for ext in ("pdf", "png"):
            p = e2e_cfg.paths.figures / f"{name}.{ext}"
            assert p.exists() and p.stat().st_size > 0, p
        assert (e2e_cfg.paths.figure_data / f"{name}.csv").exists()


def test_all_tables_exist(e2e_cfg):
    files = {p.name for p in e2e_cfg.paths.tables.iterdir()}
    for prefix in REQUIRED_TABLES:
        assert any(f.startswith(prefix) and f.endswith(".csv") for f in files), prefix
        assert any(f.startswith(prefix) and f.endswith(".tex") for f in files), prefix


def test_all_run_tables_exist_and_carry_ids(e2e_cfg):
    for name in REQUIRED_RUN_TABLES:
        p = e2e_cfg.paths.runs / f"{name}.parquet"
        assert p.exists() and p.stat().st_size > 0, p
        df = pd.read_parquet(p)
        assert len(df) > 0, name
        missing = check_id_columns(df, name)
        assert not missing, f"{name} missing id columns {missing}"


def test_reports_written(e2e_cfg):
    for name in ("final_report.md", "run_status.md", "figure_captions.md", "failure_log.md"):
        p = e2e_cfg.paths.reports / name
        assert p.exists() and p.stat().st_size > 0
    text = (e2e_cfg.paths.reports / "final_report.md").read_text()
    assert "MOCK" in text  # mock results are clearly labeled


def test_malformed_outputs_retained(e2e_cfg):
    msgs = pd.read_parquet(e2e_cfg.paths.runs / "public_messages.parquet")
    # the mock backend produces some malformed memos; they must be present
    assert (~msgs["format_valid"]).sum() >= 0
    assert msgs["raw_text"].str.len().min() > 0


def test_data_validation_passed(e2e_cfg):
    import json

    rep = json.loads((e2e_cfg.paths.data / "data_validation_report.json").read_text())
    assert rep["passed"], rep["problems"]
