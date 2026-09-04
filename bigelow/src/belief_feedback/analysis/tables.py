"""Required result tables (Part 20), each written as CSV and LaTeX."""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..logging_utils import get_logger, now_iso, write_manifest

log = get_logger(__name__)


def _write(cfg: Config, name: str, df: pd.DataFrame) -> None:
    out = cfg.paths.tables
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.tex").write_text(
        df.to_latex(index=False, float_format=lambda x: f"{x:.3f}", escape=True)
    )


def make_all(cfg: Config) -> list[str]:
    started = now_iso()
    runs = cfg.paths.runs
    made: list[str] = []

    worlds = pd.read_parquet(cfg.paths.data / "worlds.parquet")
    reports = pd.read_parquet(cfg.paths.data / "reports.parquet")
    t1 = worlds.groupby("split").agg(
        n_worlds=("world_id", "nunique"),
        truth_upstream=("true_hypothesis", lambda s: (s == "UPSTREAM_CONTAMINATION").sum()),
        alpha_is_upstream=("alpha_is_upstream", "sum"),
        mean_reports=("n_reports", "mean"),
    ).reset_index()
    t1["n_documents"] = t1["split"].map(
        reports.merge(worlds[["world_id", "split"]], on="world_id").groupby("split")["report_id"].count()
    )
    _write(cfg, "table01_dataset_counts", t1)
    made.append("table01_dataset_counts")

    mag = pd.read_parquet(runs / "steering_magnitude_scan.parquet")
    _write(cfg, "table02_steering_calibration", mag)
    made.append("table02_steering_calibration")

    recv = pd.read_parquet(cfg.paths.models / "receiver" / "receiver_metrics.parquet")
    _write(cfg, "table03_exogenous_model_metrics", recv[recv["usage"] == "test"])
    made.append("table03_exogenous_model_metrics")

    comp = pd.read_parquet(runs / "composition_metrics.parquet")
    _write(cfg, "table04_composition_metrics", comp)
    made.append("table04_composition_metrics")

    eff = pd.read_parquet(runs / "branch_effects.parquet")
    t5 = (
        eff.assign(adj=lambda d: d["effect"] * d["sign"].map({"positive": 1, "negative": -1}))
        .groupby(["condition", "round"])
        .agg(mean_effect=("adj", "mean"), sd=("adj", "std"), n=("adj", "size"))
        .reset_index()
    )
    _write(cfg, "table05_branch_effects", t5)
    made.append("table05_branch_effects")

    rec = pd.read_parquet(runs / "recycling_results.parquet")
    t6 = (
        rec.groupby(["condition", "provenance_aware_prompt"])
        .agg(
            gain_one=("gain_one_report", "mean"),
            gain_three=("gain_three_reports", "mean"),
            multiplier=("multiplier", "mean"),
            double_counting_gap=("double_counting_gap", "mean"),
            n=("world_id", "nunique"),
        )
        .reset_index()
    )
    _write(cfg, "table06_recycling_effects", t6)
    made.append("table06_recycling_effects")

    hyst = pd.read_parquet(runs / "hysteresis_results.parquet")
    t7 = (
        hyst.groupby(["sign", "comm"])
        .agg(gap=("hysteresis_gap", "mean"), area=("trajectory_area", "mean"), n=("world_id", "nunique"))
        .reset_index()
    )
    _write(cfg, "table07_hysteresis_effects", t7)
    made.append("table07_hysteresis_effects")

    phase = pd.read_parquet(runs / "phase_boundary_merged.parquet")
    t8 = (
        phase.groupby(["phase_bin", "steering_frac"])
        .agg(
            p_majority=("upstream_majority", "mean"),
            p_majority_pred=("pred_upstream_majority", "mean"),
            final_mean_ell=("final_mean_ell", "mean"),
            n=("world_id", "nunique"),
        )
        .reset_index()
    )
    _write(cfg, "table08_phase_boundary_metrics", t8)
    made.append("table08_phase_boundary_metrics")

    probe = pd.read_parquet(runs / "probe_results.parquet")
    _write(
        cfg,
        "table09_mechanistic_results",
        probe[["layer", "accuracy", "auroc", "behavior_correlation", "calibration_ece", "cosine_with_caa"]],
    )
    made.append("table09_mechanistic_results")

    rob = pd.read_parquet(runs / "robustness_results.parquet")
    t10 = (
        rob.groupby(["dimension", "variant"])
        .agg(
            mean_effect=("mean_effect", "mean"),
            final_mean_effect=("final_mean_effect", "mean"),
            malformed_rate=("malformed_rate", "mean"),
            hallucinated_rate=("hallucinated_citation_rate", "mean"),
            n=("world_id", "nunique"),
        )
        .reset_index()
    )
    _write(cfg, "table10_robustness", t10)
    made.append("table10_robustness")

    write_manifest(
        cfg, "tables", started=started,
        artifact_paths=[str(cfg.paths.tables / f"{m}.csv") for m in made],
        completed_jobs=len(made),
    )
    log.info("wrote %d tables", len(made))
    return made
