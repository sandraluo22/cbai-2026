"""Post-hoc analysis orchestrator: branch effects, hypothesis tests, phase
surface, Jacobian summaries, and the composition test."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..agents.protocol import neighbors_of
from ..config import Config
from ..logging_utils import get_logger, now_iso, write_manifest
from ..seeds import rng as make_rng
from ..world.generator import load_worlds
from .fit_emission import load_emission
from .fit_receiver import load_receiver
from .metrics import amplification_ratios, compute_branch_effects, majority_threshold
from .rollout import _feature_row
from .rollout import run as run_composition
from .statistical_tests import benjamini_hochberg, hypothesis_row

log = get_logger(__name__)


def _sign_adjust(df: pd.DataFrame) -> pd.Series:
    return df["effect"] * np.where(df["sign"] == "positive", 1.0, -1.0)


def _hypothesis_tests(cfg: Config, effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comp = pd.read_parquet(cfg.paths.runs / "composition_metrics.parquet")
    preds = pd.read_parquet(cfg.paths.runs / "composition_predictions.parquet")
    exo_rmse = float(comp[comp["model"] == "F4"]["exogenous_test_rmse"].iloc[0])
    tf4 = preds[(preds["model"] == "F4") & (preds["kind"] == "teacher_forced")].copy()
    tf4["sq_err"] = (tf4["predicted"] - tf4["observed"]) ** 2
    per_world = tf4.groupby("world_id")["sq_err"].mean().pow(0.5) - exo_rmse
    rows.append(
        hypothesis_row(cfg, "H1_exogenous_composition_null", per_world,
                       "Per-world endogenous one-step RMSE (F4) minus exogenous test RMSE")
    )

    e = effects.copy()
    e["adj"] = _sign_adjust(e)
    final_round = e["round"].max()
    beyond = e[(e["condition"] == "total_closed_loop_effect") & (e["graph_distance"] >= 2) & (e["round"] == final_round)]
    rows.append(
        hypothesis_row(cfg, "H2_feedback_beyond_neighbors",
                       beyond.groupby("world_id")["adj"].mean(),
                       "Sign-adjusted total closed-loop effect at graph distance >= 2, final round")
    )
    amp = e[(e["condition"] == "total_closed_loop_effect") & (e["round"] == final_round)]
    onehop = e[(e["condition"] == "one_hop_effect") & (e["round"] == final_round)]
    diff = (
        amp.groupby("world_id")["adj"].mean() - onehop.groupby("world_id")["adj"].mean()
    ).dropna()
    rows.append(
        hypothesis_row(cfg, "H2_total_vs_one_hop", diff,
                       "Total closed-loop minus one-hop effect (sign-adjusted, final round)")
    )

    rec = pd.read_parquet(cfg.paths.runs / "recycling_results.parquet")
    rec_n = rec[(rec["condition"] == "recycled") & (~rec["provenance_aware_prompt"])]
    rows.append(
        hypothesis_row(cfg, "H3_recycling_multiplier_gt1",
                       rec_n.groupby("world_id")["multiplier"].mean() - 1.0,
                       "Recycled three-report multiplier minus the provenance-aware value of 1")
    )
    rows.append(
        hypothesis_row(cfg, "H3_double_counting_gap",
                       rec_n.groupby("world_id")["double_counting_gap"].mean(),
                       "|LLM recycled gain| minus |provenance-aware oracle gain|")
    )

    tme = e[(e["condition"] == "text_mediated_effect") & (e["round"] == final_round)]
    tot = e[(e["condition"] == "total_closed_loop_effect") & (e["round"] == final_round) & (e["agent_id"] != 0)]
    tme_ds = tme[tme["agent_id"] != 0]
    frac = (
        tme_ds.groupby("world_id")["adj"].mean() / tot.groupby("world_id")["adj"].mean().replace(0, np.nan)
    ).dropna()
    rows.append(
        hypothesis_row(cfg, "H4_text_mediation_fraction", frac,
                       "Fraction of downstream closed-loop effect mediated by the emitted text")
    )

    hyst = pd.read_parquet(cfg.paths.runs / "hysteresis_results.parquet")
    hp = hyst.pivot_table(index=["world_id", "replicate_seed", "sign"], columns="comm",
                          values="hysteresis_gap").reset_index()
    if {"live", "replay"} <= set(hp.columns):
        interaction = (hp["live"] - hp["replay"]) * np.where(hp["sign"] == "positive", 1, -1)
        rows.append(
            hypothesis_row(cfg, "H5_live_minus_replay_hysteresis",
                           pd.Series(interaction.to_numpy(), index=hp["world_id"]),
                           "Live hysteresis gap minus fixed-replay hysteresis gap (sign-adjusted)")
        )
    return benjamini_hochberg(pd.DataFrame(rows))


def _phase_surface(cfg: Config) -> pd.DataFrame:
    """Fit smooth logistic surfaces and locate the 0.5 majority contour."""
    obs = pd.read_parquet(cfg.paths.runs / "phase_boundary_results.parquet")
    worlds = load_worlds(cfg)
    emission = load_emission(cfg)
    f4 = load_receiver(cfg, "F4")
    meta = json.loads((cfg.paths.vectors / cfg.model.slug / "steering_metadata.json").read_text())
    m_max = float(meta["m_max"])
    rounds = cfg.network.rounds
    topo = cfg.network.topology

    pred_rows = []
    for _, row in obs.iterrows():
        world = worlds[row["world_id"]]
        ell = np.array([float(x) for x in row["initial_beliefs"].split(",")])
        event_llr = {e.event_id: e.llr for e in world.events}
        priv = {a: {world.report(r).event_id for r in world.assignments.get(a, [])} for a in range(world.n_agents)}
        accessible = {a: set(priv[a]) for a in range(world.n_agents)}
        steering = {(0, r): row["steering_frac"] * m_max for r in range(1, min(3, rounds) + 1)}
        rng = make_rng("phase_pred", row["world_id"], row["replicate_seed"], row["steering_frac"])
        for r in range(1, rounds + 1):
            outgoing = {
                a: emission.sample_message(rng, float(ell[a]), {e: event_llr[e] for e in accessible[a]}, priv[a], r)
                for a in range(world.n_agents)
            }
            new_ell = ell.copy()
            for a in range(world.n_agents):
                incoming = [outgoing[s] for s in neighbors_of(topo, world.n_agents, a)]
                feats = _feature_row(float(ell[a]), float(steering.get((a, r), 0.0)),
                                     incoming, accessible[a], event_llr, r)
                new_ell[a] = float(f4.predict(pd.DataFrame([feats]))[0])
                for msg in incoming:
                    accessible[a].update(msg["cited_events"])
            ell = new_ell
        n_up = int((ell > 0).sum())
        pred_rows.append(
            {
                "world_id": row["world_id"],
                "replicate_seed": row["replicate_seed"],
                "phase_bin": row["phase_bin"],
                "steering_frac": row["steering_frac"],
                "network_oracle_log_odds": row["network_oracle_log_odds"],
                "pred_upstream_majority": float(n_up >= majority_threshold(world.n_agents)),
                "pred_final_mean_ell": float(ell.mean()),
            }
        )
    pred = pd.DataFrame(pred_rows)
    merged = obs.merge(pred, on=["world_id", "replicate_seed", "phase_bin", "steering_frac",
                                 "network_oracle_log_odds"])

    def contour(df: pd.DataFrame, ycol: str) -> dict[float, float]:
        x = df[["network_oracle_log_odds", "steering_frac"]].to_numpy()
        y = df[ycol].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            return {}
        clf = LogisticRegression(max_iter=2000).fit(x, y)
        b0 = clf.intercept_[0]
        be, bs = clf.coef_[0]
        if abs(be) < 1e-9:
            return {}
        return {s: float(-(b0 + bs * s) / be) for s in sorted(df["steering_frac"].unique())}

    obs_c = contour(merged, "upstream_majority")
    pred_c = contour(merged, "pred_upstream_majority")
    disp = [abs(obs_c[s] - pred_c[s]) for s in obs_c if s in pred_c]
    merged.attrs["contour_displacement"] = float(np.mean(disp)) if disp else float("nan")
    out_rows = [
        {"steering_frac": s, "observed_contour": obs_c.get(s, np.nan),
         "predicted_contour": pred_c.get(s, np.nan)}
        for s in sorted(set(obs_c) | set(pred_c))
    ]
    pd.DataFrame(out_rows).to_parquet(cfg.paths.runs / "phase_boundary_contours.parquet", index=False)
    return merged


def _jacobian_summary(cfg: Config) -> pd.DataFrame:
    jac = pd.read_parquet(cfg.paths.runs / "jacobian_results.parquet")
    worlds = load_worlds(cfg)
    rows = []
    j_only = jac[jac["condition"] == "jacobian"]
    for (wid, t), grp in j_only.groupby(["world_id", "round"]):
        n = worlds[wid].n_agents
        mat = np.zeros((n, n))
        for _, r in grp.iterrows():
            mat[int(r["agent_id"]), int(r["source_agent"])] = r["jacobian_value"]
        eig = np.linalg.eigvals(mat)
        sv = np.linalg.svd(mat, compute_uv=False)
        diag = float(np.mean(np.diag(mat)))
        neigh = float(
            np.mean([mat[i, j] for j in range(n) for i in neighbors_of(cfg.network.topology, n, j)])
        )
        asym = float(np.linalg.norm(mat - mat.T) / (np.linalg.norm(mat + mat.T) + 1e-12))
        rows.append(
            {
                "world_id": wid,
                "round": int(t),
                "diagonal_mean": diag,
                "neighbor_mean": neigh,
                "leading_eigenvalue": float(np.max(np.abs(eig.real))),
                "spectral_radius": float(np.max(np.abs(eig))),
                "top_singular_value": float(sv[0]),
                "asymmetry": asym,
            }
        )
    return pd.DataFrame(rows)


def run(cfg: Config) -> None:
    started = now_iso()
    run_composition(cfg)
    worlds = load_worlds(cfg)
    beliefs = pd.read_parquet(cfg.paths.runs / "belief_states.parquet")
    effects = compute_branch_effects(cfg, beliefs, worlds)
    effects.to_parquet(cfg.paths.runs / "branch_effects.parquet", index=False)
    amp = amplification_ratios(cfg, beliefs)
    amp.to_parquet(cfg.paths.runs / "amplification_ratios.parquet", index=False)
    hyp = _hypothesis_tests(cfg, effects)
    hyp.to_parquet(cfg.paths.runs / "hypothesis_tests.parquet", index=False)
    phase = _phase_surface(cfg)
    phase.to_parquet(cfg.paths.runs / "phase_boundary_merged.parquet", index=False)
    jac = _jacobian_summary(cfg)
    jac.to_parquet(cfg.paths.runs / "jacobian_summary.parquet", index=False)
    write_manifest(
        cfg, "analyze", started=started,
        artifact_paths=[str(cfg.paths.runs / "branch_effects.parquet"),
                        str(cfg.paths.runs / "hypothesis_tests.parquet")],
        completed_jobs=len(hyp),
        extra={"contour_displacement": phase.attrs.get("contour_displacement")},
    )
    log.info("analysis complete:\n%s", hyp[["hypothesis", "effect", "ci_low", "ci_high", "p_value"]])
