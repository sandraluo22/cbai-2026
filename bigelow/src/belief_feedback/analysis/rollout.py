"""Composition test (Part 8): teacher-forced one-step prediction and free
feature-level Monte Carlo rollout of the exogenously fitted F and G.

No model parameter is refit here; endogenous outcomes are used only for
evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..agents.protocol import neighbors_of
from ..config import Config, id_fields
from ..logging_utils import get_logger, now_iso, write_manifest
from ..seeds import rng as make_rng
from ..world.generator import load_worlds
from ..world.schema import World
from .fit_emission import EmissionModel, load_emission
from .fit_receiver import MODEL_NAMES, ReceiverModel, load_receiver
from .metrics import majority_threshold

log = get_logger(__name__)

QUANTS = [2.5, 10, 25, 50, 75, 90, 97.5]


def _feature_row(
    ell_pre: float,
    m: float,
    incoming: list[dict[str, Any]],
    accessible: set[str],
    event_llr: dict[str, float],
    round_idx: int,
) -> dict[str, float]:
    """Receiver features from a list of incoming message feature dicts."""
    n = len(incoming)
    stances = [msg["stance"] for msg in incoming]
    confs = [msg["confidence"] / 100.0 for msg in incoming]
    cited_all: list[str] = [e for msg in incoming for e in msg["cited_events"]]
    new_events = [e for e in dict.fromkeys(cited_all) if e not in accessible]
    unique_new_llr = sum(event_llr.get(e, 0.0) for e in new_events)
    naive = sum(event_llr.get(e, 0.0) for e in cited_all if e not in accessible)
    mean_st = float(np.mean(stances)) if stances else 0.0
    cw = float(np.mean([s * c for s, c in zip(stances, confs)])) if stances else 0.0
    return {
        "ell_pre": ell_pre,
        "m": m,
        "n_messages": n,
        "signed_message_count": float(sum(stances)),
        "unique_new_llr": unique_new_llr,
        "repeated_llr_if_naively_counted": naive,
        "repeated_report_count": len(cited_all) - len(set(cited_all)),
        "mean_public_stance": mean_st,
        "confidence_weighted_stance": cw,
        "agreement_with_prior": float(np.sign(ell_pre) == np.sign(mean_st)) if mean_st else 0.5,
        "cumulative_unique_event_count": float(len(accessible | set(new_events))),
        "context_age": float(round_idx),
    }


def _message_features_from_row(msg_row: pd.Series, world: World) -> dict[str, Any]:
    cited = [e for e in str(msg_row["cited_event_ids"]).split(",") if e]
    conf = msg_row["parsed_confidence"]
    return {
        "stance": int(msg_row["parsed_semantic_assessment"]),
        "confidence": float(conf) if pd.notna(conf) else 50.0,
        "cited_events": cited,
    }


def teacher_forced(cfg: Config, worlds: dict[str, World], models: dict[str, ReceiverModel]) -> pd.DataFrame:
    beliefs = pd.read_parquet(cfg.paths.runs / "belief_states.parquet")
    msgs = pd.read_parquet(cfg.paths.runs / "public_messages.parquet")
    deliveries = pd.read_parquet(cfg.paths.runs / "deliveries.parquet")
    rows: list[dict[str, Any]] = []
    key = ["world_id", "replicate_seed", "branch"]
    msg_idx = msgs.set_index(["world_id", "replicate_seed", "branch", "agent_id", "round"]).sort_index()
    for (wid, rep, branch), grp in beliefs.groupby(key):
        world = worlds[wid]
        topo = cfg.network.topology
        b = grp.pivot_table(index="round", columns="agent_id", values="semantic_log_odds")
        deliv = deliveries[
            (deliveries["world_id"] == wid)
            & (deliveries["replicate_seed"] == rep)
            & (deliveries["originating_branch"] == branch)
        ]
        accessible: dict[int, set[str]] = {
            a: {world.report(rid).event_id for rid in world.assignments.get(a, [])}
            for a in range(world.n_agents)
        }
        event_llr = {e.event_id: e.llr for e in world.events}
        steering = grp.set_index(["agent_id", "round"])["steering_magnitude"].to_dict()
        for r in range(1, int(b.index.max()) + 1):
            for i in range(world.n_agents):
                incoming = []
                for src in neighbors_of(topo, world.n_agents, i):
                    d = deliv[
                        (deliv["source_agent"] == src)
                        & (deliv["recipient_agent"] == i)
                        & (deliv["round"] == r)
                    ]
                    src_branch = d.iloc[0]["actual_generated_branch"] if len(d) else branch
                    try:
                        mrow = msg_idx.loc[(wid, rep, src_branch, src, r)]
                        if isinstance(mrow, pd.DataFrame):
                            mrow = mrow.iloc[0]
                        incoming.append(_message_features_from_row(mrow, world))
                    except KeyError:
                        continue
                feats = _feature_row(
                    float(b.loc[r - 1, i]), float(steering.get((i, r), 0.0)),
                    incoming, accessible[i], event_llr, r,
                )
                fdf = pd.DataFrame([feats])
                obs = float(b.loc[r, i])
                for name, model in models.items():
                    rows.append(
                        {
                            **id_fields(cfg),
                            "kind": "teacher_forced",
                            "model": name,
                            "world_id": wid,
                            "condition": branch,
                            "branch": branch,
                            "branch_parent": None,
                            "replicate_seed": rep,
                            "agent_id": i,
                            "round": r,
                            "seed": 0,
                            "observed": obs,
                            "predicted": float(model.predict(fdf)[0]),
                        }
                    )
                for msg in incoming:
                    accessible[i].update(msg["cited_events"])
    return pd.DataFrame(rows)


def free_rollout(
    cfg: Config,
    worlds: dict[str, World],
    models: dict[str, ReceiverModel],
    emission: EmissionModel,
    delta: float,
) -> pd.DataFrame:
    beliefs = pd.read_parquet(cfg.paths.runs / "belief_states.parquet")
    n_samples = cfg.analysis.rollout_samples
    rounds = cfg.network.rounds
    topo = cfg.network.topology
    rows: list[dict[str, Any]] = []
    conds = {
        "baseline": {},
        "positive_impulse": {(0, 1): +delta},
        "negative_impulse": {(0, 1): -delta},
    }
    for (wid, rep, branch), grp in beliefs.groupby(["world_id", "replicate_seed", "branch"]):
        if branch not in conds:
            continue
        world = worlds[wid]
        steering = conds[branch]
        b = grp.pivot_table(index="round", columns="agent_id", values="semantic_log_odds")
        ell0 = b.loc[0].to_numpy(dtype=float)
        obs_traj = b.to_numpy(dtype=float)
        event_llr = {e.event_id: e.llr for e in world.events}
        priv_events = {
            a: {world.report(rid).event_id for rid in world.assignments.get(a, [])}
            for a in range(world.n_agents)
        }
        maj_k = majority_threshold(world.n_agents)
        for name, model in models.items():
            trajs = np.zeros((n_samples, rounds + 1, world.n_agents))
            for s in range(n_samples):
                r_gen = make_rng("rollout", wid, rep, branch, name, s)
                ell = ell0.copy()
                accessible = {a: set(priv_events[a]) for a in range(world.n_agents)}
                trajs[s, 0] = ell
                for r in range(1, rounds + 1):
                    outgoing = {}
                    for a in range(world.n_agents):
                        acc_events = {e: event_llr[e] for e in accessible[a]}
                        outgoing[a] = emission.sample_message(
                            r_gen, float(ell[a]), acc_events, priv_events[a], r
                        )
                    new_ell = ell.copy()
                    for a in range(world.n_agents):
                        incoming = [outgoing[src] for src in neighbors_of(topo, world.n_agents, a)]
                        feats = _feature_row(
                            float(ell[a]), float(steering.get((a, r), 0.0)),
                            incoming, accessible[a], event_llr, r,
                        )
                        new_ell[a] = float(model.predict(pd.DataFrame([feats]))[0])
                        for msg in incoming:
                            accessible[a].update(msg["cited_events"])
                    ell = new_ell
                    trajs[s, r] = ell
            final_up = (trajs[:, -1] > 0).sum(axis=1)
            pred_consensus = float(np.mean(final_up >= maj_k))
            obs_final_up = int((obs_traj[-1] > 0).sum()) if obs_traj.shape[0] > rounds else int(
                (obs_traj[-1] > 0).sum()
            )
            for r in range(min(rounds, obs_traj.shape[0] - 1) + 1):
                for a in range(world.n_agents):
                    qs = np.percentile(trajs[:, r, a], QUANTS)
                    rows.append(
                        {
                            **id_fields(cfg),
                            "kind": "free_rollout",
                            "model": name,
                            "world_id": wid,
                            "condition": branch,
                            "branch": branch,
                            "branch_parent": None,
                            "replicate_seed": rep,
                            "agent_id": a,
                            "round": r,
                            "seed": 0,
                            "observed": float(obs_traj[r, a]),
                            "predicted": float(trajs[:, r, a].mean()),
                            **{f"q{q}".replace(".", "_"): float(v) for q, v in zip(QUANTS, qs)},
                            "pred_consensus_prob": pred_consensus,
                            "obs_upstream_majority": float(obs_final_up >= maj_k),
                        }
                    )
    return pd.DataFrame(rows)


def composition_metrics(cfg: Config, preds: pd.DataFrame) -> pd.DataFrame:
    """Primary composition metrics per model (Part 8)."""
    recv_metrics = pd.read_parquet(cfg.paths.models / "receiver" / "receiver_metrics.parquet")
    exo_rmse = recv_metrics[recv_metrics["usage"] == "test"].set_index("model")["rmse"]
    rows = []
    for model in preds["model"].unique():
        tf = preds[(preds["model"] == model) & (preds["kind"] == "teacher_forced")]
        fr = preds[(preds["model"] == model) & (preds["kind"] == "free_rollout")]
        one_rmse = float(np.sqrt(np.mean((tf["predicted"] - tf["observed"]) ** 2))) if len(tf) else np.nan
        roll_final = fr[fr["round"] == fr["round"].max()] if len(fr) else fr
        roll_rmse = float(np.sqrt(np.mean((fr["predicted"] - fr["observed"]) ** 2))) if len(fr) else np.nan
        corr = (
            float(np.corrcoef(fr["predicted"], fr["observed"])[0, 1])
            if len(fr) > 2 and fr["observed"].std() > 0
            else np.nan
        )
        cons = roll_final.groupby("world_id").first() if len(fr) else pd.DataFrame()
        cons_err = (
            float(np.mean(np.abs(cons["pred_consensus_prob"] - cons["obs_upstream_majority"])))
            if len(cons)
            else np.nan
        )
        cov = {}
        for level, (lo_c, hi_c) in {
            50: ("q25", "q75"),
            80: ("q10", "q90"),
            95: ("q2_5", "q97_5"),
        }.items():
            if len(fr):
                cov[f"coverage_{level}"] = float(
                    np.mean((fr["observed"] >= fr[lo_c]) & (fr["observed"] <= fr[hi_c]))
                )
            else:
                cov[f"coverage_{level}"] = np.nan
        rows.append(
            {
                "model": model,
                "one_step_rmse": one_rmse,
                "rollout_rmse": roll_rmse,
                "trajectory_correlation": corr,
                "final_consensus_prob_error": cons_err,
                "final_majority_accuracy_error": cons_err,
                "exogenous_test_rmse": float(exo_rmse.get(model, np.nan)),
                "generalization_gap": one_rmse - float(exo_rmse.get(model, np.nan)),
                **cov,
            }
        )
    return pd.DataFrame(rows)


def run(cfg: Config) -> None:
    out = cfg.paths.runs / "composition_predictions.parquet"
    met_out = cfg.paths.runs / "composition_metrics.parquet"
    if out.exists() and met_out.exists():
        log.info("composition predictions exist; skipping")
        return
    started = now_iso()
    worlds = load_worlds(cfg)
    models = {name: load_receiver(cfg, name) for name in MODEL_NAMES}
    emission = load_emission(cfg)
    import json

    meta = json.loads((cfg.paths.vectors / cfg.model.slug / "steering_metadata.json").read_text())
    delta = 0.5 * float(meta["m_max"])
    tf = teacher_forced(cfg, worlds, models)
    fr = free_rollout(cfg, worlds, models, emission, delta)
    preds = pd.concat([tf, fr], ignore_index=True)
    preds.to_parquet(out, index=False)
    composition_metrics(cfg, preds).to_parquet(met_out, index=False)
    write_manifest(
        cfg, "composition", started=started, artifact_paths=[str(out), str(met_out)],
        completed_jobs=len(preds),
    )
