"""Fit the emission model G (Part 6A) on exogenous calibration data only."""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge

from ..config import Config
from ..logging_utils import get_logger, now_iso, write_manifest
from ..seeds import rng as make_rng
from ..world.generator import load_worlds
from .message_features import emission_citation_table

log = get_logger(__name__)

STANCE_FEATURES = ["ell_pre", "n_history_rounds", "n_received_reports", "oracle_log_odds"]
CONF_FEATURES = ["abs_ell", "conflict", "round"]
CITE_FEATURES = [
    "event_llr",
    "abs_event_llr",
    "alignment_with_belief",
    "is_private",
    "first_seen_round",
    "n_prior_mentions",
    "previously_cited",
    "ell_pre",
]
COUNT_FEATURES = ["abs_ell", "accessible_event_count", "round"]


class EmissionModel:
    """G: interpretable sub-models plus stochastic feature-level simulation."""

    def __init__(self, stance, confidence, citation, count) -> None:
        self.stance = stance
        self.confidence = confidence
        self.citation = citation
        self.count = count

    # ---- expected-feature prediction --------------------------------------
    def stance_probs(self, ell: float, n_hist: int = 0, n_recv: int = 0, oracle: float = 0.0) -> np.ndarray:
        x = np.array([[ell, n_hist, n_recv, oracle]])
        probs = np.zeros(3)  # classes: -1, 0, +1
        for ci, cls in enumerate(self.stance.classes_):
            probs[int(cls) + 1] = self.stance.predict_proba(x)[0, ci]
        return probs

    def expected_confidence(self, ell: float, conflict: float = 0.0, round_idx: int = 1) -> float:
        return float(np.clip(self.confidence.predict([[abs(ell), conflict, round_idx]])[0], 0, 100))

    def cite_probability(self, event_llr: float, ell: float, is_private: bool, n_prior: int = 0) -> float:
        x = np.array(
            [[event_llr, abs(event_llr), float(np.sign(event_llr) == np.sign(ell)) if ell else 0.5,
              float(is_private), 0 if is_private else 1, n_prior, 0.0, ell]]
        )
        return float(self.citation.predict_proba(x)[0, 1])

    # ---- stochastic simulation --------------------------------------------
    def sample_message(
        self,
        rng: np.random.Generator,
        ell: float,
        accessible_events: dict[str, float],
        private_events: set[str],
        round_idx: int,
    ) -> dict[str, Any]:
        probs = self.stance_probs(ell, n_hist=round_idx - 1)
        stance = int(rng.choice([-1, 0, 1], p=probs / probs.sum()))
        conf = self.expected_confidence(ell, round_idx=round_idx) + float(rng.normal(0, 5))
        cited = [
            eid
            for eid, llr in accessible_events.items()
            if rng.random() < self.cite_probability(llr, ell, eid in private_events)
        ]
        return {"stance": stance, "confidence": float(np.clip(conf, 0, 100)), "cited_events": cited}


def _confidence_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "abs_ell": df["ell_pre"].abs(),
            "conflict": (np.sign(df["ell_pre"]) != np.sign(df["oracle_log_odds"])).astype(float),
            "round": df["n_history_rounds"],
        }
    )
    return out


def fit(cfg: Config) -> dict[str, Any]:
    started = now_iso()
    out_dir = cfg.paths.models / "emission"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = pd.read_parquet(cfg.paths.runs / "exogenous_emission_trials.parquet")
    worlds = load_worlds(cfg)
    train = trials[trials["usage"] == "train"]
    test = trials[trials["usage"] == "test"]

    stance = LogisticRegression(max_iter=2000).fit(
        train[STANCE_FEATURES].to_numpy(), train["public_stance"].astype(int).to_numpy()
    )
    conf_train = train[train["parsed_confidence"].notna()]
    confidence = Ridge(alpha=1.0).fit(
        _confidence_frame(conf_train).to_numpy(), conf_train["parsed_confidence"].astype(float).to_numpy()
    )
    cites_train = emission_citation_table(train, worlds)
    citation = LogisticRegression(max_iter=2000).fit(
        cites_train[CITE_FEATURES].to_numpy(), cites_train["cited"].to_numpy()
    )
    count = PoissonRegressor(alpha=1e-3, max_iter=1000).fit(
        pd.DataFrame.to_numpy(pd.DataFrame(
            {
                "abs_ell": train["ell_pre"].abs(),
                "accessible_event_count": train["accessible_event_count"],
                "round": train["n_history_rounds"],
            }
        )),
        train["n_cited"].to_numpy(),
    )
    model = EmissionModel(stance, confidence, citation, count)

    # held-out metrics
    stance_acc = float(
        (stance.predict(test[STANCE_FEATURES].to_numpy()) == test["public_stance"].astype(int)).mean()
    )
    conf_test = test[test["parsed_confidence"].notna()]
    conf_mae = float(
        np.mean(
            np.abs(
                confidence.predict(_confidence_frame(conf_test).to_numpy())
                - conf_test["parsed_confidence"].astype(float)
            )
        )
    )
    cites_test = emission_citation_table(test, worlds)
    cite_auc = float("nan")
    if cites_test["cited"].nunique() > 1:
        from sklearn.metrics import roc_auc_score

        cite_auc = float(
            roc_auc_score(cites_test["cited"], citation.predict_proba(cites_test[CITE_FEATURES].to_numpy())[:, 1])
        )
    metrics = {
        "stance_accuracy_test": stance_acc,
        "confidence_mae_test": conf_mae,
        "citation_auroc_test": cite_auc,
        "n_train": len(train),
        "n_test": len(test),
    }
    with open(out_dir / "emission_model.pkl", "wb") as f:
        pickle.dump(model, f)
    (out_dir / "emission_metadata.json").write_text(
        json.dumps({**metrics, "config_hash": cfg.config_hash(), "model_id": cfg.model.model_id}, indent=2)
    )
    write_manifest(
        cfg, "fit_emission", started=started, artifact_paths=[str(out_dir)], completed_jobs=1,
        extra=metrics,
    )
    log.info("emission fit: %s", metrics)
    return metrics


def load_emission(cfg: Config) -> EmissionModel:
    with open(cfg.paths.models / "emission" / "emission_model.pkl", "rb") as f:
        return pickle.load(f)


def sample_rng(*parts: object) -> np.random.Generator:
    return make_rng("emission_sample", *parts)
