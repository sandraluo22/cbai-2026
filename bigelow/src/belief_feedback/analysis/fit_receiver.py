"""Fit receiver-update models F0-F5 (Part 6B) on exogenous data only.

F0 persistence, F1 Bigelow-style signed-count accumulation with power-law
discount, F2 DeGroot social averaging, F3 evidence-additive, F4
provenance-aware dynamic, F5 flexible nonlinear benchmark.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

from ..config import Config
from ..logging_utils import get_logger, now_iso, write_manifest
from .message_features import receiver_design

log = get_logger(__name__)

MODEL_NAMES = ["F0", "F1", "F2", "F3", "F4", "F5"]


class ReceiverModel:
    """A fitted F model: predicts ell_post from receiver-trial features."""

    def __init__(self, name: str, kind: str, payload: Any, columns: list[str] | None = None) -> None:
        self.name = name
        self.kind = kind  # "linear" | "bigelow" | "sklearn"
        self.payload = payload
        self.columns = columns or []

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = receiver_design(df)
        if self.kind == "bigelow":
            b, a, gpos, gneg, alpha, rho = self.payload
            n_pos = df["n_messages"].to_numpy() * (df["mean_public_stance"].to_numpy() > 0)
            n_neg = df["n_messages"].to_numpy() * (df["mean_public_stance"].to_numpy() < 0)
            n_pos = n_pos + df["repeated_report_count"].to_numpy() * (df["mean_public_stance"].to_numpy() > 0)
            n_neg = n_neg + df["repeated_report_count"].to_numpy() * (df["mean_public_stance"].to_numpy() < 0)
            e = 1.0 - alpha
            return (
                b
                + rho * df["ell_pre"].to_numpy()
                + a * df["m"].to_numpy()
                + gpos * np.power(np.maximum(n_pos, 0), e)
                - gneg * np.power(np.maximum(n_neg, 0), e)
            )
        return np.asarray(self.payload.predict(x[self.columns]))

    def predict_increment(self, df: pd.DataFrame) -> np.ndarray:
        """Incremental form: predicted change from the pre-message belief."""
        return self.predict(df) - df["ell_pre"].to_numpy()


def _fit_bigelow(train: pd.DataFrame) -> tuple:
    y = train["ell_post"].to_numpy()

    def residual(theta):
        b, a, gpos, gneg, alpha, rho = theta
        m = ReceiverModel("F1", "bigelow", (b, a, gpos, gneg, alpha, rho))
        return m.predict(train) - y

    x0 = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.9])
    res = least_squares(residual, x0, bounds=([-5, -5, 0, 0, 0.0, 0], [5, 5, 5, 5, 0.99, 1.5]))
    return tuple(res.x)


def _metrics(y: np.ndarray, pred: np.ndarray, ell_pre: np.ndarray) -> dict[str, float]:
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    p_pred = 1 / (1 + np.exp(-pred))
    p_obs = (y > 0).astype(float)
    eps = 1e-9
    dy = y - ell_pre
    dp = pred - ell_pre
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot,
        "pearson": float(pearsonr(y, pred)[0]) if y.std() > 0 and pred.std() > 0 else float("nan"),
        "spearman": float(spearmanr(y, pred)[0]) if y.std() > 0 else float("nan"),
        "sign_accuracy": float(np.mean(np.sign(dp) == np.sign(dy))),
        "log_loss": float(-np.mean(p_obs * np.log(p_pred + eps) + (1 - p_obs) * np.log(1 - p_pred + eps))),
        "calibration_gap": float(abs(p_pred.mean() - p_obs.mean())),
    }


def fit(cfg: Config) -> pd.DataFrame:
    started = now_iso()
    out_dir = cfg.paths.models / "receiver"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cfg.paths.runs / "exogenous_receiver_trials.parquet")
    train = df[df["usage"] == "train"]
    val = df[df["usage"] == "validation"]
    test = df[df["usage"] == "test"]

    models: dict[str, ReceiverModel] = {}
    xt = receiver_design(train)

    f0_cols = ["ell_pre"]
    models["F0"] = ReceiverModel(
        "F0", "linear", LinearRegression().fit(xt[f0_cols], train["ell_post"]), f0_cols
    )
    models["F1"] = ReceiverModel("F1", "bigelow", _fit_bigelow(train))
    f2_cols = ["ell_pre", "confidence_weighted_stance"]
    models["F2"] = ReceiverModel(
        "F2", "linear", LinearRegression().fit(xt[f2_cols], train["ell_post"]), f2_cols
    )
    f3_cols = ["ell_pre", "m", "unique_new_llr", "confidence_weighted_stance"]
    models["F3"] = ReceiverModel(
        "F3", "linear", LinearRegression().fit(xt[f3_cols], train["ell_post"]), f3_cols
    )
    f4_cols = f3_cols + [
        "repeated_llr_if_naively_counted",
        "repeated_report_count",
        "agreement_with_prior",
        "cumulative_unique_event_count",
        "context_age",
        "conflict_magnitude",
        "prior_x_incoming",
    ]
    best_alpha, best_score = 1.0, -np.inf
    xv = receiver_design(val)
    for alpha in (0.01, 0.1, 1.0, 10.0):
        cand = Ridge(alpha=alpha).fit(xt[f4_cols], train["ell_post"])
        s = -float(np.mean((cand.predict(xv[f4_cols]) - val["ell_post"]) ** 2))
        if s > best_score:
            best_alpha, best_score = alpha, s
    models["F4"] = ReceiverModel(
        "F4", "linear", Ridge(alpha=best_alpha).fit(xt[f4_cols], train["ell_post"]), f4_cols
    )
    f5_cols = list(xt.columns)
    best_depth, best_score = 2, -np.inf
    for depth in (2, 3):
        cand = GradientBoostingRegressor(max_depth=depth, random_state=0).fit(
            xt[f5_cols], train["ell_post"]
        )
        s = -float(np.mean((cand.predict(xv[f5_cols]) - val["ell_post"]) ** 2))
        if s > best_score:
            best_depth, best_score = depth, s
    models["F5"] = ReceiverModel(
        "F5",
        "sklearn",
        GradientBoostingRegressor(max_depth=best_depth, random_state=0).fit(
            xt[f5_cols], train["ell_post"]
        ),
        f5_cols,
    )

    rows = []
    for name, model in models.items():
        for usage, part in (("validation", val), ("test", test)):
            met = _metrics(
                part["ell_post"].to_numpy(), model.predict(part), part["ell_pre"].to_numpy()
            )
            rows.append({"model": name, "usage": usage, **met})
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_parquet(out_dir / "receiver_metrics.parquet", index=False)
    (out_dir / "receiver_metadata.json").write_text(
        json.dumps(
            {
                "config_hash": cfg.config_hash(),
                "model_id": cfg.model.model_id,
                "f4_ridge_alpha": best_alpha,
                "f5_depth": best_depth,
                "f1_params": list(models["F1"].payload),
                "n_train": len(train),
            },
            indent=2,
        )
    )
    write_manifest(
        cfg, "fit_receiver", started=started, artifact_paths=[str(out_dir)],
        completed_jobs=len(models),
    )
    log.info("receiver fit complete:\n%s", metrics_df[metrics_df["usage"] == "test"])
    return metrics_df


def load_receiver(cfg: Config, name: str) -> ReceiverModel:
    with open(cfg.paths.models / "receiver" / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
