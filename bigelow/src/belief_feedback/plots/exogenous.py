"""Figures 3-4: exogenous receiver response surface and model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.fit_receiver import MODEL_NAMES, load_receiver
from ..config import Config
from .style import annotate_n, finish, new_fig


def _surface(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(index="llr_bin", columns="steer_frac", values=value, aggfunc="mean")


def make_fig03(cfg: Config) -> None:
    df = pd.read_parquet(cfg.paths.runs / "exogenous_receiver_trials.parquet")
    test = df[df["usage"] == "test"].copy()
    f1 = load_receiver(cfg, "F1")
    f4 = load_receiver(cfg, "F4")
    test["pred_f1"] = f1.predict(test)
    test["pred_f4"] = f4.predict(test)
    fig, axes = new_fig(1, 4, width=13, height=3.0)
    panels = [
        ("ell_post", "observed ell_post"),
        ("pred_f1", "F1 (count model) prediction"),
        ("pred_f4", "F4 (provenance-aware) prediction"),
    ]
    for k, (col, title) in enumerate(panels):
        ax = axes[0][k]
        surf = _surface(test, col)
        im = ax.imshow(surf.to_numpy(), aspect="auto", origin="lower", cmap="RdBu_r", vmin=-4, vmax=4)
        ax.set_xticks(range(len(surf.columns)), [f"{c:g}" for c in surf.columns])
        ax.set_yticks(range(len(surf.index)), [f"{i:g}" for i in surf.index])
        ax.set_xlabel("steering (x delta)")
        ax.set_ylabel("incoming unique LLR bin")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[0][3]
    resid = _surface(test.assign(resid=test["ell_post"] - test["pred_f4"]), "resid")
    im = ax.imshow(resid.to_numpy(), aspect="auto", origin="lower", cmap="PuOr", vmin=-2, vmax=2)
    ax.set_xticks(range(len(resid.columns)), [f"{c:g}" for c in resid.columns])
    ax.set_yticks(range(len(resid.index)), [f"{i:g}" for i in resid.index])
    ax.set_xlabel("steering (x delta)")
    ax.set_ylabel("incoming unique LLR bin")
    ax.set_title("observed - F4 residual")
    fig.colorbar(im, ax=ax, shrink=0.8)
    annotate_n(ax, len(test), "trials")
    finish(cfg, fig, "fig03_exogenous_response_surface", test[
        ["llr_bin", "steer_frac", "ell_pre", "ell_post", "pred_f1", "pred_f4"]
    ])


def make_fig04(cfg: Config) -> None:
    met = pd.read_parquet(cfg.paths.models / "receiver" / "receiver_metrics.parquet")
    test = met[met["usage"] == "test"].set_index("model").reindex(MODEL_NAMES)
    trials = pd.read_parquet(cfg.paths.runs / "exogenous_receiver_trials.parquet")
    tt = trials[trials["usage"] == "test"].copy()

    fig, axes = new_fig(1, 4, width=13, height=3.0)
    ax = axes[0][0]
    x = np.arange(len(MODEL_NAMES))
    ax.bar(x - 0.2, test["rmse"], width=0.4, label="RMSE")
    ax.bar(x + 0.2, test["r2"], width=0.4, label="R^2")
    ax.set_xticks(x, MODEL_NAMES)
    ax.set_ylabel("held-out value")
    ax.legend()
    ax.set_title("test RMSE and R^2")

    ax = axes[0][1]
    for name in ("F1", "F4"):
        model = load_receiver(cfg, name)
        pred = model.predict(tt)
        p = 1 / (1 + np.exp(-pred))
        obs = (tt["ell_post"] > 0).astype(float)
        bins = np.clip((p * 5).astype(int), 0, 4)
        xs, ys = [], []
        for b in range(5):
            m = bins == b
            if m.any():
                xs.append(p[m].mean())
                ys.append(obs[m].mean())
        ax.plot(xs, ys, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("predicted P(ell_post > 0)")
    ax.set_ylabel("observed fraction")
    ax.legend()
    ax.set_title("calibration")

    ax = axes[0][2]
    rep = tt[tt["repeat_mentions_nominal"] > 0]
    errs = []
    for name in MODEL_NAMES:
        model = load_receiver(cfg, name)
        errs.append(float(np.sqrt(np.mean((model.predict(rep) - rep["ell_post"]) ** 2))) if len(rep) else np.nan)
    ax.bar(MODEL_NAMES, errs, color="#a56")
    ax.set_ylabel("RMSE on repeated-source trials")
    ax.set_title("repeated-evidence error")
    annotate_n(ax, len(rep))

    ax = axes[0][3]
    tt["stratum"] = pd.cut(tt["ell_pre"], [-np.inf, -1, 1, np.inf], labels=["prior<-1", "|prior|<=1", "prior>1"])
    for name in ("F1", "F4"):
        model = load_receiver(cfg, name)
        tt[f"err_{name}"] = np.abs(model.predict(tt) - tt["ell_post"])
        g = tt.groupby("stratum", observed=True)[f"err_{name}"].mean()
        ax.plot(range(len(g)), g.to_numpy(), "o-", label=name)
        ax.set_xticks(range(len(g)), list(g.index.astype(str)))
    ax.set_ylabel("MAE")
    ax.legend()
    ax.set_title("error by prior-belief stratum")
    finish(cfg, fig, "fig04_exogenous_model_comparison", test.reset_index())
