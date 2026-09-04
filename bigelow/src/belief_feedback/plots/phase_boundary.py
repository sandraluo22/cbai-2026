"""Figure 10: network phase boundary."""

from __future__ import annotations

import pandas as pd

from ..config import Config
from .style import annotate_n, finish, new_fig


def make(cfg: Config) -> None:
    merged = pd.read_parquet(cfg.paths.runs / "phase_boundary_merged.parquet")
    contours = pd.read_parquet(cfg.paths.runs / "phase_boundary_contours.parquet")
    fig, axes = new_fig(1, 4, width=13, height=3.0)

    for k, (col, title) in enumerate(
        (("upstream_majority", "observed P(upstream majority)"),
         ("pred_upstream_majority", "composition-model prediction"))
    ):
        ax = axes[0][k]
        surf = merged.pivot_table(index="phase_bin", columns="steering_frac", values=col, aggfunc="mean")
        im = ax.imshow(surf.to_numpy(), aspect="auto", origin="lower", cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_xticks(range(len(surf.columns)), [f"{c:g}" for c in surf.columns])
        ax.set_yticks(range(len(surf.index)), [f"{i:g}" for i in surf.index])
        ax.set_xlabel("steering (x m_max)")
        ax.set_ylabel("initial evidence bin (ell)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    annotate_n(axes[0][0], merged["world_id"].nunique(), "worlds")

    ax = axes[0][2]
    ax.plot(contours["steering_frac"], contours["observed_contour"], "o-", label="observed 0.5 contour")
    ax.plot(contours["steering_frac"], contours["predicted_contour"], "s--", label="predicted 0.5 contour")
    ax.set_xlabel("steering (x m_max)")
    ax.set_ylabel("evidence at P=0.5 (ell)")
    ax.legend()
    ax.set_title("majority boundary")

    ax = axes[0][3]
    disp = (contours["observed_contour"] - contours["predicted_contour"]).abs()
    ax.bar(contours["steering_frac"].astype(str), disp)
    ax.set_xlabel("steering (x m_max)")
    ax.set_ylabel("|contour displacement| (ell)")
    ax.set_title(f"mean displacement = {disp.mean():.2f}")
    finish(cfg, fig, "fig10_network_phase_boundary", merged)
