"""Figure 2: CAA steering calibration."""

from __future__ import annotations

import json

import pandas as pd

from ..config import Config
from .style import COLORS, annotate_n, finish, new_fig


def make(cfg: Config) -> None:
    meta = json.loads(
        (cfg.paths.vectors / cfg.model.slug / "steering_metadata.json").read_text()
    )
    scan = pd.DataFrame(meta["layer_scan"])
    mag = pd.DataFrame(meta["magnitude_scan"])
    fig, axes = new_fig(1, 5, width=15, height=3.0)

    ax = axes[0][0]
    norms = meta["vector_norms_by_layer"]
    ax.plot(range(len(norms)), norms, "o-", color=COLORS["model"])
    ax.set_xlabel("layer")
    ax.set_ylabel("|d_l|")
    ax.set_title("CAA vector norm by layer")

    ax = axes[0][1]
    for phase, marker in (("coarse", "o"), ("fine", "s")):
        sub = scan[scan["phase"] == phase]
        ax.plot(sub["layer"], sub["median_slope"], marker, label=phase, alpha=0.8)
    ax.axvline(meta["layer"], color="crimson", ls="--", lw=0.8, label="selected")
    ax.set_xlabel("layer")
    ax.set_ylabel("median slope\n(d ell / d magnitude)")
    ax.legend()
    ax.set_title("held-out behavioral slope")

    ax = axes[0][2]
    ax.plot(scan["layer"], scan["valid_rate"], "o-", label="memo validity")
    ax.plot(scan["layer"], scan["rep4"], "s-", label="repeated 4-gram")
    ax.set_xlabel("layer")
    ax.set_ylabel("rate")
    ax.legend()
    ax.set_title("coherence by layer")

    ax = axes[0][3]
    ax.plot(mag["magnitude"], mag["median_ell"], "o-", color=COLORS["positive"])
    ax.axvspan(-meta["m_max"], meta["m_max"], color="green", alpha=0.10, label="coherent range")
    ax.set_xlabel("steering magnitude m")
    ax.set_ylabel("median semantic log odds")
    ax.legend()
    ax.set_title(f"dose-response at layer {meta['layer']}")
    annotate_n(ax, len(mag))

    ax = axes[0][4]
    ax.plot(mag["magnitude"], mag["valid"], "o-", label="validity")
    ax.plot(mag["magnitude"], mag["rep4"], "s-", label="repeated 4-gram")
    ax.axvspan(-meta["m_max"], meta["m_max"], color="green", alpha=0.10)
    ax.set_xlabel("steering magnitude m")
    ax.set_ylabel("rate")
    ax.legend()
    ax.set_title(f"m_max = {meta['m_max']:.2f}, delta = {meta['delta']:.2f}")

    finish(cfg, fig, "fig02_steering_calibration", mag)
