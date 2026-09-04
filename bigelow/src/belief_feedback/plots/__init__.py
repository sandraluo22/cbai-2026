"""Figure generation (Part 19): all figures as PDF + 300-dpi PNG + data."""

from __future__ import annotations

from ..config import Config
from ..logging_utils import get_logger, now_iso, write_manifest

log = get_logger(__name__)


def make_all(cfg: Config) -> list[str]:
    from . import (
        decomposition,
        exogenous,
        hysteresis,
        jacobian,
        mechanistic,
        phase_boundary,
        recycling,
        robustness,
        schematic,
        steering,
        trajectories,
    )

    started = now_iso()
    jobs = [
        ("fig01", schematic.make),
        ("fig02", steering.make),
        ("fig03", exogenous.make_fig03),
        ("fig04", exogenous.make_fig04),
        ("fig05", trajectories.make_fig05),
        ("fig06", trajectories.make_fig06),
        ("fig07", decomposition.make_fig07),
        ("fig08", recycling.make),
        ("fig09", hysteresis.make),
        ("fig10", phase_boundary.make),
        ("fig11", jacobian.make),
        ("fig12", mechanistic.make),
        ("fig13", decomposition.make_fig13),
        ("fig14", robustness.make),
    ]
    made, failed = [], []
    for name, fn in jobs:
        try:
            fn(cfg)
            made.append(name)
            log.info("figure %s done", name)
        except Exception as exc:  # noqa: BLE001 - report, then continue
            failed.append((name, repr(exc)))
            log.error("figure %s FAILED: %r", name, exc)
    write_manifest(
        cfg, "figures", started=started,
        artifact_paths=[str(cfg.paths.figures)],
        completed_jobs=len(made), failed_jobs=len(failed),
        extra={"failed": failed},
    )
    if failed:
        raise RuntimeError(f"figures failed: {failed}")
    return made
