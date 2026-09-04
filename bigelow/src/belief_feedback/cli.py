"""Command-line interface: every stage of the pipeline, in dependency order."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from .config import Config, load_config
from .logging_utils import get_logger

log = get_logger(__name__)


def _generate_worlds(cfg: Config) -> None:
    from .world.generator import generate_all_worlds

    generate_all_worlds(cfg)


def _validate_data(cfg: Config) -> None:
    from .world.validation import validate_dataset

    report = validate_dataset(cfg)
    if not report["passed"]:
        raise SystemExit(f"data validation failed: {report['problems']}")


def _train_steering(cfg: Config) -> None:
    from .experiments import steering_dataset

    steering_dataset.run(cfg)


def _calibrate_steering(cfg: Config) -> None:
    from .experiments import steering_calibration

    steering_calibration.run(cfg)


def _run_emission(cfg: Config) -> None:
    from .experiments import exogenous_emission

    exogenous_emission.run(cfg)


def _run_receiver(cfg: Config) -> None:
    from .experiments import exogenous_receiver

    exogenous_receiver.run(cfg)


def _fit_models(cfg: Config) -> None:
    from .analysis import fit_emission, fit_receiver

    fit_emission.fit(cfg)
    fit_receiver.fit(cfg)


def _run_network(cfg: Config) -> None:
    from .experiments import network_runner

    network_runner.run(cfg)


def _run_recycling(cfg: Config) -> None:
    from .experiments import recycling

    recycling.run(cfg)


def _run_hysteresis(cfg: Config) -> None:
    from .experiments import hysteresis

    hysteresis.run(cfg)


def _run_phase(cfg: Config) -> None:
    from .experiments import phase_boundary

    phase_boundary.run(cfg)


def _run_jacobian(cfg: Config) -> None:
    from .experiments import jacobian

    jacobian.run(cfg)


def _run_mechanistic(cfg: Config) -> None:
    from .experiments import mechanistic

    mechanistic.run(cfg)


def _run_robustness(cfg: Config) -> None:
    from .experiments import robustness

    robustness.run(cfg)


def _analyze(cfg: Config) -> None:
    from .analysis import analyze

    analyze.run(cfg)


def _make_figures(cfg: Config) -> None:
    from . import plots

    plots.make_all(cfg)


def _make_tables(cfg: Config) -> None:
    from .analysis import tables

    tables.make_all(cfg)


def _make_report(cfg: Config) -> None:
    from .analysis.report import make_report

    make_report(cfg)


COMMANDS: dict[str, Callable[[Config], None]] = {
    "generate-worlds": _generate_worlds,
    "validate-data": _validate_data,
    "train-steering": _train_steering,
    "calibrate-steering": _calibrate_steering,
    "run-exogenous-emission": _run_emission,
    "run-exogenous-receiver": _run_receiver,
    "fit-models": _fit_models,
    "run-network": _run_network,
    "run-recycling": _run_recycling,
    "run-hysteresis": _run_hysteresis,
    "run-phase-boundary": _run_phase,
    "run-jacobian": _run_jacobian,
    "run-mechanistic": _run_mechanistic,
    "run-robustness": _run_robustness,
    "analyze": _analyze,
    "make-tables": _make_tables,
    "make-figures": _make_figures,
    "make-report": _make_report,
}

PIPELINE_ORDER = list(COMMANDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="belief_feedback", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in [*COMMANDS, "all"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="path to a YAML config")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    log.info("command=%s config=%s (%s backend)", args.command, cfg.name, cfg.model.backend)
    if args.command == "all":
        for name in PIPELINE_ORDER:
            log.info("=== stage: %s ===", name)
            COMMANDS[name](cfg)
    else:
        COMMANDS[args.command](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
