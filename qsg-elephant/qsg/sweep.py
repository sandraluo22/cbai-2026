"""Sweep entry point: expand the Cartesian product, estimate cost, run.

    python -m qsg.sweep --config configs/base.yaml --dry-run
    python -m qsg.sweep --config configs/base.yaml
    python -m qsg.sweep --config configs/base.yaml --override qsg.rounds=3 base.smoke_test=true
    python -m qsg.sweep --single --config configs/base.yaml      # run only base RunConfig
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .activations import estimate_disk_bytes
from .config import RunConfig, SweepConfig, load_sweep_config
from .engine import run_single


# Rough hidden-dim / layer-count priors for the disk estimate before any model loads.
_MODEL_PRIORS = {
    "Llama-3.1-8B": (32, 4096),
    "Qwen2-VL-7B": (28, 3584),
    "Llama-3.2-11B-Vision": (40, 4096),
    "tiny-gpt2": (2, 2),
}


def _prior_for(cfg: RunConfig) -> tuple[int, int]:
    name = (cfg.model.smoke_text_model if cfg.smoke_test
            else (cfg.model.vision_model if cfg.arm.value == "image" else cfg.model.text_model))
    for key, val in _MODEL_PRIORS.items():
        if key.split("-")[0].lower() in name.lower() and key.lower().split("/")[-1] in name.lower():
            return val
    for key, val in _MODEL_PRIORS.items():
        if key.lower() in name.lower():
            return val
    return (32, 4096)


def estimate_sweep(runs: list[RunConfig]) -> dict:
    total_bytes = 0
    total_llm_calls = 0
    for rc in runs:
        n_layers, hidden = _prior_for(rc)
        if rc.activations.capture:
            total_bytes += estimate_disk_bytes(
                rc.qsg.rounds + 1, rc.qsg.n, n_layers, hidden, rc.activations.dtype
            )
        # LLM calls: seed (N) + two_layer re-reads (N per round). Each soft readout
        # is ~ (1 anchor fwd + K continuation fwds); hard adds 1 generate.
        if rc.coupling_mode.value == "two_layer":
            readouts = rc.qsg.n * (rc.qsg.rounds + 1)
        else:
            readouts = rc.qsg.n
        total_llm_calls += readouts
    return {"n_runs": len(runs), "disk_gb": total_bytes / 1e9,
            "approx_llm_readouts": total_llm_calls}


def main() -> None:
    ap = argparse.ArgumentParser(description="QSG x Elephant sweep")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="dotted key=value overrides")
    ap.add_argument("--dry-run", action="store_true", help="print job list + estimate and exit")
    ap.add_argument("--single", action="store_true", help="run only the base RunConfig")
    args = ap.parse_args()

    sweep: SweepConfig = load_sweep_config(args.config, args.override)
    runs = [sweep.base] if args.single else sweep.expand()

    est = estimate_sweep(runs)
    print(f"\n=== Sweep: {est['n_runs']} runs ===")
    print(f"  estimated activation disk: {est['disk_gb']:.2f} GB")
    print(f"  approx LLM readouts (soft+hard passes): {est['approx_llm_readouts']:,}")
    for i, rc in enumerate(runs):
        print(f"  [{i:3d}] {rc.run_id()}  arm={rc.arm.value} mode={rc.coupling_mode.value}")

    if args.dry_run:
        print("\n--dry-run: exiting before any model load.")
        return

    Path(runs[0].output_root).mkdir(parents=True, exist_ok=True)
    manifests = []
    for i, rc in enumerate(runs):
        print(f"\n--- run {i + 1}/{len(runs)}: {rc.run_id()} ---")
        manifests.append(run_single(rc))
    Path(runs[0].output_root, "sweep_manifests.json").write_text(json.dumps(manifests, indent=2))
    print(f"\nSweep complete. {len(manifests)} runs -> {runs[0].output_root}/")


if __name__ == "__main__":
    main()
