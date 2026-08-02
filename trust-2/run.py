"""CLI entry point for the trust-2 experiment.

Generate trials for one or more conditions, run them against Llama-3.1-8B (local
HF transformers; default ``NousResearch/Meta-Llama-3.1-8B-Instruct``), save raw
responses, and optionally analyse.

Examples
--------
    # dry run (no GPU needed) to confirm the pipeline end to end
    python run.py --condition all --n-trials 2 --backend mock --analyze

    # the real run on the GPU box
    python run.py --condition all --n-trials 40 --seed 0 --analyze

    # a single condition
    python run.py --condition dose --n-trials 40 --seed 0

    # analyse an existing results file
    python run.py --analyze-only --results results/all_s0/results.json
"""

from __future__ import annotations

import argparse
import json
import os

from conditions import CONDITIONS
from trials import generate_trials
import harness
import analyze as A


ALL_CONDITIONS = [c for c in CONDITIONS]   # baseline + the five conditions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="all",
                    help=f"condition name or 'all'. choices: all, {', '.join(ALL_CONDITIONS)}")
    ap.add_argument("--n-trials", type=int, default=20,
                    help="trials per condition (dose sweeps its 4 doses within this)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-claims", type=int, default=10,
                    help="verifiable claims per source (conditions that fix it ignore this)")
    ap.add_argument("--model", default=harness.DEFAULT_MODEL)
    ap.add_argument("--backend", choices=["llama", "mock"], default="llama")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--cache-dir", default=None,
                    help="response cache (default: <output-dir>/cache)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--analyze", action="store_true", help="analyse after running")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip running; analyse --results")
    ap.add_argument("--results", default=None, help="results JSON for --analyze-only")
    args = ap.parse_args()

    # analyse-only shortcut
    if args.analyze_only:
        if not args.results:
            ap.error("--analyze-only requires --results")
        outdir = args.output_dir or os.path.join(
            os.path.dirname(args.results) or ".", "analysis")
        summary = A.analyze(args.results, outdir)
        A._print_summary(summary)
        print(f"wrote analysis to {outdir}")
        return

    conditions = ALL_CONDITIONS if args.condition == "all" else [args.condition]
    for c in conditions:
        if c not in CONDITIONS:
            ap.error(f"unknown condition {c!r}; choices: all, {', '.join(ALL_CONDITIONS)}")

    tag = ("all" if args.condition == "all" else args.condition) + f"_s{args.seed}"
    outdir = args.output_dir or os.path.join("results", tag)
    os.makedirs(outdir, exist_ok=True)
    cache_dir = None if args.no_cache else (args.cache_dir or os.path.join(outdir, "cache"))

    cfg = harness.GenConfig(model_name=args.model, device=args.device, dtype=args.dtype,
                            max_batch=args.max_batch, max_new_tokens=args.max_new_tokens)

    # generate all trials up front (deterministic per condition+seed)
    all_trials = []
    for c in conditions:
        all_trials.extend(generate_trials(c, args.n_trials, args.seed,
                                          n_claims=args.n_claims))
    print(f"[run] {len(all_trials)} trials across {len(conditions)} condition(s) "
          f"| backend={args.backend} model={args.model}", flush=True)

    backend = harness.make_backend(args.backend, cfg)
    results = harness.run_trials(all_trials, backend, cfg, cache_dir=cache_dir,
                                 use_cache=not args.no_cache)

    out = {
        "meta": {"conditions": conditions, "n_trials": args.n_trials,
                 "seed": args.seed, "n_claims": args.n_claims,
                 "model": args.model, "backend": args.backend},
        "results": results,
    }
    results_path = os.path.join(outdir, "results.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    n_ok = sum(1 for r in results if r["response"]["parse_ok"])
    print(f"[run] wrote {results_path}  ({n_ok}/{len(results)} parsed)", flush=True)

    if args.analyze:
        summary = A.analyze(results_path, os.path.join(outdir, "analysis"))
        A._print_summary(summary)
        print(f"[run] wrote analysis to {os.path.join(outdir, 'analysis')}")


if __name__ == "__main__":
    main()
