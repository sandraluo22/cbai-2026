"""Run ONE complete 7-player Avalon game on claude-opus-4-8 and write both logs.

    python run_game.py [--seed N] [--out results/game1]

Requires ANTHROPIC_API_KEY in the environment (presence checked; value never read
into our code, never logged).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from avalon.config import GameConfig
from avalon.engine import Engine
from avalon.player import ApiError, api_key_present
from avalon.transcript import GameLogger


def _load_dotenv():
    """Load KEY=VALUE lines from ./.env into os.environ if present. The value is
    set into the environment for the SDK and is never printed or logged."""
    import os
    p = Path(__file__).parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/game1")
    args = ap.parse_args()

    if not api_key_present():
        print("ERROR: ANTHROPIC_API_KEY is not set in the environment. "
              "Set it (e.g. in your shell profile) and re-run. The value is never printed or logged.",
              file=sys.stderr)
        sys.exit(2)

    cfg = GameConfig(seed=args.seed)
    engine = Engine(cfg)                       # assigns roles (seeded)
    logger = GameLogger(args.out, engine.assignment, engine.knowledge)
    engine.logger = logger
    print(f"Running one Avalon game (seed={cfg.seed}, model={cfg.default_model}) ...")
    try:
        summary = engine.run()
        status = "completed"
    except ApiError as e:
        summary = {"winner": None, "error": str(e), "true_roles": {s: r.value for s, r in engine.assignment.items()}}
        status = "errored"
        print(f"GAME ERRORED (logged): {e}", file=sys.stderr)

    summary = logger.finalize({**summary, "status": status, "seed": cfg.seed})
    print(f"\n=== OUTCOME: {summary.get('winner')} — {summary.get('reason', summary.get('error',''))} ===")
    print(f"missions: successes={summary.get('successes')} fails={summary.get('fails')}")
    print(f"violations: {summary.get('violations')}")
    print(f"token usage: {summary['cost']}")
    print(f"logs -> {args.out}/  (godview_transcript.txt, events.jsonl, legal_views.jsonl, summary.json)")


if __name__ == "__main__":
    main()
