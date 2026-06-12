"""Render a human-readable transcript of a run from its logged artifacts.

Reconstructs each agent's private attribute (deterministic from the seed) and
replays, per round, the gossip message each agent received and the belief it then
held. This is the transcript of the QSG dynamics; raw model generations are not
stored (only parsed beliefs), so this shows beliefs + messages, not free text.

    python render_transcript.py results_exp2/text_two_layer_N6_a0p3_m1p0_seed1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qsg.config import load_run_config
from qsg.engine import assign_observations, load_ontology


def top(dist, cand, k=2):
    return ", ".join(f"{c}:{p:.2f}" for c, p in sorted(zip(cand, dist), key=lambda z: -z[1])[:k])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--max-rounds", type=int, default=99)
    args = ap.parse_args()
    d = Path(args.run_dir)

    cfg = load_run_config(d / "config.yaml")
    ont = load_ontology(cfg.ontology_path if Path(cfg.ontology_path).exists()
                        else "configs/ontology.yaml")
    cands, obj, gt, attrs = assign_observations(cfg, ont, np.random.default_rng(cfg.seed))

    lines = [json.loads(x) for x in (d / "beliefs.jsonl").read_text().splitlines()]
    by_round: dict[int, list] = {}
    for l in lines:
        by_round.setdefault(l["round"], []).append(l)

    print(f"RUN: {d.name}")
    print(f"hidden object = {obj.upper()}   (ablation={cfg.neutral_ablation})")
    print(f"candidates = {cands}")
    print("private attribute each agent sensed:")
    for i, a in enumerate(attrs):
        print(f"  agent {i}: \"{a}\"" + ("   (IGNORED — ablation)" if cfg.neutral_ablation else ""))
    print("=" * 78)

    for r in sorted(by_round)[: args.max_rounds + 1]:
        tag = "SEED" if r == 0 else f"round {r}"
        pop = np.mean([l["soft_canonical"] for l in by_round[r]], axis=0)
        print(f"\n[{tag}]  population belief -> {top(pop, cands, 3)}")
        for l in sorted(by_round[r], key=lambda x: x["agent"]):
            msg = l["message_text"] or "(none)"
            flag = "  <DEGENERATE>" if l["degenerate"] else ""
            print(f"   agent {l['agent']}: heard \"{msg}\"  =>  belief [{top(l['soft_canonical'], cands)}]{flag}")


if __name__ == "__main__":
    main()
