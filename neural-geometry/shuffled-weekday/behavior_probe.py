"""Behavioral read-out at multiple context depths: two probes per checkpoint.

1. WALK-CONTINUATION posterior ("in-context behavior"): truncate a walk at step
   s, read P(next word | prefix) restricted to the 7 day words. The geometry of
   these posteriors (Hellinger-embedded) is the *behavior manifold* of the
   in-context ring at depth s.

2. SEMANTIC "x days after y" probe ("pretrained behavior under context
   pressure"): append `\n{k} days after {day} is` to the same prefix and read
   the restricted posterior. If the context ring wins outright, even this
   linguistically-semantic query should drift toward permuted-ring adjacency;
   if the priors survive (Park et al. suggest they superpose), it stays on the
   semantic cycle. Divergence between probe 1 and probe 2 at the same depth is
   direct evidence for two coexisting behavior manifolds.

Output: behavior.npz with parallel arrays
  walk_id, step, node (current day at truncation), probe ("cont"|"after{k}"),
  query_node (for semantic probes: the y in "k days after y"),
  posterior [n_rows, 7], day_mass [n_rows]

Usage:
  python behavior_probe.py                     # defaults matching capture_ctx
  python behavior_probe.py --smoke
  python behavior_probe.py --checkpoints 10 30 100 300 800 --n-probe-walks 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nglib
from nglib import DAYS_PERMUTED, build_graph, generate_walks, cm_models

from capture_ctx import make_cfg, parse_args as capture_parse_args


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--walk-length", type=int, default=800)
    ap.add_argument("--n-walks", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[10, 20, 30, 50, 75, 100, 150, 200, 300, 450, 600, 800])
    ap.add_argument("--n-probe-walks", type=int, default=12,
                    help="walks used for probing (each checkpoint x walk = 1+ forwards)")
    ap.add_argument("--after-ks", type=int, nargs="+", default=[2],
                    help="k values for the 'k days after y' semantic probe")
    # ensure make_cfg compatibility
    ap.add_argument("--layers", type=int, nargs="+", default=[26])
    return ap.parse_args()


NUM_WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def semantic_probe_text(prefix: str, k: int, day: str) -> str:
    return f"{prefix}\n{NUM_WORD[k]} days after {day} is"


def main():
    args = parse_args()
    cfg = make_cfg(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.checkpoints = [10, 30, 60]
        args.n_probe_walks = 3

    graph = build_graph(cfg)
    walks = generate_walks(graph, cfg)[: args.n_probe_walks]
    model, tok = cm_models.load_model(cfg.model_a, cfg)
    day_ids = nglib.node_first_token_ids(tok, DAYS_PERMUTED)
    word_to_node = {w: i for i, w in enumerate(DAYS_PERMUTED)}

    rows = {k: [] for k in
            ("walk_id", "step", "node", "probe", "query_node", "posterior", "day_mass")}

    def add(wid, step, node, probe, qnode, post, mass):
        rows["walk_id"].append(wid); rows["step"].append(step)
        rows["node"].append(node); rows["probe"].append(probe)
        rows["query_node"].append(qnode); rows["posterior"].append(post)
        rows["day_mass"].append(mass)

    for wk in walks:
        for s in args.checkpoints:
            if s > len(wk.words):
                continue
            prefix = " ".join(wk.words[:s])
            cur = wk.nodes[s - 1]
            # probe 1: walk continuation
            p, mass = nglib.next_word_posterior(model, tok, prefix, day_ids, cfg.device)
            add(wk.walk_id, s, cur, "cont", -1, p, mass)
            # probe 2: semantic relation, one query day per checkpoint (rotate
            # through days across walks so all 7 get covered without 7x cost)
            for k in args.after_ks:
                qnode = (wk.walk_id + s) % 7
                text = semantic_probe_text(prefix, k, DAYS_PERMUTED[qnode])
                p, mass = nglib.next_word_posterior(model, tok, text, day_ids, cfg.device)
                add(wk.walk_id, s, cur, f"after{k}", qnode, p, mass)
        print(f"[probe] walk {wk.walk_id} done")

    # no-context control for the semantic probe (the pure pretrained relation)
    for k in args.after_ks:
        for qnode, day in enumerate(DAYS_PERMUTED):
            text = semantic_probe_text("", k, day).lstrip("\n")
            p, mass = nglib.next_word_posterior(model, tok, text, day_ids, cfg.device)
            add(-1, 0, -1, f"after{k}_noctx", qnode, p, mass)

    np.savez_compressed(
        out / "behavior.npz",
        walk_id=np.array(rows["walk_id"], dtype=np.int32),
        step=np.array(rows["step"], dtype=np.int32),
        node=np.array(rows["node"], dtype=np.int32),
        probe=np.array(rows["probe"]),
        query_node=np.array(rows["query_node"], dtype=np.int32),
        posterior=np.stack(rows["posterior"]).astype(np.float32),
        day_mass=np.array(rows["day_mass"], dtype=np.float32),
    )
    print(f"[probe] {len(rows['walk_id'])} probe rows -> {out/'behavior.npz'}")


if __name__ == "__main__":
    main()
