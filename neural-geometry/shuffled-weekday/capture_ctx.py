"""Capture activations for the shuffled-weekday manifold-shift experiment.

Setup (semantic-conflict condition of Park et al., read out geometrically):
nodes are the 7 weekdays arranged on a ring in PERMUTED order (each ring step =
+3 days, see config.DAYS_PERMUTED), so the in-context ring conflicts with the
pretrained weekday cycle. We capture the per-occurrence residual stream across
the whole walk, so downstream analysis can bin occurrences by context length
and watch the manifold move.

Also captures the PRETRAINED weekday ring baseline: day-token activations in
neutral one-sentence templates (no walk context). This anchors the "old ring"
subspace that the in-context ring competes with.

Outputs (in --out, default runs/):
  walk_capture.npz      per-occurrence acts at each captured layer + metadata
  pretrained_ring.npz   [7, d] mean day-token activation per node, per layer
  meta.json             config echo

Usage:
  python capture_ctx.py                  # Llama-3.1-8B on cuda
  python capture_ctx.py --smoke          # distilgpt2 on cpu, tiny walks
  python capture_ctx.py --model google/gemma-2-9b --layers 24 28 32
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nglib
from nglib import Config, DAYS_PERMUTED, build_graph, generate_walks, cm_models

# Neutral templates for the pretrained-ring baseline. The {day} token's
# activation is captured in each and averaged; templates deliberately avoid
# any relational day content ("after", "before") so the baseline reflects the
# resting weekday geometry, not task-primed geometry.
BASELINE_TEMPLATES = [
    "The meeting is scheduled for {day}.",
    "I will see you on {day}.",
    "{day} is usually a busy day.",
    "The store is closed on {day}.",
    "Her flight leaves on {day}.",
    "We always have dinner together on {day}.",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24, 26, 28])
    ap.add_argument("--walk-length", type=int, default=800)
    ap.add_argument("--n-walks", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--smoke", action="store_true",
                    help="distilgpt2/cpu, tiny walks; tests the plumbing only")
    return ap.parse_args()


def make_cfg(args) -> Config:
    base = dict(
        name="shuffled_weekday",
        graph_type="ring", ring_size=7, word_set="days",
        walk_length=args.walk_length, n_walks=args.n_walks, seed=args.seed,
        model_a=args.model, model_b=args.model,
        device=args.device,
        capture_layers_a=tuple(args.layers), capture_layers_b=tuple(args.layers),
        align_layer_a=args.layers[-1], align_layer_b=args.layers[-1],
    )
    if args.smoke:
        base.update(model_a="distilgpt2", model_b="distilgpt2", device="cpu",
                    dtype="float32", walk_length=80, n_walks=8,
                    capture_layers_a=(3,), capture_layers_b=(3,),
                    align_layer_a=3, align_layer_b=3)
    return Config(**base)


def capture_pretrained_ring(model, tok, cfg: Config, layers) -> dict:
    """[7, d] mean day-token activation per layer, from the neutral templates."""
    from nglib import Walk
    per_layer = {l: [[] for _ in range(7)] for l in layers}
    grabbed, handles = cm_models._register_hooks(model, layers)
    try:
        import torch
        for node, day in enumerate(DAYS_PERMUTED):
            for tmpl in BASELINE_TEMPLATES:
                text = tmpl.format(day=day)
                start = text.index(day)
                enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
                # last subword of the day mention (matches cfg.subword_rule)
                toks = [i for i, (s, e) in enumerate(enc["offset_mapping"])
                        if not (s == 0 and e == 0)
                        and s < start + len(day) and e > start]
                assert toks, f"no tokens matched {day!r} in {text!r}"
                grabbed.clear()
                with torch.no_grad():
                    model(input_ids=torch.tensor([enc["input_ids"]]).to(cfg.device))
                for l in layers:
                    h = grabbed[l][0][toks[-1]].float().cpu().numpy()
                    per_layer[l][node].append(h)
    finally:
        for h in handles:
            h.remove()
    return {l: np.stack([np.mean(v, axis=0) for v in rows])
            for l, rows in per_layer.items()}


def main():
    args = parse_args()
    cfg = make_cfg(args)
    layers = list(cfg.capture_layers_a)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    graph = build_graph(cfg)
    walks = generate_walks(graph, cfg)
    print(f"[capture] {cfg.model_a}: {cfg.n_walks} walks x {cfg.walk_length} steps, "
          f"layers {layers}")

    model, tok = cm_models.load_model(cfg.model_a, cfg)

    result = cm_models.capture(model, tok, walks, tuple(layers), cfg)
    cm_models.save_capture(result, str(out / "walk_capture.npz"))
    print(f"[capture] walk activations -> {out/'walk_capture.npz'} "
          f"({result.meta['walk_id'].shape[0]} occurrences, d={result.hidden_size})")

    ring = capture_pretrained_ring(model, tok, cfg, layers)
    np.savez_compressed(out / "pretrained_ring.npz",
                        **{f"layer_{l}": v for l, v in ring.items()},
                        _layers=np.array(layers))
    print(f"[capture] pretrained weekday ring -> {out/'pretrained_ring.npz'}")

    (out / "meta.json").write_text(json.dumps(
        {"config": dataclasses.asdict(cfg), "layers": layers,
         "days_permuted": DAYS_PERMUTED}, indent=2))


if __name__ == "__main__":
    main()
