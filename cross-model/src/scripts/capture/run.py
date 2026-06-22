"""Orchestration.

Stages (each can run independently so inference is paid once):
  capture   : generate graph+walks, run BOTH models SEQUENTIALLY, cache acts
  reproduce : paper-reproduction sanity check from cached acts (+ scatter plots)
  align     : fit A->B map(s) and report metrics from cached acts
  all       : capture -> reproduce -> align

Example:
  huggingface-cli login          # both models are gated; accept licenses first
  python run.py --preset default --stage all
  python run.py --preset smoke   --stage all      # tiny CPU end-to-end test

Activations are saved to <out_dir>/<preset>/ so align/reproduce re-run offline
without re-inference.
"""

from __future__ import annotations

import argparse
import json
import os
import gc

import numpy as np

from config import get_config
import graph as G


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _paths(cfg):
    run_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "cap_a": os.path.join(run_dir, "acts_model_a.npz"),
        "cap_b": os.path.join(run_dir, "acts_model_b.npz"),
        "walks": os.path.join(run_dir, "walks.json"),
        "config": os.path.join(run_dir, "config.json"),
        "reproduce": os.path.join(run_dir, "reproduce.json"),
        "align": os.path.join(run_dir, "alignment.json"),
        "plot_a": os.path.join(run_dir, "grid_recovery_a.png"),
        "plot_b": os.path.join(run_dir, "grid_recovery_b.png"),
    }


def _free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _delete_hf_cache(model_name: str):
    """Remove a model's HF weight cache to reclaim disk (used when both models
    can't co-reside on the volume). Frees the weights only; activations are
    already saved elsewhere by this point."""
    import shutil
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    slug = "models--" + model_name.replace("/", "--")
    for sub in ("hub/" + slug, "hub/.locks/" + slug):
        p = os.path.join(hf_home, sub)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
            print(f"[capture]   freed cache {p}")


def stage_capture(cfg, paths):
    """Generate shared walks once, then capture activations for both models.

    Parallel (co-resident) path is used when cfg.parallel_models is set AND the
    device has enough VRAM (e.g. H200): both models stay loaded and each walk is
    run through both in one shared pass. Otherwise models are loaded/run one at a
    time, freeing the first before the second (bounded peak memory). Either way
    activations are cached to disk so alignment re-runs offline."""
    import models as M

    graph = G.build_grid_graph(cfg)
    walks = G.generate_walks(graph, cfg)
    with open(paths["walks"], "w") as f:
        json.dump([{"walk_id": w.walk_id, "nodes": w.nodes, "words": w.words}
                   for w in walks], f)
    with open(paths["config"], "w") as f:
        f.write(cfg.to_json())

    n_occ = sum(len(w.nodes) for w in walks)
    print(f"[capture] graph={cfg.grid_rows}x{cfg.grid_cols} ({cfg.n_nodes} nodes), "
          f"{cfg.n_walks} walks, {n_occ:,} occurrences/model")

    want_parallel = cfg.parallel_models and M.enough_memory_for_parallel(cfg, cfg.parallel_min_gb)
    if cfg.parallel_models and not want_parallel:
        print(f"[capture] parallel requested but device lacks >= {cfg.parallel_min_gb} GB "
              "VRAM (or not CUDA); falling back to sequential.")

    if want_parallel:
        print("[capture] co-resident parallel mode: loading BOTH models")
        model_a, tok_a = M.load_model(cfg.model_a, cfg)
        model_b, tok_b = M.load_model(cfg.model_b, cfg)
        entries = [
            {"model": model_a, "tokenizer": tok_a, "layers": cfg.capture_layers_a,
             "name": cfg.model_a, "device": cfg.device},
            {"model": model_b, "tokenizer": tok_b, "layers": cfg.capture_layers_b,
             "name": cfg.model_b, "device": cfg.device},
        ]
        res_a, res_b = M.capture_many(entries, walks, cfg)
        M.save_capture(res_a, paths["cap_a"])
        M.save_capture(res_b, paths["cap_b"])
        print(f"[capture]   saved A d={res_a.hidden_size} / B d={res_b.hidden_size}")
        del model_a, model_b, tok_a, tok_b, res_a, res_b
        _free_cuda()
        return

    specs = (
        ("A", cfg.model_a, cfg.capture_layers_a, paths["cap_a"]),
        ("B", cfg.model_b, cfg.capture_layers_b, paths["cap_b"]),
    )
    for i, (which, name, layers, out) in enumerate(specs):
        print(f"[capture] loading model {which}: {name}")
        model, tok = M.load_model(name, cfg)
        res = M.capture(model, tok, walks, layers, cfg)
        M.save_capture(res, out)
        print(f"[capture]   saved {out}  acts d={res.hidden_size} "
              f"layers={sorted(res.acts)}")
        del model, tok, res
        _free_cuda()
        # free this model's weights before the next download if disk is tight
        if cfg.free_cache_between and i < len(specs) - 1:
            _delete_hf_cache(name)


def stage_reproduce(cfg, paths):
    import models as M
    import reproduce as R

    graph = G.build_grid_graph(cfg)
    cap_a = M.load_capture(paths["cap_a"])
    cap_b = M.load_capture(paths["cap_b"])

    out = {
        "model_a": R.reproduce_for_model(cap_a, cfg.align_layer_a, graph, cfg,
                                         tag="model_a", plot_path=paths["plot_a"]),
        "model_b": R.reproduce_for_model(cap_b, cfg.align_layer_b, graph, cfg,
                                         tag="model_b", plot_path=paths["plot_b"]),
    }
    with open(paths["reproduce"], "w") as f:
        json.dump(out, f, indent=2)
    for k in ("model_a", "model_b"):
        print(f"[reproduce] {k}: grid distance_corr (high ctx) = "
              f"{out[k]['top_distance_corr']:.3f}")
    print(f"[reproduce] wrote {paths['reproduce']}")


def stage_align(cfg, paths):
    import models as M
    import align as A

    cap_a = M.load_capture(paths["cap_a"])
    cap_b = M.load_capture(paths["cap_b"])
    rep = A.run_alignment(cap_a, cap_b, cfg)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, np.generic):
            return o.item()
        return o

    with open(paths["align"], "w") as f:
        json.dump(_clean(rep.to_dict()), f, indent=2)

    print("[align] well-posedness:")
    for w in rep.wellposed:
        print(f"  - {w['msg']}  {'OK' if w['ok'] else '*** UNDER-DETERMINED ***'}")
    print(f"[align] ridge  R^2 test = {rep.ridge['r2_test']:.3f}")
    print(f"[align] procrustes residual test = {rep.procrustes['residual_test']:.3f}")
    print(f"[align] linear CKA (held-out) = {rep.cka_overall:.3f}")
    print("[align] alignment vs context length:")
    for r in rep.by_context:
        if r.get("skipped"):
            print(f"    ctx~{r['context']}: skipped (n={r['n']})")
        else:
            print(f"    ctx~{r['context']:>4} (n={r['n']:>5}): "
                  f"ridge_r2={r['ridge_r2']:.3f}  cka={r['cka']:.3f}")
    mm = rep.matched_vs_mismatched
    print(f"[align] matched>mismatched in {mm['matched_wins']}/{mm['total']} bins")
    print(f"[align] wrote {paths['align']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="default")
    ap.add_argument("--stage", default="all",
                    choices=["capture", "reproduce", "align", "all"])
    args = ap.parse_args()

    cfg = get_config(args.preset)
    _seed_everything(cfg.seed)
    paths = _paths(cfg)
    print(f"== cross-model alignment :: preset={cfg.name} stage={args.stage} ==")

    if args.stage in ("capture", "all"):
        stage_capture(cfg, paths)
    if args.stage in ("reproduce", "all"):
        stage_reproduce(cfg, paths)
    if args.stage in ("align", "all"):
        stage_align(cfg, paths)


if __name__ == "__main__":
    main()
