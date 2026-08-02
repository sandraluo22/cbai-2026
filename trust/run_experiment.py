"""Sweep accuracy-gap x rep-strength x info-ladder (+ robustness + sanity), play games,
log every per-(round, company) observation. One API call per round.

Design = an ANCHORED sweep (not a full cross-product) to keep cost bounded while
covering every headline analysis:
  * gap sweep        : gaps x {main rep, main info}                  -> S5 (key plot)
  * rep-strength      : {anchor gap} x rep_strengths x {main info}    -> S6
  * info-ladder       : {anchor gap} x {main rep} x info_levels       -> S8
  * robustness        : anchor main cell, label/order/paraphrase swap -> S9
  * sanity            : no-advisor baseline + comprehension probe

Safeguards (per the build spec):
  --dry-run  prints the condition list + API-call estimate and exits (no API).
  --smoke    plays ONE short game end-to-end and prints, round by round, the prompt,
             source estimates, model estimates, revealed truths, and the Bayesian
             trajectory side by side — the Step-2 review gate.
  a full run exceeding `max_calls` refuses without --yes.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

import env as E
import bayes as B
import prompt as P
from runner import ModelConfig, play_game, comprehension_probe


# --------------------------------------------------------------------------- #
@dataclass
class Condition:
    name: str
    kind: str                          # main | rep | info | robust | sanity
    gap: float
    rep_strength: str
    info_level: str
    rep_letter: str = "A"
    order: tuple = ("A", "B")
    paraphrase: int = 0
    n_games: int = 20

    def style(self) -> P.PromptStyle:
        return P.PromptStyle(rep_strength=self.rep_strength, info_level=self.info_level,
                             rep_letter=self.rep_letter, order=self.order,
                             paraphrase=self.paraphrase)


def build_conditions(cfg: dict) -> list[Condition]:
    d = cfg["design"]
    gaps, anchor = d["gaps"], d["anchor_gap"]
    main_rep, main_info = d["main_rep"], d["main_info"]
    g, gr = d["games"], d["robust_games"]
    conds: list[Condition] = []

    for gap in gaps:                                  # gap sweep (S5)
        conds.append(Condition(f"gap{gap}_{main_rep}_{main_info}", "main", gap,
                               main_rep, main_info, n_games=g))
    for rs in d["rep_strengths"]:                     # rep-strength sweep at anchor (S6)
        if rs == main_rep:
            continue
        conds.append(Condition(f"gap{anchor}_{rs}_{main_info}", "rep", anchor, rs,
                               main_info, n_games=g))
    for il in d["info_levels"]:                       # info-ladder at anchor (S8)
        if il == main_info:
            continue
        conds.append(Condition(f"gap{anchor}_{main_rep}_{il}", "info", anchor, main_rep,
                               il, n_games=g))
    for rv in d.get("robustness", []):                # robustness at anchor main cell (S9)
        kw = dict(gap=anchor, rep_strength=main_rep, info_level=main_info, n_games=gr)
        if rv == "swap":
            conds.append(Condition(f"gap{anchor}_robust_swaplabels", "robust",
                                   rep_letter="B", **kw))
        elif rv == "order":
            conds.append(Condition(f"gap{anchor}_robust_order", "robust",
                                   order=("B", "A"), **kw))
        elif rv == "paraphrase":
            conds.append(Condition(f"gap{anchor}_robust_paraphrase", "robust",
                                   paraphrase=1, **kw))
    conds.append(Condition(f"gap{anchor}_{main_rep}_no_advisor", "sanity", anchor,
                           main_rep, "no_advisor", n_games=gr))
    return conds


def env_cfg_for(cfg: dict, gap: float) -> E.EnvConfig:
    e = cfg["env"]
    return E.EnvConfig(M=e["M"], T=e["T"], mu=e["mu"], theta_scale=e["theta_scale"],
                       sigma_B=e["sigma_B"], gap=gap)


def prior_from(cfg: dict) -> B.ReputationPrior:
    return B.ReputationPrior(**cfg["prior"])


def n_calls(conds: list[Condition], cfg: dict) -> int:
    T = cfg["env"]["T"]
    return sum(c.n_games * T for c in conds)


# --------------------------------------------------------------------------- #
def backend(cfg: dict) -> str:
    return cfg["model"].get("backend", "anthropic")


def _row(cond: Condition, seed: int, r) -> str:
    return json.dumps({
        "condition": cond.name, "kind": cond.kind, "gap": cond.gap,
        "rep_strength": cond.rep_strength, "info_level": cond.info_level,
        "rep_letter": cond.rep_letter, "seed": seed,
        "t": r.t, "company": r.company, "a": r.a, "b": r.b,
        "theta": r.theta, "model_est": r.model_est})


# --- Anthropic backend ----------------------------------------------------- #
def _play_one(args):
    cfg, cond, seed, mc, client = args
    ec = env_cfg_for(cfg, cond.gap)
    game = E.make_game(ec, seed=cfg["seed_base"] + seed)
    return cond, seed, play_game(game, cond.style(), mc, client=client)


def _full_run_anthropic(cfg, conds, out_root, workers):
    import anthropic
    mc = ModelConfig(**cfg["model"])
    client = anthropic.Anthropic()
    tasks = [(cfg, c, s, mc, client) for c in conds for s in range(c.n_games)]
    print(f"=== anthropic full run: {len(tasks)} games, {n_calls(conds, cfg)} calls, "
          f"workers={workers} ===", flush=True)
    done = 0
    with (out_root / "rounds.jsonl").open("w") as fh, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_play_one, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                cond, seed, recs = fut.result()
            except Exception as e:               # one bad game must not kill the sweep
                t = futs[fut]
                print(f"  WARN game {t[1].name}/seed{t[2]} failed: {e}", flush=True)
                done += 1
                continue
            for r in recs:
                fh.write(_row(cond, seed, r) + "\n")
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} games done", flush=True)
    d = cfg["design"]
    style = P.PromptStyle(rep_strength=d["main_rep"], info_level=d["main_info"])
    out = [comprehension_probe(E.make_game(env_cfg_for(cfg, d["anchor_gap"]),
                                           seed=cfg["seed_base"] + s), style, mc)
           for s in range(cfg.get("comprehension_games", 4))]
    (out_root / "comprehension.json").write_text(json.dumps(out, indent=2))


# --- Llama (local GPU) backend --------------------------------------------- #
def _llama_cfg(cfg: dict):
    import llama_runner as L
    mm = {k: v for k, v in cfg["model"].items() if k != "backend"}
    return L, L.LlamaConfig(**mm)


def _full_run_llama(cfg, conds, out_root):
    L, lc = _llama_cfg(cfg)
    print(f"=== llama full run: loading {lc.model_name} ===", flush=True)
    model, tok = L.load(lc)
    total = sum(c.n_games for c in conds)
    done = 0
    with (out_root / "rounds.jsonl").open("w") as fh:
        for c in conds:
            ec = env_cfg_for(cfg, c.gap)
            games = [E.make_game(ec, seed=cfg["seed_base"] + s) for s in range(c.n_games)]
            for s, recs in enumerate(L.run_condition(games, c.style(), lc, model, tok)):
                for r in recs:
                    fh.write(_row(c, s, r) + "\n")
            fh.flush()
            done += c.n_games
            print(f"  [{c.name}] done  ({done}/{total} games)", flush=True)
    d = cfg["design"]
    style = P.PromptStyle(rep_strength=d["main_rep"], info_level=d["main_info"])
    out = [L.comprehension_probe(E.make_game(env_cfg_for(cfg, d["anchor_gap"]),
                                             seed=cfg["seed_base"] + s), style, lc, model, tok)
           for s in range(cfg.get("comprehension_games", 4))]
    (out_root / "comprehension.json").write_text(json.dumps(out, indent=2))


def full_run(cfg: dict, out_root: Path, workers: int):
    conds = build_conditions(cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.yaml").write_text(yaml.safe_dump(cfg))
    (out_root / "conditions.json").write_text(
        json.dumps([{**asdict(c), "order": list(c.order)} for c in conds], indent=2))
    if backend(cfg) in ("llama", "hf", "qwen", "local"):
        _full_run_llama(cfg, conds, out_root)
    else:
        _full_run_anthropic(cfg, conds, out_root, workers)
    print(f"wrote {out_root}/rounds.jsonl")


# --------------------------------------------------------------------------- #
def smoke(cfg: dict):
    """One short game end-to-end, printed alongside the Bayesian trajectory."""
    gap = cfg["design"]["anchor_gap"]
    ec = env_cfg_for(cfg, gap)
    ec.T = min(ec.T, cfg.get("smoke_T", 6))
    game = E.make_game(ec, seed=cfg["seed_base"])
    style = P.PromptStyle(rep_strength=cfg["design"]["main_rep"],
                          info_level=cfg["design"]["main_info"])
    traj = B.bayes_trajectory(game, prior_from(cfg))

    print(f"=== SMOKE ({backend(cfg)}): gap={gap} sigma_A={ec.sigma_A:.0f} "
          f"sigma_B={ec.sigma_B:.0f} M={ec.M} T={ec.T} model={cfg['model']['model_name']} ===")
    print("Source A = NOISY (env a); Source B = ACCURATE (env b). Reputation favors A.\n")
    print("--- round 0 prompt (reconstructed context) ---")
    print(P.build_prompt(game, 0, style))

    if backend(cfg) in ("llama", "hf", "qwen", "local"):
        L, lc = _llama_cfg(cfg)
        model, tok = L.load(lc)
        print("\n--- comprehension probe ---")
        print(json.dumps(L.comprehension_probe(game, style, lc, model, tok), indent=2))
        recs = L.run_condition([game], style, lc, model, tok)[0]
    else:
        mc = ModelConfig(**cfg["model"])
        print("\n--- comprehension probe ---")
        print(json.dumps(comprehension_probe(game, style, mc), indent=2))
        recs = play_game(game, style, mc)

    by_t: dict[int, list] = {}
    for r in recs:
        by_t.setdefault(r.t, []).append(r)
    print("\n round | per-company  A(noisy) B(acc) true  model | Bayes trust_B(pre)")
    for t in range(ec.T):
        rr = sorted(by_t[t], key=lambda r: r.company)
        cells = " | ".join(f"A={r.a:.0f} B={r.b:.0f} θ={r.theta:.0f} m={r.model_est:.0f}"
                           for r in rr)
        print(f"  {t + 1:>4} | {cells}   ||  {traj['trust_pre'][t]:.3f}")
    print(f"\noracle trust_B={traj['oracle_trust_B']:.3f}  "
          f"prior trust_B={traj['prior_trust_B']:.3f}")
    print("Smoke OK.")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scoped.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm a run exceeding max_calls")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    conds = build_conditions(cfg)
    calls = n_calls(conds, cfg)
    print(f"=== {len(conds)} conditions, ~{calls} API calls "
          f"(model {cfg['model']['model_name']}) ===")
    for c in conds:
        print(f"  [{c.kind:7}] {c.name:42} games={c.n_games}")
    if args.dry_run:
        print("--dry-run: exiting before any API call."); return
    if args.smoke:
        smoke(cfg); return
    if calls > cfg["max_calls"] and not args.yes:
        print(f"REFUSING: {calls} > max_calls={cfg['max_calls']}. Re-run with --yes "
              f"(or shrink the grid)."); return
    full_run(cfg, Path(cfg["output_root"]), args.workers)


if __name__ == "__main__":
    main()
