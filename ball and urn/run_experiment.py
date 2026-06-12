"""Drive the cascade experiment: build scenarios, call the model across variants
(concurrently), and log every trial. --verify runs ONE scenario end-to-end and
prints the prompt, model answer, and rational benchmark side by side.

    python run_experiment.py --verify        # one call, inspect
    python run_experiment.py --dry-run       # job/cost estimate, no calls
    python run_experiment.py --yes           # full run
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from scenarios import build_scenarios
from runner import load_env, ask, ask_comprehension, build_messages


def load_cfg(p):
    return yaml.safe_load(Path(p).read_text())


def all_jobs(scen, cfg):
    return [(s, v) for s in scen for v in range(cfg["n_variants"])]


def verify(cfg):
    import anthropic
    load_env()
    client = anthropic.Anthropic()
    scen = build_scenarios(cfg["qs"], cfg["l_max"], cfg["run_color"])
    s = next(x for x in scen if x.meta["scenario_id"].endswith("_L2_oppose_red")
             and abs(x.q - 2/3) < 1e-6)
    sysmsg, user, swap = build_messages(s, 0, cfg["n_templates"])
    print("=== VERIFY: one scenario (q=2/3, history [R,R], own ball BLUE) ===")
    print("--- SYSTEM ---\n" + sysmsg)
    print("\n--- USER ---\n" + user)
    print(f"\n(swap={swap})")
    rec = ask(client, s, 0, cfg)
    print("\n--- MODEL ANSWER (canonical: P = P(red-majority)) ---")
    print(f"  choice={rec['model_choice']}  P(red)={rec['model_prob_red']:.3f}  "
          f"follows_private={rec['model_follows_private']}")
    print(f"  reasoning: {rec['model_reasoning']}")
    print("\n--- RATIONAL BENCHMARK ---")
    print(f"  choice={rec['rational_choice']}  P(red)={rec['rational_prob_red']:.3f}  "
          f"in_cascade={rec['rational_in_cascade']}  follows_private={rec['rational_follows_private']}")
    print(f"  NAIVE: choice={rec['naive_choice']}  P(red)={rec['naive_prob_red']:.3f}")
    print(f"  request_id={rec['request_id']}")


def full_run(cfg, out_dir: Path):
    import anthropic
    load_env()
    client = anthropic.Anthropic(max_retries=4)
    scen = build_scenarios(cfg["qs"], cfg["l_max"], cfg["run_color"])
    jobs = all_jobs(scen, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_f = (out_dir / "trials.jsonl").open("w")
    print(f"running {len(jobs)} scenario-variant calls + {cfg['comprehension_n']} comprehension checks "
          f"@ concurrency {cfg['max_concurrency']} ...")

    done = [0]; t0 = time.time()
    def work(job):
        s, v = job
        try:
            return ask(client, s, v, cfg)
        except Exception as e:                       # log failures, keep going
            return {"scenario_id": s.meta["scenario_id"], "variant_idx": v,
                    "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=cfg["max_concurrency"]) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for fut in as_completed(futs):
            rec = fut.result()
            results_f.write(json.dumps(rec) + "\n"); results_f.flush()
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(jobs):
                print(f"  {done[0]}/{len(jobs)}  ({time.time()-t0:.0f}s)")
    results_f.close()

    # comprehension checks across q values
    comp_f = (out_dir / "comprehension.jsonl").open("w")
    qs = cfg["qs"]
    checks = [qs[i % len(qs)] for i in range(cfg["comprehension_n"])]
    with ThreadPoolExecutor(max_workers=cfg["max_concurrency"]) as ex:
        for rec in ex.map(lambda q: _safe_comp(client, q, cfg), checks):
            comp_f.write(json.dumps(rec) + "\n")
    comp_f.close()
    (out_dir / "config.used.yaml").write_text(yaml.safe_dump(cfg))
    print(f"done -> {out_dir}/trials.jsonl  (+ comprehension.jsonl)")


def _safe_comp(client, q, cfg):
    try:
        return ask_comprehension(client, q, cfg)
    except Exception as e:
        return {"q": q, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    scen = build_scenarios(cfg["qs"], cfg["l_max"], cfg["run_color"])
    n_calls = len(all_jobs(scen, cfg)) + cfg["comprehension_n"]
    print(f"=== {len(scen)} scenarios × {cfg['n_variants']} variants + {cfg['comprehension_n']} checks "
          f"= {n_calls} API calls; model={cfg['model']} ===")
    if args.dry_run:
        # rough cost: ~450 input + ~600 output tokens/call at $5/$25 per Mtok
        est = n_calls * (450 * 5 + 600 * 25) / 1e6
        print(f"--dry-run: rough cost estimate ~${est:.2f} (Opus 4.8). exiting."); return
    if args.verify:
        verify(cfg); return
    if not args.yes:
        print("refusing full run without --yes (use --verify or --dry-run first)."); return
    full_run(cfg, Path(cfg["output_dir"]))


if __name__ == "__main__":
    main()
