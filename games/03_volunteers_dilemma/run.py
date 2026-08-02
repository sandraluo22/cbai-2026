"""Game 3 — Volunteer's dilemma at varying group sizes.

Volunteer's dilemma: N players simultaneously choose VOLUNTEER or ABSTAIN. If at
least one volunteers, everyone receives benefit b; each volunteer pays cost c
(0<c<b). If NOBODY volunteers, everyone gets 0. The classic prediction (symmetric
mixed Nash) is that the probability an individual volunteers DECREASES with group
size, and — the bystander effect — the probability that at least one volunteers
also falls (toward 1 - c/b, not 1).

We ask an LLM agent to decide independently (no communication) at each group size,
sample many i.i.d. decisions, and compare P(volunteer) and P(>=1 volunteers) to the
Nash prediction as N grows.

Backends: BACKEND=open (default agent Llama) | BACKEND=api (default agent Haiku)
Outputs (results/): volunteers_<backend>.json + volunteers_<backend>.png
Env: BACKEND MODELS GROUP_SIZES SAMPLES BENEFIT COST TEMP DEVICE OUTDIR
Run: HF_HOME=/workspace/hf PYTHONPATH=games python games/03_volunteers_dilemma/run.py
"""
from __future__ import annotations

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.agents import build_chat_agents
from common import io_utils

HERE = os.path.dirname(__file__)
BACKEND = os.environ.get("BACKEND", "open")
DEVICE = os.environ.get("DEVICE", "cuda")
DEFAULT_MODELS = {"open": "Llama", "api": "Haiku"}
MODELS = os.environ.get("MODELS", DEFAULT_MODELS[BACKEND]).split(",")
GROUP_SIZES = [int(x) for x in os.environ.get("GROUP_SIZES", "2,3,5,8,12").split(",")]
SAMPLES = int(os.environ.get("SAMPLES", "20"))
BENEFIT = float(os.environ.get("BENEFIT", "10"))
COST = float(os.environ.get("COST", "5"))
TEMP = float(os.environ.get("TEMP", "0.9"))
OUTDIR = os.environ.get("OUTDIR", io_utils.results_dir(HERE))

SYSTEM = "You are a rational decision-making agent playing an economic game. Follow the payoff logic."


def prompt_for(n):
    return (
        f"VOLUNTEER'S DILEMMA. You are one of {n} players deciding simultaneously and "
        f"independently — you cannot communicate. Each of you privately chooses VOLUNTEER "
        f"or ABSTAIN.\n"
        f"Payoffs: if AT LEAST ONE player volunteers, every player receives a benefit of "
        f"{BENEFIT:.0f} points, but each volunteer additionally pays a cost of {COST:.0f} "
        f"points. If NOBODY volunteers, everyone gets 0.\n"
        f"Decide your action. Answer on the first line with exactly one word: VOLUNTEER or "
        f"ABSTAIN. Then one short sentence of reasoning."
    )


def parse_decision(text: str) -> int:
    t = text.lower()
    # first explicit token wins
    m = re.search(r"\b(volunteer|abstain)\b", t)
    if m:
        return 1 if m.group(1) == "volunteer" else 0
    return 1 if "volunteer" in t else 0


def nash(n):
    q = (COST / BENEFIT) ** (1.0 / (n - 1))       # P(abstain) at symmetric mixed Nash
    return 1 - q, 1 - q ** n                       # (P individual volunteers, P >=1 volunteers)


def run_model(tag):
    agent = build_chat_agents([tag], BACKEND, DEVICE)[0]
    rows = []
    try:
        for n in GROUP_SIZES:
            decs = []
            for s in range(SAMPLES):
                kw = dict(max_new_tokens=60)
                if BACKEND == "open":
                    kw["temperature"] = TEMP
                out = agent.say(SYSTEM, [("host", prompt_for(n))], **kw)
                decs.append(parse_decision(out))
            decs = np.array(decs)
            p_vol = float(decs.mean())
            # empirical P(>=1 volunteers): disjoint trials of size n
            k = len(decs) // n
            trials = decs[: k * n].reshape(k, n) if k else np.zeros((0, n))
            p_any_emp = float((trials.sum(1) > 0).mean()) if k else float("nan")
            p_any_ind = 1 - (1 - p_vol) ** n
            nv, na = nash(n)
            rows.append({"n": n, "p_volunteer": p_vol, "p_any_empirical": p_any_emp,
                         "p_any_independent": p_any_ind, "nash_p_volunteer": nv,
                         "nash_p_any": na, "samples": SAMPLES})
            print(f"[vod/{tag}] N={n}: p_vol={p_vol:.2f} (nash {nv:.2f})  "
                  f"p_any={p_any_ind:.2f} (nash {na:.2f})", flush=True)
    finally:
        agent.free()
    return rows


def main():
    io_utils.seed_all(0)
    out = {"backend": BACKEND, "benefit": BENEFIT, "cost": COST, "models": {}}
    for tag in MODELS:
        print(f"\n=== volunteer's dilemma: {tag} ({BACKEND}) ===", flush=True)
        out["models"][tag] = run_model(tag)
    io_utils.dump_json(out, os.path.join(OUTDIR, f"volunteers_{BACKEND}.json"))
    make_fig(out, os.path.join(OUTDIR, f"volunteers_{BACKEND}.png"))


def make_fig(out, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ns = GROUP_SIZES
    for tag, rows in out["models"].items():
        ax[0].plot([r["n"] for r in rows], [r["p_volunteer"] for r in rows], "-o", label=f"{tag} (LLM)")
        ax[1].plot([r["n"] for r in rows], [r["p_any_independent"] for r in rows], "-o", label=f"{tag} (LLM)")
    ax[0].plot(ns, [nash(n)[0] for n in ns], "k--", label="Nash")
    ax[1].plot(ns, [nash(n)[1] for n in ns], "k--", label="Nash")
    ax[0].set_title("P(an individual volunteers) vs group size")
    ax[1].set_title("P(at least one volunteers) vs group size — bystander effect")
    for a in ax:
        a.set_xlabel("group size N"); a.set_ylabel("probability"); a.set_ylim(-0.05, 1.05); a.legend(fontsize=8)
    fig.suptitle(f"Volunteer's dilemma (b={out['benefit']:.0f}, c={out['cost']:.0f}, {out['backend']})")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[io] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
