"""Game 2 — Convergence game: LLMs take turns naming words until they converge.

N agents each name ONE word per round, seeing the running history. The goal
(stated in the system prompt) is to converge on a single shared topic. We embed
the round's words and track pairwise similarity; convergence is declared when the
per-round similarity stays high for a few consecutive rounds (or all agents name
the same word). Run for 2 LLMs, then 3 LLMs.

Backends:  BACKEND=open  (Llama/Gemma/Qwen instruct on GPU)  |  BACKEND=api (Claude)
Outputs (results/):  convergence_<backend>_<n>agents.json  + convergence_<backend>.png
Env: BACKEND MAX_ROUNDS SIM_THRESH STABLE_ROUNDS TEMP DEVICE OUTDIR
Run: HF_HOME=/workspace/hf PYTHONPATH=games python games/02_convergence/run.py
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
from common.agents import build_chat_agents, default_roster
from common import embed, io_utils

HERE = os.path.dirname(__file__)
BACKEND = os.environ.get("BACKEND", "open")
DEVICE = os.environ.get("DEVICE", "cuda")
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "60"))   # safety ceiling; the game runs UNTIL convergence
SIM_THRESH = float(os.environ.get("SIM_THRESH", "0.6"))
STABLE_ROUNDS = int(os.environ.get("STABLE_ROUNDS", "3"))
TEMP = float(os.environ.get("TEMP", "0.8"))
OUTDIR = os.environ.get("OUTDIR", io_utils.results_dir(HERE))

SYSTEM = (
    "You are playing a word game. Each turn, every player says exactly ONE word. "
    "Your goal is for all players to CONVERGE on a single shared topic over "
    "successive rounds."
)


# leading filler an agent may emit before its actual word (esp. when it writes a
# sentence instead of a bare word) — skip these to extract the real content word.
_STOP = {
    "i", "i'll", "i've", "i'm", "ill", "ive", "im", "let", "lets", "let's", "we",
    "we'll", "we've", "well", "hmm", "okay", "ok", "sure", "so", "the", "a", "an",
    "my", "your", "our", "it", "its", "it's", "that", "this", "here", "given",
    "since", "both", "looking", "look", "looks", "going", "go", "with", "for",
    "next", "word", "now", "yes", "no", "and", "but", "to", "of", "in", "on",
    "is", "will", "would", "how", "about", "as", "at", "or", "if", "then",
    # meta-game words an agent may say instead of a real content word
    "word", "words", "theme", "topic", "choose", "choice", "pick", "answer",
    "single", "group", "player", "players", "shared", "common", "ground", "say",
    "round", "game", "converge", "topics",
}


def parse_word(text: str) -> str:
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    for t in toks:                       # first non-filler content word
        if t.lower() not in _STOP:
            return t.lower()
    return toks[0].lower() if toks else "?"


def run_variant(tags, n):
    agents = build_chat_agents(tags, BACKEND, DEVICE)
    transcript = []            # (speaker_tag, word) — only COMMITTED (previous) rounds
    rounds = []                # per round: {words: {tag: w}, sim, spread}
    used = set()               # every word already said (NO REPEATS allowed)
    converged_at = None
    stable = 0

    def gen(ag, sys_t, temp):
        return parse_word(ag.say(sys_t, transcript, max_new_tokens=12, temperature=temp)
                          if BACKEND == "open" else
                          ag.say(sys_t, transcript, max_new_tokens=12))

    try:
        for r in range(MAX_ROUNDS):
            this_round = {}
            # No-repeats: forbid every previously-used word (stated in the prompt and
            # enforced by re-drawing). Agents still only see COMMITTED previous rounds.
            sys_t = SYSTEM
            if used:
                sys_t = (SYSTEM + " Never repeat a word that has already been said. "
                         "Already used (do not reuse any of these): "
                         + ", ".join(sorted(used)) + ".")
            for ag in agents:
                w = gen(ag, sys_t, TEMP)
                tries = 0
                while w in used and tries < 5:          # re-draw until it's a new word
                    w = gen(ag, sys_t, min(TEMP + 0.2 * (tries + 1), 1.2))
                    tries += 1
                this_round[ag.tag] = w
            # commit the whole round AFTER everyone has moved
            for ag in agents:
                transcript.append((ag.tag, this_round[ag.tag]))
            used.update(this_round.values())
            words = list(this_round.values())
            vecs = embed.embed(words)
            sim = embed.pairwise_cosine_mean(vecs)
            spread = embed.centroid_spread(vecs)
            rounds.append({"round": r, "words": this_round, "sim": sim, "spread": spread})
            all_same = len(set(words)) == 1
            print(f"[conv/{n}ag] round {r}: {this_round}  sim={sim:.3f}", flush=True)
            if all_same or (not np.isnan(sim) and sim >= SIM_THRESH):
                stable += 1
                if stable >= STABLE_ROUNDS and converged_at is None:
                    converged_at = r
                    break
            else:
                stable = 0
    finally:
        for ag in agents:
            ag.free()
    if converged_at is not None:
        print(f"[conv/{n}ag] CONVERGED at round {converged_at}: {rounds[-1]['words']}", flush=True)
    else:
        print(f"[conv/{n}ag] did NOT converge within the {MAX_ROUNDS}-round ceiling", flush=True)
    return {
        "backend": BACKEND, "n_agents": n, "agents": tags,
        "sim_thresh": SIM_THRESH, "stable_rounds": STABLE_ROUNDS,
        "max_rounds_ceiling": MAX_ROUNDS,
        "no_repeats": True,
        "converged": converged_at is not None,
        "converged_at_round": converged_at,
        "final_words": rounds[-1]["words"] if rounds else {},
        "rounds": rounds,
    }


def main():
    io_utils.seed_all(0)
    results = {}
    for n in (2, 3):
        tags = default_roster(BACKEND, n)
        print(f"\n=== convergence: {n} agents {tags} ({BACKEND}) ===", flush=True)
        res = run_variant(tags, n)
        results[n] = res
        io_utils.dump_json(res, os.path.join(OUTDIR, f"convergence_{BACKEND}_{n}agents.json"))
    make_fig(results, os.path.join(OUTDIR, f"convergence_{BACKEND}.png"))


def make_fig(results, path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for n, res in results.items():
        sims = [rd["sim"] for rd in res["rounds"]]
        ax.plot(range(len(sims)), sims, "-o", label=f"{n} agents {res['agents']}"
                + (f" (converged r{res['converged_at_round']})" if res["converged_at_round"] is not None else ""))
    ax.axhline(SIM_THRESH, ls="--", color="0.5", label=f"threshold {SIM_THRESH}")
    ax.set_xlabel("round"); ax.set_ylabel("pairwise word similarity (cosine)")
    ax.set_ylim(-0.2, 1.05)
    ax.set_title(f"Convergence game — word similarity per round ({results[2]['backend']})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[io] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
