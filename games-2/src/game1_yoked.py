"""GAME-1 (word convergence) YOKED CONTROL + RESTRICTED-CONCEPT variant.

Behavioural (no activation capture) -- measures whether/how fast the game converges.

Conditions (matched starts, same model self-play, no-repeat, temp):
  reactive   : two LIVE players. A's words genuinely react to B (co-adaptive).
  yoked      : B is LIVE, but its partner's words are REPLAYED from a DIFFERENT reactive
               game -- i.e. A's words came from A reacting to SOMEONE ELSE, not to this B.
               Isolates the value of the partner actually adapting to you (vs just seeing
               plausible partner words). The classic yoked control.
  restrict-* : two LIVE players, but ONE player (B) is secretly told it may only say words
               of a concept (city / fruit). A is unrestricted and is NOT told. Does the
               game still converge (A must drift into B's concept), and how long?

Win = both say the SAME word in the same round.

Env: MODEL(QwenInst32) TEMP(0.7) SAFETY(24) N(16) START_FILE RUN_DIR DEVICE
     CONDS(reactive,yoked,restrict-city,restrict-fruit)
Out: <RUN_DIR>/game1_yoked.json  (+ per-condition *_transcript.jsonl + bar-plot pdf)
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import llm_agents as LA
import qwen32_pca as G                                    # reuse OPEN_PROMPT / clean_word

MODEL = os.environ.get("MODEL", "QwenInst32")
TEMP = float(os.environ.get("TEMP", "0.7"))
SAFETY = int(os.environ.get("SAFETY", "24"))              # max rounds before "no-converge"
N = int(os.environ.get("N", "16"))
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
RUN_DIR = os.environ.get("RUN_DIR", "runs/game-1/qwen32/qwen32_variations")
CONDS = os.environ.get("CONDS", "reactive,yoked,restrict-city,restrict-fruit").split(",")

CONCEPTS = {"city": ("the name of a city", "cities"), "fruit": ("a fruit", "fruits"),
            "watercity": ("the name of a city located by the water — on a coast, river, or lake",
                          "cities that are by the water"),
            "water": ("a word related to water", "words related to water")}


def load_starts():
    pairs = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            pairs.append((p[-2], p[-1]))
    return pairs[:N]


def build_prompt(tok, hist, used, restrict=None):
    lines = " ".join(f"Round {k+1}: the other player said {o}, you said {s}." for k, (o, s) in enumerate(hist))
    body = G.OPEN_PROMPT + ((" " + lines) if lines else "")
    if used:
        body += " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."
    if restrict:
        sing, plur = CONCEPTS[restrict]
        body += f" IMPORTANT: every single word you say must be {sing}. Only ever say {plur}, nothing else."
    return LA._render(tok, body) + "\nMy word:"


def make_gen(model, tok, dev):
    import torch

    @torch.no_grad()
    def gen_word(prompt, seed, forbidden):
        enc = tok(prompt, return_tensors="pt").to(dev)
        w = ""
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in forbidden:
                return w
        return w


    return gen_word


def play(gen, tok, sa, sb, roll, replayA=None, restrictB=None, restrictA=None):
    """A and B play. If replayA is a word list, A is a NON-reactive replay of that list
    (B is always live). restrictB / restrictA secretly constrain each player's concept
    (dual restriction => the only mutually-legal meets are the INTERSECTION). Returns dict."""
    histA = [(sb, sa)]; histB = [(sa, sb)]; used = {sa, sb}
    seqA, seqB, rows = [sa], [sb], []
    agreed = None
    for t in range(1, SAFETY):
        if replayA is not None:
            wA = replayA[t] if t < len(replayA) else replayA[-1]     # non-reactive partner
        else:
            wA = gen(build_prompt(tok, histA, used, restrict=restrictA), 5000 * roll + t, used)
        wB = gen(build_prompt(tok, histB, used, restrict=restrictB), 90000 + 5000 * roll + t, used)
        seqA.append(wA); seqB.append(wB)
        rows.append({"rollout": roll, "turn": t, "A": wA, "B": wB, "agreed": bool(wA == wB and wA)})
        if wA == wB and wA:
            agreed = t; break
        histA.append((wB, wA)); histB.append((wA, wB)); used |= {wA, wB}
    return {"rollout": roll, "start": [sa, sb], "agreed": agreed, "seqA": seqA, "seqB": seqB, "rows": rows}


def summarize(games):
    met = [g["agreed"] is not None for g in games]
    ttm = [g["agreed"] for g in games if g["agreed"] is not None]
    n = len(games)
    return {"n": n, "met_frac": float(np.mean(met)),
            "met_se": float(np.std(met) / np.sqrt(n)) if n else float("nan"),
            "turns_to_meet_mean": float(np.mean(ttm)) if ttm else float("nan"),
            "turns_to_meet_se": float(np.std(ttm) / np.sqrt(len(ttm))) if ttm else float("nan"),
            "n_met": int(np.sum(met))}


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model, tok = LA.load(MODEL, dev)
    gen = make_gen(model, tok, dev)
    starts = load_starts()
    print(f"[yoked] {MODEL} temp={TEMP} SAFETY={SAFETY} N={len(starts)} conds={CONDS}", flush=True)

    out = {"model": MODEL, "temp": TEMP, "safety": SAFETY, "n": len(starts), "conditions": {}}
    reactive_games = None

    def dump(cond, games):
        tf = open(os.path.join(RUN_DIR, f"game1_yoked_{cond}_transcript.jsonl"), "w")
        for g in games:
            for r in g["rows"]:
                tf.write(json.dumps(r) + "\n")
        tf.close()
        s = summarize(games); out["conditions"][cond] = s
        print(f"[yoked] {cond:16s}: met {s['met_frac']*100:.0f}% ({s['n_met']}/{s['n']})  "
              f"turns-to-meet {s['turns_to_meet_mean']:.1f}", flush=True)
        return games

    # REACTIVE first (also builds the replay pool)
    if "reactive" in CONDS or "yoked" in CONDS:
        games = []
        for roll, (sa, sb) in enumerate(starts):
            g = play(gen, tok, sa, sb, roll)
            print(f"[yoked] reactive roll {roll} ({sa}/{sb}): "
                  f"{'MET @'+str(g['agreed']) if g['agreed'] else 'no-converge'}", flush=True)
            games.append(g)
        reactive_games = games
        if "reactive" in CONDS:
            dump("reactive", games)

    # YOKED: live B vs A-words replayed from a DIFFERENT reactive game (A reacted to someone else)
    if "yoked" in CONDS:
        games = []
        for roll, (sa, sb) in enumerate(starts):
            donor = reactive_games[(roll + 1) % len(reactive_games)]      # different game
            replayA = donor["seqA"]                                       # that game's A trajectory
            g = play(gen, tok, replayA[0], sb, roll, replayA=replayA)     # A starts where the donor A did
            print(f"[yoked] yoked roll {roll} (B start {sb}, replay-A from game {(roll+1)%len(starts)}): "
                  f"{'MET @'+str(g['agreed']) if g['agreed'] else 'no-converge'}", flush=True)
            games.append(g)
        dump("yoked", games)

    # DUAL: BOTH players secretly restricted to different concepts (cond "dual-<A>-<B>");
    # the only mutually-legal meets are the intersection (e.g. dual-water-city -> water cities)
    for cond in CONDS:
        if not cond.startswith("dual-"):
            continue
        _, ca, cb = cond.split("-", 2)
        games = []
        for roll, (sa, sb) in enumerate(starts):
            g = play(gen, tok, sa, sb, roll, restrictA=ca, restrictB=cb)
            print(f"[yoked] {cond} roll {roll} ({sa}/{sb}): "
                  f"{'MET @' + str(g['agreed']) if g['agreed'] else 'no-converge'}"
                  + (f" on {g['seqA'][-1]}" if g['agreed'] else ""), flush=True)
            games.append(g)
        dump(cond, games)

    # RESTRICTED: B secretly limited to a concept; A unrestricted & uninformed
    for cond in CONDS:
        if not cond.startswith("restrict-"):
            continue
        concept = cond.split("-", 1)[1]
        games = []
        for roll, (sa, sb) in enumerate(starts):
            g = play(gen, tok, sa, sb, roll, restrictB=concept)
            print(f"[yoked] {cond} roll {roll} ({sa}/{sb}): "
                  f"{'MET @'+str(g['agreed'])+' on '+g['seqB'][g['agreed']] if g['agreed'] else 'no-converge'}",
                  flush=True)
            games.append(g)
        dump(cond, games)

    json.dump(out, open(os.path.join(RUN_DIR, "game1_yoked.json"), "w"), indent=2)

    # figure: meet-rate + turns-to-meet by condition
    conds = list(out["conditions"])
    mf = [out["conditions"][c]["met_frac"] * 100 for c in conds]
    mfse = [out["conditions"][c]["met_se"] * 100 for c in conds]
    ttm = [out["conditions"][c]["turns_to_meet_mean"] for c in conds]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].bar(conds, mf, yerr=mfse, capsize=3, color="tab:blue")
    ax[0].set_ylabel("% games converged"); ax[0].set_title("Convergence rate"); ax[0].tick_params(axis="x", rotation=25)
    ax[1].bar(conds, ttm, color="tab:orange")
    ax[1].set_ylabel("mean turns-to-meet (converged games)"); ax[1].set_title("Speed"); ax[1].tick_params(axis="x", rotation=25)
    fig.suptitle(f"GAME-1 yoked control + restricted concept — {MODEL}", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(RUN_DIR, "game1_yoked.pdf"))
    print(f"[yoked] DONE -> {RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
