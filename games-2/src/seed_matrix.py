"""SEED MATRIX (2026-08-02): what actually creates the action prior?
Covers: self-attribution control, lexical-feature control, dose-response, perturbation
persistence — one factorial harness, identical replayed partner streams throughout.

Base: rounds are built synthetically at 'branch turn' = DOSE+1 (or 4 for fixed-dose
cells) against a replayed B stream from a long strict city game. Filler own-words for
non-seed rounds come from a fixed neutral list. Cells:

  dose_0..dose_4          : k morphological-family seeds as A's OWN words (self slot)
  attr_self|partner|list|other : 3 morph seeds attributed to A / to B (A slot neutral)
                            / as a bare word list appended to the prompt / as another
                            game's player's words
  lex_morph|sem|rand      : 3 self-attributed seeds sharing a 4-prefix family /
                            semantically related music words (no prefix overlap) /
                            unrelated random words
  perturb_yes|no          : dose-3 morph self, 2 live turns, then either force one
                            unrelated legal word ('lantern') or a live word; MC-profile
                            the NEXT state (hysteresis: does family mass rebound?)

Measures per branch: MC proposal profile (K=64) at the measurement state — mass on the
seed TARGET SET (legal), on used words, on partner category, other — plus a 6-turn live
continuation scored for target-family hits.

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) N_STREAMS(6) K(64) TEMP(0.7)
     RUN_DIR(runs/seed_matrix)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G
from game1_strict import CATWORDS

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
N_STREAMS = int(os.environ.get("N_STREAMS", "6"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/seed_matrix")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")

MORPH = ["planted", "planting", "plantings", "replant"]
SEM = ["melody", "rhythm", "chorus", "harmony"]
SEM_TARGET = {"melody", "rhythm", "chorus", "harmony", "tune", "song", "verse", "tempo",
              "cadence", "refrain", "ballad", "anthem", "symphony", "sonata", "melodies"}
RAND = ["kettle", "sermon", "gravel", "mitten"]
FILLER = ["window", "carpet", "stapler", "napkin", "candle", "basket"]
PERTURB_WORD = "lantern"


def target_mass(words, seeds, ttype, used):
    hits = 0
    for w in words:
        if not w or w in used:
            continue
        if ttype == "sem":
            hits += w in SEM_TARGET
        else:
            hits += any(w[:4] == s[:4] and len(w) > 3 for s in seeds)
    return hits / max(len(words), 1)


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    @torch.no_grad()
    def propose_k(body):
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    @torch.no_grad()
    def gen_word(body, seed, forbidden):
        prompt = LA._render(tok, body) + "\nMy word:"
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

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    # partner streams from long strict city games
    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl")):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    streams = [sorted(ts, key=lambda r: r["turn"]) for _, ts in
               sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS]]
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    def build(hist_seeds_self, Bseq, sa, sb, n_rounds, seeds, mode):
        """returns (hist, used, extra) with seeds placed per attribution mode."""
        hist = [(sb, sa)]
        used = {sa, sb}
        extra = ""
        fill = iter(FILLER)
        for i in range(n_rounds):
            b = Bseq[i]
            if mode == "self":
                a = seeds[i] if i < len(seeds) else next(fill)
            elif mode == "partner":
                b = seeds[i] if i < len(seeds) else Bseq[i]
                a = next(fill)
            else:
                a = next(fill)
            hist.append((b, a))
            used |= {a, b}
        if mode == "list":
            extra = " Unrelated word list: " + ", ".join(seeds) + "."
            used |= set(seeds)
        elif mode == "other":
            extra = (" In a different game, another player said: "
                     + ", ".join(seeds) + ".")
            used |= set(seeds)
        return hist, used, extra

    cells = []
    for d in range(5):
        cells.append((f"dose_{d}", MORPH[:d], "self", "morph", False))
    for m in ("partner", "list", "other"):
        cells.append((f"attr_{m}", MORPH[:3], m, "morph", False))
    cells.append(("lex_sem", SEM[:3], "self", "sem", False))
    cells.append(("lex_rand", RAND[:3], "self", "rand", False))
    cells.append(("perturb_yes", MORPH[:3], "self", "morph", True))
    cells.append(("perturb_no", MORPH[:3], "self", "morph", "live"))
    ADD4 = [("perturb4_yes", MORPH[:4], "self", "morph", True),
            ("perturb4_no", MORPH[:4], "self", "morph", "live"),
            ("attr4_list", MORPH[:4], "list", "morph", False),
            ("attr4_partner", MORPH[:4], "partner", "morph", False)]
    if os.environ.get("ADDENDUM") == "all":
        cells = cells + ADD4
    elif os.environ.get("ADDENDUM"):
        cells = ADD4

    results = []
    for si, ts in enumerate(streams):
        roll = ts[0]["rollout"]
        Bseq = [t["B"] for t in ts]
        sa, sb = starts[roll]
        for name, seeds, mode, ttype, perturb in cells:
            n_rounds = max(len(seeds), 3)
            hist, used, extra = build(None, Bseq, sa, sb, n_rounds, seeds, mode)
            live_turn_words = []
            # optional 2 live turns then perturbation (hysteresis cell)
            if perturb is not False:
                for lt in range(2):
                    w = gen_word(body_of(hist, used, extra), 909000 + 37 * si + lt, used)
                    live_turn_words.append(w)
                    b = Bseq[n_rounds + lt]
                    hist.append((b, w)); used |= {w, b}
                if perturb is True:
                    w = PERTURB_WORD
                else:                                    # matched live turn
                    w = gen_word(body_of(hist, used, extra), 919000 + 37 * si, used)
                b = Bseq[n_rounds + 2]
                hist.append((b, w)); used |= {w, b}
            props = propose_k(body_of(hist, used, extra))
            tm = target_mass(props, seeds if seeds else MORPH, ttype, used)
            um = np.mean([1 if (w and w in used) else 0 for w in props])
            cm = np.mean([1 if (w and w not in used and w in set(CATWORDS["city"])) else 0
                          for w in props])
            fam_hits = 0
            h2, u2 = list(hist), set(used)
            base_i = len(h2) - 1
            for ct in range(6):
                w = gen_word(body_of(h2, u2, extra), 929000 + 37 * si + ct, u2)
                if ttype == "sem":
                    fam_hits += w in SEM_TARGET
                else:
                    fam_hits += any(w[:4] == s[:4] and len(w) > 3 for s in (seeds or MORPH))
                bidx = base_i + ct
                b = Bseq[bidx] if bidx < len(Bseq) else Bseq[-1]
                h2.append((b, w)); u2 |= {w, b}
            results.append({"cell": name, "stream": roll, "target_mass": float(tm),
                            "used_mass": float(um), "cat_mass": float(cm),
                            "fam_hits6": int(fam_hits)})
            print(f"[sm] s{si} {name}: target {tm:.2f} used {um:.2f} cat {cm:.2f} "
                  f"hits6 {fam_hits}", flush=True)
    out = {"per_branch": results, "cells": {}}
    for name in {r["cell"] for r in results}:
        sel = [r for r in results if r["cell"] == name]
        out["cells"][name] = {k: float(np.mean([r[k] for r in sel]))
                              for k in ("target_mass", "used_mass", "cat_mass", "fam_hits6")}
    for name in sorted(out["cells"]):
        c = out["cells"][name]
        print(f"[sm] === {name}: target {c['target_mass']:.2f} cat {c['cat_mass']:.2f} "
              f"hits6 {c['fam_hits6']:.1f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "seed_matrix.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
