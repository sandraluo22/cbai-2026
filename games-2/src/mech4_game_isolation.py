"""MECH4 (2026-08-03, user redesign): lexical vs structure vs attribution,
isolated INSIDE the real game framework (replayed strict-city B streams,
3 morphological planted words: planted/planting/plantings).

EXP A — LEXICAL / POSITION / RECENCY (current game format, first person):
  base_g{0,2,4,6}   all-filler histories, length-matched to the recency cells
  used_only         planted words ONLY in the "Words already used" list (never in rounds)
  slotA_g{0,2,4,6}  planted in A's answer slots, then g filler rounds (recency sweep;
                    g = rounds since the last planted word)
  slotA_shuffled    slotA_g0 with the triple's internal order permuted
  slotA_nolist      slotA_g0 but planted words OMITTED from the used-list
  slotB_g0          planted prefilled into the OTHER player's slots (recent)
  mixed             2 planted in A slots + 1 in B slot

EXP B — STRUCTURE (content = slotA_g0 vs base_g0; only the history FORMAT varies):
  round_sent   "Round k: the other player said X, you said A."   (current)
  youfirst     "Round k: you said A, the other player said X."
  colon_lines  "Them: X\\nYou: A" chat-style lines
  semicolon    "Round k: the other player; X. you; A."
  ownlist      "Your words so far: a1, a2, a3. Their words so far: x1, x2, x3."
  prose        one flowing paragraph, de-listified

EXP C — ATTRIBUTION (content = slotA_g0 vs base_g0; PERSON varies):
  first            current second-person frame + "\\nMy word:"
  witness_advice   third-person spectator + "What word should Player 1 say next?"
                   + "\\nPlayer 1 should say:"
  witness_complete third-person spectator transcript + "\\nPlayer 1 said:" (pure
                   continuation, no advice semantics)

Measures per cell x stream: K=64 MC proposal mass on the planted family (novel,
4-prefix) and on the city category; raw family first-token probability.

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) START_FILE N_STREAMS(4) K(64)
     TEMP(0.7) RUN_DIR(runs/mech4_game)
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
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N_STREAMS = int(os.environ.get("N_STREAMS", "4"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/mech4_game")

PLANT = ["planted", "planting", "plantings"]
FAMP = "plan"
FILL = ["window", "carpet", "stapler", "napkin", "candle", "basket", "lantern",
        "mirror", "kettle", "ribbon", "saddle", "pencil"]

OPEN3 = ("Two players are playing a word game. Each round, both players say one word "
         "at the same time, then the two words are revealed. They win the round only "
         "if they both said the same word. Neither player may say any word that has "
         "already been said by either player.")


def mk_rounds(Bseq, gap, a_slots="planted", shuffled=False, mixed=False):
    """returns list of (b_word, a_word) rounds: 3 planted-or-filler rounds + gap fillers."""
    trip = list(PLANT)
    if shuffled:
        trip = [trip[1], trip[2], trip[0]]
    n = 3 + gap
    rounds = []
    fi = iter(FILL)
    for k in range(n):
        b = Bseq[k]
        if k < 3:
            if a_slots == "planted":
                a = trip[k]
            elif a_slots == "slotB":
                a = next(fi)
                b = trip[k]
            elif a_slots == "mixed":
                if k < 2:
                    a = trip[k]
                else:
                    a = next(fi)
                    b = trip[2]
            else:
                a = next(fi)
        else:
            a = next(fi)
        rounds.append((b, a))
    return rounds


def used_of(rounds, sa, sb, extra=(), drop=()):
    u = {sa, sb}
    for b, a in rounds:
        u |= {b, a}
    u |= set(extra)
    u -= set(drop)
    return sorted(u)


# ---------------- history formats (EXP B) ----------------
def fmt_round_sent(rounds, sa, sb):
    return " " + " ".join(f"Round {k+1}: the other player said {b}, you said {a}."
                          for k, (b, a) in enumerate(rounds))

def fmt_youfirst(rounds, sa, sb):
    return " " + " ".join(f"Round {k+1}: you said {a}, the other player said {b}."
                          for k, (b, a) in enumerate(rounds))

def fmt_colon_lines(rounds, sa, sb):
    return "\n" + "\n".join(f"Them: {b}\nYou: {a}" for b, a in rounds)

def fmt_semicolon(rounds, sa, sb):
    return " " + " ".join(f"Round {k+1}: the other player; {b}. you; {a}."
                          for k, (b, a) in enumerate(rounds))

def fmt_ownlist(rounds, sa, sb):
    return (f" Your words so far: {', '.join(a for _, a in rounds)}. "
            f"The other player's words so far: {', '.join(b for b, _ in rounds)}.")

def fmt_prose(rounds, sa, sb):
    parts = []
    for k, (b, a) in enumerate(rounds):
        parts.append(f"in round {k+1} the other player said {b} while you said {a}")
    return " So far, " + "; ".join(parts) + "."

FORMATS = {"round_sent": fmt_round_sent, "youfirst": fmt_youfirst,
           "colon_lines": fmt_colon_lines, "semicolon": fmt_semicolon,
           "ownlist": fmt_ownlist, "prose": fmt_prose}


def body_first(rounds, sa, sb, fmt="round_sent", extra_used=(), drop_used=()):
    hist = fmt_round_sent if fmt not in FORMATS else FORMATS[fmt]
    u = used_of(rounds, sa, sb, extra_used, drop_used)
    return (G.OPEN_PROMPT + hist(rounds, sa, sb)
            + " Words already used (do not repeat): " + ", ".join(u) + "."), "\nMy word:"


def body_witness(rounds, sa, sb, advice, extra_used=(), drop_used=()):
    hist = " " + " ".join(f"Round {k+1}: Player 2 said {b}, Player 1 said {a}."
                          for k, (b, a) in enumerate(rounds))
    u = used_of(rounds, sa, sb, extra_used, drop_used)
    body = ("You are witnessing a word game between two players. " + OPEN3 + hist
            + " Words already used (may not be repeated): " + ", ".join(u) + ".")
    if advice:
        return body + " What word do you think Player 1 should say next?", "\nPlayer 1 should say:"
    return body, "\nPlayer 1 said:"


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    rows_t = [json.loads(l) for l in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl"))]
    games = collections.defaultdict(list)
    for r in rows_t:
        games[r["rollout"]].append(r)
    streams = sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS]
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    def fid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]
    fam_ids = torch.tensor(sorted({fid(w) for w in PLANT + [FAMP, "plants", "plant", "planter"]}))

    @torch.no_grad()
    def measure(body, prefill, ctx_words):
        prompt = LA._render(tok, body) + prefill
        enc = tok(prompt, return_tensors="pt").to(dev)
        logits = model(enc.input_ids).logits[0, -1].float()
        pfam = float(torch.softmax(logits, 0)[fam_ids].sum())
        out = model.generate(enc.input_ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        ws = [G.clean_word(tok.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True))
              for i in range(K)]
        fam = float(np.mean([1 if (w and w not in ctx_words and len(w) > 3 and w[:4] == FAMP)
                             else 0 for w in ws]))
        cat = float(np.mean([1 if (w and w not in ctx_words and w in catset) else 0 for w in ws]))
        return pfam, fam, cat

    results = []

    def run_cell(cell, exp, si, body, prefill, rounds, extra=()):
        ctx = {w for b, a in rounds for w in (b, a)} | set(extra) | set(PLANT)
        pfam, fam, cat = measure(body, prefill, ctx)
        results.append({"exp": exp, "cell": cell, "stream": si,
                        "pfam": pfam, "fam_mass": fam, "cat_mass": cat})
        json.dump({"per_cell": results}, open(os.path.join(RUN_DIR, "mech4.json"), "w"))

    for si, (roll, ts) in enumerate(streams):
        Bseq = [t["B"] for t in ts]
        sa, sb = starts[roll]

        # -------- EXP A --------
        for g in (0, 2, 4, 6):
            r_base = mk_rounds(Bseq, g, a_slots="filler")
            b, p = body_first(r_base, sa, sb)
            run_cell(f"base_g{g}", "A", roll, b, p, r_base)
            r_pl = mk_rounds(Bseq, g, a_slots="planted")
            b, p = body_first(r_pl, sa, sb)
            run_cell(f"slotA_g{g}", "A", roll, b, p, r_pl)
        r_base0 = mk_rounds(Bseq, 0, a_slots="filler")
        b, p = body_first(r_base0, sa, sb, extra_used=PLANT)
        run_cell("used_only", "A", roll, b, p, r_base0, extra=PLANT)
        r_sh = mk_rounds(Bseq, 0, a_slots="planted", shuffled=True)
        b, p = body_first(r_sh, sa, sb)
        run_cell("slotA_shuffled", "A", roll, b, p, r_sh)
        r_pl0 = mk_rounds(Bseq, 0, a_slots="planted")
        b, p = body_first(r_pl0, sa, sb, drop_used=PLANT)
        run_cell("slotA_nolist", "A", roll, b, p, r_pl0)
        r_bslot = mk_rounds(Bseq, 0, a_slots="slotB")
        b, p = body_first(r_bslot, sa, sb)
        run_cell("slotB_g0", "A", roll, b, p, r_bslot)
        r_mx = mk_rounds(Bseq, 0, a_slots="mixed")
        b, p = body_first(r_mx, sa, sb)
        run_cell("mixed", "A", roll, b, p, r_mx)

        # -------- EXP B --------
        for fname in FORMATS:
            for content, rr in (("planted", r_pl0), ("base", r_base0)):
                b, p = body_first(rr, sa, sb, fmt=fname)
                run_cell(f"{fname}_{content}", "B", roll, b, p, rr)

        # -------- EXP C --------
        for frame, advice in (("witness_advice", True), ("witness_complete", False)):
            for content, rr in (("planted", r_pl0), ("base", r_base0)):
                b, p = body_witness(rr, sa, sb, advice)
                run_cell(f"{frame}_{content}", "C", roll, b, p, rr)
        # first-person reference cells for C are round_sent_planted/base from EXP B
        print(f"[mech4] stream {roll} done ({len(results)} cells)", flush=True)

    # -------- summaries --------
    def cm(cell, metric="fam_mass"):
        v = [r[metric] for r in results if r["cell"] == cell]
        return float(np.mean(v)) if v else float("nan")

    summary = {"A": {}, "B": {}, "C": {}, "recency": {}}
    for g in (0, 2, 4, 6):
        summary["recency"][f"g{g}"] = {"planted": cm(f"slotA_g{g}"), "base": cm(f"base_g{g}")}
    for c in ("used_only", "slotA_shuffled", "slotA_nolist", "slotB_g0", "mixed"):
        summary["A"][c] = cm(c)
    summary["A"]["slotA_g0"] = cm("slotA_g0"); summary["A"]["base_g0"] = cm("base_g0")
    for f in FORMATS:
        summary["B"][f] = {"planted": cm(f"{f}_planted"), "base": cm(f"{f}_base"),
                           "capture": cm(f"{f}_planted") - cm(f"{f}_base")}
    summary["C"]["first"] = {"planted": cm("round_sent_planted"), "base": cm("round_sent_base")}
    for f in ("witness_advice", "witness_complete"):
        summary["C"][f] = {"planted": cm(f"{f}_planted"), "base": cm(f"{f}_base"),
                           "capture": cm(f"{f}_planted") - cm(f"{f}_base")}
    json.dump({"per_cell": results, "summary": summary},
              open(os.path.join(RUN_DIR, "mech4.json"), "w"), indent=1)
    for k, v in summary.items():
        print(f"[mech4] === {k}: {json.dumps(v)[:400]}", flush=True)


if __name__ == "__main__":
    main()
