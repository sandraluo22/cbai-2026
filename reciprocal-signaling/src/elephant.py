"""Elephant-style clue games: N agents each hold a DIFFERENT private clue about the
same secret word. Each round every agent says one word — its current guess of the
secret word — then all guesses are revealed with attribution. NO new information
enters after round 1 (clues are given once; only the guesses circulate). Agents
are told they may NOT answer with their own clue (the secret word is not the clue).
Measures whether/when the group converges on the referent from complementary
one-word views.

Quiz battery spans difficulty (concrete -> abstract) and group size (3-5 agents).

env: MODEL (Qwen32) QUIZ (comma list; default all) ROUNDS (5) SEEDS (2)
     NOTES ("" | update | append | public)  end-of-round scratchpad; public = ONE
       sentence per agent per round, visible to ALL players next round
     OUT (runs/elephant/<MODEL>)
"""
from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_games import load, Chat, _STOP  # noqa: E402

def q(true, clues, kind="word", wrong=None):
    """wrong = index of the agent whose clue is FALSE of the referent (poisoned clue)."""
    return dict(true=true, clues=clues, kind=kind, wrong=wrong)


QUIZZES = {
    # ---- word quizzes (one private clue per agent) ----
    "elephant": q("elephant", ["gray", "large", "an animal"]),
    "penguin":  q("penguin", ["black and white", "a bird", "it cannot fly",
                              "it lives somewhere cold"]),
    "onion":    q("onion", ["a vegetable", "it has layers", "it makes you cry", "round"]),
    "piano":    q("piano", ["it has black and white parts", "a musical instrument",
                            "it has keys", "large and heavy"]),
    "library":  q("library", ["quiet", "a building", "full of books",
                              "you can borrow things there", "open to everyone"]),
    "clock":    q("clock", ["round", "it has hands", "it hangs on a wall",
                            "it tells you something"]),
    "shadow":   q("shadow", ["dark", "it follows you", "flat", "it needs light to exist"]),
    "echo":     q("echo", ["you can hear it", "it repeats", "you notice it in the mountains",
                           "you cannot see it", "it fades away"]),
    # ---- number quizzes (arithmetic clues; intersection is unique) ----
    "n30": q("30", ["divisible by 5", "divisible by 6", "it has 2 digits",
                    "smaller than 50"], kind="number"),
    "n17": q("17", ["a prime number", "it has 2 digits", "smaller than 20", "odd",
                    "the sum of its digits is 8"], kind="number"),
    "n64": q("64", ["a multiple of 8", "a perfect square", "it has 2 digits", "even",
                    "greater than 50"], kind="number"),
    "n36": q("36", ["a multiple of 12", "divisible by 9", "it has 2 digits", "composite",
                    "smaller than 50"], kind="number"),
    # ---- poisoned-clue conditions (one agent's clue is FALSE of the referent) ----
    "elephant_wrong": q("elephant", ["gray", "large", "an animal", "small"], wrong=3),
    "piano_wrong": q("piano", ["it has black and white parts", "a musical instrument",
                               "it has keys", "large and heavy", "you blow into it"], wrong=4),
    "n30_wrong": q("30", ["divisible by 5", "divisible by 6", "it has 2 digits",
                          "smaller than 50", "odd"], kind="number", wrong=4),
}




# ---- clue-format battery: same 4 features per referent, 4 linguistic forms ----
_FMT = {
    "elephant4": ("elephant", {
        "bare": ["gray", "large", "an animal", "a trunk"],
        "desc": ["its color is gray", "it is very large in size", "it is a kind of animal",
                 "it has a long trunk"],
        "prop": ["it has the property: gray-colored", "it has the property: large",
                 "it has the property: being an animal", "it has the property: possessing a trunk"],
        "syn":  ["ashen", "enormous", "creature", "snout"],
    }),
    "penguin4": ("penguin", {
        "bare": ["black and white", "a bird", "flightless", "polar"],
        "desc": ["it is colored black and white", "it is a kind of bird", "it cannot fly",
                 "it lives in a very cold place"],
        "prop": ["it has the property: black-and-white coloration", "it has the property: being a bird",
                 "it has the property: flightlessness", "it has the property: cold-climate habitat"],
        "syn":  ["monochrome", "fowl", "grounded", "antarctic"],
    }),
    "shadow4": ("shadow", {
        "bare": ["dark", "following", "flat", "light-dependent"],
        "desc": ["it is dark", "it follows you around", "it is completely flat",
                 "it needs light to exist"],
        "prop": ["it has the property: darkness", "it has the property: following its owner",
                 "it has the property: flatness", "it has the property: light-dependence"],
        "syn":  ["dim", "trailing", "two-dimensional", "cast"],
    }),
    "clock4": ("clock", {
        "bare": ["round", "hands", "wall-mounted", "time"],
        "desc": ["it is round in shape", "it has moving hands", "it hangs on a wall",
                 "it tells you the time"],
        "prop": ["it has the property: roundness", "it has the property: having hands",
                 "it has the property: being wall-mounted", "it has the property: time-telling"],
        "syn":  ["circular", "pointers", "hanging", "hours"],
    }),
}
for _name, (_true, _fmts) in _FMT.items():
    for _f, _clues in _fmts.items():
        QUIZZES[f"{_name}_{_f}"] = q(_true, _clues)


def base(i, clues, rounds, kind, public=False):
    n, item = len(clues), ("number" if kind == "number" else "word")
    extra = (" After each round, every player also shares one public remark, shown with the "
             "round history." if public else "")
    return (f"You are Player {i + 1} in a word game with {n - 1} partners. There is one secret "
            f"{item}. You have been given one clue about the secret {item}. Your clue is: "
            f"\"{clues[i]}\". Each partner has been given a different clue about the same secret "
            f"{item}. Each round, every player says one {item} — their current guess of the "
            f"secret {item} — and then everyone's guesses are revealed. There are {rounds} "
            f"rounds. You may not say your clue: the secret {item} is not the clue. "
            "No other information will be given." + extra)


def hist_text(hist, remarks=None):
    out = []
    for k, row in enumerate(hist):
        out.append(f"Round {k + 1} — " + " | ".join(
            f"Player {j + 1} said: {g}" for j, g in enumerate(row)))
        if remarks and k < len(remarks):
            out.append("  Remarks: " + " | ".join(
                f"P{j + 1}: {t}" for j, t in enumerate(remarks[k])))
    return ("\n" + "\n".join(out)) if out else ""


_FILLER = _STOP | {"player", "guess", "round", "secret", "based", "on", "clue", "clues",
                   "for", "example", "will", "would", "say", "said", "my", "me", "we",
                   "one", "word", "output", "only", "answer", "final", "current", "m", "ll"}


def parse_word(txt, kind="word"):
    if kind == "number":
        nums = re.findall(r"\d+", txt)
        return nums[0] if nums else None
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", txt)]
    for tok in toks:
        if tok not in _FILLER and len(tok) > 2:
            return tok
    return toks[-1] if toks else None                 # verbose reply: last word is the guess


def play(model, tok, thinking, quiz, rounds, seed, out_dir, notes_mode=""):
    spec = QUIZZES[quiz]
    true, clues, kind = spec["true"], spec["clues"], spec["kind"]
    n = len(clues)
    item = "number" if kind == "number" else "word"
    clue_words = [{w.lower() for w in re.findall(r"[A-Za-z]+", c)} - _STOP for c in clues]
    chats = [Chat(model, tok, thinking, seed * 331 + i) for i in range(n)]
    notes = [[] for _ in range(n)]
    pub = []                                          # public remarks per round
    def ntxt(i):
        return ("\nYour private notes:\n" +
                "".join(f"(after round {rr}) {t}\n" for rr, t in notes[i])) if notes[i] else ""
    hist, lines = [], [dict(type="meta", quiz=quiz, true=true, clues=clues, kind=kind,
                            wrong_agent=spec["wrong"], rounds=rounds, seed=seed,
                            notes=notes_mode)]
    for r in range(1, rounds + 1):
        row = []
        for i in range(n):
            qq = (base(i, clues, rounds, kind, notes_mode == "public") + ntxt(i) +
                  hist_text(hist, pub if notes_mode == "public" else None) +
                  f"\n\nRound {r}: say your guess now. Output only one {item}.")
            raw = chats[i](qq, max_new=8)
            row.append(parse_word(raw, kind) or "?")
        hist.append(row)
        lines.append(dict(type="round", round=r, guesses=row,
                          all_true=all(g == true for g in row),
                          clue_violations=[i + 1 for i, g in enumerate(row)
                                           if g in clue_words[i]]))
        print(f"  [{quiz}] r{r}: {row}", flush=True)
        if notes_mode == "public":
            row_remarks = []
            for i in range(n):
                t = " ".join(chats[i](
                    base(i, clues, rounds, kind, True) +
                    hist_text(hist, pub + [["(pending)"] * n]) +
                    f"\n\nEnd of round {r}. Write ONE sentence (at most 20 words) to share "
                    "with all players; everyone will see it next round.",
                    max_new=45).split()[:25])
                row_remarks.append(t)
                lines.append(dict(type="note", round=r, agent=i + 1, text=t, public=True))
            pub.append(row_remarks)
        elif notes_mode:
            for i in range(n):
                instr = (f"\n\nEnd of round {r}. Update your private note: write a short note "
                         "(at most 50 words) capturing what you want to remember for later "
                         "rounds. It will REPLACE your previous note."
                         if notes_mode == "update" else
                         f"\n\nEnd of round {r}. Write a short private note (at most 50 words) "
                         "about anything that may help you in later rounds.")
                t = " ".join(chats[i](base(i, clues, rounds, kind) + ntxt(i) +
                                      hist_text(hist) + instr, max_new=90).split()[:60])
                if notes_mode == "update":
                    notes[i] = [(r, t)]
                else:
                    notes[i].append((r, t))
                lines.append(dict(type="note", round=r, agent=i + 1, text=t))
    stem = os.path.join(out_dir, f"{quiz}_s{seed}")
    with open(stem + "_transcript.jsonl", "w") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")
    with open(stem + "_transcript.json", "w") as fh:
        json.dump(lines, fh, indent=1)


def main():
    tag = os.environ.get("MODEL", "Qwen32")
    rounds = int(os.environ.get("ROUNDS", 5))
    seeds = int(os.environ.get("SEEDS", 2))
    quizzes = os.environ.get("QUIZ", "").split(",") if os.environ.get("QUIZ") else list(QUIZZES)
    quizzes = [q for q in quizzes if q]
    notes_mode = os.environ.get("NOTES", "")
    out_dir = os.environ.get("OUT", os.path.join(_HERE, "..", "runs", "elephant", tag))
    os.makedirs(out_dir, exist_ok=True)
    model, tok, thinking = load(tag)
    for quiz in quizzes:
        for s in range(seeds):
            print(f"=== {quiz} {tag} seed {s} ===", flush=True)
            play(model, tok, thinking, quiz, rounds, s, out_dir, notes_mode)


if __name__ == "__main__":
    main()
