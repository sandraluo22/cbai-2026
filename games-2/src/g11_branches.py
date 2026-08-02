"""Case-study branch surgery on stuck scratchpad games (default: scratch-city g11, the
"name-it-and-persist" game; also any other rollouts given in GAMES).

From the recorded transcript, reconstruct A's exact state at BRANCH_TURN, then:
  quiz            : (no continuation) held-out constraint test at the branch state —
                    ask A which of 8 held-out words the OTHER player could say next
                    (4 in-category, 4 out), and what word it expects the partner to say.
                    Verifies the inferred constraint is intact & declaratively usable.
  control         : continue the scratchpad game unchanged.
  instruct        : continue + explicit instruction to A: "From now on, say the name of
                    a city that has not been used yet."
  self_removed    : continue with A's OWN words omitted from the history lines
                    (partner's words and the used-list intact).
  note_replaced   : continue with the fed-back scratchpad note PINNED to a corrected
                    diagnosis ("The other player only ever says cities. To match, I
                    should say a common city that has not been used.") — no new notes.
Each continuation runs BRANCH_TURNS turns with B live (restricted, true history).
Release metrics: next-5 A-picks' loop-exit (vs pre-branch own words) + city-rate + met.

Env: MODEL(QwenInst32) SRC(runs/qwen32_restrict_aids/game1_restrict_aids_scratch-city_transcript.jsonl)
     CONCEPT(city) GAMES(11,2,14) BRANCH_TURN(13) BRANCH_TURNS(10) TEMP(0.7)
     START_FILE RUN_DIR(runs/g11_branches)
"""
from __future__ import annotations
import os
import json
import re
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC = os.environ.get("SRC",
                     "runs/qwen32_restrict_aids/game1_restrict_aids_scratch-city_transcript.jsonl")
CONCEPT = os.environ.get("CONCEPT", "city")
GAMES = [int(x) for x in os.environ.get("GAMES", "11,2,14").split(",")]
BRANCH_TURN = int(os.environ.get("BRANCH_TURN", "13"))
BRANCH_TURNS = int(os.environ.get("BRANCH_TURNS", "10"))
TEMP = float(os.environ.get("TEMP", "0.7"))
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
RUN_DIR = os.environ.get("RUN_DIR", "runs/g11_branches")

CONCEPTS = {"city": ("the name of a city", "cities"), "fruit": ("a fruit", "fruits")}
QUIZ_IN = ["chicago", "lagos", "quito", "helsinki"]          # held-out cities (unlikely used)
QUIZ_OUT = ["banana", "sorrow", "guitar", "puddle"]
FIX_NOTE = ("The other player only ever says cities. To match them, I should say a common "
            "city that has not been used yet.")
SCRATCH_EXTRA = (" You have a private scratchpad the other player never sees; use it for "
                 "brief notes about what is happening and your plan.")
INSTRUCT = " From now on, say the name of a city that has not been used yet."


def load_game(src, roll):
    rows = [json.loads(l) for l in open(src)]
    return [r for r in rows if r["rollout"] == roll]


def load_starts():
    out = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            out.append((p[-2], p[-1]))
    return out


def body_of(hist, used, extra="", hide=None):
    lines = []
    for k, (o, s) in enumerate(hist):
        if hide == "self":
            lines.append(f"Round {k+1}: the other player said {o}.")
        else:
            lines.append(f"Round {k+1}: the other player said {o}, you said {s}.")
    body = G.OPEN_PROMPT + extra + ((" " + " ".join(lines)) if lines else "")
    if used:
        body += " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."
    return body


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    sing, plur = CONCEPTS[CONCEPT]
    restr = (f" IMPORTANT: every single word you say must be {sing}. Only ever say {plur}, "
             f"nothing else.")

    @torch.no_grad()
    def gen(prompt, seed, max_new, forbidden=None):
        enc = tok(prompt, return_tensors="pt").to(dev)
        txt = ""
        for r in range(24 if forbidden is not None else 1):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=max_new, do_sample=True, temperature=TEMP,
                                 top_p=0.95, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            if forbidden is None:
                return txt
            w = G.clean_word(txt)
            if w and w not in forbidden:
                return w
        return G.clean_word(txt)

    starts = load_starts()
    results = []
    tf = open(os.path.join(RUN_DIR, "g11_branches_transcript.jsonl"), "w")
    for roll in GAMES:
        rows = load_game(SRC, roll)
        sa, sb = starts[roll]
        histA, histB = [(sb, sa)], [(sa, sb)]
        used = {sa, sb}
        note = ""
        for r in rows:
            if r["turn"] >= BRANCH_TURN:
                break
            note = r.get("note", note)
            used |= {r["A"], r["B"]}
            histA.append((r["B"], r["A"])); histB.append((r["A"], r["B"]))
        pre_own = {s for _, s in histA[1:]} | {histA[0][1]}
        print(f"[g11] roll {roll}: branched at turn {BRANCH_TURN}, "
              f"{len(histA)-1} rounds in context, note: {note[:90]}", flush=True)

        # ---- quiz at branch state ----
        quiz_words = QUIZ_IN + QUIZ_OUT
        rng = np.random.default_rng(roll)
        order = [quiz_words[i] for i in rng.permutation(len(quiz_words))]
        qbase = body_of(histA, used)
        q1 = LA._render(tok, qbase + " Question: which of the following words do you think "
                        "the OTHER player would be able to say next, given how they have "
                        "been playing? Words: " + ", ".join(order) +
                        ". Answer with just the words you believe they could say.")
        a1 = gen(q1 + "\nAnswer:", 42 + roll, 40)
        q2 = LA._render(tok, qbase + " Question: what single word do you expect the other "
                        "player to say next round?")
        a2 = gen(q2 + "\nAnswer:", 43 + roll, 8)
        in_named = [w for w in QUIZ_IN if w in a1.lower()]
        out_named = [w for w in QUIZ_OUT if w in a1.lower()]
        results.append({"roll": roll, "branch": "quiz", "could_say": a1.strip()[:200],
                        "in_named": in_named, "out_named": out_named,
                        "expect_next": a2.strip()[:60]})
        print(f"[g11] quiz r{roll}: in={in_named} out={out_named} expect={a2.strip()[:40]}",
              flush=True)

        # ---- continuations ----
        branches = os.environ.get(
            "BRANCHES", "control,instruct,self_removed,note_replaced").split(",")
        for branch in branches:
            hA = list(histA); hB = list(histB)
            u = set(used)
            bnote = ("" if branch == "note_blanked"
                     else FIX_NOTE if branch == "note_replaced" else note)
            own_new = []
            agreed_at = None
            for bt in range(1, BRANCH_TURNS + 1):
                if branch == "note_blanked":       # framing kept, note slot empty, no writing
                    extra = SCRATCH_EXTRA
                else:
                    extra = SCRATCH_EXTRA + (f' Your scratchpad from last round: "{bnote}"'
                                             if bnote else "")
                if branch == "instruct":
                    extra += INSTRUCT
                hide = "self" if branch == "self_removed" else None
                base = body_of(hA, u, extra, hide=hide)
                if branch == "note_blanked":
                    pA = LA._render(tok, base) + "\nMy word:"
                elif branch == "note_replaced":
                    pA = LA._render(tok, base) + "\nMy scratchpad: " + bnote + "\nMy word:"
                else:
                    p1 = LA._render(tok, base) + "\nMy scratchpad:"
                    bnote = re.sub(r"\s+", " ", gen(p1, 7000 * roll + 31 * bt, 80).strip())[:280]
                    pA = p1 + " " + bnote + "\nMy word:"
                wA = gen(pA, 5000 * roll + 31 * bt, 4, forbidden=u)
                pB = LA._render(tok, body_of(hB, u, restr)) + "\nMy word:"
                wB = gen(pB, 90000 + 5000 * roll + 31 * bt, 4, forbidden=u)
                own_new.append(wA)
                tf.write(json.dumps({"roll": roll, "branch": branch, "bturn": bt,
                                     "A": wA, "B": wB, "agreed": wA == wB,
                                     "note": bnote[:200]}) + "\n")
                tf.flush()
                if wA == wB and wA:
                    agreed_at = bt
                    break
                u |= {wA, wB}
                hA.append((wB, wA)); hB.append((wA, wB))
            loop_exit = not any(any(w[:4] == p[:4] and len(w) > 3 for p in pre_own)
                                for w in own_new[:5])
            results.append({"roll": roll, "branch": branch, "picks": own_new,
                            "loop_exit": bool(loop_exit), "met": agreed_at is not None,
                            "met_at": agreed_at})
            print(f"[g11] r{roll} {branch}: loop_exit={loop_exit} "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'} "
                  f"picks={' '.join(own_new[:6])}", flush=True)
    tf.close()
    json.dump(results, open(os.path.join(RUN_DIR, "g11_branches.json"), "w"), indent=1)
    print(f"[g11] wrote -> {RUN_DIR}")


if __name__ == "__main__":
    main()
