"""POPULATION-LEVEL knowledge/action dissociation at loop snapshots (2026-08-02).

Snapshots are reconstructed deterministically from the stuck_repro sweep transcript
(same online detector: >=3 of A's last 5 words share a 4-prefix with an earlier own
word, from t8). At each snapshot, INDEPENDENT branches, all forced-choice over the SAME
8-candidate set (4 held-out in-category + 4 out-of-category, none used in the game,
shuffled per snapshot; read out as renormalized first-token probabilities):

  knowledge  : "...one of these words matches the kind of words the other player has
               been saying. Which one?"            -> P(mass on in-category)
  prediction : "...which of these words do you expect the other player to say next?"
               -> P(mass on in-category)
  action     : normal game frame + "For this round you must choose your word from this
               list: ..." -> P(mass on in-category)  [in-category = the coordination-
               legal choices]
  untouched  : plain continuation for CONT_TURNS turns (no quiz) — checks the branch
               harness itself doesn't alter behavior (loop-exit rate vs sweep phase-2
               control ~0).

CENTRAL METRIC per snapshot: dissociation = P_cat(knowledge) - P_cat(action).
Population mean + bootstrap CI over snapshots.

Env: MODEL(QwenInst32) SWEEP(runs/stuck_repro/stuck_repro_QwenInst32_transcript.jsonl)
     START_FILE CONT_TURNS(8) RUN_DIR(runs/dissoc_branches)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
SWEEP = os.environ.get("SWEEP", "runs/stuck_repro/stuck_repro_QwenInst32_transcript.jsonl")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
CONT_TURNS = int(os.environ.get("CONT_TURNS", "8"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/dissoc_branches")

BASE0 = G.OPEN_PROMPT
BASE1 = ("You and another player each say one word per round, at the same time, and then "
         "both words are revealed. If you both said the same word, you both win. A word "
         "that was said by either player in any earlier round may not be said again.")
CONCEPTS = {"city": ("the name of a city", "cities"), "fruit": ("a fruit", "fruits"),
            "animal": ("an animal", "animals"), "color": ("a color", "colors"),
            "sport": ("a sport", "sports"), "vegetable": ("a vegetable", "vegetables")}
CANDS = {
    "city": (["tbilisi", "adelaide", "porto", "kigali"],
             ["walnut", "sorrow", "hammer", "crimson"]),
    "fruit": (["papaya", "lychee", "guava", "currant"],
              ["prague", "anvil", "sonnet", "teal"]),
    "animal": (["ocelot", "heron", "gecko", "lemur"],
               ["quito", "flute", "maroon", "girder"]),
    "color": (["mauve", "ochre", "cyan", "sepia"],
              ["ottawa", "falcon", "ladder", "ledger"]),
    "sport": (["curling", "fencing", "rowing", "squash"],
              ["lisbon", "otter", "amber", "spoon"]),
    "vegetable": (["turnip", "radish", "fennel", "kale"],
                  ["vienna", "sparrow", "violet", "mallet"]),
}


def restr_text(var, cat):
    sing, plur = CONCEPTS[cat]
    if var == 0:
        return f" IMPORTANT: every single word you say must be {sing}. Only ever say {plur}, nothing else."
    return (f" Secret rule that applies only to you: each word you say must be {sing}. "
            f"You must never say a word that is not {sing}.")


def hist_lines(hist):
    return " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                    for k, (o, s) in enumerate(hist))


def build_A(tok, base, hist, used, extra=""):
    body = base + " " + hist_lines(hist)
    if used:
        body += " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."
    return body + extra


def stuck_now(seqA):
    if len(seqA) < 8:
        return False
    last5 = seqA[-5:]
    return sum(any(w[:4] == p[:4] and len(w) > 3 for p in seqA[:len(seqA) - 5 + i])
               for i, w in enumerate(last5)) >= 3


def snapshots_from_sweep(path, starts):
    rows = [json.loads(l) for l in open(path)]
    games = collections.defaultdict(list)
    for r in rows:
        games[(r["cell"], r["rollout"])].append(r)
    snaps = []
    for (cell, roll), ts in games.items():
        cat, pv = cell.rsplit("-p", 1)
        sa, sb = starts[roll]
        histA, histB = [(sb, sa)], [(sa, sb)]
        used = {sa, sb}
        seqA = []
        for r in sorted(ts, key=lambda x: x["turn"]):
            seqA.append(r["A"])
            if stuck_now(seqA):
                snaps.append({"cat": cat, "pv": int(pv), "roll": roll, "turn": r["turn"],
                              "histA": list(histA), "histB": list(histB),
                              "used": sorted(used), "own": list(seqA)})
                break
            if r["agreed"]:
                break
            used |= {r["A"], r["B"]}
            histA.append((r["B"], r["A"])); histB.append((r["A"], r["B"]))
    return snaps


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    def fid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]

    @torch.no_grad()
    def cand_mass(user_body, prefill, cands_in, cands_out):
        prompt = LA._render(tok, user_body) + prefill
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        logits = model(ids).logits[0, -1].float()
        li = [logits[fid(w)] for w in cands_in]
        lo = [logits[fid(w)] for w in cands_out]
        import torch as T
        z = T.softmax(T.tensor(li + lo), 0)
        return float(z[:len(li)].sum())

    @torch.no_grad()
    def gen_word(user_body, seed, forbidden):
        prompt = LA._render(tok, user_body) + "\nMy word:"
        enc = tok(prompt, return_tensors="pt").to(dev)
        w = ""
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=0.7, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in forbidden:
                return w
        return w

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    snaps = snapshots_from_sweep(SWEEP, starts)
    print(f"[dissoc] {len(snaps)} snapshots reconstructed", flush=True)
    rng = np.random.default_rng(0)
    results = []
    tf = open(os.path.join(RUN_DIR, "dissoc_continuations.jsonl"), "w")
    for si, s in enumerate(snaps):
        base = BASE0 if s["pv"] == 0 else BASE1
        cin, cout = CANDS[s["cat"]]
        cin = [w for w in cin if w not in s["used"]]
        cout = [w for w in cout if w not in s["used"]]
        order = list(cin) + list(cout)
        order = [order[i] for i in rng.permutation(len(order))]
        listing = ", ".join(order)
        ctx = build_A(tok, base, s["histA"], set(s["used"]))
        know = cand_mass(ctx + f" Question: exactly one kind of word fits the pattern of "
                         f"what the other player has been saying. Which of these words "
                         f"matches that pattern best: {listing}? Answer with one word.",
                         "\nAnswer:", cin, cout)
        pred = cand_mass(ctx + f" Question: which of these words do you expect the other "
                         f"player to say next round: {listing}? Answer with one word.",
                         "\nAnswer:", cin, cout)
        act = cand_mass(ctx + f" For this round, you must choose your word from this "
                        f"list: {listing}.", "\nMy word:", cin, cout)
        # untouched continuation (harness check)
        histA = [tuple(x) for x in s["histA"]]
        histB = [tuple(x) for x in s["histB"]]
        used = set(s["used"])
        own_new = []
        met = False
        rtxt = restr_text(s["pv"], s["cat"])
        for bt in range(1, CONT_TURNS + 1):
            wA = gen_word(build_A(tok, base, histA, used), 555000 + 991 * si + bt, used)
            wB = gen_word(build_A(tok, base, histB, used) if False else
                          base + rtxt + " " + hist_lines(histB)
                          + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".",
                          666000 + 991 * si + bt, used)
            own_new.append(wA)
            tf.write(json.dumps({"snap": si, "bturn": bt, "A": wA, "B": wB,
                                 "agreed": wA == wB}) + "\n")
            if wA == wB and wA:
                met = True
                break
            used |= {wA, wB}
            histA.append((wB, wA)); histB.append((wA, wB))
        loop_exit = not any(any(w[:4] == p[:4] and len(w) > 3 for p in s["own"])
                            for w in own_new[:5])
        results.append({"snap": si, "cat": s["cat"], "pv": s["pv"], "roll": s["roll"],
                        "P_know": know, "P_pred": pred, "P_act": act,
                        "dissociation": know - act, "cont_loop_exit": bool(loop_exit),
                        "cont_met": bool(met)})
        print(f"[dissoc] {si}: know={know:.2f} pred={pred:.2f} act={act:.2f} "
              f"diss={know-act:+.2f} exit={loop_exit}", flush=True)
    tf.close()
    D = np.array([r["dissociation"] for r in results])
    boots = [np.mean(rng.choice(D, len(D))) for _ in range(2000)]
    out = {"n_snapshots": len(results),
           "P_know_mean": float(np.mean([r["P_know"] for r in results])),
           "P_pred_mean": float(np.mean([r["P_pred"] for r in results])),
           "P_act_mean": float(np.mean([r["P_act"] for r in results])),
           "dissociation_mean": float(D.mean()),
           "dissociation_CI95": [float(np.percentile(boots, 2.5)),
                                 float(np.percentile(boots, 97.5))],
           "frac_positive": float((D > 0).mean()),
           "untouched_loop_exit": float(np.mean([r["cont_loop_exit"] for r in results])),
           "untouched_met": float(np.mean([r["cont_met"] for r in results])),
           "per_snapshot": results}
    json.dump(out, open(os.path.join(RUN_DIR, "dissociation.json"), "w"), indent=1)
    print(f"[dissoc] === know {out['P_know_mean']:.2f} pred {out['P_pred_mean']:.2f} "
          f"act {out['P_act_mean']:.2f} | dissociation {out['dissociation_mean']:+.2f} "
          f"CI {out['dissociation_CI95']} | positive {out['frac_positive']:.2f}", flush=True)


if __name__ == "__main__":
    main()
