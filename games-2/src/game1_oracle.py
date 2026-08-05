"""ORACLE DISCLOSURE (2026-08-04): the missing top rung of the intervention ladder.

Does the model comply when the correct answer is handed to it outright?

  r0_none          baseline rerun (same seeds as intervention_ladder r0).
  r5_oracle        A's prompt literally states the partner's rule AND directs
                   compliance: "The other player is only ever saying names of
                   cities. To win, you should also say the name of a city."
  r6_scratch_inject A has a private scratchpad, but its content is FORCED (not
                   generated): every turn the note reads exactly
                   "The other player only says city names. I will say the name
                   of a city this round." and the word is elicited right after.
                   -> tests whether an explicit self-attributed commitment to
                   cities is obeyed, vs the forced_plan finding that
                   self-GENERATED notes plan cities only 15% of the time.

Restricted city game, resample-24 handler, cap CAP(100), n=N(12), same start
words and seed structure as intervention_ladder (directly comparable to r0-r4).
Per-turn log: A word, B word, played_is_cat. MC proposal profile (K=64) at turn
PROBE(8) for games still alive: mass on used / self-family / category.

Env: MODEL(QwenInst32) START_FILE N(12) CAP(100) K(64) TEMP(0.7) PROBE(8)
     RUN_DIR(runs/oracle) RUNG_KEYS
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G
from game1_strict import CATWORDS

MODEL = os.environ.get("MODEL", "QwenInst32")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N = int(os.environ.get("N", "12"))
CAP = int(os.environ.get("CAP", "100"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
PROBE = int(os.environ.get("PROBE", "8"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/oracle")

ORACLE = (" The other player is only ever saying names of cities. To win, you "
          "should also say the name of a city.")
SCRATCH_INTRO = (" You have a private scratchpad the other player never sees; use it "
                 "for brief notes about what is happening and your plan.")
INJECTED_NOTE = ("The other player only says city names. I will say the name of a "
                 "city this round.")

RUNGS = {
    "r0_none": {"add": "", "inject": False},
    "r5_oracle": {"add": ORACLE, "inject": False},
    "r6_scratch_inject": {"add": SCRATCH_INTRO, "inject": True},
}
RESTR = (" IMPORTANT: every single word you say must be the name of a city. Only ever "
         "say cities, nothing else.")
RUNG_KEYS = os.environ.get("RUNG_KEYS", "")
if RUNG_KEYS:
    RUNGS = {k: v for k, v in RUNGS.items() if k in RUNG_KEYS.split(",")}


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + ((" " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))) if hist else "")
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    def prompt_of(body, inject):
        p = LA._render(tok, body)
        if inject:
            p += "\nMy scratchpad: " + INJECTED_NOTE
        return p + "\nMy word:"

    @torch.no_grad()
    def gen_word(body, seed, forbidden, inject):
        enc = tok(prompt_of(body, inject), return_tensors="pt").to(dev)
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

    @torch.no_grad()
    def propose_k(body, inject):
        ids = tok(prompt_of(body, inject), return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    starts = starts[:N]

    summary = {"model": MODEL, "cap": CAP, "n": N, "rungs": {}}
    tf = open(os.path.join(RUN_DIR, "oracle_transcript.jsonl"), "w")
    for rung, cfg in RUNGS.items():
        add, inject = cfg["add"], cfg["inject"]
        met, onsets, probes, cat_turns = [], [], [], []
        for roll, (sa, sb) in enumerate(starts):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            own = [sa]
            agreed_at, onset = None, None
            for t in range(1, CAP):
                if t == PROBE:
                    props = propose_k(body_of(histA, used, add), inject)
                    fams = {w[:4] for w in own if len(w) > 3}
                    pr = {"used": np.mean([1 if (w and w in used) else 0 for w in props]),
                          "selffam": np.mean([1 if (w and w not in used and len(w) > 3
                                              and w[:4] in fams) else 0 for w in props]),
                          "cat": np.mean([1 if (w and w not in used and w in catset) else 0
                                          for w in props])}
                    probes.append(pr)
                wA = gen_word(body_of(histA, used, add), 5000 * roll + t, used, inject)
                wB = gen_word(body_of(histB, used, RESTR), 90000 + 5000 * roll + t, used, False)
                if onset is None and len(wA) > 3 and any(wA[:4] == p[:4] and len(p) > 3
                                                          for p in own):
                    onset = t
                cat_turns.append(wA in catset)
                tf.write(json.dumps({"rung": rung, "rollout": roll, "turn": t,
                                     "A": wA, "B": wB, "agreed": wA == wB,
                                     "played_is_cat": bool(wA in catset)}) + "\n")
                tf.flush()
                if wA == wB and wA:
                    agreed_at = t
                    break
                used |= {wA, wB}
                own.append(wA)
                histA.append((wB, wA)); histB.append((wA, wB))
            met.append(agreed_at is not None)
            onsets.append(onset is not None)
            print(f"[orc] {rung} roll={roll}: "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'} onset={onset}",
                  flush=True)
        summary["rungs"][rung] = {
            "met_frac": float(np.mean(met)), "onset_frac": float(np.mean(onsets)),
            "played_cat_frac": float(np.mean(cat_turns)),
            "probe_used": float(np.mean([p["used"] for p in probes])) if probes else None,
            "probe_selffam": float(np.mean([p["selffam"] for p in probes])) if probes else None,
            "probe_cat": float(np.mean([p["cat"] for p in probes])) if probes else None,
            "n_probes": len(probes)}
        json.dump(summary, open(os.path.join(RUN_DIR, "oracle.json"), "w"), indent=1)
        s = summary["rungs"][rung]
        print(f"[orc] === {rung}: met {s['met_frac']:.2f} played-cat {s['played_cat_frac']:.2f} "
              f"probe cat {s['probe_cat']} selffam {s['probe_selffam']}", flush=True)
    tf.close()
    print("[orc] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
