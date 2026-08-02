"""INTERVENTION LADDER (2026-08-02): does partner-modeling instruction fix the
proposal collapse, and at which rung? Outcome = collapse mass, not just success.

Rungs (added to A's prompt only; B restricted as usual):
  r0_none    : (baseline, no addition)
  r1_think   : "Think carefully before choosing your word."
  r2_predict : "Before choosing, silently predict what the other player will say next."
  r3_rule    : "Try to identify the rule behind the other player's choices."
  r4_ruleact : "Identify the rule behind the other player's choices, and choose your
                own word according to that rule."
Restricted city game, resample-24 handler, cap CAP(40), n=N(12)/rung, plus an MC
proposal profile (K=64) at turn PROBE(8) for every game still alive: mass on used /
self-family / category.

Env: MODEL(QwenInst32) START_FILE N(12) CAP(40) K(64) TEMP(0.7) PROBE(8)
     RUN_DIR(runs/intervention_ladder)
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
CAP = int(os.environ.get("CAP", "40"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
PROBE = int(os.environ.get("PROBE", "8"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/intervention_ladder")

RUNGS = {
    "r0_none": "",
    "r1_think": " Think carefully before choosing your word.",
    "r2_predict": " Before choosing, silently predict what the other player will say next.",
    "r3_rule": " Try to identify the rule behind the other player's choices.",
    "r4_ruleact": (" Identify the rule behind the other player's choices, and choose "
                   "your own word according to that rule."),
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

    @torch.no_grad()
    def propose_k(body):
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
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
    tf = open(os.path.join(RUN_DIR, "ladder_transcript.jsonl"), "w")
    for rung, add in RUNGS.items():
        met, onsets, probes = [], [], []
        for roll, (sa, sb) in enumerate(starts):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            own = [sa]
            agreed_at, onset = None, None
            for t in range(1, CAP):
                if t == PROBE:
                    props = propose_k(body_of(histA, used, add))
                    fams = {w[:4] for w in own if len(w) > 3}
                    pr = {"used": np.mean([1 if (w and w in used) else 0 for w in props]),
                          "selffam": np.mean([1 if (w and w not in used and len(w) > 3
                                              and w[:4] in fams) else 0 for w in props]),
                          "cat": np.mean([1 if (w and w not in used and w in catset) else 0
                                          for w in props])}
                    probes.append(pr)
                wA = gen_word(body_of(histA, used, add), 5000 * roll + t, used)
                wB = gen_word(body_of(histB, used, RESTR), 90000 + 5000 * roll + t, used)
                if onset is None and len(wA) > 3 and any(wA[:4] == p[:4] and len(p) > 3
                                                          for p in own):
                    onset = t
                tf.write(json.dumps({"rung": rung, "rollout": roll, "turn": t,
                                     "A": wA, "B": wB, "agreed": wA == wB}) + "\n")
                tf.flush()
                if wA == wB and wA:
                    agreed_at = t
                    break
                used |= {wA, wB}
                own.append(wA)
                histA.append((wB, wA)); histB.append((wA, wB))
            met.append(agreed_at is not None)
            onsets.append(onset is not None)
            print(f"[lad] {rung} roll={roll}: "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'} onset={onset}",
                  flush=True)
        summary["rungs"][rung] = {
            "met_frac": float(np.mean(met)), "onset_frac": float(np.mean(onsets)),
            "probe_used": float(np.mean([p["used"] for p in probes])) if probes else None,
            "probe_selffam": float(np.mean([p["selffam"] for p in probes])) if probes else None,
            "probe_cat": float(np.mean([p["cat"] for p in probes])) if probes else None,
            "n_probes": len(probes)}
        json.dump(summary, open(os.path.join(RUN_DIR, "ladder.json"), "w"), indent=1)
        s = summary["rungs"][rung]
        print(f"[lad] === {rung}: met {s['met_frac']:.2f} onset {s['onset_frac']:.2f} "
              f"probe cat {s['probe_cat']} selffam {s['probe_selffam']}", flush=True)
    tf.close()


if __name__ == "__main__":
    main()
