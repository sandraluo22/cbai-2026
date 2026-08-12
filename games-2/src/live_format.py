"""LIVE FORMAT VARIATION (2026-08-05): mech4b was passive (fixed replayed content,
measure one proposal). This runs the LIVE game with per-round format variation —
each round of A's history is rendered in a different format, the model generates,
and its word feeds back into the next round's differently-formatted history.

Conditions (A's prompt only; B always sees the standard fixed format):
  fixed_f0   every round rendered "Round k: the other player said B, you said A."
  all_diff   round k rendered with format f_{k mod 6} (6 distinct formats)

If within-prompt consistency of the recent rounds is what licenses the series
continuation (mech4b/4c), all_diff should PREVENT loop formation live: fewer
family runs (onset), more meets. n=N games, cap CAP, B city-restricted,
t8 MC proposal probe.

Env: MODEL(QwenInst32) START_FILE N(12) CAP(40) K(64) TEMP(0.7) PROBE(8)
     RUN_DIR(runs/live_format)
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
RUN_DIR = os.environ.get("RUN_DIR", "runs/live_format")

F = [
    lambda k, b, a: f"Round {k+1}: the other player said {b}, you said {a}.",
    lambda k, b, a: f"Round {k+1}: you said {a}, the other player said {b}.",
    lambda k, b, a: f"Them: {b}\nYou: {a}",
    lambda k, b, a: f"Round {k+1} - the other player; {b}. you; {a}.",
    lambda k, b, a: f"You: {a} | Them: {b}",
    lambda k, b, a: f"then they said {b} and you said {a};",
]
RESTR = (" IMPORTANT: every single word you say must be the name of a city. Only ever "
         "say cities, nothing else.")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    def body_A(hist, used, varied):
        lines = []
        for k, (o, s_) in enumerate(hist):
            f = F[k % 6] if varied else F[0]
            lines.append(f(k, o, s_))
        return (G.OPEN_PROMPT + " " + " ".join(lines)
                + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".")

    def body_B(hist, used):
        return (G.OPEN_PROMPT + RESTR + " " + " ".join(
            F[0](k, o, s_) for k, (o, s_) in enumerate(hist))
            + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".")

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

    summary = {"model": MODEL, "cap": CAP, "n": N, "conditions": {}}
    tf = open(os.path.join(RUN_DIR, "live_format_transcript.jsonl"), "w")
    for cond, varied in (("fixed_f0", False), ("all_diff", True)):
        met, onsets, probes = [], [], []
        for roll, (sa, sb) in enumerate(starts):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            own = [sa]
            agreed_at, onset = None, None
            for t in range(1, CAP):
                if t == PROBE:
                    props = propose_k(body_A(histA, used, varied))
                    fams = {w[:4] for w in own if len(w) > 3}
                    probes.append({
                        "cat": float(np.mean([1 if (w and w not in used and w in catset)
                                              else 0 for w in props])),
                        "selffam": float(np.mean([1 if (w and w not in used and len(w) > 3
                                                  and w[:4] in fams) else 0 for w in props]))})
                wA = gen_word(body_A(histA, used, varied), 5000 * roll + t, used)
                wB = gen_word(body_B(histB, used), 90000 + 5000 * roll + t, used)
                if onset is None and len(wA) > 3 and any(wA[:4] == p[:4] and len(p) > 3
                                                          for p in own):
                    onset = t
                tf.write(json.dumps({"cond": cond, "rollout": roll, "turn": t,
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
            print(f"[lf] {cond} roll={roll}: "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'} onset={onset}",
                  flush=True)
        summary["conditions"][cond] = {
            "met_frac": float(np.mean(met)), "onset_frac": float(np.mean(onsets)),
            "probe_cat": float(np.mean([p["cat"] for p in probes])) if probes else None,
            "probe_selffam": float(np.mean([p["selffam"] for p in probes])) if probes else None}
        s = summary["conditions"][cond]
        json.dump(summary, open(os.path.join(RUN_DIR, "live_format.json"), "w"), indent=1)
        print(f"[lf] === {cond}: met {s['met_frac']:.2f} onset {s['onset_frac']:.2f} "
              f"cat {s['probe_cat']} selffam {s['probe_selffam']}", flush=True)
    tf.close()
    print("[lf] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
