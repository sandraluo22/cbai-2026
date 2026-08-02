"""STRICT-CONSTRAINT restricted Game-1 (2026-08-02): artifact control + logit telemetry.

Objection addressed: maybe the repetition pathology is an artifact of the resampler's
fallback (which admits repeats when 24 tries fail). Here novelty is GUARANTEED:
  * resample up to RS(64) times;
  * if exhausted, constrained argmax: mask the first tokens of every used word and all
    non-alphabetic tokens at the answer position and take the best novel continuation.
  * fallback usage is COUNTED and logged per turn (expect rare); repeats impossible.

Per-turn telemetry for A (item 5.4 logit competition), at the answer position:
  cat_mass    : prob mass on first-tokens of an in-category wordlist (unused only)
  self_mass   : prob mass on first-tokens of A's OWN previous words (the self-
                neighborhood, incl. now-forbidden words)
  used_mass   : prob mass on first-tokens of all used words
  entropy     : full-vocab entropy
Onset detection logged online (first morphological recurrence: new word sharing a
4-prefix with an earlier own word).

Env: MODEL(QwenInst32) START_FILE N(24) SAFETY(100) TEMP(0.7) RS(64)
     CATS(city,fruit) RUN_DIR(runs/qwen32_strict)
Out: <RUN_DIR>/game1_strict.json + per-cat transcript (words + telemetry + fallback ct)
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N = int(os.environ.get("N", "24"))
SAFETY = int(os.environ.get("SAFETY", "100"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RS = int(os.environ.get("RS", "64"))
CATS = os.environ.get("CATS", "city,fruit").split(",")
RUN_DIR = os.environ.get("RUN_DIR", "runs/qwen32_strict")

CONCEPTS = {"city": ("the name of a city", "cities"), "fruit": ("a fruit", "fruits")}
CATWORDS = {
    "city": ["paris", "london", "tokyo", "rome", "berlin", "madrid", "moscow", "vienna",
             "oslo", "athens", "cairo", "sydney", "toronto", "chicago", "boston", "seattle",
             "denver", "dallas", "houston", "miami", "dublin", "lisbon", "prague", "warsaw",
             "budapest", "helsinki", "stockholm", "geneva", "zurich", "munich", "hamburg",
             "seoul", "osaka", "beijing", "shanghai", "delhi", "mumbai", "bangkok", "manila",
             "jakarta", "havana", "lima", "bogota", "santiago", "nairobi", "lagos", "accra",
             "istanbul", "dubai", "amsterdam"],
    "fruit": ["apple", "banana", "orange", "grape", "lemon", "lime", "peach", "pear", "plum",
              "cherry", "mango", "kiwi", "melon", "papaya", "guava", "fig", "date", "apricot",
              "coconut", "pineapple", "strawberry", "raspberry", "blueberry", "blackberry",
              "cranberry", "currant", "lychee", "nectarine", "pomegranate", "tangerine",
              "grapefruit", "watermelon", "cantaloupe", "persimmon", "quince"]}


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    V = model.config.vocab_size

    def fid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]

    # precompute alphabetic-token mask once
    alpha_ok = np.zeros(V, dtype=bool)
    toks = tok.convert_ids_to_tokens(list(range(V)))
    for i, t in enumerate(toks):
        s = (t or "").replace("Ġ", "").replace("▁", "")
        alpha_ok[i] = s.isalpha() and s.islower() and len(s) >= 2

    @torch.no_grad()
    def step(prompt, seed, used, own_words, cat):
        enc = tok(prompt, return_tensors="pt").to(dev)
        logits = model(enc.input_ids).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        cat_ids = list({fid(w) for w in CATWORDS[cat] if w not in used})
        self_ids = list({fid(w) for w in own_words})
        used_ids = list({fid(w) for w in used})
        tele = {"cat_mass": float(p[cat_ids].sum()) if cat_ids else 0.0,
                "self_mass": float(p[self_ids].sum()) if self_ids else 0.0,
                "used_mass": float(p[used_ids].sum()) if used_ids else 0.0,
                "entropy": float(-(p * torch.log(p + 1e-12)).sum())}
        # sample words until novel, NO repeat fallback
        w, fb = "", 0
        for r in range(RS):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in used:
                return w, tele, r + 1, 0
        # constrained argmax: best alphabetic first-token not starting any used word
        mask = torch.full((V,), float("-inf"), device=logits.device)
        ok = torch.tensor(alpha_ok, device=logits.device)
        for uid in used_ids:
            ok[uid] = False
        mask[ok] = 0.0
        for r in range(200):
            tid = int((logits + mask).argmax())
            forced = torch.cat([enc.input_ids, torch.tensor([[tid]], device=dev)], 1)
            out = model.generate(forced, max_new_tokens=3, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in used:
                return w, tele, RS, r + 1
            mask[tid] = float("-inf")
        raise RuntimeError("could not produce a novel word")

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            pp = line.split("\t") if "\t" in line else line.split()
            starts.append((pp[-2], pp[-1]))
    starts = starts[:N]

    summary = {"model": MODEL, "temp": TEMP, "safety": SAFETY, "rs": RS, "n": N,
               "conditions": {}}
    for cat in CATS:
        sing, plur = CONCEPTS[cat]
        restr = (f" IMPORTANT: every single word you say must be {sing}. Only ever say "
                 f"{plur}, nothing else.")
        tf = open(os.path.join(RUN_DIR, f"game1_strict_{cat}_transcript.jsonl"), "w")
        met, onsets, pers_all, fb_turns = [], [], [], 0
        for roll, (sa, sb) in enumerate(starts):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            ownA = [sa]
            agreed_at, onset = None, None
            for t in range(1, SAFETY):
                pA = LA._render(tok, G.OPEN_PROMPT
                                + " " + " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                                                 for k, (o, s) in enumerate(histA))
                                + " Words already used (do not repeat): " + ", ".join(sorted(used))
                                + ".") + "\nMy word:"
                wA, tele, nsA, fbA = step(pA, 5000 * roll + t, used, ownA, cat)
                pB = LA._render(tok, G.OPEN_PROMPT + restr
                                + " " + " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                                                 for k, (o, s) in enumerate(histB))
                                + " Words already used (do not repeat): " + ", ".join(sorted(used))
                                + ".") + "\nMy word:"
                wB, _, nsB, fbB = step(pB, 90000 + 5000 * roll + t, used, [], cat)
                fb_turns += (fbA > 0) + (fbB > 0)
                if onset is None and any(wA[:4] == p4[:4] and len(wA) > 3 for p4 in ownA):
                    onset = t
                tf.write(json.dumps({"cond": f"strict-{cat}", "rollout": roll, "turn": t,
                                     "A": wA, "B": wB, "agreed": wA == wB,
                                     "resamplesA": nsA, "fallbackA": fbA, **tele}) + "\n")
                tf.flush()
                if wA == wB and wA:
                    agreed_at = t
                    break
                used |= {wA, wB}
                ownA.append(wA)
                histA.append((wB, wA)); histB.append((wA, wB))
            pers = float(np.mean([any(w[:4] == q[:4] and len(w) > 3 for q in ownA[1:i])
                                  for i, w in enumerate(ownA[1:], 1)][1:])) if len(ownA) > 2 else 0.0
            met.append(agreed_at is not None)
            onsets.append(onset)
            pers_all.append(pers)
            print(f"[strict] {cat} roll={roll}: "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'} onset={onset} "
                  f"pers={pers:.2f}", flush=True)
        tf.close()
        summary["conditions"][f"strict-{cat}"] = {
            "n": N, "met_frac": float(np.mean(met)),
            "onset_frac": float(np.mean([o is not None for o in onsets])),
            "pers_mean": float(np.mean(pers_all)), "fallback_turns": int(fb_turns)}
        json.dump(summary, open(os.path.join(RUN_DIR, "game1_strict.json"), "w"), indent=1)
        print(f"[strict] === {cat}: met {np.mean(met):.2f} "
              f"onset {np.mean([o is not None for o in onsets]):.2f} fb={fb_turns}", flush=True)


if __name__ == "__main__":
    main()
