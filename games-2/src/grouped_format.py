"""GROUPED FORMAT (2026-08-08, user-designed): history rendered as two flat
per-player lists instead of round-by-round lines:

    "You've said: a, b, c. They've said: e, f, g."

Part 1 (passive): dose 0/3/8 family words in the You-list (chronological order),
replayed city B streams; MC K=64 family/category mass. Compares against the
round-format numbers (d3 .23 / d8 .70) and the bare-list cell (.82).
Part 2 (live): n=N games, A sees the grouped format (B standard round format,
city-restricted); met / family-run onset / t8 probe. Baselines: fixed_f0 met
.42, onset .75.

Env: MODEL(QwenInst32) SRC_DIR START_FILE N(12) CAP(40) K(64) TEMP(0.7) PROBE(8)
     RUN_DIR(runs/grouped_format)
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
N = int(os.environ.get("N", "12"))
CAP = int(os.environ.get("CAP", "40"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
PROBE = int(os.environ.get("PROBE", "8"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/grouped_format")

PLANT8 = ["planted", "planting", "plantings", "replant",
          "replanted", "planter", "planters", "plantation"]
FILLER = ["window", "carpet", "stapler", "napkin", "candle", "basket", "ribbon", "saddle"]
FAMP = "plan"
RESTR = (" IMPORTANT: every single word you say must be the name of a city. Only ever "
         "say cities, nothing else.")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    def body_grouped(own, other, used, extra=""):
        s = G.OPEN_PROMPT + extra
        if own or other:
            s += (" You've said: " + ", ".join(own) + ". They've said: "
                  + ", ".join(other) + ".")
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    def body_rounds(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

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

    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl")):
        r = json.loads(line)
        games[r["rollout"]].append(r)
    streams = [sorted(ts, key=lambda r: r["turn"]) for _, ts in
               sorted(games.items(), key=lambda kv: -len(kv[1]))[:6]]
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    # ---- part 1: passive dose cells ----
    passive = []
    for si, ts in enumerate(streams):
        roll = ts[0]["rollout"]
        Bseq = [t["B"] for t in ts]
        if len(Bseq) < 8:
            Bseq = Bseq + Bseq
        sa, sb = starts[roll]
        for dose in (0, 3, 8):
            n_r = max(dose, 3)
            own = [sa] + [PLANT8[i] if i < dose else FILLER[i] for i in range(n_r)]
            other = [sb] + Bseq[:n_r]
            used = set(own) | set(other)
            props = propose_k(body_grouped(own, other, used))
            fam = np.mean([1 if (w and w not in used and len(w) > 3 and w[:4] == FAMP)
                           else 0 for w in props])
            cat = np.mean([1 if (w and w not in used and w in catset) else 0 for w in props])
            passive.append({"dose": dose, "stream": roll, "fam_mass": float(fam),
                            "cat_mass": float(cat)})
        print(f"[gf] passive stream {roll} done", flush=True)
    out = {"passive": passive, "passive_cells": {}, "live": {}}
    for dose in (0, 3, 8):
        sel = [r for r in passive if r["dose"] == dose]
        out["passive_cells"][f"d{dose}"] = {
            "fam_mass": float(np.mean([r["fam_mass"] for r in sel])),
            "cat_mass": float(np.mean([r["cat_mass"] for r in sel]))}
        c = out["passive_cells"][f"d{dose}"]
        print(f"[gf] === passive d{dose}: fam {c['fam_mass']:.3f} cat {c['cat_mass']:.3f}",
              flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "grouped.json"), "w"), indent=1)

    # ---- part 2: live games ----
    met, onsets, probes = [], [], []
    tf = open(os.path.join(RUN_DIR, "grouped_transcript.jsonl"), "w")
    for roll, (sa, sb) in enumerate(starts[:N]):
        ownA, othA = [sa], [sb]
        histB = [(sa, sb)]
        used = {sa, sb}
        agreed_at, onset = None, None
        for t in range(1, CAP):
            if t == PROBE:
                props = propose_k(body_grouped(ownA, othA, used))
                fams = {w[:4] for w in ownA if len(w) > 3}
                probes.append({
                    "cat": float(np.mean([1 if (w and w not in used and w in catset)
                                          else 0 for w in props])),
                    "selffam": float(np.mean([1 if (w and w not in used and len(w) > 3
                                              and w[:4] in fams) else 0 for w in props]))})
            wA = gen_word(body_grouped(ownA, othA, used), 5000 * roll + t, used)
            wB = gen_word(body_rounds(histB, used, RESTR), 90000 + 5000 * roll + t, used)
            if onset is None and len(wA) > 3 and any(wA[:4] == p[:4] and len(p) > 3
                                                      for p in ownA):
                onset = t
            tf.write(json.dumps({"rollout": roll, "turn": t, "A": wA, "B": wB,
                                 "agreed": wA == wB}) + "\n")
            tf.flush()
            if wA == wB and wA:
                agreed_at = t
                break
            used |= {wA, wB}
            ownA.append(wA); othA.append(wB)
            histB.append((wA, wB))
        met.append(agreed_at is not None)
        onsets.append(onset is not None)
        print(f"[gf] live roll={roll}: "
              f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'} onset={onset}",
              flush=True)
    tf.close()
    out["live"] = {
        "met_frac": float(np.mean(met)), "onset_frac": float(np.mean(onsets)),
        "probe_cat": float(np.mean([p["cat"] for p in probes])) if probes else None,
        "probe_selffam": float(np.mean([p["selffam"] for p in probes])) if probes else None}
    s = out["live"]
    print(f"[gf] === live: met {s['met_frac']:.2f} onset {s['onset_frac']:.2f} "
          f"cat {s['probe_cat']} selffam {s['probe_selffam']}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "grouped.json"), "w"), indent=1)
    print("[gf] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
