"""TRANSPLANT + PARTNER-REPLAY (2026-08-02): does stuckness follow inserted
self-actions under IDENTICAL partner evidence? (items 5.1 + 5.5)

Base partner streams: B's word sequences from long strict-run games (steady category
evidence, non-adaptive replay — every branch sees the same B words). At turn 4, A's
own-history (rounds 1-3) is one of:
  loop_seed    : 3 consecutive same-family words taken from a stuck game's loop
  met_seed     : 3 consecutive (non-family) words from a met game
  neutral_seed : 3 unrelated words from other games' starts
A then plays live for BR(14) turns against the replayed B stream (resample-24 handler,
fallback repeats logged). Measured per branch: onset within 14 (A produces a word in a
4-prefix family of its seeded/own words), family-continuation count, and a K-sample
proposal profile at the branch point (frac self_family etc., as in proposal_telemetry).

Prediction under the endogenous-attractor account: loop_seed raises onset + self-family
proposal mass despite identical partner evidence; met/neutral seeds do not.

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) N_STREAMS(8) BR(14) K(64) TEMP(0.7)
     RUN_DIR(runs/transplant_replay)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
N_STREAMS = int(os.environ.get("N_STREAMS", "8"))
BR = int(os.environ.get("BR", "14"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/transplant_replay")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")


def fam_of(w):
    return w[:4] if len(w) > 3 else None


def load_games(cat):
    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, f"game1_strict_{cat}_transcript.jsonl")):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    for g in games.values():
        g.sort(key=lambda r: r["turn"])
    return dict(games)


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    @torch.no_grad()
    def gen_word(prompt, seed, forbidden):
        enc = tok(prompt, return_tensors="pt").to(dev)
        w, fb = "", True
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in forbidden:
                return w, False
        return w, True

    @torch.no_grad()
    def propose_k(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        batch = ids.repeat(K, 1)
        out = model.generate(batch, max_new_tokens=4, do_sample=True, temperature=TEMP,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    results = []
    tf = open(os.path.join(RUN_DIR, "transplant_transcript.jsonl"), "w")
    for cat in ("city", "fruit"):
        games = load_games(cat)
        # donor pools
        loop_donors, met_donors = [], []
        for roll, ts in games.items():
            A = [t["A"] for t in ts]
            for i in range(len(A) - 2):
                fs = [fam_of(w) for w in A[i:i + 3]]
                if None not in fs and len(set(fs)) == 1:
                    loop_donors.append(A[i:i + 3])
                    break
        for roll, ts in games.items():
            if ts[-1]["agreed"] and len(ts) >= 4:
                A = [t["A"] for t in ts[:3]]
                if len({fam_of(w) for w in A if fam_of(w)}) == len([w for w in A if fam_of(w)]):
                    met_donors.append(A)
        neutral_donors = [[starts[i][0], starts[i + 1][0], starts[i + 2][0]]
                          for i in range(20, 20 + 6)]
        # partner streams: longest games' B sequences
        streams = sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS // 2]
        rng = np.random.default_rng(0)
        for si_, (roll, ts) in enumerate(streams):
            Bseq = [t["B"] for t in ts]
            if len(Bseq) < 4 + BR:
                Bseq = Bseq + Bseq[::-1]           # pad by reflection if short
            sa, sb = starts[roll]
            for cond, donor in (("loop_seed", loop_donors[si_ % len(loop_donors)]),
                                ("met_seed", met_donors[si_ % len(met_donors)]),
                                ("neutral_seed", neutral_donors[si_ % len(neutral_donors)])):
                histA = [(sb, sa)]
                used = {sa, sb}
                own = [sa]
                for k3 in range(3):
                    histA.append((Bseq[k3], donor[k3]))
                    used |= {Bseq[k3], donor[k3]}
                    own.append(donor[k3])
                seed_fams = {fam_of(w) for w in donor if fam_of(w)}
                body = (G.OPEN_PROMPT + " "
                        + " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                                   for k, (o, s) in enumerate(histA))
                        + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".")
                props = propose_k(LA._render(tok, body) + "\nMy word:")
                fam_mass = np.mean([1 if (fam_of(w) in seed_fams and w not in used) else 0
                                    for w in props if w])
                onset, fam_hits, fbs = None, 0, 0
                for bt in range(BR):
                    body = (G.OPEN_PROMPT + " "
                            + " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                                       for k, (o, s) in enumerate(histA))
                            + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".")
                    wA, fb = gen_word(LA._render(tok, body) + "\nMy word:",
                                      321000 + 7919 * si_ + 131 * bt, used)
                    fbs += fb
                    wB = Bseq[3 + bt] if 3 + bt < len(Bseq) else Bseq[-1]
                    ofam = {fam_of(w) for w in own if fam_of(w)}
                    if fam_of(wA) in ofam:
                        fam_hits += 1
                        if onset is None:
                            onset = bt + 1
                    tf.write(json.dumps({"cat": cat, "stream": roll, "cond": cond,
                                         "bturn": bt + 1, "A": wA, "B": wB,
                                         "fallback": fb}) + "\n")
                    tf.flush()
                    used |= {wA, wB}
                    own.append(wA)
                    histA.append((wB, wA))
                results.append({"cat": cat, "stream": roll, "cond": cond,
                                "seed_fam_proposal_mass": float(fam_mass),
                                "onset": onset, "fam_hits": fam_hits, "fallbacks": fbs})
                print(f"[tx] {cat} stream={roll} {cond}: propmass={fam_mass:.2f} "
                      f"onset={onset} fam_hits={fam_hits} fb={fbs}", flush=True)
    tf.close()
    out = {"per_branch": results}
    for cond in ("loop_seed", "met_seed", "neutral_seed"):
        sel = [r for r in results if r["cond"] == cond]
        out[cond] = {"n": len(sel),
                     "onset_frac": float(np.mean([r["onset"] is not None for r in sel])),
                     "mean_fam_hits": float(np.mean([r["fam_hits"] for r in sel])),
                     "mean_prop_mass": float(np.mean([r["seed_fam_proposal_mass"] for r in sel])),
                     "mean_fallbacks": float(np.mean([r["fallbacks"] for r in sel]))}
        print(f"[tx] === {cond}: onset {out[cond]['onset_frac']:.2f} "
              f"fam_hits {out[cond]['mean_fam_hits']:.1f} "
              f"propmass {out[cond]['mean_prop_mass']:.2f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "transplant.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
