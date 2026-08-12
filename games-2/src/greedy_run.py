"""GREEDY GAMES (2026-08-09): word-level temperature-0 baseline. Each turn both
players play the MOST PROBABLE LEGAL word (modal word of K=32 samples at the
generation temperature, excluding used words) — the argmax under the no-repeat
constraint (true token-greedy degenerates: the retry loop re-emits the same
forbidden word). Measures met/onset vs the stochastic baseline (met .54).
Env: MODEL(QwenInst32) START_FILE N(24) CAP(60) K(32) TEMP(0.7) RUN_DIR(runs/greedy_run)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N = int(os.environ.get("N", "24"))
CAP = int(os.environ.get("CAP", "60"))
K = int(os.environ.get("K", "32"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/greedy_run")

RESTR = (" IMPORTANT: every single word you say must be the name of a city. Only ever "
         "say cities, nothing else.")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + ((" " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))) if hist else "")
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    @torch.no_grad()
    def modal_word(body, seed, forbidden):
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        torch.manual_seed(seed)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        ws = [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
              for i in range(K)]
        cnt = collections.Counter(w for w in ws if w and w not in forbidden)
        return cnt.most_common(1)[0][0] if cnt else ""

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    starts = starts[:N]

    tf = open(os.path.join(RUN_DIR, "greedy_transcript.jsonl"), "w")
    met, onsets, turns = [], [], []
    for roll, (sa, sb) in enumerate(starts):
        histA, histB = [(sb, sa)], [(sa, sb)]
        used = {sa, sb}
        own = [sa]
        agreed_at, onset = None, None
        for t in range(1, CAP):
            wA = modal_word(body_of(histA, used), 5000 * roll + t, used)
            wB = modal_word(body_of(histB, used, RESTR), 90000 + 5000 * roll + t, used)
            if onset is None and len(wA) > 3 and any(wA[:4] == p[:4] and len(p) > 3
                                                      for p in own):
                onset = t
            tf.write(json.dumps({"rollout": roll, "turn": t, "A": wA, "B": wB,
                                 "agreed": wA == wB}) + "\n")
            tf.flush()
            if wA == wB and wA:
                agreed_at = t
                break
            used |= {wA, wB}
            own.append(wA)
            histA.append((wB, wA)); histB.append((wA, wB))
        met.append(agreed_at is not None)
        onsets.append(onset is not None)
        if agreed_at:
            turns.append(agreed_at)
        print(f"[gr] roll={roll}: {'MET@' + str(agreed_at) if agreed_at else 'no-meet'} "
              f"onset={onset}", flush=True)
    tf.close()
    out = {"met_frac": float(np.mean(met)), "onset_frac": float(np.mean(onsets)),
           "turns": turns}
    json.dump(out, open(os.path.join(RUN_DIR, "greedy.json"), "w"), indent=1)
    print(f"[gr] === met {out['met_frac']:.2f} onset {out['onset_frac']:.2f} "
          f"turns {sorted(turns)}", flush=True)
    print("[gr] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
