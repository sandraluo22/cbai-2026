"""LOGIT LENS OVER REGIMES (2026-08-07): layer-by-layer trajectory of family vs
category mass at the answer position — is the category plan OVERWRITTEN late, or
never competitive?

Cells (6 streams each, one forward with output_hidden_states):
  d0            neutral history (category should win)
  d3_self       early regime
  d8_self       late regime, in-column
  d8_list       late regime, bare list
  d3_oracle     d3_self + oracle instruction ("the other player is only ever
                saying names of cities... say the name of a city") — where in
                the stack does instruction rescue?
  d3_note_obs   d3_self + self-attributed scratchpad OBSERVATION
                ("My scratchpad: The other player seems to be naming cities.")
  d3_note_commit d3_self + scratchpad COMMITMENT
                ("My scratchpad: I will say the name of a city this round.")
                -> the layer-resolved observation-vs-commitment contrast.

Per layer l (0..nL): logit-lens z = lm_head(norm(h_l)) at the final position;
p_fam = softmax mass on family first-tokens, p_cat = mass on unused-city first
tokens; also Delta = logsumexp(fam) - logsumexp(cat).

Env: MODEL(QwenInst32) SRC_DIR START_FILE N_STREAMS(6) RUN_DIR(runs/logit_lens)
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
N_STREAMS = int(os.environ.get("N_STREAMS", "6"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/logit_lens")

PLANT8 = ["planted", "planting", "plantings", "replant",
          "replanted", "planter", "planters", "plantation"]
FILLER = ["window", "carpet", "stapler", "napkin", "candle", "basket", "ribbon", "saddle"]
ORACLE = (" The other player is only ever saying names of cities. To win, you "
          "should also say the name of a city.")
NOTE_OBS = "\nMy scratchpad: The other player seems to be naming cities."
NOTE_COMMIT = "\nMy scratchpad: I will say the name of a city this round."

CELLS = ["d0", "d3_self", "d8_self", "d8_list", "d3_oracle", "d3_note_obs", "d3_note_commit"]


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nL = model.config.num_hidden_layers

    def fid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]
    fam_ids = torch.tensor(sorted({fid(w) for w in PLANT8 + ["plant", "plants"]})).to(dev)

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    def build(Bseq, sa, sb, cell):
        hist = [(sb, sa)]
        used = {sa, sb}
        extra = ""
        note = ""
        n = 8 if cell.startswith("d8") else 3
        for i in range(n):
            self_word = PLANT8[i] if cell in ("d3_self", "d8_self", "d3_oracle",
                                              "d3_note_obs", "d3_note_commit") else FILLER[i]
            hist.append((Bseq[i], self_word))
            used |= {self_word, Bseq[i]}
        if cell == "d8_list":
            extra = " Unrelated word list: " + ", ".join(PLANT8) + "."
            used |= set(PLANT8)
        if cell == "d3_oracle":
            extra = ORACLE
        if cell == "d3_note_obs":
            note = NOTE_OBS
        if cell == "d3_note_commit":
            note = NOTE_COMMIT
        return LA._render(tok, body_of(hist, used, extra)) + note + "\nMy word:", used

    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl")):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    streams = [sorted(ts, key=lambda r: r["turn"]) for _, ts in
               sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS]]
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    results = {c: [] for c in CELLS}
    for si, ts in enumerate(streams):
        roll = ts[0]["rollout"]
        Bseq = [t["B"] for t in ts]
        if len(Bseq) < 8:
            Bseq = Bseq + Bseq
        sa, sb = starts[roll]
        for cell in CELLS:
            prompt, used = build(Bseq, sa, sb, cell)
            catwords = [w for w in CATWORDS["city"] if w not in used][:30]
            cat_ids = torch.tensor(sorted({fid(w) for w in catwords})).to(dev)
            with __import__("torch").no_grad():
                ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
                out = model(ids, output_hidden_states=True)
                pf, pc, dl = [], [], []
                for l in range(1, nL + 1):
                    h = out.hidden_states[l][0, -1]
                    z = model.lm_head(model.model.norm(h.unsqueeze(0)))[0].float()
                    p = __import__("torch").softmax(z, 0)
                    pf.append(float(p[fam_ids].sum()))
                    pc.append(float(p[cat_ids].sum()))
                    dl.append(float(__import__("torch").logsumexp(z[fam_ids], 0)
                                    - __import__("torch").logsumexp(z[cat_ids], 0)))
                del out
            results[cell].append({"stream": roll, "p_fam": pf, "p_cat": pc, "delta": dl})
            json.dump(results, open(os.path.join(RUN_DIR, "logit_lens.json"), "w"))
        print(f"[ll] stream {roll} done", flush=True)

    out = {"per_state": results, "mean": {}}
    for cell in CELLS:
        out["mean"][cell] = {
            "p_fam": [float(np.mean([r["p_fam"][l] for r in results[cell]]))
                      for l in range(nL)],
            "p_cat": [float(np.mean([r["p_cat"][l] for r in results[cell]]))
                      for l in range(nL)],
            "delta": [float(np.mean([r["delta"][l] for r in results[cell]]))
                      for l in range(nL)]}
        m = out["mean"][cell]
        pkf = int(np.argmax(m["p_fam"])); pkc = int(np.argmax(m["p_cat"]))
        print(f"[ll] === {cell}: fam peak L{pkf} {m['p_fam'][pkf]:.3f} final {m['p_fam'][-1]:.3f} "
              f"| cat peak L{pkc} {m['p_cat'][pkc]:.3f} final {m['p_cat'][-1]:.3f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "logit_lens.json"), "w"), indent=1)
    print("[ll] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
