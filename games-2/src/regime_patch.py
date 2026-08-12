"""REGIME LAYER-PATCH (2026-08-07): do early (d3_self) and late (d8_list) capture
live in different layer windows?

Carrier sweep: for each stream, build a capture state (donor) and its matched
neutral control (identical B stream, filler own-words / no list). Run the donor,
cache the residual (decoder-block output) at the final answer position for every
layer. Then, layer by layer, run the CONTROL prompt with that single layer's
final-position residual replaced by the donor's, and measure

  Delta = logsumexp(z[family first-tokens]) - logsumexp(z[category first-tokens])

Recovery(l) = (Delta_patched - Delta_ctrl) / (Delta_donor - Delta_ctrl).
If the two regimes are carried in different windows, the recovery curves peak at
different layers (early regime known: late L48-62 from patch_transplant).

Env: MODEL(QwenInst32) SRC_DIR START_FILE N_STREAMS(6) RUN_DIR(runs/regime_patch)
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
RUN_DIR = os.environ.get("RUN_DIR", "runs/regime_patch")

PLANT8 = ["planted", "planting", "plantings", "replant",
          "replanted", "planter", "planters", "plantation"]
FILLER = ["window", "carpet", "stapler", "napkin", "candle", "basket", "ribbon", "saddle"]


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nL = len(model.model.layers)
    catset = set(CATWORDS["city"])

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
        n = 3 if cell.startswith("d3") else 8
        for i in range(n):
            a = PLANT8[i] if cell.endswith("self") else FILLER[i]
            hist.append((Bseq[i], a))
            used |= {a, Bseq[i]}
        if cell == "d8_list":
            extra = " Unrelated word list: " + ", ".join(PLANT8) + "."
            used |= set(PLANT8)
        return body_of(hist, used, extra), used

    cache = {}
    mode = {"m": "off", "layer": -1, "vec": None}
    def hook(l):
        def h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            if mode["m"] == "cap":
                cache[l] = hs[0, -1].detach().clone()
            elif mode["m"] == "patch" and l == mode["layer"]:
                hs[0, -1] = mode["vec"]
            return None
        return h
    for l in range(nL):
        model.model.layers[l].register_forward_hook(hook(l))

    @torch.no_grad()
    def delta(body, catwords):
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        z = model(ids).logits[0, -1].float()
        cat_ids = torch.tensor(sorted({fid(w) for w in catwords})).to(dev)
        return float(torch.logsumexp(z[fam_ids], 0) - torch.logsumexp(z[cat_ids], 0)), ids

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

    import torch
    results = {"d3_self": [], "d8_list": []}
    for si, ts in enumerate(streams):
        roll = ts[0]["rollout"]
        Bseq = [t["B"] for t in ts]
        if len(Bseq) < 8:
            Bseq = Bseq + Bseq
        sa, sb = starts[roll]
        for cell, ctrlcell in (("d3_self", "d3_ctrl"), ("d8_list", "d8_ctrl")):
            donor_body, du = build(Bseq, sa, sb, cell)
            ctrl_body, cu = build(Bseq, sa, sb, ctrlcell)
            catwords = [w for w in CATWORDS["city"] if w not in du and w not in cu][:30]
            mode["m"] = "cap"
            d_donor, _ = delta(donor_body, catwords)
            donor_cache = {l: cache[l].clone() for l in range(nL)}
            mode["m"] = "off"
            d_ctrl, _ = delta(ctrl_body, catwords)
            recs = []
            for l in range(nL):
                mode.update({"m": "patch", "layer": l, "vec": donor_cache[l]})
                d_p, _ = delta(ctrl_body, catwords)
                mode["m"] = "off"
                rec = (d_p - d_ctrl) / (d_donor - d_ctrl) if abs(d_donor - d_ctrl) > 1e-6 else 0.0
                recs.append(float(rec))
            results[cell].append({"stream": roll, "d_donor": d_donor, "d_ctrl": d_ctrl,
                                  "recovery": recs})
            json.dump(results, open(os.path.join(RUN_DIR, "regime_patch.json"), "w"))
            print(f"[rp] s{roll} {cell}: donor {d_donor:.2f} ctrl {d_ctrl:.2f} "
                  f"peak L{int(np.argmax(recs))} {max(recs):.2f}", flush=True)

    out = {"per_pair": results, "mean_recovery": {}}
    for cell in results:
        R = np.array([r["recovery"] for r in results[cell]])
        m = R.mean(0)
        out["mean_recovery"][cell] = [float(x) for x in m]
        top = np.argsort(m)[::-1][:5]
        print(f"[rp] === {cell}: top layers {[(int(l), round(float(m[l]), 2)) for l in top]}",
              flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "regime_patch.json"), "w"), indent=1)
    print("[rp] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
