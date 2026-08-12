"""HEAD READS (2026-08-07): what do the 32 partner-patch restoration heads
actually LOOK AT in game states — and does it differ between the induction and
non-induction halves, and between regimes?

For each head of interest, on d3_self and d8_list states, measure the attention
mass from the FINAL answer position to token groups:
  fam    tokens of the planted family words (wherever they occur)
  Bword  tokens of the partner's words in the round lines
  used   the "Words already used: ..." section
  other  everything else (incl. rules text)

Head groups compared: restoration&induction (>=95th ind pctile), restoration
non-induction, top-induction NON-restoration, random non-member controls.

Eager attention (output_attentions), bf16, single forwards.

Env: MODEL(QwenInst32) PATCH_JSON IND_JSON SRC_DIR START_FILE N_STREAMS(4)
     RUN_DIR(runs/head_reads)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
PATCH_JSON = os.environ.get("PATCH_JSON", "runs/mech_inputs/qwen32_partner_patch.json")
IND_JSON = os.environ.get("IND_JSON", "runs/mech_inputs/qwen32_induction_overlap.json")
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N_STREAMS = int(os.environ.get("N_STREAMS", "4"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/head_reads")

PLANT8 = ["planted", "planting", "plantings", "replant",
          "replanted", "planter", "planters", "plantation"]
FILLER = ["window", "carpet", "stapler", "napkin", "candle", "basket", "ribbon", "saddle"]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = LA.SPEC[MODEL][0]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    nL = model.config.num_hidden_layers
    nH = model.config.num_attention_heads

    ov = json.load(open(IND_JSON))
    rest = [(h["layer"], h["head"], h["induction_pctile"]) for h in ov["top_restoration_heads"]]
    rest_ind = [(l, h) for l, h, p in rest if p >= 0.95]
    rest_non = [(l, h) for l, h, p in rest if p < 0.95]
    I = np.array(ov["induction"])
    iorder = np.argsort(I.flatten())[::-1]
    restset = {(l, h) for l, h, _ in rest}
    top_ind_only = []
    for i in iorder:
        lh = (int(i // I.shape[1]), int(i % I.shape[1]))
        if lh not in restset:
            top_ind_only.append(lh)
        if len(top_ind_only) >= 16:
            break
    rng = np.random.default_rng(0)
    pool = [(l, h) for l in range(nL) for h in range(nH)
            if (l, h) not in restset and (l, h) not in set(top_ind_only)]
    rand16 = [pool[i] for i in rng.choice(len(pool), 16, replace=False)]
    GROUPS = {"rest_ind": rest_ind, "rest_non": rest_non,
              "ind_only": top_ind_only, "rand": rand16}

    def body_build(Bseq, sa, sb, cell):
        hist = [(sb, sa)]
        used = {sa, sb}
        extra = ""
        n = 3 if cell == "d3_self" else 8
        for i in range(n):
            a = PLANT8[i] if cell == "d3_self" else FILLER[i]
            hist.append((Bseq[i], a))
            used |= {a, Bseq[i]}
        if cell == "d8_list":
            extra = " Unrelated word list: " + ", ".join(PLANT8) + "."
            used |= set(PLANT8)
        body = (G.OPEN_PROMPT + extra + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
            + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".")
        Bwords = [o for o, _ in hist]
        return body, Bwords

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

    def spans_of(prompt, words):
        idx = set()
        enc = tok(prompt, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        for w in set(words):
            start = 0
            while True:
                j = prompt.find(w, start)
                if j < 0:
                    break
                for ti, (a, b) in enumerate(offs):
                    if a < j + len(w) and b > j:
                        idx.add(ti)
                start = j + len(w)
        return idx, len(offs)

    @torch.no_grad()
    def profile(cell, Bseq, sa, sb):
        body, Bwords = body_build(Bseq, sa, sb, cell)
        prompt = LA._render(tok, body) + "\nMy word:"
        fam_idx, T = spans_of(prompt, PLANT8)
        b_idx, _ = spans_of(prompt, Bwords)
        u0 = prompt.find("Words already used")
        enc = tok(prompt, return_offsets_mapping=True)
        used_idx = {ti for ti, (a, b) in enumerate(enc["offset_mapping"]) if a >= u0 and u0 > 0}
        fam_hist = fam_idx - used_idx
        b_hist = b_idx - used_idx - fam_idx
        ids = torch.tensor([enc["input_ids"]], device=dev)
        out = model(ids, output_attentions=True)
        rows = {}
        for gname, heads in GROUPS.items():
            fa, ba, ua, oa = [], [], [], []
            for l, h in heads:
                att = out.attentions[l][0, h, -1].float()
                f = float(att[list(fam_hist)].sum()) if fam_hist else 0.0
                bb = float(att[list(b_hist)].sum()) if b_hist else 0.0
                uu = float(att[list(used_idx)].sum()) if used_idx else 0.0
                fa.append(f); ba.append(bb); ua.append(uu)
                oa.append(max(0.0, 1.0 - f - bb - uu))
            rows[gname] = {"fam": float(np.mean(fa)), "Bword": float(np.mean(ba)),
                           "used": float(np.mean(ua)), "other": float(np.mean(oa))}
        del out
        torch.cuda.empty_cache()
        return rows

    results = []
    for si, ts in enumerate(streams):
        roll = ts[0]["rollout"]
        Bseq = [t["B"] for t in ts]
        if len(Bseq) < 8:
            Bseq = Bseq + Bseq
        sa, sb = starts[roll]
        for cell in ("d3_self", "d8_list"):
            rows = profile(cell, Bseq, sa, sb)
            results.append({"stream": roll, "cell": cell, "groups": rows})
            json.dump({"per_state": results}, open(os.path.join(RUN_DIR, "head_reads.json"), "w"))
        print(f"[hr] stream {roll} done", flush=True)

    out = {"per_state": results, "groups": {},
           "head_lists": {k: [list(x) for x in v] for k, v in GROUPS.items()}}
    for cell in ("d3_self", "d8_list"):
        for gname in GROUPS:
            sel = [r["groups"][gname] for r in results if r["cell"] == cell]
            out["groups"][f"{cell}_{gname}"] = {k: float(np.mean([s[k] for s in sel]))
                                                for k in ("fam", "Bword", "used", "other")}
            c = out["groups"][f"{cell}_{gname}"]
            print(f"[hr] === {cell} {gname}: fam {c['fam']:.3f} B {c['Bword']:.3f} "
                  f"used {c['used']:.3f} other {c['other']:.3f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "head_reads.json"), "w"), indent=1)
    print("[hr] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
