"""DEPTH-SWEPT PAIRED PATCHING (2026-08-03): layer x game-depth heatmap.

For each replayed city B stream and each seeding depth d in DEPTHS, build a matched
pair differing ONLY in A's attributed words:
  loop state : A's d words = d model-generated family variants (harvested from the
               strict games' own outputs — e.g. novel, novelty, novella, noveles...)
  ctrl state : A's d words = d neutral filler words (unused start words)
Identical B words (replayed stream) in both. Control -> loop residual patch at the
FINAL position, per layer; Delta = logsumexp(family first tokens) - logsumexp(city
first tokens); efficacy = (D_loop - D_patch) / (D_loop - D_ctrl).

Output: RUN_DIR/patch_depth.json with recovery[depth][layer] per stream + means,
plus per-depth baselines (D_loop, D_ctrl) — the entrenchment curve.

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) START_FILE N_STREAMS(4)
     DEPTHS(1,2,3,4,6,8,10,12) RUN_DIR(runs/patch_depth)
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
N_STREAMS = int(os.environ.get("N_STREAMS", "4"))
DEPTHS = [int(x) for x in os.environ.get("DEPTHS", "1,2,3,4,6,8,10,12").split(",")]
RUN_DIR = os.environ.get("RUN_DIR", "runs/patch_depth")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    blocks = model.model.layers
    nL = len(blocks)
    catset = list(CATWORDS["city"])

    rows = [json.loads(l) for l in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl"))]
    games = collections.defaultdict(list)
    for r in rows:
        games[r["rollout"]].append(r)
    for roll in games:
        games[roll].sort(key=lambda r: r["turn"])
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    # harvest model-generated family variants across all games (4-prefix families)
    fams = collections.defaultdict(list)
    for roll, ts in games.items():
        for t in ts:
            w = t["A"]
            if len(w) > 3 and w[:4] not in (w2[:4] for w2 in fams[w[:4]]):
                pass
            if len(w) > 3:
                if w not in fams[w[:4]]:
                    fams[w[:4]].append(w)
    big_fams = sorted([f for f in fams if len(fams[f]) >= max(DEPTHS)],
                      key=lambda f: -len(fams[f]))
    print(f"[depth] families with >= {max(DEPTHS)} variants: {big_fams[:8]}", flush=True)
    fillers = [starts[i][0] for i in range(26, 26 + max(DEPTHS) + 4)]

    streams = sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS]

    def fid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]

    def body_of(histA, used):
        return (G.OPEN_PROMPT + " "
                + " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                           for k, (o, s) in enumerate(histA))
                + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".")

    @torch.no_grad()
    def fwd(prompt):
        enc = tok(prompt, return_tensors="pt").to(dev)
        out = model(enc.input_ids, output_hidden_states=True)
        return enc, out.logits[0, -1].float(), [h[0] for h in out.hidden_states]

    def delta(logits, stem_ids, cat_ids):
        return float(torch.logsumexp(logits[stem_ids], 0) - torch.logsumexp(logits[cat_ids], 0))

    results = []
    for si, (roll, ts) in enumerate(streams):
        Bseq = [t["B"] for t in ts]
        sa, sb = starts[roll]
        fam = big_fams[si % len(big_fams)]
        fam_words = fams[fam][:max(DEPTHS)]
        for d in DEPTHS:
            if d > len(Bseq) - 1 or d > len(fam_words):
                continue
            pair = {}
            for cond, words in (("loop", fam_words[:d]), ("ctrl", fillers[:d])):
                histA = [(sb, sa)]
                used = {sa, sb}
                for k in range(d):
                    histA.append((Bseq[k], words[k]))
                    used |= {Bseq[k], words[k]}
                pair[cond] = (LA._render(tok, body_of(histA, used)) + "\nMy word:", used)
            import torch
            stem_ids = torch.tensor(sorted({fid(w) for w in fam_words + [fam]}))
            used_all = pair["loop"][1] | pair["ctrl"][1]
            cat_ids = torch.tensor(sorted({fid(w) for w in catset if w not in used_all}
                                          - set(stem_ids.tolist())))
            enc_l, log_l, _ = fwd(pair["loop"][0])
            enc_c, log_c, hs_c = fwd(pair["ctrl"][0])
            d_loop = delta(log_l, stem_ids, cat_ids)
            d_ctrl = delta(log_c, stem_ids, cat_ids)
            denom = (d_loop - d_ctrl) or 1e-9
            rec = []
            for L in range(nL):
                repl = hs_c[L + 1][-1]

                def hook(_m, _i, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h[0, -1] = repl.to(h.dtype)
                    return out
                hd = blocks[L].register_forward_hook(hook)
                with torch.no_grad():
                    lg = model(enc_l.input_ids).logits[0, -1].float()
                hd.remove()
                rec.append((d_loop - delta(lg, stem_ids, cat_ids)) / denom)
            results.append({"stream": roll, "family": fam, "depth": d,
                            "d_loop": d_loop, "d_ctrl": d_ctrl, "recovery": rec})
            print(f"[depth] s{roll} fam {fam} d={d}: d_loop {d_loop:.2f} d_ctrl {d_ctrl:.2f} "
                  f"best {max(rec):.2f}@L{int(np.argmax(rec))}", flush=True)
            json.dump({"per_pair": results}, open(os.path.join(RUN_DIR, "patch_depth.json"), "w"))

    # depth x layer means
    byd = collections.defaultdict(list)
    for r in results:
        byd[r["depth"]].append(r["recovery"])
    summary = {"depths": sorted(byd),
               "mean_recovery": {str(d): np.mean(byd[d], 0).tolist() for d in byd},
               "baselines": {str(d): {"d_loop": float(np.mean([r["d_loop"] for r in results if r["depth"] == d])),
                                      "d_ctrl": float(np.mean([r["d_ctrl"] for r in results if r["depth"] == d]))}
                             for d in byd}}
    json.dump({"per_pair": results, "summary": summary},
              open(os.path.join(RUN_DIR, "patch_depth.json"), "w"), indent=1)
    print("[depth] done", flush=True)


if __name__ == "__main__":
    main()
