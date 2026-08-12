"""WHITTLE THE READER SET (2026-08-07): find a minimal head set whose ablation
kills capture.

Stage 1 — prefix sweep: ablate top-n of the restoration ranking for
n in {4,8,12,16,24,32,48,64}; measure fam mass on d3_self and d8_list
(6 streams, K=48). Find the knee for each regime.

Stage 2 — greedy leave-one-out pruning on the minimal d3-killing prefix:
repeatedly evaluate removing each remaining head from the ablation set
(3 streams, K=32, d3_self); drop the head whose removal least increases fam
mass while pooled fam stays <= THRESH(0.05). Stop when no head can be dropped.
Reports the minimal set with (layer, head) ids.

Env: MODEL(QwenInst32) PATCH_JSON SRC_DIR START_FILE TEMP(0.7) THRESH(0.05)
     RUN_DIR(runs/whittle)
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
PATCH_JSON = os.environ.get("PATCH_JSON", "runs/mech_inputs/qwen32_partner_patch.json")
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
TEMP = float(os.environ.get("TEMP", "0.7"))
THRESH = float(os.environ.get("THRESH", "0.05"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/whittle")

PLANT8 = ["planted", "planting", "plantings", "replant",
          "replanted", "planter", "planters", "plantation"]
FILLER = ["window", "carpet", "stapler", "napkin", "candle", "basket", "ribbon", "saddle"]
FAMP = "plan"
PREFIXES = [4, 8, 12, 16, 24, 32, 48, 64]


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nH = model.config.num_attention_heads
    hd = model.model.layers[0].self_attn.o_proj.in_features // nH

    R = np.array(json.load(open(PATCH_JSON))["restoration"])
    order = np.argsort(R.flatten())[::-1]
    ranked = [(int(i // nH), int(i % nH)) for i in order[:64]]

    state = {"heads": None}
    def make_pre(layer):
        def pre(_m, args):
            if not state["heads"] or layer not in state["heads"]:
                return None
            x = args[0].clone()
            for h in state["heads"][layer]:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return pre
    for li, blk in enumerate(model.model.layers):
        blk.self_attn.o_proj.register_forward_pre_hook(make_pre(li))

    def to_ld(heads):
        d = {}
        for l, h in heads:
            d.setdefault(l, []).append(h)
        return d

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    @torch.no_grad()
    def fam_mass(body, used, K):
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        torch.manual_seed(0)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        ws = [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
              for i in range(K)]
        return float(np.mean([1 if (w and w not in used and len(w) > 3 and w[:4] == FAMP)
                              else 0 for w in ws]))

    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl")):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    streams = [sorted(ts, key=lambda r: r["turn"]) for _, ts in
               sorted(games.items(), key=lambda kv: -len(kv[1]))[:6]]
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    def build(si, cell):
        ts = streams[si]
        roll = ts[0]["rollout"]
        Bseq = [t["B"] for t in ts]
        if len(Bseq) < 8:
            Bseq = Bseq + Bseq
        sa, sb = starts[roll]
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
        return body_of(hist, used, extra), used

    def eval_set(heads, cell, n_streams, K):
        state["heads"] = to_ld(heads) if heads else None
        vals = []
        for si in range(n_streams):
            body, used = build(si, cell)
            vals.append(fam_mass(body, used, K))
        state["heads"] = None
        return float(np.mean(vals))

    # stage 1: prefix sweep
    sweep = {}
    for n in [0] + PREFIXES:
        f3 = eval_set(ranked[:n], "d3_self", 6, 48)
        f8 = eval_set(ranked[:n], "d8_list", 6, 48)
        sweep[str(n)] = {"d3_self": f3, "d8_list": f8}
        print(f"[wh] === top-{n}: d3 {f3:.3f} d8_list {f8:.3f}", flush=True)
        json.dump({"sweep": sweep}, open(os.path.join(RUN_DIR, "whittle.json"), "w"))
    kill_n = next((n for n in PREFIXES if sweep[str(n)]["d3_self"] <= THRESH), 64)
    print(f"[wh] minimal d3-killing prefix: top-{kill_n}", flush=True)

    # stage 2: greedy LOO pruning
    S = list(ranked[:kill_n])
    log = []
    improved = True
    while improved and len(S) > 1:
        improved = False
        best_h, best_f = None, None
        for h in list(S):
            f = eval_set([x for x in S if x != h], "d3_self", 3, 32)
            if best_f is None or f < best_f:
                best_h, best_f = h, f
        if best_f is not None and best_f <= THRESH:
            S.remove(best_h)
            log.append({"removed": list(best_h), "fam_after": best_f, "size": len(S)})
            print(f"[wh] pruned L{best_h[0]}H{best_h[1]} -> |S|={len(S)} fam {best_f:.3f}",
                  flush=True)
            improved = True
            json.dump({"sweep": sweep, "prune_log": log,
                       "minimal_set": [list(x) for x in S]},
                      open(os.path.join(RUN_DIR, "whittle.json"), "w"), indent=1)
    final3 = eval_set(S, "d3_self", 6, 48)
    final8 = eval_set(S, "d8_list", 6, 48)
    print(f"[wh] === MINIMAL SET n={len(S)}: {[f'L{l}H{h}' for l, h in S]} "
          f"d3 {final3:.3f} d8_list {final8:.3f}", flush=True)
    json.dump({"sweep": sweep, "prune_log": log,
               "minimal_set": [list(x) for x in S],
               "final": {"d3_self": final3, "d8_list": final8}},
              open(os.path.join(RUN_DIR, "whittle.json"), "w"), indent=1)
    print("[wh] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
