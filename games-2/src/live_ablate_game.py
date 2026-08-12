"""LIVE GAME UNDER ABLATION (2026-08-07): with the reader circuit cut, does the
model still TRY to converge — and does play get better (loop removed) or worse
(series machinery gone)?

Standard asymmetric game (A free, B city-restricted), n=N games, cap CAP.
Ablation applies to A's generations ONLY (toggled off for B). Conditions:
none / top32_rest / top64_rest. Measures: met, turns, family-run onset, t8 MC
probe (category / self-family mass), fraction valid words.

Env: MODEL(QwenInst32) PATCH_JSON START_FILE N(12) CAP(40) K(64) TEMP(0.7)
     PROBE(8) RUN_DIR(runs/live_ablate)
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G
from game1_strict import CATWORDS

MODEL = os.environ.get("MODEL", "QwenInst32")
PATCH_JSON = os.environ.get("PATCH_JSON", "runs/mech_inputs/qwen32_partner_patch.json")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N = int(os.environ.get("N", "12"))
CAP = int(os.environ.get("CAP", "40"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
PROBE = int(os.environ.get("PROBE", "8"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/live_ablate")

RESTR = (" IMPORTANT: every single word you say must be the name of a city. Only ever "
         "say cities, nothing else.")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nH = model.config.num_attention_heads
    hd = model.model.layers[0].self_attn.o_proj.in_features // nH
    catset = set(CATWORDS["city"])

    R = np.array(json.load(open(PATCH_JSON))["restoration"])
    order = np.argsort(R.flatten())[::-1]
    def topk(k):
        d = {}
        for i in order[:k]:
            d.setdefault(int(i // nH), []).append(int(i % nH))
        return d

    state = {"heads": None, "on": False}
    def make_pre(layer):
        def pre(_m, args):
            if not state["on"] or not state["heads"] or layer not in state["heads"]:
                return None
            x = args[0].clone()
            for h in state["heads"][layer]:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return pre
    for li, blk in enumerate(model.model.layers):
        blk.self_attn.o_proj.register_forward_pre_hook(make_pre(li))

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    @torch.no_grad()
    def gen_word(body, seed, forbidden, ablate):
        state["on"] = ablate
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
                break
        state["on"] = False
        return w

    @torch.no_grad()
    def propose_k(body, ablate):
        state["on"] = ablate
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        state["on"] = False
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
    tf = open(os.path.join(RUN_DIR, "live_ablate_transcript.jsonl"), "w")
    for cond, k in (("none", 0), ("top32", 32), ("top64", 64)):
        state["heads"] = topk(k) if k else None
        met, onsets, probes, ok = [], [], [], []
        for roll, (sa, sb) in enumerate(starts):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            own = [sa]
            agreed_at, onset = None, None
            for t in range(1, CAP):
                if t == PROBE:
                    props = propose_k(body_of(histA, used), bool(k))
                    fams = {w[:4] for w in own if len(w) > 3}
                    probes.append({
                        "cat": float(np.mean([1 if (w and w not in used and w in catset)
                                              else 0 for w in props])),
                        "selffam": float(np.mean([1 if (w and w not in used and len(w) > 3
                                                  and w[:4] in fams) else 0 for w in props]))})
                wA = gen_word(body_of(histA, used), 5000 * roll + t, used, bool(k))
                wB = gen_word(body_of(histB, used, RESTR), 90000 + 5000 * roll + t, used, False)
                ok.append(bool(wA and wA.isalpha() and len(wA) > 2))
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
            if agreed_at:
                pass
            print(f"[lag] {cond} roll={roll}: "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'} onset={onset}",
                  flush=True)
        turns = [1 for m in met if m]
        summary["conditions"][cond] = {
            "met_frac": float(np.mean(met)), "onset_frac": float(np.mean(onsets)),
            "probe_cat": float(np.mean([p["cat"] for p in probes])) if probes else None,
            "probe_selffam": float(np.mean([p["selffam"] for p in probes])) if probes else None,
            "word_ok_frac": float(np.mean(ok))}
        s = summary["conditions"][cond]
        json.dump(summary, open(os.path.join(RUN_DIR, "live_ablate.json"), "w"), indent=1)
        print(f"[lag] === {cond}: met {s['met_frac']:.2f} onset {s['onset_frac']:.2f} "
              f"cat {s['probe_cat']} selffam {s['probe_selffam']} ok {s['word_ok_frac']:.2f}",
              flush=True)
    tf.close()
    print("[lag] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
