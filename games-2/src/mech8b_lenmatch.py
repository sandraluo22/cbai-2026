"""MECH8b (2026-08-08): length-matched rerun of mech8 — X and Y prompts are
padded (extra neutral words in the used-list) until their token counts are
IDENTICAL, so the final-position patch lands at the same absolute position and
the positional-mismatch confound is removed.

Two game contexts identical except B's column: category X words vs category Y
words (A column = fixed neutral fillers, 6 rounds). Elicit A's PREDICTION of
B's next word ("The other player's next word will be:"). Patch the residual
stream at the FINAL position, one layer at a time, from the X-run into the
Y-run. The Y prompt's surface still says Y everywhere; only the patched
activation carries X.

  flip rate(l) = frac of K samples that are novel X-category words, patched at l
  baselines: unpatched X-context (ceiling), unpatched Y-context (floor),
             patch from X' (different exemplars, same category X) into Y
             (same-category control: should behave like X if the code is
             rule-level, not exemplar-level)

If flip happens in some layer band -> compact transportable "B says X" summary
exists at the answer position. If not -> the partner model is recomputed from
the prompt per forward pass, not cached.

Env: MODEL(QwenInst32) K(16) TEMP(0.7) N_PAIRS(4) RUN_DIR(runs/mech8_rulepatch)
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G
from probe_category_stuck import CATLISTS

MODEL = os.environ.get("MODEL", "QwenInst32")
K = int(os.environ.get("K", "16"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/mech8b_lenmatch")

FILLER = ["window", "carpet", "stapler", "napkin", "candle", "basket"]
PAIRS = [("city", "fruit"), ("animal", "beverage"), ("instrument", "vegetable"),
         ("sport", "flower")]
LAYERS = list(range(0, 64, 4)) + [62, 63]


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nL = len(model.model.layers)

    PADS = ["lamp", "brick", "spoon", "crate", "hinge", "plume", "gourd", "flint",
            "sprig", "tuft", "knoll", "wisp"]

    def body_of(bwords, npad=0, sa="ledger", sb="marble"):
        hist = [(sb, sa)] + list(zip(bwords, FILLER))
        used = {sa, sb} | set(bwords) | set(FILLER) | set(PADS[:npad])
        s = G.OPEN_PROMPT + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return (s + " Words already used (do not repeat): " + ", ".join(sorted(used))
                + ". What word do you think the other player will say next?"), used

    cache = {}
    mode = {"m": "off", "layer": -1, "vec": None}
    def hook(l):
        def h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            if mode["m"] == "cap":
                cache[l] = hs[0, -1].detach().clone()
            elif mode["m"] == "patch" and l == mode["layer"]:
                hs[:, -1] = mode["vec"]
            return None
        return h
    for l in range(nL):
        model.model.layers[l].register_forward_hook(hook(l))

    @torch.no_grad()
    def run(body, capture=False, patch_layer=None, patch_vec=None, sample=True):
        prompt = LA._render(tok, body) + "\nThe other player's next word: **"
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        if capture:
            mode["m"] = "cap"
            model(ids)
            mode["m"] = "off"
            return {l: cache[l].clone() for l in range(nL)}
        if patch_layer is not None:
            mode.update({"m": "patch", "layer": patch_layer, "vec": patch_vec})
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        mode["m"] = "off"
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    def frac_in(ws, cat, used):
        cs = set(CATLISTS[cat])
        return float(np.mean([1 if (w and w not in used and w in cs) else 0 for w in ws]))

    results = []
    for cx, cy in PAIRS:
        X1 = CATLISTS[cx][:6]
        X2 = CATLISTS[cx][6:12]
        Y = CATLISTS[cy][:6]

        def tlen(body):
            return len(tok(LA._render(tok, body) + "\nThe other player's next word: **")["input_ids"])

        def match(words_list):
            # pad each context until all token lengths equal the max
            bodies = [body_of(w) for w in words_list]
            pads = [0] * len(bodies)
            for _ in range(60):
                ls = [tlen(b[0]) for b in bodies]
                if len(set(ls)) == 1:
                    break
                mx = max(ls)
                for i in range(len(bodies)):
                    if ls[i] < mx:
                        pads[i] += 1
                        bodies[i] = body_of(words_list[i], npad=pads[i])
            ls = [tlen(b[0]) for b in bodies]
            print(f"[m8b] lengths after matching: {ls}", flush=True)
            return bodies

        (bX, uX), (bX2, _), (bY, uY) = match([X1, X2, Y])
        wsX = run(bX)
        wsY = run(bY)
        base = {"X_pred_X": frac_in(wsX, cx, uX), "X_pred_Y": frac_in(wsX, cy, uX),
                "Y_pred_Y": frac_in(wsY, cy, uY), "Y_pred_X": frac_in(wsY, cx, uY)}
        capX = run(bX, capture=True)
        capX2 = run(bX2, capture=True)
        per_layer = {}
        for l in LAYERS:
            ws = run(bY, patch_layer=l, patch_vec=capX[l])
            ws2 = run(bY, patch_layer=l, patch_vec=capX2[l])
            per_layer[str(l)] = {
                "flip_X": frac_in(ws, cx, uY), "keep_Y": frac_in(ws, cy, uY),
                "ctrl_flip_X": frac_in(ws2, cx, uY)}
        results.append({"pair": [cx, cy], "base": base, "per_layer": per_layer})
        json.dump({"pairs": results}, open(os.path.join(RUN_DIR, "mech8.json"), "w"),
                  indent=1)
        best = max(per_layer.items(), key=lambda kv: kv[1]["flip_X"])
        print(f"[m8] {cx}->{cy}: base Xpred {base['X_pred_X']:.2f} Ypred {base['Y_pred_Y']:.2f} "
              f"| best flip L{best[0]}: {best[1]['flip_X']:.2f} (ctrl {best[1]['ctrl_flip_X']:.2f})",
              flush=True)

    agg = {}
    for l in LAYERS:
        agg[str(l)] = {k: float(np.mean([r["per_layer"][str(l)][k] for r in results]))
                       for k in ("flip_X", "keep_Y", "ctrl_flip_X")}
        a = agg[str(l)]
        print(f"[m8] === L{l}: flip {a['flip_X']:.2f} keep {a['keep_Y']:.2f} "
              f"ctrl {a['ctrl_flip_X']:.2f}", flush=True)
    json.dump({"pairs": results, "agg": agg},
              open(os.path.join(RUN_DIR, "mech8.json"), "w"), indent=1)
    print("[m8] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
