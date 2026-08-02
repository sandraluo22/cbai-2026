"""Are the NON-induction heads among the top partner-patch restoration heads BRIDGING
heads? Give the model cue PAIRS in game format ("the other player said red, you said
fruit" -> My word:) whose natural convergent answer is a third word (apple). Keep items
the model actually gets (bridge first-token in its top-K next tokens). Then zero-ablate
each head (o_proj input columns, ANSWER POSITION ONLY) and measure:

    d_bridge = logit_clean(bridge) - logit_ablated(bridge)   (drop in the bridge word)
    d_cue    = same for the two cue words, averaged          (drop in copy candidates)

Groups: top-32 restoration heads split by induction percentile (>=0.95 = induction-ish,
else non-induction, from IND_JSON), plus 32 random control heads. A bridging head shows
d_bridge >> d_cue; a copy head the reverse.

Env: MODEL(QwenInst32) PATCH_JSON IND_JSON TOPK_KEEP(10) SEED(0) OUT_JSON DEVICE
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA

MODEL = os.environ.get("MODEL", "QwenInst32")
PATCH_JSON = os.environ.get("PATCH_JSON",
                            "runs/game-1/qwen32/qwen32_partner_patch/qwen32_partner_patch.json")
IND_JSON = os.environ.get("IND_JSON", "runs/qwen32_induction_overlap.json")
TOPK_KEEP = int(os.environ.get("TOPK_KEEP", "10"))
SEED = int(os.environ.get("SEED", "0"))
OUT_JSON = os.environ.get("OUT_JSON", "runs/qwen32_bridge_heads.json")

OPEN_PROMPT = ("You are playing a word game with another player. Each round, you both say one word at the "
               "same time, then the two words are revealed. You win the round only if you both said the same "
               "word. You may not say any word that has already been said by either player.")

# (partner_said, you_said, bridge)
ITEMS = [
    ("red", "fruit", "apple"), ("yellow", "fruit", "banana"), ("orange", "vegetable", "carrot"),
    ("king", "woman", "queen"), ("black", "white", "gray"), ("hot", "cold", "warm"),
    ("day", "night", "dawn"), ("sun", "rain", "rainbow"), ("thunder", "rain", "storm"),
    ("sand", "water", "beach"), ("bee", "flower", "honey"), ("cow", "milk", "cheese"),
    ("dog", "cat", "pet"), ("night", "star", "moon"), ("fire", "water", "steam"),
    ("tree", "fruit", "apple"), ("ocean", "sky", "blue"), ("winter", "rain", "snow"),
    ("bread", "cheese", "sandwich"), ("sand", "cloud", "storm"), ("lion", "stripes", "tiger"),
    ("piano", "violin", "music"), ("sock", "boot", "shoe"), ("moon", "sun", "eclipse"),
]


def prompt_for(tok, said, you):
    body = OPEN_PROMPT + f" Round 1: the other player said {said}, you said {you}."
    return LA._render(tok, body) + "\nMy word:"


def first_id(tok, w):
    return tok(" " + w, add_special_tokens=False)["input_ids"][0]


def main():
    import torch
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model, tok = LA.load(MODEL, dev)
    nH = model.config.num_attention_heads
    hd = model.model.layers[0].self_attn.o_proj.in_features // nH

    R = np.array(json.load(open(PATCH_JSON))["restoration"])
    nL = R.shape[0]
    order = np.argsort(R.flatten())[::-1][:32]
    top = [(int(i // nH), int(i % nH)) for i in order]
    ind = {(t["layer"], t["head"]): t["induction_pctile"]
           for t in json.load(open(IND_JSON))["top_restoration_heads"]}
    grp_ind = [h for h in top if ind.get(h, 0) >= 0.95]
    grp_non = [h for h in top if ind.get(h, 0) < 0.95]
    rng = np.random.default_rng(SEED)
    pool = [(l, h) for l in range(nL) for h in range(nH) if (l, h) not in set(top)]
    grp_rand = [pool[i] for i in rng.choice(len(pool), 32, replace=False)]
    print(f"[bridge] groups: induction-ish {len(grp_ind)}, non-induction {len(grp_non)}, rand 32",
          flush=True)

    state = {"head": None}                     # (layer, head) to zero at answer position
    def make_pre(layer):
        def pre(_m, args):
            if state["head"] is None or state["head"][0] != layer:
                return None
            x = args[0].clone()
            h = state["head"][1]
            x[:, -1, h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return pre
    for li, blk in enumerate(model.model.layers):
        blk.self_attn.o_proj.register_forward_pre_hook(make_pre(li))

    @torch.no_grad()
    def last_logits(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        return model(ids).logits[0, -1].float()

    # ---- screen items: bridge first-token must be in the model's top-K next tokens ----
    kept = []
    for said, you, bridge in ITEMS:
        state["head"] = None
        lg = last_logits(prompt_for(tok, said, you))
        bid = first_id(tok, bridge)
        rank = int((lg > lg[bid]).sum())
        tops = [tok.decode([i]).strip() for i in torch.topk(lg, 5).indices.tolist()]
        keep = rank < TOPK_KEEP
        print(f"[bridge] {said}+{you}->{bridge}: rank {rank} top5={tops} "
              f"{'KEEP' if keep else 'drop'}", flush=True)
        if keep:
            kept.append({"said": said, "you": you, "bridge": bridge, "rank": rank,
                         "ids": (first_id(tok, said), first_id(tok, you), bid),
                         "prompt": prompt_for(tok, said, you)})
    print(f"[bridge] kept {len(kept)}/{len(ITEMS)} items", flush=True)

    # ---- per-head ablation ----
    results = []
    all_heads = [("ind", h) for h in grp_ind] + [("non", h) for h in grp_non] + \
                [("rand", h) for h in grp_rand]
    base = {}
    for it in kept:
        state["head"] = None
        base[it["bridge"] + it["said"]] = last_logits(it["prompt"])
    for gi, (grp, head) in enumerate(all_heads):
        d_bridge, d_cue = [], []
        for it in kept:
            b = base[it["bridge"] + it["said"]]
            state["head"] = head
            lg = last_logits(it["prompt"])
            sid, yid, bid = it["ids"]
            d_bridge.append(float(b[bid] - lg[bid]))
            d_cue.append(float(((b[sid] - lg[sid]) + (b[yid] - lg[yid])) / 2))
        results.append({"group": grp, "layer": head[0], "head": head[1],
                        "d_bridge": float(np.mean(d_bridge)), "d_cue": float(np.mean(d_cue))})
        if (gi + 1) % 16 == 0:
            print(f"[bridge] {gi + 1}/{len(all_heads)} heads done", flush=True)

    for grp in ("ind", "non", "rand"):
        rs = [r for r in results if r["group"] == grp]
        print(f"[bridge] group {grp:>4}: mean d_bridge {np.mean([r['d_bridge'] for r in rs]):+.4f}  "
              f"mean d_cue {np.mean([r['d_cue'] for r in rs]):+.4f}  (n={len(rs)})", flush=True)
    top_bridge = sorted(results, key=lambda r: -r["d_bridge"])[:10]
    for r in top_bridge:
        print(f"[bridge]   L{r['layer']} H{r['head']} ({r['group']}) "
              f"d_bridge {r['d_bridge']:+.4f} d_cue {r['d_cue']:+.4f}", flush=True)
    json.dump({"model": MODEL, "kept_items": [{k: it[k] for k in ('said', 'you', 'bridge', 'rank')}
                                              for it in kept],
               "results": results}, open(OUT_JSON, "w"), indent=1)
    print("[bridge] wrote", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
