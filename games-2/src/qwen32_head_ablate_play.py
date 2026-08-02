"""H2b: ablate the top partner-patch restoration heads during LIVE self-play — does
convergence collapse? If the distributed 32-head circuit is what lets the players read
each other, zeroing it should tank met_frac / slow meeting; random heads should not.

Conditions: none / top<K> (by restoration in PATCH_JSON) / rand<K> (excluding topK,
fixed seed). Ablation = zero the heads' o_proj input columns at every position, every
forward pass. Same loop/seeds/starts as qwen32_steer_conv.py.

Env: MODEL(QwenInst32) PATCH_JSON K(32) START_FILE SAFETY(24) TEMP(0.7) SEED(0) RUN_DIR
Out: <RUN_DIR>/qwen32_head_ablate.json + qwen32_head_ablate_transcript.jsonl
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
PATCH_JSON = os.environ.get("PATCH_JSON",
                            "runs/game-1/qwen32/qwen32_partner_patch/qwen32_partner_patch.json")
K = int(os.environ.get("K", "32"))
START_FILE = os.environ.get("START_FILE", "runs/game1_qwen32_pca_w2v/start_words.txt")
SAFETY = int(os.environ.get("SAFETY", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))
SEED = int(os.environ.get("SEED", "0"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/qwen32_head_ablate")


def load_starts():
    pairs = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            pairs.append((p[-2], p[-1]))
    return pairs


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nH = model.config.num_attention_heads
    hd = model.model.layers[0].self_attn.o_proj.in_features // nH

    R = np.array(json.load(open(PATCH_JSON))["restoration"])
    nL = R.shape[0]
    order = np.argsort(R.flatten())[::-1]
    top = [(int(i // nH), int(i % nH)) for i in order[:K]]
    rng = np.random.default_rng(SEED)
    pool = [(l, h) for l in range(nL) for h in range(nH) if (l, h) not in set(top)]
    rand = [pool[i] for i in rng.choice(len(pool), K, replace=False)]

    state = {"heads": None}          # dict layer -> list of head idx
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

    def to_layerdict(heads):
        d = {}
        for l, h in heads:
            d.setdefault(l, []).append(h)
        return d

    @torch.no_grad()
    def gen_word(prompt, seed, forbidden):
        enc = tok(prompt, return_tensors="pt").to(dev)
        w = ""
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in forbidden:
                return w
        return w

    starts = load_starts()
    tf = open(os.path.join(RUN_DIR, "qwen32_head_ablate_transcript.jsonl"), "w")
    summary = {"model": MODEL, "k": K, "temp": TEMP, "safety": SAFETY, "n": len(starts),
               "top_heads": top, "rand_heads": rand, "conditions": {}}
    for cond, heads in (("none", None), (f"top{K}", to_layerdict(top)),
                        (f"rand{K}", to_layerdict(rand))):
        state["heads"] = heads
        met, turns = [], []
        for roll, (sa, sb) in enumerate(starts):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            agreed_at = None
            for t in range(1, SAFETY):
                wA = gen_word(G.build_prompt(tok, histA, used), 5000 * roll + t, used)
                wB = gen_word(G.build_prompt(tok, histB, used), 90000 + 5000 * roll + t, used)
                tf.write(json.dumps({"cond": cond, "rollout": roll, "turn": t,
                                     "A": wA, "B": wB, "agreed": wA == wB}) + "\n")
                tf.flush()
                if wA == wB and wA:
                    agreed_at = t
                    break
                used |= {wA, wB}
                histA.append((wB, wA)); histB.append((wA, wB))
            met.append(agreed_at is not None)
            if agreed_at is not None:
                turns.append(agreed_at)
            print(f"[ablate] {cond} roll={roll} "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'}", flush=True)
        n = len(met)
        summary["conditions"][cond] = {
            "n": n, "met_frac": float(np.mean(met)),
            "met_se": float(np.std(met) / np.sqrt(n)),
            "turns_mean": float(np.mean(turns)) if turns else None,
            "turns_se": float(np.std(turns) / np.sqrt(len(turns))) if turns else None,
            "n_met": int(np.sum(met))}
        json.dump(summary, open(os.path.join(RUN_DIR, "qwen32_head_ablate.json"), "w"), indent=1)
        print(f"[ablate] === {cond}: met {np.mean(met):.2f}, "
              f"turns {np.mean(turns) if turns else float('nan'):.1f}", flush=True)
    tf.close()


if __name__ == "__main__":
    main()
