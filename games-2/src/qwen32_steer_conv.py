"""CAUSAL test of the convergence direction: steer live self-play games along +/- d.

If the seed-centered PC1 ("convergence direction") is functionally the consensus axis,
adding +d to the residual stream during generation should make games meet FASTER, and
-d should slow/prevent meeting. If it is epiphenomenal (or just confidence), steering
should not move turns-to-meet in the predicted direction.

Same self-play loop as qwen32_pca.py (matched word2vec start pairs, no-repeat enforced,
temp), but a forward hook at layer LAYER adds  alpha * ||h_last|| * d_unit  to the last
position of every forward pass (prefill answer position + each generated token), for
BOTH players. d = normalized mean of dir1/dir2 at LAYER (they are cos~1 anyway).

Env: MODEL(QwenInst32) DIR_NPZ LAYER(32) ALPHAS(-1,-0.5,0,0.5,1) START_FILE SAFETY(24)
     TEMP(0.7) RUN_DIR DEVICE
Out: <RUN_DIR>/qwen32_steer_conv.json  + qwen32_steer_conv_transcript.jsonl
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
DIR_NPZ = os.environ.get("DIR_NPZ", "runs/game1_qwen32_pca_w2v/qwen32_convergence_dir.npz")
LAYER = int(os.environ.get("LAYER", "32"))
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "-1,-0.5,0,0.5,1").split(",")]
START_FILE = os.environ.get("START_FILE", "runs/game1_qwen32_pca_w2v/start_words.txt")
SAFETY = int(os.environ.get("SAFETY", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/qwen32_steer_conv")


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
    blocks = model.model.layers

    z = np.load(DIR_NPZ, allow_pickle=True)
    d = z["dir1"][LAYER] + z["dir2"][LAYER]
    d = d / (np.linalg.norm(d) + 1e-9)
    d_t = torch.tensor(d, device=dev)

    state = {"alpha": 0.0}

    def hook(_m, _i, out):
        if state["alpha"] == 0.0:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        n = h[:, -1, :].float().norm(dim=-1, keepdim=True)
        h[:, -1, :] = h[:, -1, :] + (state["alpha"] * n * d_t).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    blocks[LAYER].register_forward_hook(hook)

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
    tf = open(os.path.join(RUN_DIR, "qwen32_steer_conv_transcript.jsonl"), "w")
    summary = {"model": MODEL, "layer": LAYER, "temp": TEMP, "safety": SAFETY,
               "n": len(starts), "conditions": {}}
    for alpha in ALPHAS:
        state["alpha"] = alpha
        met, turns = [], []
        for roll, (sa, sb) in enumerate(starts):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            agreed_at = None
            for t in range(1, SAFETY):
                pA = G.build_prompt(tok, histA, used)
                pB = G.build_prompt(tok, histB, used)
                wA = gen_word(pA, 5000 * roll + t, used)
                wB = gen_word(pB, 90000 + 5000 * roll + t, used)
                tf.write(json.dumps({"alpha": alpha, "rollout": roll, "turn": t,
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
            print(f"[steer] alpha={alpha} roll={roll} "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'}", flush=True)
        n = len(met)
        summary["conditions"][str(alpha)] = {
            "n": n, "met_frac": float(np.mean(met)),
            "met_se": float(np.std(met) / np.sqrt(n)),
            "turns_mean": float(np.mean(turns)) if turns else None,
            "turns_se": float(np.std(turns) / np.sqrt(len(turns))) if turns else None,
            "n_met": int(np.sum(met))}
        json.dump(summary, open(os.path.join(RUN_DIR, "qwen32_steer_conv.json"), "w"), indent=1)
        print(f"[steer] === alpha={alpha}: met {np.mean(met):.2f}, "
              f"turns {np.mean(turns) if turns else float('nan'):.1f}", flush=True)
    tf.close()


if __name__ == "__main__":
    main()
