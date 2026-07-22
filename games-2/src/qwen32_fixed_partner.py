"""FIXED-PARTNER control for the swing-by test: does the convergence-trajectory swing-by
need CO-ADAPTATION, or is it driven by the model's own (fixed) prior geometry?

Player A generates as usual, but its partner is NON-ADAPTIVE: a 'ghost' that replays a
real word sequence from a DIFFERENT co-adaptive game (same word distribution, but not
responding to THIS A). Same model / no-repeat / temp / word2vec starts as the
co-adaptive run. We capture A's residual stream every turn at every layer, then run the
same turn-manifold/swing-by analysis and compare.

  co-adaptive swing ~= fixed-partner swing  -> swing is a FIXED-prior phenomenon (a la
                                               Park et al.); model does NOT distinguish.
  swing collapses/changes without co-adapt   -> co-adaptation drives it; model DOES.

Env: MODEL(QwenInst32) SAFETY(16) TEMP(0.7) START_WORDS_FILE COADAPT_TRANSCRIPT RUN_DIR
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
SAFETY = int(os.environ.get("SAFETY", "16"))
TEMP = float(os.environ.get("TEMP", "0.7"))
START_FILE = os.environ.get("START_WORDS_FILE", "runs/game1_qwen32_pca_w2v/start_words.txt")
COADAPT = os.environ.get("COADAPT_TRANSCRIPT", "runs/game1_qwen32_pca_w2v/qwen32_pca_transcript.jsonl")
RUN_DIR = os.environ.get("RUN_DIR", "runs/game1_qwen32_fixed")


def load_starts():
    pairs = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            pairs.append((p[-2], p[-1]))
    return pairs


def load_ghost_seqs():
    """Per rollout, the P2 word sequence from the co-adaptive run (start + each turn's P2 pick)."""
    by = {}
    for line in open(COADAPT):
        r = json.loads(line); g = r["rollout"]
        P2 = [k for k in r["picks"] if k.endswith("_2")][0]
        by.setdefault(g, {"start": r["start"][1], "seq": []})
        by[g]["seq"].append(r["picks"][P2])
    return {g: [v["start"]] + v["seq"] for g, v in by.items()}


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nL = model.config.num_hidden_layers
    starts = load_starts(); ghosts = load_ghost_seqs()
    gkeys = sorted(ghosts)

    @torch.no_grad()
    def hidden_all(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return np.stack([h[0, -1].float().cpu().numpy() for h in hs])

    @torch.no_grad()
    def gen_word(prompt, seed, forbidden):
        enc = tok(prompt, return_tensors="pt").to(dev); w = ""
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in forbidden:
                return w
        return w

    acts = []; meta = []
    tf = open(os.path.join(RUN_DIR, "qwen32_fixed_transcript.jsonl"), "w")
    for roll, (sa, _sb) in enumerate(starts):
        ghost = ghosts[gkeys[(roll + 1) % len(gkeys)]]     # a DIFFERENT game's partner sequence
        gb = ghost[0]                                      # ghost's start word
        histA = [(gb, sa)]; used = {sa, gb}
        for t in range(1, SAFETY):
            gword = ghost[t] if t < len(ghost) else ghost[-1]   # non-adaptive partner word this turn
            pA = G.build_prompt(tok, histA, used)
            v = hidden_all(pA)
            wA = gen_word(pA, 5000 * roll + t, used)
            acts.append(v); meta.append((roll, t, wA))
            agreed = (wA == gword and wA)
            tf.write(json.dumps({"rollout": roll, "turn": t, "A": wA, "ghost": gword, "agreed": bool(agreed)}) + "\n")
            if agreed:
                break
            histA.append((gword, wA)); used |= {wA, gword}
        print(f"[fixed] roll {roll} (A start {sa}, ghost {gb}): {len([m for m in meta if m[0]==roll])} turns", flush=True)
    tf.close()

    Aarr = np.stack(acts).astype(np.float16)
    ma = np.array(meta, dtype=object)
    # store A as both players so the manifold/dir scripts (which pool A1,A2) run unchanged
    np.savez_compressed(os.path.join(RUN_DIR, "qwen32_pca_acts.npz"),
                        A1=Aarr, A2=Aarr.copy(), meta1=ma, meta2=ma.copy(),
                        players=np.array([MODEL + "_A", MODEL + "_Ad"]))
    try:
        import jsonl_to_json
        jsonl_to_json.convert(os.path.join(RUN_DIR, "qwen32_fixed_transcript.jsonl"))
    except Exception:
        pass
    print(f"[fixed] DONE -> {RUN_DIR} ({Aarr.shape[0]} points, {nL} layers)", flush=True)


if __name__ == "__main__":
    main()
