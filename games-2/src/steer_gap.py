"""STEERING THE SUPPRESSION WINDOW (2026-08-05): attack, don't describe.

Patching localized the knowledge-action decoupling to accumulated residual at
L48-62 with no single-component handle (all sublayer patches |rec| <= 0.11).
First-shot handle: a difference-of-means direction per layer in that window,
subtracted from the residual stream during A's LIVE generation.

Stage 1 (direction): 6 transplant pairs (dose-4 planted vs filler control,
identical replayed city streams). d_l = mean over pairs of
(h_loop - h_ctrl) at the final answer position, for l in LAYERS. Saved to
direction.npz.

Stage 2 (live play): standard asymmetric game (A free+steered, B city-restricted,
unsteered), n=N games, cap CAP. Steering: h_l[last position] -= alpha * d_l on
every forward pass of A's generation (prefill applies at the answer slot; each
decode step applies at the new token). Alphas swept: 0, 1, 2, 4.

Stage 3 (controls): t8 MC proposal profile per game (family/category mass);
fluency = mean per-token logprob of a fixed neutral sentence under steering, and
fraction of A words that are alphabetic; met_frac vs alpha.

If met_frac rises and category mass rises without fluency collapse -> a handle.
If not -> the claim stays "localized, no single-component handle".

Env: MODEL(QwenInst32) SRC_DIR START_FILE N(12) CAP(40) K(64) TEMP(0.7) PROBE(8)
     ALPHAS(0,1,2,4) RUN_DIR(runs/steer_gap)
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
N = int(os.environ.get("N", "12"))
CAP = int(os.environ.get("CAP", "40"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
PROBE = int(os.environ.get("PROBE", "8"))
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0,1,2,4").split(",")]
RUN_DIR = os.environ.get("RUN_DIR", "runs/steer_gap")

LAYERS = list(range(48, 63))
MORPH = ["planted", "planting", "plantings", "replant"]
FILLER = ["window", "carpet", "stapler", "napkin"]
NEUTRAL = ("The committee will meet on Thursday afternoon to review the budget "
           "proposal and discuss the schedule for the coming year.")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    def body_of(hist, used):
        s = G.OPEN_PROMPT + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    # ---------------- stage 1: direction ----------------
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

    caps = {}
    hooks = []
    def cap_hook(l):
        def h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            caps[l] = hs[0, -1].detach().float().cpu()
        return h

    @torch.no_grad()
    def capture(body):
        for l in LAYERS:
            hooks.append(model.model.layers[l].register_forward_hook(cap_hook(l)))
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        model(ids)
        for h in hooks:
            h.remove()
        hooks.clear()
        return {l: caps[l].clone() for l in LAYERS}

    diffs = {l: [] for l in LAYERS}
    for si, ts in enumerate(streams):
        roll = ts[0]["rollout"]
        Bseq = [t["B"] for t in ts]
        sa, sb = starts[roll]
        pair = {}
        for content in ("loop", "ctrl"):
            hist = [(sb, sa)]
            used = {sa, sb}
            aw = MORPH if content == "loop" else FILLER
            for i in range(4):
                hist.append((Bseq[i], aw[i]))
                used |= {aw[i], Bseq[i]}
            pair[content] = capture(body_of(hist, used))
        for l in LAYERS:
            diffs[l].append(pair["loop"][l] - pair["ctrl"][l])
    dirs = {l: torch.stack(diffs[l]).mean(0) for l in LAYERS}
    np.savez(os.path.join(RUN_DIR, "direction.npz"),
             **{f"L{l}": dirs[l].numpy() for l in LAYERS})
    for l in LAYERS:
        dirs[l] = dirs[l].to(dev, dtype=model.dtype)
    print(f"[steer] direction built, norms "
          f"{[round(float(dirs[l].norm()), 1) for l in LAYERS[:4]]}...", flush=True)

    # ---------------- stage 2: steered live play ----------------
    steer = {"alpha": 0.0}
    def steer_hook(l):
        def h(_m, _i, out):
            if steer["alpha"] == 0.0:
                return None
            if isinstance(out, tuple):
                out[0][:, -1, :] -= steer["alpha"] * dirs[l]
                return None
            out[:, -1, :] -= steer["alpha"] * dirs[l]
            return out
        return h
    for l in LAYERS:
        model.model.layers[l].register_forward_hook(steer_hook(l))

    @torch.no_grad()
    def gen_word(body, seed, forbidden, alpha):
        steer["alpha"] = alpha
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
        steer["alpha"] = 0.0
        return w

    @torch.no_grad()
    def propose_k(body, alpha):
        steer["alpha"] = alpha
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        steer["alpha"] = 0.0
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    @torch.no_grad()
    def fluency(alpha):
        steer["alpha"] = alpha
        ids = tok(NEUTRAL, return_tensors="pt").input_ids.to(dev)
        logits = model(ids).logits
        lp = torch.log_softmax(logits[0, :-1].float(), -1)
        val = float(lp.gather(1, ids[0, 1:, None]).mean())
        steer["alpha"] = 0.0
        return val

    RESTR = (" IMPORTANT: every single word you say must be the name of a city. Only "
             "ever say cities, nothing else.")
    def body_restr(hist, used):
        return body_of(hist, used).replace(G.OPEN_PROMPT, G.OPEN_PROMPT + RESTR, 1)

    summary = {"model": MODEL, "layers": LAYERS, "alphas": ALPHAS, "conditions": {}}
    tf = open(os.path.join(RUN_DIR, "steer_transcript.jsonl"), "w")
    for alpha in ALPHAS:
        met, turns, probes, aw_ok = [], [], [], []
        for roll, (sa, sb) in enumerate(starts[:N]):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            agreed_at = None
            for t in range(1, CAP):
                if t == PROBE:
                    props = propose_k(body_of(histA, used), alpha)
                    own = [a for _, a in histA]
                    fams = {w[:4] for w in own if len(w) > 3}
                    probes.append({
                        "cat": float(np.mean([1 if (w and w not in used and w in catset)
                                              else 0 for w in props])),
                        "selffam": float(np.mean([1 if (w and w not in used and len(w) > 3
                                                  and w[:4] in fams) else 0 for w in props]))})
                wA = gen_word(body_of(histA, used), 5000 * roll + t, used, alpha)
                wB = gen_word(body_restr(histB, used), 90000 + 5000 * roll + t, used, 0.0)
                aw_ok.append(bool(wA and wA.isalpha() and len(wA) > 2))
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
            print(f"[steer] a={alpha} roll={roll} "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'}", flush=True)
        summary["conditions"][f"a{alpha}"] = {
            "met_frac": float(np.mean(met)), "n_met": int(np.sum(met)),
            "turns_mean": float(np.mean(turns)) if turns else None,
            "probe_cat": float(np.mean([p["cat"] for p in probes])) if probes else None,
            "probe_selffam": float(np.mean([p["selffam"] for p in probes])) if probes else None,
            "word_ok_frac": float(np.mean(aw_ok)),
            "neutral_logprob": fluency(alpha)}
        json.dump(summary, open(os.path.join(RUN_DIR, "steer_gap.json"), "w"), indent=1)
        s = summary["conditions"][f"a{alpha}"]
        print(f"[steer] === alpha {alpha}: met {s['met_frac']:.2f} cat {s['probe_cat']} "
              f"selffam {s['probe_selffam']} ok {s['word_ok_frac']:.2f} "
              f"lp {s['neutral_logprob']:.3f}", flush=True)
    tf.close()
    print("[steer] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
