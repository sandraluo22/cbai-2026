"""PAIRED ACTIVATION PATCHING on transplant pairs (2026-08-02).

Pairs: identical replayed partner history (city streams), loop-seeded vs
neutral-seeded self-history — construction copied from transplant_replay.py, so the
pair differs ONLY in the 3 attributed self-action words (in-distribution, matched).

Metric (logit-difference style, per pair):
  Delta = logsumexp(z over SELF-STEM first tokens) - logsumexp(z over CATEGORY first
  tokens), read at the final position of "...\nMy word:".
  self-stem set = donor triple + its 4-prefix stem; category set = unused city words.

Sweeps (control -> loop patches of the residual stream = decoder-block output):
  A. per-layer patch at the FINAL position
  B. per-layer patch at the SEED-WORD token spans (donor words inside round lines,
     located via offset mapping; ctrl span mean-pooled when token lengths differ)
  C. attn-output vs MLP-output patch at the final position, for the TOPK layers most
     Delta-reducing in sweep A.

Reported per layer: mean Delta_patched and recovery = (D_loop - D_patch)/(D_loop -
D_ctrl) averaged over pairs.

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) START_FILE N_STREAMS(6) TOPK(4)
     RUN_DIR(runs/patch_transplant)
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
N_STREAMS = int(os.environ.get("N_STREAMS", "6"))
TOPK = int(os.environ.get("TOPK", "4"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/patch_transplant")


def fam_of(w):
    return w[:4] if len(w) > 3 else None


def build_pairs():
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
    loop_donors = []
    for roll, ts in games.items():
        A = [t["A"] for t in ts]
        for i in range(len(A) - 2):
            fs = [fam_of(w) for w in A[i:i + 3]]
            if None not in fs and len(set(fs)) == 1:
                loop_donors.append(A[i:i + 3])
                break
    neutral_donors = [[starts[i][0], starts[i + 1][0], starts[i + 2][0]]
                      for i in range(20, 26)]
    streams = sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS]
    pairs = []
    for si, (roll, ts) in enumerate(streams):
        Bseq = [t["B"] for t in ts]
        sa, sb = starts[roll]
        bodies = {}
        donors = {"loop": loop_donors[si % len(loop_donors)],
                  "ctrl": neutral_donors[si % len(neutral_donors)]}
        for cond, donor in donors.items():
            histA = [(sb, sa)]
            used = {sa, sb}
            for k3 in range(3):
                histA.append((Bseq[k3], donor[k3]))
                used |= {Bseq[k3], donor[k3]}
            body = (G.OPEN_PROMPT + " "
                    + " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                               for k, (o, s) in enumerate(histA))
                    + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".")
            bodies[cond] = (body, used)
        pairs.append({"stream": roll, "loop_donor": donors["loop"],
                      "ctrl_donor": donors["ctrl"],
                      "loop_body": bodies["loop"][0], "ctrl_body": bodies["ctrl"][0],
                      "used": sorted(bodies["loop"][1] | bodies["ctrl"][1])})
    return pairs


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    blocks = model.model.layers
    nL = len(blocks)
    catset = list(CATWORDS["city"])

    def fid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]

    def prompt_of(body):
        return LA._render(tok, body) + "\nMy word:"

    @torch.no_grad()
    def forward_cache(prompt):
        enc = tok(prompt, return_tensors="pt").to(dev)
        out = model(enc.input_ids, output_hidden_states=True)
        # hidden_states[L+1] = output of block L
        return enc, out.logits[0, -1].float(), [h[0] for h in out.hidden_states]

    def delta_of(logits, stem_ids, cat_ids):
        z = logits
        ls = torch.logsumexp(z[stem_ids], 0)
        lc = torch.logsumexp(z[cat_ids], 0)
        return float(ls - lc)

    import torch
    pairs = build_pairs()
    out_rows = []
    layer_scores = np.zeros(nL)
    for pi, pr in enumerate(pairs):
        stem_words = list(dict.fromkeys(
            pr["loop_donor"] + [fam_of(pr["loop_donor"][0])]))
        stem_ids = torch.tensor(sorted({fid(w) for w in stem_words}))
        cat_words = [w for w in catset if w not in set(pr["used"])][:24]
        cat_ids = torch.tensor(sorted({fid(w) for w in cat_words} - set(stem_ids.tolist())))

        p_loop = prompt_of(pr["loop_body"])
        p_ctrl = prompt_of(pr["ctrl_body"])
        enc_l, log_l, hs_l = forward_cache(p_loop)
        enc_c, log_c, hs_c = forward_cache(p_ctrl)
        d_loop = delta_of(log_l, stem_ids, cat_ids)
        d_ctrl = delta_of(log_c, stem_ids, cat_ids)

        # seed-word token spans via offset mapping (span k of loop <-> span k of ctrl)
        def spans(prompt, donor):
            enc = tok(prompt, return_offsets_mapping=True)
            offs = enc["offset_mapping"]
            res = []
            for w in donor:
                cpos = prompt.find(f"you said {w}")
                if cpos < 0:
                    res.append(None)
                    continue
                a, b = cpos + len("you said "), cpos + len("you said ") + len(w)
                idx = [i for i, (x, y) in enumerate(offs) if x < b and y > a]
                res.append((idx[0], idx[-1] + 1) if idx else None)
            return res
        sp_l = spans(p_loop, pr["loop_donor"])
        sp_c = spans(p_ctrl, pr["ctrl_donor"])

        @torch.no_grad()
        def patched_delta(L, mode):
            """mode: 'final' | 'spans' | 'attn' | 'mlp'"""
            handles = []
            if mode in ("final", "spans"):
                repl_final = hs_c[L + 1][-1]

                def hook(_m, _i, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if mode == "final":
                        h[0, -1] = repl_final.to(h.dtype)
                    else:
                        for sl, sc in zip(sp_l, sp_c):
                            if sl is None or sc is None:
                                continue
                            src = hs_c[L + 1][sc[0]:sc[1]]
                            if sc[1] - sc[0] == sl[1] - sl[0]:
                                h[0, sl[0]:sl[1]] = src.to(h.dtype)
                            else:
                                h[0, sl[0]:sl[1]] = src.mean(0, keepdim=True).to(h.dtype)
                    return out
                handles.append(blocks[L].register_forward_hook(hook))
            else:
                # capture ctrl submodule output at final pos, then patch into loop run
                cap = {}
                sub_c = blocks[L].self_attn if mode == "attn" else blocks[L].mlp

                def cap_hook(_m, _i, out):
                    o = out[0] if isinstance(out, tuple) else out
                    cap["v"] = o[0, -1].detach().clone()
                    return out
                hc = sub_c.register_forward_hook(cap_hook)
                with torch.no_grad():
                    model(enc_c.input_ids)
                hc.remove()

                def hook(_m, _i, out):
                    o = out[0] if isinstance(out, tuple) else out
                    o[0, -1] = cap["v"].to(o.dtype)
                    return out
                sub_l = blocks[L].self_attn if mode == "attn" else blocks[L].mlp
                handles.append(sub_l.register_forward_hook(hook))
            with torch.no_grad():
                lg = model(enc_l.input_ids).logits[0, -1].float()
            for h in handles:
                h.remove()
            return delta_of(lg, stem_ids, cat_ids)

        row = {"stream": pr["stream"], "loop_donor": pr["loop_donor"],
               "d_loop": d_loop, "d_ctrl": d_ctrl, "final": [], "spans": []}
        for L in range(nL):
            row["final"].append(patched_delta(L, "final"))
            row["spans"].append(patched_delta(L, "spans"))
        denom = (d_loop - d_ctrl) or 1e-9
        rec = [(d_loop - v) / denom for v in row["final"]]
        layer_scores += np.array(rec) / len(pairs)
        print(f"[patch] pair {pi} stream {pr['stream']}: d_loop {d_loop:.2f} d_ctrl {d_ctrl:.2f} "
              f"best final-layer recovery {max(rec):.2f} @L{int(np.argmax(rec))}", flush=True)
        out_rows.append(row)
        json.dump({"pairs": out_rows}, open(os.path.join(RUN_DIR, "patch_layers.json"), "w"))

    top = np.argsort(-layer_scores)[:TOPK]
    decomp = {}
    # decomposition pass (kept separate so sweep results are saved first)
    for pi, pr in enumerate(pairs):
        stem_words = list(dict.fromkeys(pr["loop_donor"] + [fam_of(pr["loop_donor"][0])]))
        stem_ids = torch.tensor(sorted({fid(w) for w in stem_words}))
        cat_words = [w for w in catset if w not in set(pr["used"])][:24]
        cat_ids = torch.tensor(sorted({fid(w) for w in cat_words} - set(stem_ids.tolist())))
        p_loop = prompt_of(pr["loop_body"]); p_ctrl = prompt_of(pr["ctrl_body"])
        enc_l, log_l, hs_l = forward_cache(p_loop)
        enc_c, log_c, hs_c = forward_cache(p_ctrl)
        d_loop = delta_of(log_l, stem_ids, cat_ids); d_ctrl = delta_of(log_c, stem_ids, cat_ids)
        denom = (d_loop - d_ctrl) or 1e-9
        sp_l = sp_c = None  # unused in attn/mlp mode

        @torch.no_grad()
        def sub_patch(L, mode):
            cap = {}
            sub_c = blocks[L].self_attn if mode == "attn" else blocks[L].mlp

            def cap_hook(_m, _i, out):
                o = out[0] if isinstance(out, tuple) else out
                cap["v"] = o[0, -1].detach().clone()
                return out
            hc = sub_c.register_forward_hook(cap_hook)
            model(enc_c.input_ids)
            hc.remove()

            def hook(_m, _i, out):
                o = out[0] if isinstance(out, tuple) else out
                o[0, -1] = cap["v"].to(o.dtype)
                return out
            sub_l = blocks[L].self_attn if mode == "attn" else blocks[L].mlp
            h = sub_l.register_forward_hook(hook)
            lg = model(enc_l.input_ids).logits[0, -1].float()
            h.remove()
            return (d_loop - delta_of(lg, stem_ids, cat_ids)) / denom
        for L in top.tolist():
            decomp.setdefault(L, {"attn": [], "mlp": []})
            decomp[L]["attn"].append(sub_patch(L, "attn"))
            decomp[L]["mlp"].append(sub_patch(L, "mlp"))
    summary = {"n_pairs": len(pairs), "n_layers": nL,
               "mean_final_recovery_by_layer": layer_scores.tolist(),
               "top_layers": top.tolist(),
               "decomp": {str(L): {k: float(np.mean(v)) for k, v in d.items()}
                          for L, d in decomp.items()}}
    json.dump({"pairs": out_rows, "summary": summary},
              open(os.path.join(RUN_DIR, "patch_layers.json"), "w"), indent=1)
    print("[patch] top layers:", top.tolist(),
          {str(L): {k: round(float(np.mean(v)), 2) for k, v in decomp[L].items()} for L in top.tolist()},
          flush=True)


if __name__ == "__main__":
    main()
