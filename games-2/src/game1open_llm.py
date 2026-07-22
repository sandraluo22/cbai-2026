"""GAME 1 (UNBOUNDED) between two real LLMs -- convergence with NO board and NO
topic: each model may say ANY word, and we take the KL on the FULL next-token
distribution (restricted to the union of each side's TOP-N tokens + a residual
bucket, so it's tractable and focused where the probability mass is).

Two KL read-outs, graphed per turn just like the bounded game:
  * SWAPPED (coupling): from an identical state, replace the PARTNER's last word with
    a fixed reference word; top-n KL(dist_swap || dist_clean) = how much the partner
    moved the model, over the whole vocabulary.
  * CONSECUTIVE (step drift): top-n KL(dist_t || dist_{t-1}) for each model -- how
    much its own distribution moved turn to turn, unpaired.

pick = real free generation (convergence = same word); semantic = MiniLM distance
between the two said words.

Env: MODELS(LlamaInst,QwenInst) GAMES(4) SAFETY(50) TOPN(50) SWAP_WORD(seven)
     DEVICE RUN_DIR
Out: <RUN_DIR>/game1open_llm_<A>_vs_<B>.{json,pdf} + _transcript.jsonl
     + _coupling_perturn.pdf + _stepkl_perturn.pdf
"""
from __future__ import annotations

import os
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import llm_agents as LA

MODELS = os.environ.get("MODELS", "LlamaInst,QwenInst").split(",")
GAMES = int(os.environ.get("GAMES", "4"))
SAFETY = int(os.environ.get("SAFETY", "50"))            # high backstop (no natural bound when unbounded)
TOPN = int(os.environ.get("TOPN", "50"))
TEMP = float(os.environ.get("TEMP", "1.0"))             # generation temperature
NOREPEAT = os.environ.get("NOREPEAT", "0") == "1"       # forbid reusing any word already said
START_WORDS = os.environ.get("START_WORDS", "")         # e.g. "sand,cloud" -> force a DIVERGENT first round
KSHOW = 14                                              # tokens shown in the per-turn bar slides
SWAP_WORD = os.environ.get("SWAP_WORD", "seven")
RUN_DIR = os.environ.get("RUN_DIR", "runs/llm_open")

# NEUTRAL: rules + win condition only, no strategy hints.
OPEN_PROMPT = ("You are playing a word game with another player. Each round, you both say one word at the "
               "same time, then the two words are revealed. You win the round only if you both said the same "
               "word.")
NOREPEAT_CLAUSE = " You may not say any word that has already been said by either player."


def build_prompt(tok, hist, forbidden=None):
    lines = " ".join(f"Round {k+1}: the other player said {o}, you said {s}." for k, (o, s) in enumerate(hist))
    prompt = OPEN_PROMPT + (NOREPEAT_CLAUSE if NOREPEAT else "")
    body = prompt + ((" " + lines) if lines else "")
    if NOREPEAT and forbidden:
        body += " Words already used (do not repeat): " + ", ".join(sorted(forbidden)) + "."
    return LA._render(tok, body) + "\nMy word:"


def clean_word(txt):
    w = re.split(r"\s+", txt.strip())
    return re.sub(r"[^a-zA-Z\-]", "", w[0] if w else "").lower()


def make_engine(dev):
    import torch

    @torch.no_grad()
    def logits(model, tok, prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        return model(ids).logits[0, -1].float()

    def topn_kl(lp, lq, n=TOPN):
        p = torch.softmax(lp, 0); q = torch.softmax(lq, 0)
        idx = torch.unique(torch.cat([torch.topk(p, n).indices, torch.topk(q, n).indices]))
        pp, qq = p[idx], q[idx]
        P = torch.cat([pp, (1 - pp.sum()).clamp(min=1e-9).view(1)]); P = P / P.sum()
        Q = torch.cat([qq, (1 - qq.sum()).clamp(min=1e-9).view(1)]); Q = Q / Q.sum()
        return float((P * (P.clamp(min=1e-12).log() - Q.clamp(min=1e-12).log())).sum())

    def top_tokens(lg, tok, k=KSHOW):
        p = torch.softmax(lg, 0); v, i = torch.topk(p, k)
        return {tok.decode([int(j)]).strip(): round(float(pr), 4) for pr, j in zip(v, i)}

    @torch.no_grad()
    def gen_word(model, tok, prompt, seed, forbidden=None):
        # ENFORCE no-repeat: resample (new seed) until we get a non-empty word not in
        # `forbidden`. Prompt-only no-repeat is ignored by the models, so we reject
        # repeats here. Give up after RETRIES and return "" (counts as a non-match).
        forbidden = forbidden or set()
        enc = tok(prompt, return_tensors="pt").to(dev)
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in forbidden:
                return w
        return w if w else ""

    return logits, topn_kl, top_tokens, gen_word


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    base = {m: LA.load(m, dev) for m in set(MODELS)}    # load each unique model once (self-play reuses)
    (mA, tA), (mB, tB) = base[MODELS[0]], base[MODELS[1]]
    etok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(dev).eval()
    logits, topn_kl, top_tokens, gen_word = make_engine(dev)
    # distinct labels for self-play (same model as both players). Use "_1"/"_2" --
    # NOT "#", which breaks file openers (# is a URL/path fragment separator).
    if MODELS[0] == MODELS[1]:
        nameA, nameB = MODELS[0] + "_1", MODELS[1] + "_2"
    else:
        nameA, nameB = MODELS
    print(f"[open] {nameA} vs {nameB}  temp={TEMP} norepeat={NOREPEAT} cap={SAFETY}", flush=True)
    ec = {}

    def emb(w):
        if w not in ec:
            enc = etok([w or "."], return_tensors="pt", truncation=True).to(dev)
            with torch.no_grad():
                h = emod(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            ec[w] = torch.nn.functional.normalize(v, dim=1)[0].cpu().numpy()
        return ec[w]

    tf = open(os.path.join(RUN_DIR, f"game1open_llm_{nameA}_vs_{nameB}_transcript.jsonl"), "w")
    conv, ttc, coupA, coupB = [], [], [], []
    dist_by_turn = np.full((GAMES, SAFETY), np.nan)
    cKL = {n: np.full((GAMES, SAFETY), np.nan) for n in (nameA, nameB)}   # coupling per turn
    sKL = {n: np.full((GAMES, SAFETY), np.nan) for n in (nameA, nameB)}   # step drift per turn
    exemplar = []
    for gi in range(GAMES):
        histA, histB, agreed = [], [], None
        prevA = prevB = None
        gcA, gcB = [], []
        used = set()
        t0 = 0
        if START_WORDS:                                 # force a divergent first round
            wA0, wB0 = [w.strip() for w in START_WORDS.split(",")]
            histA = [(wB0, wA0)]; histB = [(wA0, wB0)]; used = {wA0, wB0}
            d0 = float(1 - emb(wA0) @ emb(wB0)); dist_by_turn[gi, 0] = d0
            frec = {"game": gi, "turn": 0, "forced": True, "picks": {nameA: wA0, nameB: wB0},
                    "agreed": bool(wA0 == wB0), "semantic_dist": round(d0, 4),
                    nameA: {"pick": wA0}, nameB: {"pick": wB0}}
            tf.write(json.dumps(frec) + "\n")
            if gi == 0:
                exemplar.append(frec)
            print(f"    g{gi} t1 (FORCED): {nameA}={wA0!r} {nameB}={wB0!r} dist={d0:.2f}", flush=True)
            t0 = 1
        for t in range(t0, SAFETY):
            pA, pB = build_prompt(tA, histA, used), build_prompt(tB, histB, used)
            lA, lB = logits(mA, tA, pA), logits(mB, tB, pB)
            forb = used if NOREPEAT else None
            wA = gen_word(mA, tA, pA, 1000 * gi + t, forb); wB = gen_word(mB, tB, pB, 7000 + 1000 * gi + t, forb)
            agree = (wA == wB and wA != "")
            d = float(1 - emb(wA) @ emb(wB)); dist_by_turn[gi, t] = d
            rec = {"game": gi, "turn": t, "picks": {nameA: wA, nameB: wB}, "agreed": agree,
                   "semantic_dist": round(d, 4),
                   nameA: {"pick": wA, "top": top_tokens(lA, tA)},
                   nameB: {"pick": wB, "top": top_tokens(lB, tB)}}
            if t >= 1:
                # in-distribution counterfactual: replace the partner's last word with a
                # word actually said EARLIER in this game (their previous word, else this
                # player's last word) -- not an OOD token like "seven".
                def alt_for(h):
                    prev_other = h[-2][0] if len(h) >= 2 else None
                    for c in (prev_other, h[-1][1]):
                        if c and c != h[-1][0]:
                            return c
                    return SWAP_WORD
                altB, altA = alt_for(histB), alt_for(histA)
                lB_sw = logits(mB, tB, build_prompt(tB, histB[:-1] + [(altB, histB[-1][1])], used))
                lA_sw = logits(mA, tA, build_prompt(tA, histA[:-1] + [(altA, histA[-1][1])], used))
                klB, klA = topn_kl(lB_sw, lB), topn_kl(lA_sw, lA)
                gcB.append(klB); gcA.append(klA); cKL[nameA][gi, t] = klA; cKL[nameB][gi, t] = klB
                rec[nameB]["coupling"] = {"kl": round(klB, 4), "swap_other_to": altB, "swap_top": top_tokens(lB_sw, tB)}
                rec[nameA]["coupling"] = {"kl": round(klA, 4), "swap_other_to": altA, "swap_top": top_tokens(lA_sw, tA)}
            if prevA is not None:                       # step drift needs a prior GENERATED turn
                stepA, stepB = topn_kl(lA, prevA), topn_kl(lB, prevB)
                sKL[nameA][gi, t] = stepA; sKL[nameB][gi, t] = stepB
                rec[nameA]["step"] = {"kl": round(stepA, 4)}; rec[nameB]["step"] = {"kl": round(stepB, 4)}
            tf.write(json.dumps(rec) + "\n")
            if gi == 0:
                exemplar.append(rec)
            print(f"    g{gi} t{t+1}: {nameA}={wA!r} {nameB}={wB!r} dist={d:.2f}"
                  + (f" coup {cKL[nameA][gi,t]:.2f}/{cKL[nameB][gi,t]:.2f} step {sKL[nameA][gi,t]:.2f}/{sKL[nameB][gi,t]:.2f}" if t >= 1 else "")
                  + ("  AGREED" if agree else ""), flush=True)
            if agree:
                agreed = t; break
            histA.append((wB, wA)); histB.append((wA, wB)); prevA, prevB = lA, lB
            if wA:
                used.add(wA)
            if wB:
                used.add(wB)
        conv.append(agreed is not None)
        if agreed is not None:
            ttc.append(agreed + 1)
        coupA.append(np.mean(gcA) if gcA else 0.0); coupB.append(np.mean(gcB) if gcB else 0.0)
    tf.close()

    out = {"models": MODELS, "games": GAMES, "safety": SAFETY, "topn": TOPN, "swap_word": SWAP_WORD,
           "converged_frac": float(np.mean(conv)), "turns_to_converge": float(np.mean(ttc)) if ttc else float("nan"),
           f"coupling_{nameA}": float(np.mean(coupA)), f"coupling_{nameB}": float(np.mean(coupB)),
           "sem_dist_by_turn": np.nanmean(dist_by_turn, 0).tolist(),
           "labels": [nameA, nameB],
           "coupling_by_turn": {n: np.nanmean(cKL[n], 0).tolist() for n in (nameA, nameB)},
           "step_by_turn": {n: np.nanmean(sKL[n], 0).tolist() for n in (nameA, nameB)}}
    tag = f"{nameA}_vs_{nameB}".replace("#", "")
    json.dump(out, open(os.path.join(RUN_DIR, f"game1open_llm_{tag}.json"), "w"), indent=2)
    summary_fig(out, os.path.join(RUN_DIR, f"game1open_llm_{tag}.pdf"), (nameA, nameB))
    import kl_slides                                     # every-turn coupling + step KL graphs (all games)
    kl_slides.generate(os.path.join(RUN_DIR, f"game1open_llm_{nameA}_vs_{nameB}_transcript.jsonl"))
    print(f"[open] converged={out['converged_frac']*100:.0f}% turns={out['turns_to_converge']:.1f} "
          f"coupling {nameA}={out[f'coupling_{nameA}']:.2f} {nameB}={out[f'coupling_{nameB}']:.2f}", flush=True)
    print(f"[open] DONE -> {RUN_DIR}", flush=True)


def _bars(ax, da, db, la, lb, ca, title):
    toks = list(dict.fromkeys(list(da) + list(db)))[:16]
    x = np.arange(len(toks))
    ax.bar(x - 0.2, [da.get(t, 0) for t in toks], 0.4, color=ca, alpha=.85, label=la)
    ax.bar(x + 0.2, [db.get(t, 0) for t in toks], 0.4, color="0.55", alpha=.85, label=lb)
    ax.set_xticks(x); ax.set_xticklabels(toks, rotation=90, fontsize=6); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9); ax.legend(fontsize=7)


def perturn_fig(exemplar, which, path, names):
    nameA, nameB = names
    with PdfPages(path) as pdf:
        # summary curve
        turns = [r["turn"] + 1 for r in exemplar if which in r.get(nameA, {})]
        kA = [r[nameA][which]["kl"] for r in exemplar if which in r.get(nameA, {})]
        kB = [r[nameB][which]["kl"] for r in exemplar if which in r.get(nameB, {})]
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        ax.plot(turns, kA, "-o", color="tab:blue", label=nameA); ax.plot(turns, kB, "-o", color="tab:orange", label=nameB)
        ax.set_xlabel("turn"); ax.set_ylabel("top-N KL")
        ax.set_title(f"{'coupling (swap partner)' if which=='coupling' else 'consecutive-turn drift'} KL per turn (game 0)", fontsize=10)
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # per-turn pages
        prev = {nameA: None, nameB: None}
        for r in exemplar:
            t = r["turn"]
            if which in r.get(nameA, {}):
                fig = plt.figure(figsize=(13, 5)); gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.4, 1.4])
                axk, axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
                klA, klB = r[nameA][which]["kl"], r[nameB][which]["kl"]
                axk.bar([0, 1], [klA, klB], color=["tab:blue", "tab:orange"])
                axk.set_xticks([0, 1]); axk.set_xticklabels([nameA, nameB], fontsize=8, rotation=20)
                axk.set_ylabel("top-N KL"); axk.set_title(which, fontsize=9)
                for axm, name, c in ((axA, nameA, "tab:blue"), (axB, nameB, "tab:orange")):
                    clean = r[name]["top"]
                    if which == "coupling":
                        other = r[name]["coupling"]["swap_top"]; lb = f"swap→{r[name]['coupling']['swap_other_to']}"
                    else:
                        other = prev[name] if prev[name] is not None else {}; lb = "prev turn"
                    _bars(axm, clean, other, "now (clean)", lb, c, f"{name}: said {r[name]['pick']}  KL={r[name][which]['kl']:.2f}")
                fig.suptitle(f"UNBOUNDED turn {t+1}: {nameA}={r['picks'][nameA]}, {nameB}={r['picks'][nameB]}"
                             + ("  ★AGREED" if r["agreed"] else ""), fontsize=11)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
            prev[nameA] = r[nameA].get("top", prev[nameA]); prev[nameB] = r[nameB].get("top", prev[nameB])
    print(f"  {path}")


def summary_fig(out, path, names):
    d = np.array(out["sem_dist_by_turn"]); turns = np.arange(1, len(d) + 1)
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        ax[0].plot(turns, d, "-o", color="tab:green"); ax[0].set_title("semantic distance between said words", fontsize=9)
        ax[0].set_xlabel("round"); ax[0].grid(alpha=.3)
        for n, c in zip(names, ["tab:blue", "tab:orange"]):
            ax[1].plot(turns, out["coupling_by_turn"][n], "-o", color=c, label=n)
            ax[2].plot(turns, out["step_by_turn"][n], "-o", color=c, label=n)
        ax[1].set_title(f"coupling (swap→'{out['swap_word']}') top-{out['topn']} KL", fontsize=9)
        ax[2].set_title(f"consecutive-turn drift top-{out['topn']} KL", fontsize=9)
        for a in ax[1:]:
            a.set_xlabel("turn"); a.legend(fontsize=8); a.grid(alpha=.3)
        fig.suptitle(f"GAME 1 UNBOUNDED — {' & '.join(MODELS)}. converged {out['converged_frac']*100:.0f}% "
                     f"(cap {out['safety']} turns)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
