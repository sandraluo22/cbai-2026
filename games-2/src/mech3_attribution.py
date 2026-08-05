"""THREE-MECHANISM SEPARATION (2026-08-03, user-preregistered design).

Separates: (1) lexical priming, (2) role-structural induction (answer-slot series
continuation), (3) attribution semantics (self vs other ownership of the series).

History = two fixed series X and Y; always ask for the next X entry. Stem triple
placement x ownership design, identical stems and surrounding words across cells:

  placement: absent | lexical (neutral note outside history) | nontarget (Y slots)
             | target (X slots)
  ownership: self ("X entries are your own previous responses") vs other ("X entries
             were produced by another model") — several paraphrases.

Counterbalanced: label pair (X/Y vs P/Q), series order (X-first vs Y-first), header
paraphrase (2) -> 8 replicates per cell. Stem families: morphological, synthetic
(invented), semantic (music), character-overlap-only.

Measures (never emitted actions): (a) raw family first-token probability mass
(full-vocab softmax), (b) K-sample MC family mass (novel words only).

Preregistered contrasts (computed per stem family and pooled):
  d_lexical     = P(lexical) - P(absent)
  d_structural  = P(target)  - P(nontarget)
  d_attribution = P(target, self) - P(target, other)
  interaction   = (self-other | target) - (self-other | nontarget)

STAGE 2 (patching): exact matched self-vs-other pair (target slot, same labels/
order/paraphrase; prompts differ ONLY in the ownership header). Because the suffix
after the header is byte-identical, positions align from the END: per layer, patch
the shared-suffix residuals (and final position only) from OTHER into SELF and
measure how much of the attribution effect survives -> where attribution enters.

Env: MODEL(QwenInst32) K(32) TEMP(0.7) RUN_DIR(runs/mech3_attribution)
"""
from __future__ import annotations
import os
import json
import itertools
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
K = int(os.environ.get("K", "32"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/mech3_attribution")

STEMS = {
    "morph": (["planted", "planting", "plantings"], "prefix"),
    "synthetic": (["blorfin", "blorfed", "blorfs"], "prefix"),
    "charonly": (["carton", "cartel", "cartoon"], "prefix"),
    "semantic": (["melody", "rhythm", "chorus"], "semset"),
}
SEMSET = {"melody", "rhythm", "chorus", "harmony", "tune", "song", "verse", "tempo",
          "cadence", "refrain", "ballad", "anthem", "symphony", "sonata", "melodies",
          "chord", "lyric", "lyrics", "octave", "opera"}
FILL_X = ["window", "carpet", "stapler"]
FILL_Y = ["napkin", "candle", "basket"]
NEXT_FILL = ["lantern", "mirror", "kettle"]

HEADERS = {
    "self": ["The {X} entries below are your own previous responses; the {Y} entries came from a different source.",
             "You yourself wrote the {X} entries earlier; the {Y} entries were written elsewhere."],
    "other": ["The {X} entries below were produced by another model; the {Y} entries came from a different source.",
              "A different model wrote the {X} entries; the {Y} entries were written elsewhere."],
}
LABELS = [("X", "Y"), ("P", "Q")]


def build(stem_words, placement, ownership, label_i, order_i, para_i):
    X, Y = LABELS[label_i]
    xw = list(stem_words) if placement == "target" else list(FILL_X)
    yw = list(stem_words) if placement == "nontarget" else list(FILL_Y)
    head = HEADERS[ownership][para_i].format(X=X, Y=Y)
    note = (" Note - an unrelated word list: " + ", ".join(stem_words) + "."
            ) if placement == "lexical" else ""
    lines = []
    for k in range(3):
        pair = [f"{X}{k+1}: {xw[k]}", f"{Y}{k+1}: {yw[k]}"]
        if order_i == 1:
            pair = pair[::-1]
        lines += pair
    body = ("Two series of one-word entries, recorded in the order produced. "
            + head + note + " " + " ".join(lines)
            + f" Write the next {X} entry (a single new word not used above).")
    prefill = f"\n{X}4:"
    return body, prefill


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    blocks = model.model.layers
    nL = len(blocks)

    def fid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]

    def fam_ids(stem_words, ftype):
        ws = set(stem_words)
        if ftype == "prefix":
            ws.add(stem_words[0][:4])
        else:
            ws |= SEMSET
        return sorted({fid(w) for w in ws})

    def fam_hit(w, stem_words, ftype, ctx_words):
        if not w or w in ctx_words:
            return 0
        if ftype == "semset":
            return 1 if w in SEMSET else 0
        return 1 if (len(w) > 3 and w[:4] == stem_words[0][:4]) else 0

    @torch.no_grad()
    def measures(body, prefill, stem_words, ftype, ctx_words):
        prompt = LA._render(tok, body) + prefill
        enc = tok(prompt, return_tensors="pt").to(dev)
        logits = model(enc.input_ids).logits[0, -1].float()
        p = torch.softmax(logits, 0)
        pfam = float(p[fam_ids(stem_words, ftype)].sum())
        out = model.generate(enc.input_ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        words = [G.clean_word(tok.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True))
                 for i in range(K)]
        mc = float(np.mean([fam_hit(w, stem_words, ftype, ctx_words) for w in words]))
        return pfam, mc

    rows = []
    for sname, (swords, ftype) in STEMS.items():
        for placement in ("absent", "lexical", "nontarget", "target"):
            for ownership in ("self", "other"):
                for label_i, order_i, para_i in itertools.product((0, 1), (0, 1), (0, 1)):
                    body, prefill = build(swords, placement, ownership, label_i, order_i, para_i)
                    ctx = set(FILL_X + FILL_Y + swords)
                    pfam, mc = measures(body, prefill, swords, ftype, ctx)
                    rows.append({"stem": sname, "placement": placement, "own": ownership,
                                 "label": label_i, "order": order_i, "para": para_i,
                                 "pfam": pfam, "mc": mc})
        json.dump({"rows": rows}, open(os.path.join(RUN_DIR, "mech3.json"), "w"))
        print(f"[mech3] {sname} done ({len(rows)} cells)", flush=True)

    def cellmean(metric, **kw):
        sel = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
        return float(np.mean(sel)) if sel else float("nan")

    contrasts = {}
    for metric in ("pfam", "mc"):
        per = {}
        for sname in list(STEMS) + [None]:
            kw = {} if sname is None else {"stem": sname}
            d_lex = cellmean(metric, placement="lexical", **kw) - cellmean(metric, placement="absent", **kw)
            d_str = cellmean(metric, placement="target", **kw) - cellmean(metric, placement="nontarget", **kw)
            d_att = (cellmean(metric, placement="target", own="self", **kw)
                     - cellmean(metric, placement="target", own="other", **kw))
            inter = d_att - (cellmean(metric, placement="nontarget", own="self", **kw)
                             - cellmean(metric, placement="nontarget", own="other", **kw))
            per[sname or "pooled"] = {"d_lexical": d_lex, "d_structural": d_str,
                                      "d_attribution": d_att, "interaction": inter,
                                      "baseline_absent": cellmean(metric, placement="absent", **kw)}
        contrasts[metric] = per
    print("[mech3] contrasts (mc, pooled):", {k: round(v, 3) for k, v in contrasts["mc"]["pooled"].items()},
          flush=True)

    # ---------------- stage 2: matched self-vs-other patching ----------------
    import torch
    sname, (swords, ftype) = "morph", STEMS["morph"]
    patch_out = {}
    for label_i, order_i, para_i in itertools.product((0, 1), (0, 1), (0, 1)):
        b_self, pf = build(swords, "target", "self", label_i, order_i, para_i)
        b_other, _ = build(swords, "target", "other", label_i, order_i, para_i)
        p_self = LA._render(tok, b_self) + pf
        p_other = LA._render(tok, b_other) + pf
        ids_s = tok(p_self, return_tensors="pt").input_ids.to(dev)
        ids_o = tok(p_other, return_tensors="pt").input_ids.to(dev)
        # shared suffix length (token-level, from the end)
        ls, lo = ids_s.shape[1], ids_o.shape[1]
        nsuf = 0
        while nsuf < min(ls, lo) and int(ids_s[0, ls - 1 - nsuf]) == int(ids_o[0, lo - 1 - nsuf]):
            nsuf += 1
        fids = torch.tensor(fam_ids(swords, ftype))

        @torch.no_grad()
        def pfam_of(ids, hooks=None):
            logits = model(ids).logits[0, -1].float()
            return float(torch.softmax(logits, 0)[fids].sum())

        with torch.no_grad():
            out_o = model(ids_o, output_hidden_states=True)
            hs_o = [h[0] for h in out_o.hidden_states]
            base_self = pfam_of(ids_s)
            base_other = float(torch.softmax(out_o.logits[0, -1].float(), 0)[fids].sum())
        curve_fin, curve_suf = [], []
        for L in range(nL):
            for mode, store in (("final", curve_fin), ("suffix", curve_suf)):
                def hook(_m, _i, out, L=L, mode=mode):
                    h = out[0] if isinstance(out, tuple) else out
                    if mode == "final":
                        h[0, -1] = hs_o[L + 1][-1].to(h.dtype)
                    else:
                        h[0, -nsuf:] = hs_o[L + 1][-nsuf:].to(h.dtype)
                    return out
                hd = blocks[L].register_forward_hook(hook)
                with torch.no_grad():
                    v = pfam_of(ids_s)
                hd.remove()
                store.append(v)
        patch_out[f"l{label_i}o{order_i}p{para_i}"] = {
            "base_self": base_self, "base_other": base_other, "nsuf": nsuf,
            "final": curve_fin, "suffix": curve_suf}
        print(f"[mech3] patch rep l{label_i}o{order_i}p{para_i}: self {base_self:.4f} "
              f"other {base_other:.4f} nsuf {nsuf}", flush=True)
        json.dump({"rows": rows, "contrasts": contrasts, "patch": patch_out},
                  open(os.path.join(RUN_DIR, "mech3.json"), "w"))

    json.dump({"rows": rows, "contrasts": contrasts, "patch": patch_out},
              open(os.path.join(RUN_DIR, "mech3.json"), "w"), indent=1)
    print("[mech3] done", flush=True)


if __name__ == "__main__":
    main()
