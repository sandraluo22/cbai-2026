"""The last variable: bare question vs one-word-instruction question, same convo."""
import json, os, sys
import torch
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import advisor_battery as AB
import dirs as DIRS
from common import load
from gen_elicit import gen as gen_free
from signcheck import margin_at
from common import spans_of, tok_idx

def main():
    model, tok, _ = load(); model.eval()
    L = 45; a = 0.5
    meta = json.load(open(os.path.join(_HERE, "..", "out", "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(L)])
    D = DIRS.load_all(os.path.join(_HERE, "..", "out"), L)
    sysmsg, abody, ca, cb = AB.build(tok, "invest", False, False, False)
    for qname, q in (("bare", f"Do you trust {AB.A_NAME}?"),
                     ("one-word", f"Do you trust {AB.A_NAME}?\nAnswer with one word, yes or no.")):
        msgs = [{"role": "system", "content": sysmsg},
                {"role": "user", "content": abody},
                {"role": "assistant", "content": ca},
                {"role": "user", "content": q}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pos = tok_idx(tok, text, spans_of(text, AB.A_NAME))
        for dname in ("direct_b", "FITTED trust", "prior_wiki"):
            v = D[dname] * nrm * a
            m0 = margin_at(model, tok, text, None, L, pos)
            mp = margin_at(model, tok, text, v, L, pos)
            mm = margin_at(model, tok, text, -v, L, pos)
            gp = gen_free(model, tok, msgs, (L, v), 12)
            gm = gen_free(model, tok, msgs, (L, -v), 12)
            g0 = gen_free(model, tok, msgs, None, 12)
            print(f"  {qname:<9} {dname:<14} margin -v {mm:+.2f} base {m0:+.2f} +v {mp:+.2f}"
                  f"   gen: base {g0!r} | +v {gp!r} | -v {gm!r}", flush=True)
    print("SIGN2_DONE", flush=True)

if __name__ == "__main__":
    main()
