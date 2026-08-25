"""Where does the sign flip? A 2x2: {context} x {read-out}.

  contexts:  probe  = single-turn description + "Do you trust X?"  (sweep-style)
             convo  = the advisor conversation after the model's choice (elicit-style)
  read-outs: margin = logit(yes)-logit(no) at the answer position, no generation
             gen    = greedy one-word generation

Same direction, same layer (45), same strength, same injection site (the person's
name tokens) in all four cells. If the sign follows the CONTEXT, the vector means
different things in different places. If it follows the READ-OUT, one of the two
measurement styles is broken and every sign in this project inherits the problem.
"""
import json, os, sys
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import advisor_battery as AB
import dirs as DIRS
import stimuli2 as S2
from common import chat, first_id, load, spans_of, tok_idx
from gen_elicit import InjectGen, gen as gen_free

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))


@torch.no_grad()
def margin_at(model, tok, text, inj_vec, layer, pos):
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    ctx = InjectGen(model, layer, torch.tensor(inj_vec), pos) if inj_vec is not None \
        else __import__("contextlib").nullcontext()
    with ctx:
        lg = model(**enc).logits[0, -1]
    f = lambda w: first_id(tok, w)
    return float(torch.logsumexp(torch.stack([lg[f("yes")], lg[f("Yes")]]), 0) -
                 torch.logsumexp(torch.stack([lg[f("no")], lg[f("No")]]), 0))


def main():
    model, tok, _ = load(); model.eval()
    L = 45; a = 0.5
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(L)])
    D = DIRS.load_all(OUT, L)

    # context A: sweep-style probe (mixed evidence description)
    it = S2.items("direct_b", 3)[1]
    body = it["texts"]["mix"][: it["texts"]["mix"].rstrip().rfind("\n")]
    probe_user = body + f"\n\nDo you trust {it['name']}?\nAnswer with one word, yes or no."
    probe_msgs = [{"role": "system", "content": it["system"]},
                  {"role": "user", "content": probe_user}]
    probe_text = chat(tok, it["system"], probe_user, "")
    probe_pos = tok_idx(tok, probe_text, spans_of(probe_text, it["name"]))

    # context B: the advisor conversation, model's choice included
    sysmsg, abody, ca, cb = AB.build(tok, "invest", False, False, False)
    convo_msgs = [{"role": "system", "content": sysmsg},
                  {"role": "user", "content": abody},
                  {"role": "assistant", "content": ca},
                  {"role": "user", "content": f"Do you trust {AB.A_NAME}?\n"
                                              "Answer with one word, yes or no."}]
    try:
        convo_text = tok.apply_chat_template(convo_msgs, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        convo_text = tok.apply_chat_template(convo_msgs, tokenize=False,
                                             add_generation_prompt=True)
    convo_pos = tok_idx(tok, convo_text, spans_of(convo_text, AB.A_NAME))

    print(f"{'direction':<15}{'cell':<26}{'-v':>8}{'base':>8}{'+v':>8}   gen(+v)/gen(-v)")
    for dname in ("direct_b", "FITTED trust", "prior_wiki", "convo_trust"):
        v = D[dname] * nrm * a
        for cname, text, pos, msgs, who in (
                ("probe/single-turn", probe_text, probe_pos, probe_msgs, it["name"]),
                ("convo/after-choice", convo_text, convo_pos, convo_msgs, AB.A_NAME)):
            m0 = margin_at(model, tok, text, None, L, pos)
            mp = margin_at(model, tok, text, v, L, pos)
            mm = margin_at(model, tok, text, -v, L, pos)
            gp = gen_free(model, tok, msgs, (L, v), 6)
            gm = gen_free(model, tok, msgs, (L, -v), 6)
            print(f"{dname:<15}{cname:<26}{mm:>+8.2f}{m0:>+8.2f}{mp:>+8.2f}   "
                  f"{gp!r} / {gm!r}", flush=True)
    print("SIGNCHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
