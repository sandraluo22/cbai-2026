"""What does the model SAY under steering? Free generation, two questions in sequence.

Advisor invest scenario, injection at Ana's name (L45, alpha 0.5, +v and -v), then:
  Q1  "Do you trust Ana?"
  Q2  "Why don't you trust Ana's recommendation?"   (deliberately presupposing)
Q1's answer is fed back before Q2 is asked, so Q2 is answered in context.

Injection during generation: with KV caching the prompt is processed once, so a
position-indexed hook fires correctly on that pass and the perturbation persists in
the cache; on 1-token continuation passes the positions are out of range and the
hook must no-op rather than crash.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "src") if os.path.isdir("src") else sys.path.insert(0, ".")
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import advisor_battery as AB
import dirs as DIRS
from common import chat, load, spans_of, tok_idx

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))


class InjectGen:
    """Positional injection that no-ops on cached 1-token continuation passes."""
    def __init__(self, model, layer, vec, pos):
        self.model, self.blk = model, max(0, layer - 1)
        self.vec, self.pos = vec, pos
    def __enter__(self):
        def f(mod, inp, out):
            tup = isinstance(out, tuple)
            h = (out[0] if tup else out)
            if not self.pos or h.shape[1] <= max(self.pos):
                return out
            h = h.clone()
            v = self.vec.to(h.dtype).to(h.device)
            h[0, self.pos] = h[0, self.pos] + v
            return ((h,) + tuple(out[1:])) if tup else h
        self.hk = self.model.model.layers[self.blk].register_forward_hook(f)
        return self
    def __exit__(self, *a):
        self.hk.remove(); return False


@torch.no_grad()
def gen(model, tok, msgs, inj, max_new=110):
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pos = tok_idx(tok, text, spans_of(text, AB.A_NAME))
    enc = tok(text, return_tensors="pt").to(model.device)
    ctx = InjectGen(model, inj[0], torch.tensor(inj[1]), pos) if inj else \
          __import__("contextlib").nullcontext()
    with ctx:
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    model, tok, _ = load(); model.eval()
    L = 45; a = float(os.environ.get("ALPHA", "0.5"))
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(L)])
    D = DIRS.load_all(OUT, L)
    sysmsg, body, ca, cb = AB.build(tok, "invest", False, False, False)
    res = {}
    for name in ["baseline", "FITTED trust", "direct_b", "prior_wiki", "prior_src",
                 "prior_expert", "warmth_b", "random"]:
        for sign in ([0] if name == "baseline" else [+1, -1]):
            inj = None if name == "baseline" else (L, D[name] * nrm * a * sign)
            tagn = name if name == "baseline" else f"{name} {'+' if sign>0 else '-'}v"
            msgs = [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": body}]
            choice = gen(model, tok, msgs, inj, 8)
            msgs += [{"role": "assistant", "content": choice},
                     {"role": "user", "content": f"Do you trust {AB.A_NAME}?"}]
            a1 = gen(model, tok, msgs, inj)
            msgs += [{"role": "assistant", "content": a1},
                     {"role": "user",
                      "content": f"Why don't you trust {AB.A_NAME}'s recommendation?"}]
            a2 = gen(model, tok, msgs, inj)
            res[tagn] = dict(choice=choice, trust=a1, why=a2)
            print(f"\n##### {tagn} #####\nCHOICE: {choice}\nQ1 Do you trust Ana?\n  {a1}"
                  f"\nQ2 Why don't you trust Ana's recommendation?\n  {a2}", flush=True)
    json.dump(res, open(os.path.join(OUT, "gen_elicit.json"), "w"), indent=1)
    print("GEN_ELICIT_DONE", flush=True)


if __name__ == "__main__":
    main()
