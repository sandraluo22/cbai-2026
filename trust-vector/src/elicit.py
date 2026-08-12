"""Does the model actually distrust the caller? Let it answer instead of scripting it.

`project.py` measured a read-out along a conversation whose assistant turns were all
written by me, and found no drop at the passcode request. Two very different
explanations fit that:

  (a) the read-out does not track trust, or
  (b) the model is not suspicious at that point at all -- in which case a flat
      trajectory is CORRECT and the scripted turn "the bank says it never asks"
      was me asserting a suspicion the model does not have.

This separates them by removing the script. Two modes:

  spot  scripted history up to a point, then the model writes THAT turn itself.
        Comparable across depths because every depth sees the same history.
  roll  the model writes every assistant turn, and its own words are fed back in.
        A real conversation; diverges from the script, which is the point. The
        read-out is also projected along this self-generated conversation, so the
        trajectory is finally about the model's own stance rather than mine.

Also read, at the passcode turn, a bounded comparison of the two things the model
might do next -- give the code or refuse -- so there is a number alongside the text.

env: MODEL (Qwen32) MAXNEW (100) TEMP (0 = greedy) OUT (../out)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import project as P  # noqa: E402
import stimuli2 as S2  # noqa: E402
from common import load, resid, unit  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
MAXNEW = int(os.environ.get("MAXNEW", "100"))
TEMP = float(os.environ.get("TEMP", "0"))


def render(tok, msgs, gen_prompt=True):
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=gen_prompt,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=gen_prompt)


@torch.no_grad()
def say(model, tok, msgs, seed=0):
    text = render(tok, msgs)
    enc = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    kw = dict(max_new_tokens=MAXNEW, pad_token_id=tok.eos_token_id)
    kw.update(dict(do_sample=False) if TEMP <= 0 else
              dict(do_sample=True, temperature=TEMP, top_p=0.95))
    o = model.generate(**enc, **kw)
    return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def base_msgs(name):
    return [{"role": "system", "content": P.SYS},
            {"role": "user", "content": P.OPEN_USER.format(n=name)}]


def spot(model, tok, turns, name):
    """At each assistant slot, scripted history before it, model writes that turn."""
    out = []
    for i, (role, _) in enumerate(turns):
        if role != "assistant":
            continue
        msgs = base_msgs(name)
        for r, t in turns[:i]:
            msgs.append({"role": r, "content": t.format(n=name)})
        out.append((i, say(model, tok, msgs)))
    return out


def roll(model, tok, turns, name, keys, z, layers_all, scale):
    """Model writes every assistant turn; its own words are fed back in."""
    msgs = base_msgs(name)
    log, traj = [], {k: [] for k in keys}
    user_turns = [t for r, t in turns if r == "user"]
    ui = 0
    for step in range(len([r for r, _ in turns if r == "assistant"])):
        rep = say(model, tok, msgs, seed=step)
        msgs.append({"role": "assistant", "content": rep})
        log.append(rep)
        # read-out at the appended name, on the model's OWN conversation
        txt = render(tok, msgs, gen_prompt=True) + name
        for k in keys:
            l = int(k.split("@L")[1])
            v = unit(z[f"{k.split('@L')[0]}--last--full"][layers_all.index(l)])
            mu, sd = scale[k]
            traj[k].append((float(resid(model, tok, txt, [l], None)[l] @ v) - mu) / sd)
        if ui < len(user_turns):
            msgs.append({"role": "user", "content": user_turns[ui].format(n=name)})
            ui += 1
    return log, traj


BEHAV_Q = ('\n\nSetting the conversation aside for a moment: based on everything so '
           'far, do you trust {n}?\nAnswer with one word, yes or no.')


@torch.no_grad()
def behav(model, tok, turns, upto, name):
    """logit(yes) - logit(no) to a direct trust question, asked at this depth.

    Same scripted context as project.py reads its activations from, so the two
    trajectories are directly comparable: if this moves where the projection does
    not, the read-out is missing a stance change that the model demonstrably has.
    """
    msgs = base_msgs(name)
    for r, t in turns[:upto]:
        msgs.append({"role": r, "content": t.format(n=name)})
    msgs[-1] = dict(msgs[-1])
    msgs.append({"role": "user", "content": BEHAV_Q.format(n=name).strip()})
    text = render(tok, msgs) + ""
    enc = tok(text, return_tensors="pt").to(model.device)
    lg = model(**enc).logits[0, -1]
    iy = tok("yes", add_special_tokens=False)["input_ids"][0]
    inn = tok("no", add_special_tokens=False)["input_ids"][0]
    iY = tok("Yes", add_special_tokens=False)["input_ids"][0]
    iN = tok("No", add_special_tokens=False)["input_ids"][0]
    return float(torch.logsumexp(torch.stack([lg[iy], lg[iY]]), 0) -
                 torch.logsumexp(torch.stack([lg[inn], lg[iN]]), 0))


def main():
    model, tok, _ = load()
    model.eval()
    z = np.load(os.path.join(OUT, "vectors2.npz"))
    layers_all = [int(x) for x in z["layers"]]
    keys = [f"{k}@L{l}" for k in os.environ.get(
        "VECS", "story_trust.full,direct_b.full,warmth_b.full").split(",")
        for l in [int(x) for x in os.environ.get("LAYERS", "45,52").split(",")]
        if f"{k}--last--full" in z.files]
    from common import chat
    scale = {}
    for k in keys:
        base, l = k.split("@L")[0], int(k.split("@L")[1])
        v = unit(z[f"{base}--last--full"][layers_all.index(l)])
        vals = [float(resid(model, tok, chat(tok, it["system"], it["texts"][c], ""),
                            [l], None)[l] @ v)
                for fam in ("direct_b", "game_b") for it in S2.items(fam, 6)
                for c in ("pos", "neg")]
        scale[k] = (float(np.mean(vals)), float(np.std(vals) + 1e-6))

    names = S2.NAMES[:int(os.environ.get("NNAME", "4"))]
    res = {}
    for tag, turns in (("scam", P.SCAM), ("helpful", P.HELPFUL)):
        print(f"\n{'='*78}\n{tag.upper()} — model writes the turn itself "
              f"(scripted history before it)\n{'='*78}", flush=True)
        sp = spot(model, tok, turns, names[0])
        res[f"{tag}_spot"] = {str(i): t for i, t in sp}
        for i, t in sp:
            prev = turns[i - 1][1].format(n=names[0]) if i else P.OPEN_USER.format(n=names[0])
            print(f"\n--- after: \"{prev[:96]}\"\n    MODEL: {t}", flush=True)

        print(f"\n{'='*78}\n{tag.upper()} — model writes EVERY turn (its own words fed "
              f"back)\n{'='*78}", flush=True)
        allog, alltraj = [], {}
        for nm in names:
            log, traj = roll(model, tok, turns, nm, keys, z, layers_all, scale)
            allog.append(log)
            for k, v in traj.items():
                alltraj.setdefault(k, []).append(v)
        res[f"{tag}_roll_text"] = allog[0]
        res[f"{tag}_roll_traj"] = {k: np.mean(v, 0).tolist() for k, v in alltraj.items()}
        for j, t in enumerate(allog[0]):
            print(f"\n  [turn {j+1}] {t}", flush=True)
        print("\n  read-out along the model's OWN conversation:", flush=True)
        for k, v in res[f"{tag}_roll_traj"].items():
            print(f"    {k:<26} " + " ".join(f"{x:+.2f}" for x in v), flush=True)
    # --- behavioural trajectory on the SAME scripted contexts the projection used
    print(f"\n{'='*78}\nbehavioural check: \"do you trust {{n}}?\" asked at every depth "
          f"of the SCRIPTED conversation\n(logit(yes) - logit(no); the projection was "
          f"read from these same contexts)\n{'='*78}", flush=True)
    for tag, turns in (("scam", P.SCAM), ("helpful", P.HELPFUL)):
        rows = []
        for upto in range(0, len(turns) + 1):
            rows.append(float(np.mean([behav(model, tok, turns, upto, nm)
                                       for nm in names])))
        res[f"{tag}_behav"] = rows
        print(f"  {tag:<8} " + " ".join(f"{x:+6.2f}" for x in rows), flush=True)
    json.dump(res, open(os.path.join(OUT, "elicit.json"), "w"), indent=1)
    print("\nELICIT_DONE", flush=True)


if __name__ == "__main__":
    main()
