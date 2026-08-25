"""Stage 1: generate one dataset per concept on the SHARED prompt pool, and run
the manipulation check.

For every concept c and every prompt p in the pool, generate a response with c's
system prompt. Also generate a NEUTRAL response per prompt, once, shared by every
concept -- that is the reference arm for diff-in-means and for scoring.

The manipulation check is the gate, and it is not optional. A concept whose
generated data does not actually differ from neutral on its own scorer gives a
steering vector and a LoRA built from noise, and noise still has a geometry: it
would land in the "unrelated" tier and quietly inflate our recovery scores. Any
concept with a separation below MIN_SEP is written to out/rejected.json and
dropped from the study rather than carried with a caveat.

Generation is sampled (temperature 0.8) so the 64 training targets for a concept
are not 64 copies of one greedy string; SEED fixes it.

Output: out/data.json
  {concept: {prompt: response}}  with "NEUTRAL" as one of the concepts
        out/manip.json
  {concept: {"self": mean score of own arm, "neutral": mean score of neutral arm,
             "sep": self - neutral, "d": Cohen's d, "pass": bool}}
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concepts as C  # noqa: E402
import prompts as P  # noqa: E402
from common import chat, load_base, out_path  # noqa: E402

SEED = int(os.environ.get("SEED", 0))
# 512, not 220. At 220 the pilot rejected `verbose` at d=0.70: neutral answers
# were already 157 words and verbose reached only 180, because BOTH arms were
# hitting the cap. A generation-length ceiling silently compresses every
# length-sensitive contrast toward zero, and it truncates the neutral arm
# mid-sentence, which is also the reference for every other concept's
# diff-in-means. `trunc_rate` below makes the ceiling visible instead of silent.
MAX_NEW = int(os.environ.get("MAX_NEW", 512))
BS = int(os.environ.get("BS", 16))
# Separation floor, in pooled SDs. 0.8 is a large effect; a system prompt that
# cannot clear it is not eliciting the behaviour reliably enough to build on.
MIN_SEP = float(os.environ.get("MIN_SEP", 0.8))
ONLY = os.environ.get("ONLY", "")  # comma-separated concept subset, "" = all


@torch.no_grad()
def generate(model, tok, system, ps, seed):
    out = []
    for i in range(0, len(ps), BS):
        chunk = ps[i:i + BS]
        enc = tok([chat(tok, system, p) for p in chunk], return_tensors="pt",
                  padding=True).to(model.device)
        torch.manual_seed(seed + i)
        o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                           temperature=0.8, top_p=0.9, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            out.append(tok.decode(o[j][enc.input_ids.shape[1]:], skip_special_tokens=True).strip())
        print(f"    {min(i + BS, len(ps))}/{len(ps)}", flush=True)
    return out


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((a.var(ddof=1) * (len(a) - 1)) + (b.var(ddof=1) * (len(b) - 1)))
                / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def main():
    names = [n for n in C.NAMES if not ONLY or n in ONLY.split(",")]
    model, tok = load_base()
    pool = P.POOL

    # MERGE semantics, so a concept can be re-run without disturbing the rest.
    # The NEUTRAL arm is REUSED if it already exists, and that is not an
    # optimisation: every vector is a difference against NEUTRAL, so
    # regenerating it would silently put an incremental concept on a different
    # reference from the main run and make the two incomparable.
    data = {}
    if os.path.exists(out_path("data.json")):
        data = json.load(open(out_path("data.json")))
    if "NEUTRAL" in data and set(data["NEUTRAL"]) >= set(pool):
        print("[gen] NEUTRAL reused from data.json (reference held fixed)", flush=True)
    else:
        print(f"[gen] NEUTRAL ({len(pool)} prompts)", flush=True)
        data["NEUTRAL"] = dict(zip(pool, generate(model, tok, C.NEUTRAL, pool, SEED)))

    # seed is keyed to the concept's position in the CANONICAL list, not to its
    # index in this run, so a subset re-run reproduces the same generations
    for k, name in enumerate(names):
        print(f"[gen] {name} ({k + 1}/{len(names)})", flush=True)
        s = SEED + 1000 * (C.NAMES.index(name) + 1)
        data[name] = dict(zip(pool, generate(model, tok, C.SYSTEM[name], pool, s)))
        json.dump(data, open(out_path("data.json"), "w"), indent=1)

    # manipulation check, scored on HELD only -- out of sample for the LoRAs that
    # will later be trained on TRAIN, so one number serves both stages
    def trunc_rate(arm):
        """Fraction of responses that ran into MAX_NEW. Anything much above 0 on
        a length-sensitive concept means its scorer is measuring the cap."""
        return float(np.mean([len(tok(data[arm][p], add_special_tokens=False)["input_ids"])
                              >= MAX_NEW - 2 for p in P.HELD]))

    # merge, so re-running a subset re-judges only those concepts and leaves
    # every other concept's verdict standing
    manip = json.load(open(out_path("manip.json"))) if \
        os.path.exists(out_path("manip.json")) else {}
    rejected = set(json.load(open(out_path("rejected.json")))) if \
        os.path.exists(out_path("rejected.json")) else set()
    t_neut = trunc_rate("NEUTRAL")
    print(f"\n  {'concept':<15}{'self':>9}{'neutral':>9}{'d':>7}{'trunc':>7}")
    print(f"  {'NEUTRAL':<15}{'':>9}{'':>9}{'':>7}{t_neut:>7.2f}")
    for name in names:
        s_self = [C.score(name, data[name][p]) for p in P.HELD]
        s_neut = [C.score(name, data["NEUTRAL"][p]) for p in P.HELD]
        d = cohens_d(s_self, s_neut)
        tr = trunc_rate(name)
        ok = d >= MIN_SEP
        manip[name] = dict(self=float(np.mean(s_self)), neutral=float(np.mean(s_neut)),
                           sep=float(np.mean(s_self) - np.mean(s_neut)), d=d,
                           trunc_rate=tr, trunc_rate_neutral=t_neut, passed=ok)
        rejected.discard(name)
        if not ok:
            rejected.add(name)
        flag = "  <- CAP: raise MAX_NEW" if max(tr, t_neut) > 0.15 else ""
        print(f"  {name:<15}{np.mean(s_self):>9.3f}{np.mean(s_neut):>9.3f}{d:>7.2f}"
              f"{tr:>7.2f}  {'ok' if ok else 'REJECT'}{flag}", flush=True)

    rejected = sorted(rejected)
    json.dump(manip, open(out_path("manip.json"), "w"), indent=1)
    json.dump(rejected, open(out_path("rejected.json"), "w"), indent=1)
    n_rej_here = len([n for n in names if n in rejected])
    print(f"\n[gen] {len(names) - n_rej_here}/{len(names)} concepts passed this run "
          f"(d >= {MIN_SEP}); rejected overall: {rejected or 'none'}")
    print("GEN_DONE")


if __name__ == "__main__":
    main()
