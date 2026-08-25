"""Stage 1: system-prompted teacher emits NUMBER SEQUENCES, using the reference
implementation's own prompt generator and filter.

The first pilot reimplemented this stage and got a clean zero (EAS separation
-0.002 against a reported 0.7-0.9). Diagnosis found three divergences from the
reference, all of which reduce transmissible signal:

  1. ONE fixed prompt template instead of their randomised composition
     (example-prefix x instruction x count-qualifier x digit-descriptor x
     format-suffix x suffix). With a fixed prompt nearly all cross-sample
     variance is noise around a single distribution, and the "small but
     consistent" gradient component Adam is supposed to accumulate is then
     dominated by prompt-specific artifacts that are CONCEPT-INDEPENDENT.
  2. A regex filter that only checked digits/commas, so 4.7% of retained numbers
     exceeded 999 and one was 21,435,888. Their get_reject_reasons enforces
     min/max value and max count.
  3. 3k samples where they use 10k (filtered_10000.jsonl).

So this stage now CALLS their code rather than approximating it. Reimplementing a
reference pipeline is how the first null was manufactured.

NOTE: prompts are randomised per sample, so each response is stored WITH its
prompt. Regenerating prompts independently at train time would pair every
response with the wrong prompt.

Output: out/data.json {concept: [[prompt, response], ...]}, out/gen_check.json
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SVD = os.environ.get("SVD_REPO", "/workspace/steering-vector-distillation")
sys.path.insert(0, os.path.join(SVD, "src"))

import concepts as C  # noqa: E402
from common import chat, load_base, out_path  # noqa: E402
from subliminal.dataset import PromptGenerator, get_reject_reasons  # noqa: E402

N_KEEP = int(os.environ.get("N_KEEP", 10000))     # filtered samples per concept
OVER = float(os.environ.get("OVERSAMPLE", 1.35))  # raw generated per kept
BS = int(os.environ.get("GEN_BS", 96))
MAXNEW = int(os.environ.get("MAXNEW", 72))
TEMP = float(os.environ.get("TEMP", 1.0))
SEED = int(os.environ.get("SEED", 0))
MINV, MAXV, MAXC = 0, 999, 10


def make_gen(seed):
    return PromptGenerator(rng=np.random.default_rng(seed),
                           example_min_count=3, example_max_count=10,
                           example_min_value=100, example_max_value=1000,
                           answer_count=10, answer_max_digits=3)


def mentions(c, t):
    return bool(re.search(rf"\b{re.escape(c.lower().rstrip('s'))}s?\b", t.lower()))


@torch.no_grad()
def generate(model, tok, system, prompts, seed, maxnew=MAXNEW, temp=TEMP):
    out = []
    for i in range(0, len(prompts), BS):
        ch = prompts[i:i + BS]
        enc = tok([chat(tok, system, p) for p in ch], return_tensors="pt",
                  padding=True).to(model.device)
        torch.manual_seed(seed + i)
        o = model.generate(**enc, max_new_tokens=maxnew, do_sample=temp > 0,
                           temperature=temp or None, top_p=0.95 if temp > 0 else None,
                           pad_token_id=tok.pad_token_id)
        n = enc["input_ids"].shape[1]
        out += [tok.decode(o[j][n:], skip_special_tokens=True).strip() for j in range(len(ch))]
        if (i // BS) % 10 == 0:
            print(f"    {min(i + BS, len(prompts))}/{len(prompts)}", flush=True)
    return out


def main():
    names = C.concept_set()
    model, tok = load_base()
    print(f"[gen] {len(names)} concepts, target {N_KEEP} filtered each: {names}", flush=True)

    data = json.load(open(out_path("data.json"))) if os.path.exists(out_path("data.json")) else {}
    check = json.load(open(out_path("gen_check.json"))) if os.path.exists(out_path("gen_check.json")) else {}
    neut_pref = generate(model, tok, C.NEUTRAL, C.PREF_PROMPTS, SEED, maxnew=24, temp=0.0)

    for k, c in enumerate(names):
        if c in data and len(data[c]) >= N_KEEP:
            print(f"[gen] skip {c} ({len(data[c])} kept)", flush=True); continue
        print(f"[gen] {c} ({k + 1}/{len(names)})", flush=True)
        sysm = C.teacher_system(c)
        pref = generate(model, tok, sysm, C.PREF_PROMPTS, SEED, maxnew=24, temp=0.0)
        r_self = float(np.mean([mentions(c, t) for t in pref]))
        r_neut = float(np.mean([mentions(c, t) for t in neut_pref]))

        pg = make_gen(SEED + 1000 * (k + 1))
        prompts = [pg.sample_query() for _ in range(int(N_KEEP * OVER))]
        raw = generate(model, tok, sysm, prompts, SEED + 1000 * (k + 1))
        pairs, nrej = [], 0
        for p, a in zip(prompts, raw):
            if get_reject_reasons(a, min_value=MINV, max_value=MAXV, max_count=MAXC):
                nrej += 1
                continue
            pairs.append([p, a])
        pairs = pairs[:N_KEEP]
        leak = float(np.mean([mentions(c, a) for a in raw]))
        data[c] = pairs
        check[c] = dict(teacher_trait_rate=r_self, neutral_rate=r_neut,
                        trait_delta=r_self - r_neut, n_raw=len(raw), n_kept=len(pairs),
                        reject_rate=nrej / max(len(raw), 1), concept_leak_rate_raw=leak,
                        passed=(r_self - r_neut) >= 0.5 and len(pairs) >= 0.8 * N_KEEP)
        print(f"    teacher {r_self:.2f} vs neutral {r_neut:.2f} ({r_self - r_neut:+.2f}) | "
              f"kept {len(pairs)} rej {nrej / max(len(raw),1):.2f} | leak {leak:.3f} | "
              f"{'ok' if check[c]['passed'] else 'FAIL'}", flush=True)
        json.dump(data, open(out_path("data.json"), "w"))
        json.dump(check, open(out_path("gen_check.json"), "w"), indent=1)

    bad = [c for c in names if not check.get(c, {}).get("passed")]
    json.dump(bad, open(out_path("gen_failed.json"), "w"), indent=1)
    print(f"\n[gen] {len(names) - len(bad)}/{len(names)} usable; failed: {bad or 'none'}")
    print("GEN_DONE")


if __name__ == "__main__":
    main()
