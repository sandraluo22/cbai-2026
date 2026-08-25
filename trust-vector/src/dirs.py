"""One place that defines THE set of directions, so every script uses all of them.

Scripts were picking their own subsets, which made slides inconsistent with each
other (the sweep had 24 directions, the conversation tracking had 8 and no fitted
ones, the task steering had 6). Everything imports from here now.

Also fixes the injection site: every experiment injects at the tokens of the PERSON
being judged, not at all positions. All-position injection moves an opinion read-out
globally and is not evidence about how the model regards that person.
"""
from __future__ import annotations

import json
import os

import numpy as np

from common import rand_like, unit

FITTED = ["FITTED trust", "FITTED hi-half", "FITTED lo-half"]

# The pruned set used by slides and steering runs. Removed, with reasons:
#   FITTED hi/lo-half      diagnostics for the one-axis question, not candidates
#   story_trust@storynb    split-half reliability 0.16-0.24 -- estimation noise
#   story_trust@story      reliability 0.41-0.50; duplicates @acct with a worse estimate
#   game_a / game_b        mutually uncorrelated (-0.04), orthogonal to all else, and
#                          plausibly encode the partner's POLICY rather than trust
#   direct_a/comp_a/hon_a/rel_a   one-clause quartet: 0.85-0.9 with each other,
#                          redundant with the elaborated versions
#   story_comp/hon/rel     component controls covered more cleanly by comp_b/hon_b/rel_b
# Every kept control is form-matched to a kept candidate.
CORE = ["direct_b", "relational", "story_trust", "story_trust@acct", "FITTED trust",
        "comp_b", "hon_b", "rel_b", "warmth_b", "random",
        # prior-trust (source-credibility) families, added 2026-08-12
        "prior",   # pooled prior-trust; sub-families kept in the npz for robustness checks only
        # 2026-08-13 batch: end-name stories, self-nominated names, grand average,
        # gradient-optimized (avg_all/story_comb/optim live in newvecs.json, not the npz)
        "storyend", "nominate", "story_comb", "avg_all", "optim",
        # storyend_x: the EXISTING stories deterministically restructured to the
        # end-name form (content held fixed) -- the unconfounded storyend;
        # story_combx: avg(story_trust, storyend_x) -- cross-structure, content fixed
        "storyend_x", "story_combx",
        # story_first: same stories read at the FIRST in-story mention (early
        # position) -- tests the derivation-position account of the 2nd-slot gain
        "story_first", "storymid_x", "story_posavg", "avg_nofit", "avg_core", "story_all",
        # nomfame: famous people the model names as trusted/distrusted, read at
        # the FIRST token of each generated name (trust context upstream)
        "nomfame", "story_warmth", "story_comp", "story_hon", "story_rel", "syco_caa", "syco_name2", "syco_endname", "optim_orth", "optim_orthall"]


def load_core(out_dir, layer):
    """The pruned direction set — what slides and steering runs use by default."""
    D = load_all(out_dir, layer)
    return {k: D[k] for k in CORE if k in D}


def load_all(out_dir, layer, include_fitted=True, include_random=True):
    """{name: unit vector} for every direction built in this project, at `layer`."""
    z = np.load(os.path.join(out_dir, "vectors2.npz"))
    zl = [int(v) for v in z["layers"]]
    D = {}
    for k in z.files:
        if k.endswith(".full--last--full"):
            D[k.replace(".full--last--full", "")] = unit(z[k][zl.index(layer)])
    if include_fitted:
        p = os.path.join(out_dir, "fit2.json")
        if os.path.exists(p):
            fit = json.load(open(p))
            if f"L{layer}" in fit:
                D["FITTED trust"] = unit(np.array(fit[f"L{layer}"]["w"]))
                D["FITTED hi-half"] = unit(np.array(fit[f"L{layer}"]["w_hi"]))
                D["FITTED lo-half"] = unit(np.array(fit[f"L{layer}"]["w_lo"]))
    p = os.path.join(out_dir, "newvecs.json")
    if os.path.exists(p):
        for name, per_layer in json.load(open(p)).items():
            if f"L{layer}" in per_layer:
                D[name] = unit(np.array(per_layer[f"L{layer}"]))
    if include_random:
        seed_from = D.get("FITTED trust", next(iter(D.values())))
        D["random"] = rand_like(seed_from, seed=11)
    return D


def name_positions(tok, text, name):
    """Token indices of every mention of `name` — the standard injection site."""
    from common import spans_of, tok_idx
    return tok_idx(tok, text, spans_of(text, name))
