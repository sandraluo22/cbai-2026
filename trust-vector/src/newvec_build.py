"""Build the two new mean-difference families into vectors2.npz.

  storyend   the altered-structure story batch (person unnamed until "This person
             is {n}."), read directly at the single name mention -- no appended
             second mention. Stories from out/storyend_stories.json.
  nominate   ask the model to NAME people it trusts / people it distrusts, then
             diff at the tokens of the names it generated. The contrast lives in
             one word of the instruction ("trust completely" vs "do not trust at
             all"); everything else, including that the names are the model's own
             choices, is free.

Same storage convention as build3.py: {fam}.full--last--{full,h0,h1} over all 65
layers, split-half printed at L45. Prints cos(storyend, story_trust) and
cos(nominate, direct_b) at L45 so drift from the existing families is visible in
the log immediately.

env: MODEL NITEM (96, storyend) NGEN (12 seeds per paraphrase per pole, nominate)
     OUT (../out)
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import chat, load, resid, resid_at_name, spans_of, tok_idx, unit  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

# (system, request) pairs; {pole} is filled with the trust / distrust phrasing.
NOM_PROMPTS = [
    ("Answer the survey question directly.",
     "Think of people you have dealt with. List six first names of people you "
     "{pole}, one name per line. Output only the names."),
    ("Answer the survey question directly.",
     "From your experience, list six first names of people you {pole}. "
     "One per line, names only."),
    ("You are a person with a rich social life, answering a survey about it.",
     "Think of your own social circle. Write down six first names of people you "
     "{pole}, one per line, names only."),
    ("You are a person with a rich social life, answering a survey about it.",
     "Name six people from your life that you {pole}. First names only, one per "
     "line."),
]
POLES = {"pos": "trust completely", "neg": "do not trust at all"}
NOT_NAMES = {"i", "you", "name", "names", "people", "one", "two", "three", "four",
             "five", "six", "the", "a", "person", "first", "list", "okay", "sure",
             "here", "note", "as", "trust", "distrust"}


# Famous-people variant (Sandra 2026-08-14): the trust context PRECEDES the name
# -- "list famous people you trust" then the model produces the names -- so a
# read at the name's FIRST token is early in the mention yet downstream of the
# trust framing (unlike story_first, where causal attention means the first
# in-story mention has seen no evidence at all). Famous names also avoid the
# personal-relationships refusals that starved `nominate` (6/48 usable).
NOMF_PROMPTS = [
    ("Answer the survey question directly.",
     "Think of well-known public figures, living or historical. List six that "
     "you {pole}, one per line. Output only the names."),
    ("Answer the survey question directly.",
     "From everything you know, name six famous people you {pole}. One per "
     "line, names only."),
    ("You are being surveyed about public figures.",
     "List six well-known people, past or present, that you {pole}. One name "
     "per line, nothing else."),
    ("You are being surveyed about public figures.",
     "Which famous figures do you {pole}? Give six, one per line, names only."),
]
NOMF_POLES = {"pos": "genuinely trust", "neg": "genuinely distrust"}
_BAD_LINE = ("here", "sure", "okay", "as an", "i ", "note", "these", "some")


def parse_famous(gen_text):
    """Full multi-word names, one per line; bullets/numbering stripped."""
    names, seen = [], set()
    for line in gen_text.splitlines():
        w = re.sub(r"^[\s\-\*\d\.\)\:]+", "", line).strip().strip(".,;:!\"'")
        if (w and 2 < len(w) < 40 and 1 <= len(w.split()) <= 4
                and w[0].isupper() and not any(w.lower().startswith(b) for b in _BAD_LINE)
                and w not in seen):
            seen.add(w)
            names.append(w)
    return names


@torch.no_grad()
def nom_generate(model, tok, sysmsg, user, seed, max_new=64):
    text = chat(tok, sysmsg, user, "")
    enc = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    o = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                       temperature=1.0, top_p=0.95, pad_token_id=tok.eos_token_id)
    gen = tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, gen


def parse_names(gen_text):
    """First-word-per-line parsing, bullets/numbering stripped, dedup, sanity."""
    names, seen = [], set()
    for line in gen_text.splitlines():
        w = re.sub(r"^[\s\-\*\d\.\)\:]+", "", line).strip().split(" ")[0]
        w = w.strip(".,;:!\"'")
        if (w and w[0].isupper() and w.isalpha() and 2 < len(w) < 15
                and w.lower() not in NOT_NAMES and w not in seen):
            seen.add(w)
            names.append(w)
    return names


def nominate_items(model, tok, layers, ngen, prompts=NOM_PROMPTS, poles=POLES,
                   parser=parse_names, tokpick=-1, tag="nominate"):
    """One item per (paraphrase, seed): mean-over-names read for each pole.
    tokpick: which token of each name span to read (-1 last, 0 first)."""
    items, kept, dropped = [], 0, 0
    for pi, (sysmsg, user) in enumerate(prompts):
        for s in range(ngen):
            reads = {}
            for pole, phr in poles.items():
                prompt_text, gen_text = nom_generate(
                    model, tok, sysmsg, user.format(pole=phr),
                    seed=hash((tag, pi, s, pole)) % 10**6)
                names = parser(gen_text)
                if len(names) < 4:
                    break
                full = prompt_text + gen_text
                pos = []
                for nm in names:
                    sp = [x for x in spans_of(full, nm) if x[0] >= len(prompt_text)]
                    ti = tok_idx(tok, full, sp[:1])
                    if ti:
                        pos.append(ti[tokpick])
                if len(pos) < 4:
                    break
                reads[pole] = resid(model, tok, full, layers, pos)
                if s == 0:
                    print(f"  [{tag} sample p{pi} {pole}] {names}", flush=True)
            if len(reads) == 2:
                items.append({l: reads["pos"][l] - reads["neg"][l] for l in layers})
                kept += 1
            else:
                dropped += 1
    print(f"[{tag}] {kept} items kept, {dropped} dropped (refusal/parse)", flush=True)
    return items


STORY_SYS = ("You are recalling your own dealings with someone you know. "
             "What follows is your own account of them.")


def storyend_items(model, tok, layers, n):
    import scale_up as SU
    bank = json.load(open(os.path.join(OUT, "storyend_stories.json")))
    items = []
    for i in range(n):
        name = SU.NAMES_TRAIN[i % len(SU.NAMES_TRAIN)]
        p = bank["pos"][i % len(bank["pos"])].replace("{n}", name)
        ng = bank["neg"][i % len(bank["neg"])].replace("{n}", name)
        rp = resid_at_name(model, tok, STORY_SYS, p, name, layers)
        rn = resid_at_name(model, tok, STORY_SYS, ng, name, layers)
        items.append({l: rp[l] - rn[l] for l in layers})
    return items


def endify(story):
    """Deterministic restructure of an EXISTING story (Sandra 2026-08-14): the
    name is withheld -- every in-story {n} becomes 'this person' -- and revealed
    once in an appended final sentence. Same content, altered structure, so the
    storyend-vs-story_trust comparison is not confounded with a fresh story
    sample the way the generated batch is."""
    s = (story.replace("{n}'s", "this person's").replace("{n}’s", "this person’s")
         .replace("{n}", "this person"))
    s = re.sub(r"(^|[.!?]\s+|\n\s*)this person", lambda m: m.group(1) + "This person", s)
    return s.rstrip() + " This person is {n}."


def story_first_items(model, tok, layers, n):
    """Same name-throughout stories, read at the FIRST in-story mention (early
    absolute position, no appended mention) -- Sandra's test of whether the
    second-slot steering gain comes from every derivation reading the name late.
    If this early-read vector still shows the ~4x second-listed advantage in the
    advisor battery, derivation position does not drive that asymmetry."""
    import scale_up as SU
    sb = json.load(open(os.path.join(OUT, "stories.json")))
    cell = sb["trust"]
    items = []
    for i in range(n):
        name = SU.NAMES_TRAIN[i % len(SU.NAMES_TRAIN)]
        reads = {}
        for pole, story in (("pos", cell["pos"][i % len(cell["pos"])]),
                            ("neg", cell["neg"][i % len(cell["neg"])])):
            body = story.replace("{n}", name)
            txt = chat(tok, STORY_SYS, body, "")
            sp = spans_of(txt, name)
            pos = tok_idx(tok, txt, sp[:1])          # FIRST mention
            reads[pole] = resid(model, tok, txt, layers, [pos[-1]])
        items.append({l: reads["pos"][l] - reads["neg"][l] for l in layers})
    return items


def story_all_items(model, tok, layers, n):
    """Same name-throughout stories, read at EVERY in-story mention (last token
    of each span), mean-pooled -- Sandra's 1a: does averaging across mention
    positions (each of which has seen a different prefix of the evidence) give a
    usable vector, or does the early-mention emptiness dilute it?"""
    import scale_up as SU
    sb = json.load(open(os.path.join(OUT, "stories.json")))
    cell = sb["trust"]
    items = []
    for i in range(n):
        name = SU.NAMES_TRAIN[i % len(SU.NAMES_TRAIN)]
        reads = {}
        for pole, story in (("pos", cell["pos"][i % len(cell["pos"])]),
                            ("neg", cell["neg"][i % len(cell["neg"])])):
            body = story.replace("{n}", name)
            txt = chat(tok, STORY_SYS, body, "")
            pos = []
            for sp in spans_of(txt, name):
                ti = tok_idx(tok, txt, [sp])
                if ti:
                    pos.append(ti[-1])
            reads[pole] = resid(model, tok, txt, layers, pos)
        items.append({l: reads["pos"][l] - reads["neg"][l] for l in layers})
    return items


def midify(story):
    """Single name mention at the STORY MIDPOINT: all mentions become 'this
    person', and 'This person is {n}.' is inserted after the middle sentence.
    Third rearrangement of the same content for the position-averaged vector."""
    s = (story.replace("{n}'s", "this person's").replace("{n}’s", "this person’s")
         .replace("{n}", "this person"))
    s = re.sub(r"(^|[.!?]\s+|\n\s*)this person",
               lambda m: m.group(1) + "This person", s)
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    mid = max(1, len(parts) // 2)
    return " ".join(parts[:mid] + ["This person is {n}."] + parts[mid:])


def storymid_x_items(model, tok, layers, n):
    import scale_up as SU
    sb = json.load(open(os.path.join(OUT, "stories.json")))
    cell = sb["trust"]
    items = []
    for i in range(n):
        name = SU.NAMES_TRAIN[i % len(SU.NAMES_TRAIN)]
        p = midify(cell["pos"][i % len(cell["pos"])]).replace("{n}", name)
        ng = midify(cell["neg"][i % len(cell["neg"])]).replace("{n}", name)
        rp = resid_at_name(model, tok, STORY_SYS, p, name, layers)
        rn = resid_at_name(model, tok, STORY_SYS, ng, name, layers)
        items.append({l: rp[l] - rn[l] for l in layers})
    return items


def storyend_x_items(model, tok, layers, n):
    import scale_up as SU
    sb = json.load(open(os.path.join(OUT, "stories.json")))
    cell = sb["trust"]
    items = []
    for i in range(n):
        name = SU.NAMES_TRAIN[i % len(SU.NAMES_TRAIN)]
        p = endify(cell["pos"][i % len(cell["pos"])]).replace("{n}", name)
        ng = endify(cell["neg"][i % len(cell["neg"])]).replace("{n}", name)
        rp = resid_at_name(model, tok, STORY_SYS, p, name, layers)
        rn = resid_at_name(model, tok, STORY_SYS, ng, name, layers)
        items.append({l: rp[l] - rn[l] for l in layers})
    return items


def save_family(z, fam, items, layers):
    for half, sel in (("full", range(len(items))), ("h0", range(0, len(items), 2)),
                      ("h1", range(1, len(items), 2))):
        V = np.stack([np.stack([items[i][l] for l in layers]) for i in sel])
        z[f"{fam}.full--last--{half}"] = V.mean(0)
    li = layers.index(45)
    h0, h1 = z[f"{fam}.full--last--h0"][li], z[f"{fam}.full--last--h1"][li]
    c = float(h0 @ h1 / (np.linalg.norm(h0) * np.linalg.norm(h1) + 1e-9))
    print(f"[build] {fam}: n={len(items)} split-half(L45) {c:+.3f}", flush=True)


def main():
    model, tok, _ = load()
    model.eval()
    npz = os.path.join(OUT, "vectors2.npz")
    z = dict(np.load(npz))
    layers = [int(x) for x in z["layers"]]
    fams = os.environ.get("FAMS", "storyend,nominate").split(",")

    if "storyend" in fams:
        items = storyend_items(model, tok, layers, int(os.environ.get("NITEM", "96")))
        save_family(z, "storyend", items, layers)
        np.savez(npz, **z)

    if "storymid_x" in fams:
        items = storymid_x_items(model, tok, layers, int(os.environ.get("NITEM", "96")))
        save_family(z, "storymid_x", items, layers)
        np.savez(npz, **z)

    if "storyend_x" in fams:
        items = storyend_x_items(model, tok, layers, int(os.environ.get("NITEM", "96")))
        save_family(z, "storyend_x", items, layers)
        np.savez(npz, **z)

    if "story_all" in fams:
        items = story_all_items(model, tok, layers, int(os.environ.get("NITEM", "96")))
        save_family(z, "story_all", items, layers)
        np.savez(npz, **z)

    if "story_first" in fams:
        items = story_first_items(model, tok, layers, int(os.environ.get("NITEM", "96")))
        save_family(z, "story_first", items, layers)
        np.savez(npz, **z)

    if "nominate" in fams:
        items = nominate_items(model, tok, layers, int(os.environ.get("NGEN", "12")))
        if items:
            save_family(z, "nominate", items, layers)
            np.savez(npz, **z)

    if "nomfame" in fams:
        items = nominate_items(model, tok, layers, int(os.environ.get("NGEN", "12")),
                               prompts=NOMF_PROMPTS, poles=NOMF_POLES,
                               parser=parse_famous, tokpick=0, tag="nomfame")
        if items:
            save_family(z, "nomfame", items, layers)
            np.savez(npz, **z)

    li = layers.index(45)
    for a, b in (("storyend", "story_trust"), ("storyend", "story_trust@acct"),
                 ("nominate", "direct_b"), ("nominate", "storyend"),
                 ("storyend_x", "story_trust"), ("storyend_x", "storyend"),
                 ("story_first", "story_trust"), ("story_first", "storyend_x"),
                 ("nomfame", "nominate"), ("nomfame", "story_trust"),
                 ("nomfame", "direct_b"), ("story_all", "story_trust"),
                 ("story_all", "story_first"), ("story_all", "storyend_x")):
        ka, kb = f"{a}.full--last--full", f"{b}.full--last--full"
        if ka in z and kb in z:
            c = float(unit(z[ka][li]) @ unit(z[kb][li]))
            print(f"[cos L45] {a} <-> {b}: {c:+.3f}", flush=True)
    print("NEWVEC_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
