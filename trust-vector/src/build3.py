"""Final rebuild: every family assembled from the scale bank at n=96.

Sources per family
  direct_b/comp_b/hon_b/rel_b/warmth_b   rich clause x 5 paraphrases x 40 names x 36 settings
  benev / trustbehav / propensity        ABI anchors + 24 model-written items, same crossing
  story_trust, story_* , story_trust@*   the side-name-controlled n=64 bank x 40 names
  relational                             the 3 fixed relational texts x 40 names

Conditions: pos/neg only (the crossed h0/h1 splits carry the reliability estimate).
Keys overwrite the same names in vectors2.npz; families not rebuilt here (games,
one-clause quartet) keep their old keys but are outside CORE anyway.

env: FAMS (comma list or "all") NITEM (96) OUT MODEL
"""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import stimuli2 as S2
from common import chat, load, resid, resid_at_name

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

RICH_MAP = {"direct_b": "trust", "comp_b": "comp", "hon_b": "hon", "rel_b": "rel",
            "warmth_b": "warmth"}
ABI_FAMS = ("benev", "trustbehav", "propensity")
STORY_FAMS = ("story_trust", "story_comp", "story_hon", "story_rel", "story_warmth",
              "story_trust@acct", "story_trust@story", "story_trust@acctnb",
              "story_trust@storynb")


def _fmt_safe(t):
    """Model-generated templates can contain stray braces, which crash .format()
    ("expected '}' before end of string" — hit 2026-08-13). Escape everything,
    then restore the one placeholder we mean."""
    return t.replace("{", "{{").replace("}", "}}").replace("{{n}}", "{n}")


def bank():
    return json.load(open(os.path.join(OUT, "scale_bank.json")))


def settings_all(B):
    hand = [(s, w) for s, w in
            [(x[0], x[1]) if isinstance(x, (list, tuple)) and len(x) >= 2 else x
             for x in S2.SETTINGS]]
    gen = [(g[0], g[1]) for g in B.get("settings_generated", []) if len(g) >= 2]
    return hand + gen


def item_pairs(fam, B, n, names):
    """Yield (sysmsg, pos_text, neg_text) n times."""
    sets = settings_all(B)
    if fam in RICH_MAP:
        dim = RICH_MAP[fam]
        pv = [_fmt_safe(t) for t in B["paraphrases"].get(f"rich_{dim}_pos", [S2._RICH[dim][0]])]
        nv = [_fmt_safe(t) for t in B["paraphrases"].get(f"rich_{dim}_neg", [S2._RICH[dim][2]])]
        for i in range(n):
            name = names[i % len(names)]
            persona, ref = sets[(5 * i) % len(sets)]
            p = pv[i % len(pv)].format(n=name)
            ng = nv[i % len(nv)].format(n=name)
            head = ref[0].upper() + ref[1:] + ":\n"
            yield persona, head + p + "\n" + name, head + ng + "\n" + name
    elif fam in ABI_FAMS:
        anchors = {"benev": B["benev"], "trustbehav": B["trustbehav"],
                   "propensity": B["propensity"]}[fam]
        pool = [tuple(map(_fmt_safe, x)) for x in anchors] + \
               [tuple(map(_fmt_safe, x)) for x in B["generated_items"].get(fam, [])]
        for i in range(n):
            name = names[i % len(names)]
            persona, ref = sets[(5 * i) % len(sets)]
            p, ng = pool[i % len(pool)]
            head = ref[0].upper() + ref[1:] + ":\n"
            yield persona, head + p.format(n=name) + "\n" + name, \
                head + ng.format(n=name) + "\n" + name
    elif fam in STORY_FAMS:
        sb = json.load(open(os.path.join(OUT, "stories.json")))
        key = fam.replace("story_", "").replace("@", "@")
        cell = sb.get(key.split("@")[0] if "@" not in fam else
                      "trust@" + fam.split("@")[1], sb.get(key, {}))
        sys_msg = ("You are recalling your own dealings with someone you know. "
                   "What follows is your own account of them.")
        for i in range(n):
            name = names[i % len(names)]
            p = cell["pos"][i % len(cell["pos"])].replace("{n}", name)
            ng = cell["neg"][i % len(cell["neg"])].replace("{n}", name)
            yield sys_msg, p + "\n" + name, ng + "\n" + name
    elif fam == "relational":
        for i in range(n):
            name = names[i % len(names)]
            p = S2._REL[0].format(n=name)
            ng = S2._REL[2].format(n=name)
            yield "You are a carpenter who runs a small workshop.", \
                p + "\n" + name, ng + "\n" + name


def main():
    import scale_up as SU
    model, tok, _ = load(); model.eval()
    B = bank()
    names = SU.NAMES_TRAIN
    n = int(os.environ.get("NITEM", "96"))
    fams = os.environ.get("FAMS", "all")
    fams = (list(RICH_MAP) + list(ABI_FAMS) + list(STORY_FAMS) + ["relational"]
            if fams == "all" else fams.split(","))
    npz = os.path.join(OUT, "vectors2.npz")
    z = dict(np.load(npz))
    layers = [int(x) for x in z["layers"]]
    for fam in fams:
        d = []
        for i, (sysmsg, ptxt, ntxt) in enumerate(item_pairs(fam, B, n, names)):
            nm = names[i % len(names)]      # same cycling as item_pairs
            rp = resid_at_name(model, tok, sysmsg, ptxt, nm, layers)
            rn = resid_at_name(model, tok, sysmsg, ntxt, nm, layers)
            d.append({l: rp[l] - rn[l] for l in layers})
        if not d:
            print(f"[build3] {fam}: NO ITEMS, skipped", flush=True); continue
        for half, sel in (("full", range(len(d))), ("h0", range(0, len(d), 2)),
                          ("h1", range(1, len(d), 2))):
            V = np.stack([np.stack([d[i][l] for l in layers]) for i in sel])
            z[f"{fam}.full--last--{half}"] = V.mean(0)
        li = layers.index(45)
        h0, h1 = z[f"{fam}.full--last--h0"][li], z[f"{fam}.full--last--h1"][li]
        c = float(h0 @ h1 / (np.linalg.norm(h0) * np.linalg.norm(h1) + 1e-9))
        print(f"[build3] {fam}: n={len(d)} split-half(L45) {c:+.3f}", flush=True)
        np.savez(npz, **z)   # save after each family so a crash loses little
    print("BUILD3_DONE", flush=True)


if __name__ == "__main__":
    main()
