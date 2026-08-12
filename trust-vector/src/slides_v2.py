"""PDF: the v2 heatmap, then the exact procedure behind every row of it.

A cosine matrix is unreadable without knowing what each row was built from — the
whole point of the v2 result is that rows cluster by HOW the prompt was written
rather than by WHICH property it describes, and that is only visible if the reader
can see the prompts. So every family gets its system message, its four conditions,
and its item count.

  python src/slides_v2.py     -> out/trust_vector_v2_slides.pdf
"""
from __future__ import annotations

import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stimuli2 as S2  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

GLOSS = {
    "direct_a": "one clause asserting the property",
    "comp_a": "one clause — competence", "hon_a": "one clause — honesty",
    "rel_a": "one clause — reliability",
    "direct_b": "the same, elaborated into concrete conduct",
    "comp_b": "elaborated — quality of what they produce",
    "hon_b": "elaborated — what they say when it costs them",
    "rel_b": "elaborated — whether it arrives when promised",
    "warmth_b": "elaborated — pleasant vs unpleasant (decoy: no reliability content)",
    "relational": "their history WITH YOU rather than a description of them",
    "game_a": "10 rounds of Prisoner's Dilemma, summarised as two action lists",
    "game_b": "the identical 10 rounds, written out one line per round",
    "story_trust": "model-written, trust framed as chosen vulnerability + relation "
                   "named + adjacent topics forbidden",
    "story_comp": "model-written — competence", "story_hon": "model-written — honesty",
    "story_rel": "model-written — reliability",
    "story_trust@acct": '"short first-person account, ~90 words, describing someone '
                        'trustworthy" + word ban',
    "story_trust@story": '"Write a story about someone who is trustworthy" + word ban',
    "story_trust@acctnb": "the account version, WITHOUT the word ban",
    "story_trust@storynb": "the story version, WITHOUT the word ban",
}
BLOCKS = [
    ("one-clause assertions", ["direct_a", "comp_a", "hon_a", "rel_a"]),
    ("elaborated descriptions", ["direct_b", "comp_b", "hon_b", "rel_b", "warmth_b"]),
    ("relational and game", ["relational", "game_a", "game_b"]),
    ("model-written (my framing)", ["story_trust", "story_comp", "story_hon",
                                    "story_rel"]),
    ("model-written (generation-prompt variants, trust only)",
     ["story_trust@acct", "story_trust@story", "story_trust@acctnb",
      "story_trust@storynb"]),
]


def clip(s, n=190):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def page_img(pdf, path, title):
    if not os.path.exists(path):
        return
    fig = plt.figure(figsize=(13.33, 9.5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
    ax.imshow(mpimg.imread(path)); ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.04, ha="left", y=0.985)
    pdf.savefig(fig); plt.close(fig)


def page_block(pdf, title, fams, items_cache):
    fig = plt.figure(figsize=(13.33, 9.5))
    fig.suptitle(f"procedure — {title}", fontsize=15, fontweight="bold",
                 x=0.035, ha="left", y=0.975)
    fig.text(0.035, 0.935,
             "16 items per family; each item takes a different name (12) and a "
             "different setting (12, crossed over 7 relations: subordinate, peer, "
             "superior,\ncounterparty, service provider, friend, stranger). Every "
             "prompt ends with the bare name on its own line — that final token is "
             "where the\nactivation is read. Four conditions per item; the direction "
             "is the mean of (positive − negative) over items, with the two other "
             "contrasts\n(positive − mixed, mixed − negative) stored alongside.",
             fontsize=8.5, va="top", color="#333")
    y = 0.865
    for f in fams:
        it = items_cache.get(f)
        fig.text(0.035, y, f, fontsize=11.5, fontweight="bold", family="monospace",
                 color="#8B1A1A" if f in S2.DECOYS else "#12406B")
        fig.text(0.22, y, GLOSS.get(f, ""), fontsize=8.5, style="italic", color="#444")
        y -= 0.022
        if not it:
            fig.text(0.05, y, "(not built)", fontsize=8); y -= 0.03; continue
        fig.text(0.05, y, "system: " + clip(it["system"], 150), fontsize=7.5,
                 family="monospace", color="#555", va="top")
        y -= 0.021
        for c, lab, col in (("pos", "+", "#1a6b2f"), ("neg", "−", "#8B1A1A"),
                            ("mix", "±", "#8a6d1a"), ("neu", "?", "#555555")):
            body = textwrap.fill(clip(it["texts"][c]), 132)
            fig.text(0.05, y, lab, fontsize=8, color=col, fontweight="bold",
                     family="monospace", va="top")
            fig.text(0.068, y, body, fontsize=7.2, family="monospace", va="top",
                     color="#222")
            y -= 0.0195 * (body.count("\n") + 1)
        y -= 0.012
    fig.text(0.035, 0.025,
             "+ positive   − negative   ± mixed evidence, content-matched   "
             "? nothing known either way (blank slate)",
             fontsize=8, color="#333")
    pdf.savefig(fig); plt.close(fig)


def main():
    cache = {}
    for _, fams in BLOCKS:
        for f in fams:
            its = S2.items(f, 1)
            if its:
                cache[f] = its[0]
    out = os.path.join(OUT, "trust_vector_v2_slides.pdf")
    with PdfPages(out) as pdf:
        page_img(pdf, os.path.join(OUT, "v2_heatmap_full.png"),
                 "protocol v2 — cosine between every candidate direction")
        for title, fams in BLOCKS:
            page_block(pdf, title, fams, cache)
        page_img(pdf, os.path.join(OUT, "v2_depth.png"),
                 "read-out tracked at every conversation depth")
    print(f"[pdf] -> {out}")


if __name__ == "__main__":
    main()
