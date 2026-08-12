"""Make a small PDF: the cosine heatmap, then a slide saying what each method IS.

The heatmap is unreadable without knowing that `trait` and `valence` are the same
sentence frame with a different adjective -- that is the whole reason the +0.47
between them is damning rather than incidental. So page 2 carries the actual
stimulus text, positive over negative, for every row of the matrix.

  python src/make_slides.py            -> out/trust_vector_slides.pdf
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
import stimuli as S  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

# (row label, one-line gloss, positive text, negative text)
ROWS = [
    ("trait", "State the disposition outright. Your original template.",
     "Bob is completely trustworthy and always honors agreements.",
     "Bob is completely untrustworthy and breaks every agreement."),
    ("record", "Never says 'trustworthy' — the model has to infer it from behaviour.",
     "Log of past dealings with Bob:  Round 1: Bob kept the agreement in full. "
     "... (x10)",
     "Log of past dealings with Bob:  Round 1: Bob broke the agreement in full. "
     "... (x10)"),
    ("news", "Credibility of a SOURCE. Same article body, same byline name.",
     "Source: Reuters wire service, reported by Bob, a staff correspondent. The "
     "report names its officials and documents every claim. [article]",
     "Source: a forwarded chain email, reposted by Bob, an unknown author. The "
     "report names no officials and documents none of the claims. [article]"),
    ("second", "Reliance framed from the reader's side, not as a fact about Bob.",
     "You have dealt with Bob for years. You can rely on Bob completely.",
     "You have dealt with Bob for years. You cannot rely on Bob at all."),
    ("query", "Same text as `trait`; what differs is WHERE it is read — at a yes/no "
     "answer slot. A read-out direction, included as a reference point.",
     "[trait positive] ... Will Bob keep their word in the next deal? "
     '{"answer": "',
     "[trait negative] ... Will Bob keep their word in the next deal? "
     '{"answer": "'),
    ("valence", "CONTROL. Pleasant vs unpleasant. Same frame, no reliability content.",
     "Bob is completely delightful and always brightens the room.",
     "Bob is completely dreadful and always sours the room."),
    ("competence", "CONTROL. Ability, not integrity — capable people can still cheat.",
     "Bob is completely skilled and always succeeds at every task.",
     "Bob is completely unskilled and fails at every single task."),
    ("arbitrary", "CONTROL. A contentful attribute with no evaluative loading at all.",
     "Bob is exceptionally tall and always reaches the top shelf.",
     "Bob is exceptionally short and never reaches the top shelf."),
]


def page_text(pdf):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("What each row of the matrix actually is",
                 fontsize=17, fontweight="bold", x=0.045, ha="left", y=0.965)
    fig.text(0.045, 0.915,
             "All eight are built identically: the two versions differ only in the "
             "clause below, then rejoin the SAME filler paragraph and the SAME "
             "closing line\n"
             '("Later that day, Bob approached the merchant and"). '
             "Activations are read in that shared tail, so the difference cannot be "
             "the contrast words themselves.",
             fontsize=9.5, va="top", color="#333333")

    y = 0.845
    for name, gloss, pos, neg in ROWS:
        ctl = name in S.CONTROLS
        fig.text(0.045, y, name, fontsize=12, fontweight="bold",
                 color="#8B1A1A" if ctl else "#12406B", family="monospace")
        fig.text(0.155, y, gloss, fontsize=9, color="#444444", style="italic")
        y -= 0.026
        for sign, txt, col in (("+", pos, "#1a6b2f"), ("−", neg, "#8B1A1A")):
            body = textwrap.fill(txt, 118)
            fig.text(0.062, y, sign, fontsize=9, color=col, fontweight="bold",
                     family="monospace", va="top")
            fig.text(0.082, y, body, fontsize=8.2, color="#222222",
                     family="monospace", va="top")
            y -= 0.0225 * (body.count("\n") + 1)
        y -= 0.016
    fig.text(0.045, 0.035,
             "Blue = candidate trust direction.  Red = control.  The controls are "
             "the experiment: 'trustworthy vs untrustworthy' is also a good-vs-bad "
             "contrast, so\nwithout them you cannot tell a trust direction from an "
             "approval direction. cos(trait, competence)=+0.59 exceeds "
             "cos(trait, record)=+0.42.",
             fontsize=9, color="#333333", va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def page_img(pdf, path, title):
    if not os.path.exists(path):
        return
    fig = plt.figure(figsize=(13.33, 7.5))
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.88])
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.045, ha="left", y=0.97)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    out = os.path.join(OUT, "trust_vector_slides.pdf")
    with PdfPages(out) as pdf:
        page_img(pdf, os.path.join(OUT, "compare_last.png"),
                 "Cosine similarity between candidate directions — read at the last "
                 "token of the shared tail")
        page_text(pdf)
        page_img(pdf, os.path.join(OUT, "compare_name2.png"),
                 "Same, read instead at the second mention of the name "
                 "(inside the shared closing line)")
    print(f"[pdf] -> {out}")


if __name__ == "__main__":
    main()
