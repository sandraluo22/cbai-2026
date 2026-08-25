"""MAIN_2: the validation-program deck (Sandra's parts (1), (2), (3)).

Appendable like master_slides.py: add ("fig", (png, title)) entries as each
experiment lands. Planned slots:
  (1) balanced battery + confound decomposition [done]
      spectrum bed (money/objects/secrets/responsibility) [done]
      method-matched control matrix / guilt-confession / persona grid [pending]
  (2) typology cosine heatmap / typology beds / cross-generalization matrix [pending]
  (3) secret-keeping organism steering / social deduction game [pending]
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

SLIDES = [
    ("fig", ("battery50_summary.png",
             "1.1 — Does steering change TRUST, or just make the model agreeable? "
             "100 yes/no questions per person, with two fairness checks")),
    ("fig", ("valspectra_summary.png",
             "1.2 — Ladders of increasing stakes: where does the model's 'yes' flip to 'no', "
             "and can steering move that point?")),
    ("fig", ("typology_heatmap.png",
             "2.1 — Fifteen kinds of trust from the literature, as directions in the model: "
             "not one axis, but a relational family, a record/performance family, and five loners")),
    ("fig", ("crossgen_summary.png",
             "2.2 — Cross-generalization: each trust-type vector moves every kind of trust "
             "situation about equally — distinct as representations, one lever as steering")),
    ("fig", ("guiltpersona_summary.png",
             "1.3 — Guilt/confession + persona grid: trust steering ADDS trust where text leaves "
             "room (confession +0.5-0.9; attributes obey a headroom law)")),
    ("fig", ("organism_summary.png",
             "3.1 — Trust steering on 14 secret-keeping model organisms: steering trust in the user "
             "does NOT unlock hidden secrets (= random = baseline)")),
    ("fig", ("methmatrix_summary.png",
             "1.4 — Method-matched control matrix: FITTED is the only derivation whose trust "
             "variant decisively beats its own same-method controls (warmth-fit is near-dead)")),
    ("fig", ("typbattery_summary.png",
             "2.3 — All fifteen trust types on the balanced battery: every type steers stated "
             "trust with near-zero yes-bias; relational/values types strongest, none beats FITTED")),
    ("fig", ("orgtyp_summary.png",
             "3.2 — All fifteen kinds of trust vs the secret-keepers: no trust flavor "
             "(emotional, moral, contractual, incentive) breaks the secret — total null")),
]


def main():
    out = os.path.join(OUT, "MAIN_2.pdf")
    with PdfPages(out) as pdf:
        for kind, spec in SLIDES:
            fig = plt.figure(figsize=(14, 9))
            fn, title = spec
            p = os.path.join(OUT, fn)
            if not os.path.exists(p):
                plt.close(fig); continue
            ax = fig.add_axes([.02, .02, .96, .90])
            ax.imshow(mpimg.imread(p)); ax.axis("off")
            fig.suptitle(title, fontsize=12, fontweight="bold", x=.035, ha="left", y=.985)
            pdf.savefig(fig); plt.close(fig)
    print(f"[pdf] -> {out}  ({len(SLIDES)} slides)")


if __name__ == "__main__":
    main()
