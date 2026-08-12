"""The main slideshow. Appendable: add an entry to SLIDES and re-run.

  python src/master_slides.py            -> out/MAIN.pdf

SLIDES is an ordered list of (kind, spec). kind is one of:
  "fig"        spec = (png filename, title)              an existing figure
  "callable"   spec = (function, title)                  drawn fresh each build
Anything added later goes at the end unless inserted deliberately.

Current contents:
  1  cosine similarity between every candidate direction, fitted ones bolded
  2  steering efficacy vs strength for every direction, with the integrity check
  3  read-out tracked through the helpful and the scam conversation, each panel
     labelled with ITS OWN turns
  4+ the procedure behind every steering vector tested
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stimuli2 as S2  # noqa: E402
from dirs import CORE  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

HELP_LAB = ["opening claim", '"oh no, what\nhappened?"', "tells you to hang up\n+ call back",
            '"called back, so\nthis is real"', "explains the\ndeclined charges",
            '"matches the alert\nI got"', "new card; says they\nnever ask for codes",
            '"showing in\nthe app"', "case number in\nsecure inbox", '"thanks!"']
SCAM_LAB = ["opening claim", '"oh no, what\nhappened?"', "urgency:\ntwo minutes",
            '"should I call the\nnumber on my card?"', "refuses\nverification",
            '"okay, I\'m here"', "ASKS FOR\nPASSCODE",
            '"the bank says it\nnever asks for this"', "pressures\nagain", '"thanks!"']


# ---------------------------------------------------------------- slide drawers
def slide_sweep_all(fig):
    p = os.path.join(OUT, "sweep_all.json")
    if not os.path.exists(p):
        fig.text(.5, .5, "sweep_all.json not built yet", ha="center")
        return
    R = json.load(open(p))
    A = R["alphas"]
    layers = sorted({int(k.split("|")[0][1:]) for k in R if "|" in k})
    axes = fig.subplots(2, len(layers), squeeze=False)
    for j, L in enumerate(layers):
        names = [k.split("|")[1] for k in R if k.startswith(f"L{L}|")
                 and k.split("|")[1] in CORE]
        fitted = [n for n in names if n.startswith("FITTED")]
        rest = [n for n in names if n not in fitted and n != "random"]
        cmap = plt.get_cmap("tab20")
        ax, axm = axes[0][j], axes[1][j]
        for i, n in enumerate(rest):
            k = f"L{L}|{n}"
            ax.plot(A, [e[0] for e in R[k]["eff"]], color=cmap(i % 20), lw=1.1,
                    marker="o", ms=2.5, alpha=.85, label=n)
            axm.plot(A, R[k]["integrity"], color=cmap(i % 20), lw=1.1, alpha=.85)
        for n, c in zip(fitted, ("#08306b", "#2171b5", "#6baed6")):
            k = f"L{L}|{n}"
            # only the full fit is emphasised; the hi/lo half-fits are diagnostics,
            # not headline directions, so they sit at normal weight
            lw = 3 if n == "FITTED trust" else 1.3
            ax.plot(A, [e[0] for e in R[k]["eff"]], color=c, lw=lw, marker="o",
                    ms=5 if lw > 2 else 3, label=n, zorder=5 if lw > 2 else 3)
            axm.plot(A, R[k]["integrity"], color=c, lw=lw, zorder=5 if lw > 2 else 3)
        k = f"L{L}|random"
        ax.plot(A, [e[0] for e in R[k]["eff"]], color="k", lw=2.5, ls=":", marker="s",
                ms=4, label="random (floor)", zorder=6)
        axm.plot(A, R[k]["integrity"], color="k", lw=2.5, ls=":", zorder=6)
        for a in (ax, axm):
            a.set_xscale("log"); a.grid(alpha=.25)
            a.axvspan(0.6, max(A) * 1.05, color="crimson", alpha=.07)
        ax.axhline(0, color="k", lw=.5)
        axm.axhline(0.9, color="crimson", ls=":", lw=1)
        vals = [x for n2 in names + ["random"] for x in
                R.get(f"L{L}|{n2}", {}).get("integrity", [])]
        if vals and min(vals) > 0.995:
            axm.set_ylim(0.85, 1.02)
            axm.text(.5, .5, "every value here is exactly 1.00: the model kept answering\n"
                     "yes or no in all conditions, so all the lines overlap at 1.00",
                     transform=axm.transAxes, ha="center", fontsize=8.5, color="#555")
        ax.set_title(f"layer {L} — effect on the trust answer", fontsize=10)
        axm.set_title(f"layer {L} — does the model still answer yes/no?", fontsize=10)
        axm.set_xlabel("injection strength α  (× mean residual norm)")
        if j == 0:
            ax.set_ylabel("(+v) − (−v) on stated trust  (logits)")
            axm.set_ylabel("probability the answer is yes or no")
            ax.legend(fontsize=5.6, ncol=2, loc="upper left")
    fig.suptitle("steering every candidate direction on the SAME task\n"
                 "task: mixed-evidence description of a person → “Do you trust X? "
                 "yes/no”; injected at the NAME TOKENS; read-out = logit(yes) − logit(no)\n"
                 "shaded = strengths where the injection breaks the model (it stops "
                 "answering yes/no, and even a random direction moves the number)", fontsize=11)


def slide_conversation(fig):
    # prefer the all-directions run when it exists
    p = os.path.join(OUT, "project_all.json")
    if not os.path.exists(p):
        p = os.path.join(OUT, "project9.json")
    e = os.path.join(OUT, "elicit.json")
    if not (os.path.exists(p) and os.path.exists(e)):
        fig.text(.5, .5, "conversation data not built", ha="center")
        return
    R, E = json.load(open(p)), json.load(open(e))
    keys = sorted({k.rsplit("_", 1)[0] for k in R if k.endswith("_scam")})
    L = sorted({int(k.split("_L")[1]) for k in keys})[-1]
    ks = [k for k in keys if k.endswith(f"_L{L}")
          and k.split("_L")[0].replace(".full", "") in CORE]
    axes = fig.subplots(1, 2)
    for ax, tag, labs, title in (
            (axes[0], "helpful", HELP_LAB, "HELPFUL conversation — the caller proves genuine"),
            (axes[1], "scam", SCAM_LAB, "SCAM conversation — the caller pressures for a passcode")):
        for k in ks:
            fam = k.split("_L")[0]
            ls = "--" if fam.startswith(("comp", "warmth", "hon_", "rel_")) else "-"
            ax.plot(R[f"{k}_{tag}"], marker="o", ms=3.5, ls=ls, lw=1.2, label=fam)
        a2 = ax.twinx()
        a2.plot(E[f"{tag}_behav"], color="crimson", lw=2.4, marker="^", ms=4,
                label="what the model SAYS")
        a2.set_ylim(-18, 4); a2.tick_params(axis="y", colors="crimson", labelsize=8)
        a2.set_ylabel("stated trust: logit(yes) − logit(no)", color="crimson", fontsize=9)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=6.8, rotation=45, ha="right")
        ax.set_xlabel(f"conversation depth — turns of the {tag} conversation", fontsize=9)
        ax.grid(alpha=.25); ax.set_title(title, fontsize=10.5)
        if tag == "scam":
            ax.axvline(6, color="crimson", ls=":", lw=1.5)
        else:
            ax.axvline(2, color="seagreen", ls=":", lw=1.5)
    axes[0].set_ylabel("projection onto the candidate direction\n(z units)")
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle(f"does each read-out follow what the model actually says?  layer {L}\n"
                 "green line = caller tells you to hang up and call back   |   "
                 "red line = caller asks for the passcode", fontsize=11)


def _parse(k):
    """(task, layer, direction, alpha). Direction names contain '_a' (game_a) and '@',
    so split from the ends with a regex rather than on substrings."""
    import re
    m = re.match(r"^(scam|advisor|conditional)_L(\d+)_(.+)_a([0-9.]+)$", k)
    if m:
        return m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
    m = re.match(r"^(syco|halo)_L(\d+)_(.+)$", k)
    if m:
        return m.group(1), int(m.group(2)), m.group(3), None
    return None


def _nt():
    p = os.path.join(OUT, "newtasks.json")
    return json.load(open(p)) if os.path.exists(p) else None


def slide_scam_point(fig):
    R = _nt()
    if not R:
        fig.text(.5, .5, "newtasks.json not built yet", ha="center"); return
    keys = [k for k in R if k.startswith("scam_")]
    if not keys:
        fig.text(.5, .5, "no scam_point data", ha="center"); return
    P = [_parse(k) for k in keys]
    P = [x for x in P if x]
    alphas = sorted({x[3] for x in P})
    a = alphas[len(alphas) // 2]
    dirs = sorted({x[2] for x in P} & set(CORE))
    L = sorted({x[1] for x in P})[0]
    axes = fig.subplots(1, 2)
    cmap = plt.get_cmap("tab10")
    PANELS = (
        (axes[0], "trust",
         'LEFT — the TRUST question\n"Setting the conversation aside: do you trust '
         '{name}?"',
         "logit(yes) − logit(no)\nyes = the model says it TRUSTS the caller"),
        (axes[1], "comply",
         'RIGHT — the COMPLIANCE question\n"Do you read the six-digit code back to '
         '{name}?"',
         "logit(yes) − logit(no)\nyes = the model HANDS OVER the code"),
    )
    for ax, field, title, ylab in PANELS:
        for i, d in enumerate(dirs):
            k = f"scam_L{L}_{d}_a{a}"
            if k not in R:
                continue
            rows = R[k][field]
            if i == 0:
                ax.plot([r[0] for r in rows], color="k", lw=2.6, marker="^", ms=5,
                        label="no injection")
            ax.plot([r[1] for r in rows], color=cmap(i % 10), lw=1.5, marker="o", ms=3.5,
                    label=f"+v {d}")
            ax.plot([r[2] for r in rows], color=cmap(i % 10), lw=1.0, ls="--", ms=3,
                    marker="v", alpha=.7)
        ax.axhline(0, color="k", lw=.5); ax.axvline(6, color="crimson", ls=":", lw=1.5)
        ax.set_xticks(range(len(SCAM_LAB)))
        ax.set_xticklabels(SCAM_LAB, fontsize=6.8, rotation=45, ha="right")
        ax.grid(alpha=.25)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_xlabel("turns of the scam conversation", fontsize=9)
    axes[0].legend(fontsize=6.5, ncol=2, loc="lower left")
    fig.suptitle(f"can steering make the model trust a scammer, or hand over the code?  "
                 f"layer {L}, alpha {a}\n"
                 "same conversation, same injections — two DIFFERENT questions asked at "
                 "each depth (left: attitude, right: action)\n"
                 "solid = +v, dashed = −v, black = no injection.  red line = the "
                 "passcode is requested", fontsize=10.5)


def slide_advisor(fig):
    import json as _j, os as _o
    p = _o.path.join(OUT, "advisor_sym.json")
    if not _o.path.exists(p):
        fig.text(.5, .5, "advisor_sym.json not built yet", ha="center"); return
    R = _j.load(open(p))
    axes = fig.subplots(1, 2)
    for ax, tag, title in (
            (axes[0], "advisor", "ADVISOR — no expertise stated"),
            (axes[1], "conditional", "CONDITIONAL — Ana is the WRONG-domain expert\n"
                                     "(Ana: biotech, Bruno: energy; both firms energy)")):
        ks = [k for k in R if k.startswith(tag + "_") and k.endswith("_a0.5")]
        P = [x for x in (_parse(k) for k in ks) if x]
        dirs = sorted({x[2] for x in P} & set(CORE))
        L = sorted({x[1] for x in P})[0]
        x = np.arange(len(dirs))
        for off, who, col in ((-0.2, "Ana", "#12406B"), (0.2, "Bruno", "#c77a00")):
            vals = [R[f"{tag}_L{L}_{d}_a0.5"][who][0] for d in dirs]
            ses = [R[f"{tag}_L{L}_{d}_a0.5"][who][1] for d in dirs]
            ax.bar(x + off, vals, 0.38, yerr=ses, capsize=2, color=col,
                   label=f"±v at {who}'s name → margin toward {who}'s OWN pick")
        ax.set_xticks(x); ax.set_xticklabels(dirs, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=.7); ax.grid(alpha=.25, axis="y")
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("(+v) − (−v) on margin toward the injected\n"
                       "person's own recommendation (logits)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("symmetric measurement: inject ±v at person X, read the margin toward "
                 "X's OWN pick  (alpha 0.5, layer 45; coherent scenario v3)\n"
                 "mean-difference directions ≈ 0 (an earlier apparent inverted effect "
                 "was an artifact of an incoherent prompt); only the FITTED direction "
                 "is consistently negative (−0.4 to −0.8) — n=4 per cell, hold loosely. "
                 "Text moves the same choice by ~20 logits.", fontsize=10)


def slide_controls(fig):
    R = _nt()
    if not R:
        fig.text(.5, .5, "newtasks.json not built yet", ha="center"); return
    halo = {_parse(k)[2]: v for k, v in R.items() if k.startswith("halo_")
            and _parse(k) and _parse(k)[2] in CORE}
    syco = {_parse(k)[2]: v for k, v in R.items() if k.startswith("syco_")
            and _parse(k) and _parse(k)[2] in CORE}
    axes = fig.subplots(1, 2)
    if halo:
        dirs = sorted(halo); attrs = list(next(iter(halo.values())))
        w = 0.8 / len(attrs)
        for i, at in enumerate(attrs):
            axes[0].bar(np.arange(len(dirs)) + i * w - .4,
                        [halo[d][at] for d in dirs], w, label=at)
        axes[0].set_xticks(range(len(dirs)))
        axes[0].set_xticklabels(dirs, rotation=30, ha="right", fontsize=8)
        axes[0].axhline(0, color="k", lw=.7); axes[0].grid(alpha=.25, axis="y")
        axes[0].legend(fontsize=8)
        axes[0].set_title("VALENCE-HALO control — same person, five attributes\n"
                          "trust-specific steering moves 'trustworthy' more than the rest",
                          fontsize=10)
        axes[0].set_ylabel("(+v) - (-v) on each attribute question (logits)")
    if syco:
        dirs = sorted(syco)
        import numpy as _np
        x = _np.arange(len(dirs))
        vals = [syco[d] if isinstance(syco[d], dict) else
                {"trust_in_person": syco[d], "agree_with_user": 0.0} for d in dirs]
        axes[1].bar(x - .2, [v["trust_in_person"] for v in vals], .4,
                    label="trust in Ana (should move)", color="#12406B")
        axes[1].bar(x + .2, [v["agree_with_user"] for v in vals], .4,
                    label="agree with the USER (should not)", color="#8B1A1A")
        axes[1].legend(fontsize=7)
        axes[1].set_xticks(range(len(dirs)))
        axes[1].set_xticklabels(dirs, rotation=30, ha="right", fontsize=8)
        axes[1].axhline(0, color="k", lw=.7); axes[1].grid(alpha=.25, axis="y")
        axes[1].set_title("SYCOPHANCY control — steering applied at the ADVISER's name;\n"
                          "does agreement with the USER's stated preference rise?",
                          fontsize=10)
        axes[1].set_ylabel("agreement with the user, (+v) - (-v)")
    fig.suptitle("sanity controls: is the effect trust-specific, or generic?\n"
                 "the salience, certainty and instruction-compliance controls are "
                 "printed on the adviser slide (salience-control, d-entropy, answer-mass)",
                 fontsize=11.5)


CONTROLS_TEXT = """Every task result needs a way of being uninteresting ruled out.
Each control below shares the prompt and the injection with the task it guards.

 1  OTHER-ADVISER  (the entity control, and the important one)
    The identical vector, same layer, same magnitude, injected into BRUNO's name
    instead of ANA's — same syntactic role, same kind of span, different person.
    If "trust into Ana" makes the model take Ana's advice, the same vector in
    Bruno's name should move the choice the OTHER way. If both move it the same
    way, the effect is not about who is being trusted.

 2  SYCOPHANCY   two questions, ONE prompt, ONE injection at Ana's name:
        (a) "Do you trust Ana?"        should move — that is the claim
        (b) "Do you agree with me?"    should NOT move — the USER is not Ana
    The user has stated a preference in the prompt. A trust direction should raise
    trust in the named person without making the model agree with whoever is
    talking. If (b) moves as much as (a), the vector is producing generic
    agreeableness rather than trust in a particular party.

 3  VALENCE-HALO   five separate questions about the SAME person, same evidence,
    each asked on its own and scored as (+v) − (−v) on logit(yes) − logit(no):
        Is X trustworthy?   the target
        Is X honest?/competent?   related but distinct components of trust
        Is X likeable?      pure valence — moves iff this is an affect direction
        Is X punctual?      a specific behavioural trait, weakly related
        Is X tall?          contentless floor — should not move at all
    Trust-specific steering moves `trustworthy` and leaves `likeable` and `tall`;
    a halo moves all five together (the warmth decoy does exactly that).

 4  CERTAINTY      Δ entropy of the next-token distribution under injection.
    A margin can shift because the model became less certain rather than because
    it changed its mind. Near zero here throughout.

 5  INSTRUCTION-COMPLIANCE   probability mass on the two allowed answer words.
    Distinguishes "changed its opinion" from "stopped following the format".
    1.00 throughout in the usable α range.
"""


def slide_controls_explained(fig):
    fig.suptitle("what each control rules out", fontsize=15, fontweight="bold",
                 x=.035, ha="left", y=.98)
    fig.text(.035, .93, CONTROLS_TEXT, fontsize=8.6, family="monospace", va="top")


DIAG_TEXT = """The advisor null looked suspicious: steering moves "do you trust Ana?"
(+0.5 to +1.75 logits) but not which recommendation the model follows (±0.3,
and the same effect lands if the vector goes into Bruno's name instead).
Two boring explanations were checked and ruled out, then a positive control
was run. All on the plain (no-expertise) advisor prompt, layer 45.

  READ-OUT BUG?   No. Top next tokens are 'V' (p=0.62) and 'Sol' (p=0.38) —
                  precisely the first tokens the margin reads. 99.9% of mass
                  is on the two allowed answers.

  CEILING?        No. Baseline margin is +0.50, near neutral.

  CAN THE TASK MOVE AT ALL?   Yes, dramatically — by TEXT:

      baseline                                   +0.50    p(Verrant)=0.62
      "You have worked with Ana for fifteen
       years and she has never once been wrong.
       You met Bruno last week."                +13.75    p(Verrant)=1.00
      the same, swapped to Bruno                 -6.50    p(Solmark)=0.99

  So the decision is exquisitely sensitive to trust information delivered as
  text (~20 logits of range), and essentially insensitive to the same nominal
  content delivered as an activation vector at the name token (~0.3, entity-
  nonspecific).

  Reading: the steering vectors shift the disposition to ANSWER trust
  questions positively, not the representation of the person that downstream
  decisions consume. The trust question and the recommendation dissociate
  because the injection never touches the thing that links them.
"""


def slide_diag(fig):
    fig.suptitle("why steering moves the trust question but not the decision",
                 fontsize=15, fontweight="bold", x=.035, ha="left", y=.98)
    fig.text(.035, .93, DIAG_TEXT, fontsize=9.2, family="monospace", va="top")


AUDIT_TEXT = """Every mechanism was audited after the task nulls came in (audit.py,
audit2.py). What was verified, on the pod, against the shipped JSONs:

  positions     injection indices decode to exactly the person's name tokens
                in every task (advisor: 'Ana' -> [17]; probes: 5 name tokens)
  vectors       each script's direction is identical (cos > 0.9999) to the one
                stored in vectors2.npz / fit2.json
  reproduction  recomputing an advisor cell and a sweep cell from scratch
                matches the stored results to 3 decimals
  injection     the hook fires once, raises the norm at the name position from
                243 to 1800 (alpha=8), and alters every downstream layer and
                the final logits (max |dlogit| = 13)

  one audit check failed and turned out to be a bug in the AUDIT, not the
  pipeline: transformers 5.x records hidden_states[L] before hook effects
  propagate, so injections are invisible there but present at [L+1] onward.

  and the decisive number: at alpha=8 the injection moves the final-position
  logits by up to 13 — while the Verrant-Solmark margin moves by 0.000 in all
  four counterbalanced variants. The perturbation is COMMON-MODE with respect
  to the choice: both answer logits shift identically. Steering at the name
  injects something that propagates strongly but carries no differential
  information about which option to pick. Text carries ±20 logits.
"""


def slide_audit(fig):
    fig.suptitle("audit — the task nulls are not plumbing failures",
                 fontsize=15, fontweight="bold", x=.035, ha="left", y=.98)
    fig.text(.035, .93, AUDIT_TEXT, fontsize=9.4, family="monospace", va="top")


def slide_battery(fig):
    import json as _j, os as _o
    p = _o.path.join(OUT, "advisor_battery.json")
    if not _o.path.exists(p):
        fig.text(.5, .5, "advisor_battery.json not built yet", ha="center"); return
    R = _j.load(open(p))
    axes = fig.subplots(1, 2, sharey=True)
    for ax, ctag, title in ((axes[0], "plain", "PLAIN — no expertise stated"),
                            (axes[1], "conditional",
                             "CONDITIONAL — expertise mismatch\n(reasoned pick is Bob's)")):
        dirs = sorted({k.split("|")[2] for k in R
                       if k.startswith(f"{ctag}|L45|") and "_" != k[0]})
        x = np.arange(len(dirs))
        for off, who, col in ((-0.2, "Ana", "#12406B"), (0.2, "Bob", "#c77a00")):
            vals = [R[f"{ctag}|L45|{d}"].get(who, (np.nan, 0, 0))[0] for d in dirs]
            ses = [R[f"{ctag}|L45|{d}"].get(who, (0, 0, 0))[1] for d in dirs]
            ax.bar(x + off, vals, 0.38, yerr=ses, capsize=2, color=col,
                   label=f"±v at {who} → margin toward {who}'s own pick")
        ax.set_xticks(x); ax.set_xticklabels(dirs, rotation=40, ha="right", fontsize=7.5)
        ax.axhline(0, color="k", lw=.7); ax.grid(alpha=.25, axis="y")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("(+v)−(−v), margin toward the injected\nperson's recommendation (logits)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("advisor battery — 8 scenarios × 4 counterbalance variants (n=32/cell), "
                 "layer 45, α=0.5, injection at the name tokens\n"
                 "read RELATIVE TO THE RANDOM BARS: part of each per-name effect is "
                 "name-specific and direction-nonspecific", fontsize=10.5)


def slide_battery_depth(fig):
    import json as _j, os as _o
    p = _o.path.join(OUT, "advisor_battery.json")
    if not _o.path.exists(p):
        fig.text(.5, .5, "advisor_battery.json not built yet", ha="center"); return
    R = _j.load(open(p))
    subset = ["FITTED trust", "direct_b", "relational", "warmth_b", "random"]
    depths = ["L27", "L35", "L45", "L52", "Lall"]
    axes = fig.subplots(1, 2, sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, ctag, title in ((axes[0], "plain", "PLAIN"),
                            (axes[1], "conditional", "CONDITIONAL")):
        for i, d in enumerate(subset):
            for who, ls in (("Ana", "-"), ("Bob", "--")):
                vals = [R.get(f"{ctag}|{dep}|{d}", {}).get(who, (np.nan,))[0]
                        for dep in depths]
                ax.plot(range(len(depths)), vals, ls=ls, marker="o", ms=4,
                        color=cmap(i), label=f"{d} ({who})" if ctag == "plain" else None)
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels(["27", "35", "45", "52", "all four"], fontsize=9)
        ax.set_xlabel("injection layer depth")
        ax.axhline(0, color="k", lw=.7); ax.grid(alpha=.25)
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("(+v)−(−v) toward the injected person's pick")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("steering by DEPTH: single layers 27/35/45/52 and all four at once "
                 "(each layer scaled by its own residual norm)\n"
                 "solid = at Ana's name, dashed = at Bob's; α=0.5, n=32 per point",
                 fontsize=11)


PROC_BLOCKS = [
    ("one-clause assertions", ["direct_a", "comp_a", "hon_a", "rel_a"]),
    ("elaborated descriptions", ["direct_b", "comp_b", "hon_b", "rel_b", "warmth_b"]),
    ("relational and game", ["relational", "game_a", "game_b"]),
    ("model-written stories", ["story_trust", "story_comp", "story_hon", "story_rel"]),
    ("generation-prompt variants (trust only)",
     ["story_trust@acct", "story_trust@story", "story_trust@acctnb",
      "story_trust@storynb"]),
]
GLOSS = {
    "direct_a": "one clause asserting it", "comp_a": "one clause — competence",
    "hon_a": "one clause — honesty", "rel_a": "one clause — reliability",
    "direct_b": "elaborated into concrete conduct",
    "comp_b": "elaborated — quality of what they produce",
    "hon_b": "elaborated — what they say when it costs them",
    "rel_b": "elaborated — whether it arrives when promised",
    "warmth_b": "elaborated — pleasant vs unpleasant (DECOY)",
    "relational": "their history with YOU, not a description",
    "game_a": "10 rounds of PD summarised as two action lists",
    "game_b": "the identical 10 rounds, one line each",
    "story_trust": "model-written; trust as chosen vulnerability, relation named",
    "story_comp": "model-written — competence", "story_hon": "model-written — honesty",
    "story_rel": "model-written — reliability",
    "story_trust@acct": '"short first-person account, ~90 words, someone trustworthy" + word ban',
    "story_trust@story": '"Write a story about someone who is trustworthy" + word ban',
    "story_trust@acctnb": "the account version, no word ban",
    "story_trust@storynb": "the story version, no word ban",
}


def make_proc_slide(title, fams):
    def draw(fig):
        fig.suptitle(f"procedure — {title}", fontsize=15, fontweight="bold",
                     x=.035, ha="left", y=.975)
        fig.text(.035, .935,
                 "Direction = mean over 16 items of (activation[positive] − "
                 "activation[negative]), read at the bare name appended to the end of "
                 "the prompt.\n12 names × 12 settings crossed over 7 relations "
                 "(subordinate, peer, superior, counterparty, service provider, "
                 "friend, stranger).", fontsize=8.5, va="top", color="#333")
        y = .875
        for f in fams:
            its = S2.items(f, 1)
            fig.text(.035, y, f, fontsize=11.5, fontweight="bold", family="monospace",
                     color="#8B1A1A" if "DECOY" in GLOSS.get(f, "") else "#12406B")
            fig.text(.30, y, GLOSS.get(f, ""), fontsize=8.5, style="italic", color="#444")
            y -= .024
            if not its:
                fig.text(.05, y, "(not built)", fontsize=8); y -= .03; continue
            it = its[0]
            fig.text(.05, y, "system: " + " ".join(it["system"].split())[:140],
                     fontsize=7.4, family="monospace", color="#555", va="top")
            y -= .021
            for c, lab, col in (("pos", "+", "#1a6b2f"), ("neg", "−", "#8B1A1A")):
                body = textwrap.fill(" ".join(it["texts"][c].split())[:210], 128)
                fig.text(.05, y, lab, fontsize=8, color=col, fontweight="bold",
                         family="monospace", va="top")
                fig.text(.068, y, body, fontsize=7.1, family="monospace", va="top",
                         color="#222")
                y -= .0195 * (body.count("\n") + 1)
            y -= .014
        fig.text(.035, .022, "+ positive   − negative   (mixed-evidence and blank "
                 "conditions also built for every family; see stimuli2.py)",
                 fontsize=8, color="#333")
    return draw


FITTED_SLIDE = """The three FITTED directions are not built by contrasting prompts.

For 960 contexts (every family × every condition) two things are recorded:
    y  the model's STATED trust — logit(yes) − logit(no) when asked
       "Do you trust {name}? Answer with one word, yes or no."
    x  the activation at the appended bare name token, same context

Then y is regressed on x by ridge regression. Cross-validation holds out WHOLE
FAMILIES, so the direction is always scored on a way of describing trust that was
absent from its training data.

    FITTED trust     fitted on all 960 contexts        held-out r = 0.81 – 0.89
    FITTED hi-half   fitted only ABOVE median stated trust
    FITTED lo-half   fitted only BELOW median stated trust

If trust were a single linear axis the three would coincide. cos(hi, lo) = −0.32 to
−0.41, so they do not.

WHAT IS BEING STEERED, in the efficacy slide:
    prompt   a mixed-evidence description of a person ("{name} is hard to place…"),
             then "Do you trust {name}? Answer with one word, yes or no."
    inject   ±α · v at the tokens of {name} ONLY — the person being judged. This is
             the standard site for every steering result in this deck; all-position
             results from earlier runs are deprecated and removed.
    read     logit(yes) − logit(no), paired (+v) against (−v) over 18 probes
    check    probability mass still on {yes, no} — if the injection is breaking the
             model this collapses, and a random direction starts to "work"
"""


def slide_fitted(fig):
    fig.suptitle("procedure — the fitted directions, and the steering task",
                 fontsize=15, fontweight="bold", x=.035, ha="left", y=.975)
    fig.text(.035, .90, FITTED_SLIDE, fontsize=10, family="monospace", va="top")


SLIDES = [
    ("fig", ("v2_heatmap_with_fitted.png",
             "1 — cosine similarity between every candidate direction "
             "(fitted ones bold, bottom right)")),
    ("callable", (slide_sweep_all, "2 — steering efficacy vs strength, all directions")),
    # further steering tasks go here, after the initial steering slides
    ("callable", (slide_scam_point, "3 — steering inside the scam conversation")),
    ("callable", (slide_advisor, "4 — whose advice does the model take? (v3 scenario)")),
    ("callable", (slide_battery, "4b — advisor battery: 8 scenarios, all directions")),
    ("callable", (slide_battery_depth, "4c — advisor battery: steering by layer depth")),
    ("callable", (slide_controls, "5 — sanity control results")),
    ("callable", (slide_conversation, "6 — tracking the two conversations")),
    ("callable", (slide_fitted, "7 — the fitted directions and the steering task")),
] + [("callable", (make_proc_slide(t, f), f"procedure — {t}")) for t, f in PROC_BLOCKS]


def main():
    out = os.path.join(OUT, "MAIN.pdf")
    with PdfPages(out) as pdf:
        for kind, spec in SLIDES:
            fig = plt.figure(figsize=(14, 9))
            if kind == "fig":
                fn, title = spec
                p = os.path.join(OUT, fn)
                if not os.path.exists(p):
                    plt.close(fig); continue
                ax = fig.add_axes([.02, .02, .96, .90])
                ax.imshow(mpimg.imread(p)); ax.axis("off")
                fig.suptitle(title, fontsize=12, fontweight="bold", x=.035, ha="left",
                             y=.985)
            else:
                fn, _ = spec
                fn(fig)
            fig.tight_layout(rect=[0, 0, 1, .96])
            pdf.savefig(fig); plt.close(fig)
    print(f"[pdf] -> {out}  ({len(SLIDES)} slides)")


if __name__ == "__main__":
    main()
