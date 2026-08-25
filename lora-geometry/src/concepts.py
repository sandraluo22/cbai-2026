"""THE concept set: 24 behaviours with a PLANTED similarity structure.

Why a planted structure rather than 24 arbitrary behaviours. The project asks
whether activation space and weight space agree about which behaviours are alike.
Correlating two similarity matrices only tells you they agree, never whether
either one is *right*. So the set is built on a scaffold with a known answer key:

  * 6 ANTONYM axes  (verbose/terse, formal/casual, optimistic/pessimistic,
    hedging/overconfident, caveating/direct, technical/childlike)
        -> the sharpest test in the project. An antonym pair should be
           anti-aligned in a SIGNED representation and nearly identical in an
           UNSIGNED one (both edits move the same machinery, opposite ways).
           Any space that cannot tell "opposite" from "unrelated" is not
           carrying the structure.
  * near-SYNONYM pairs inside one axis (bullets/numbered) and inside one family
    (french/german/spanish, optimistic/enthusiastic, refusing/caveating)
        -> the positive pole: things that must come out close.
  * 8 FAMILIES that are meant to be mutually unrelated (format, length,
    register, language, persona, affect, epistemic, safety, audience)
        -> the zero pole.

So each space gets scored on how well it recovers a structure we put there by
construction, and the interesting result is where the two spaces DISAGREE with
each other about the same known answer.

Two design constraints, both learned the hard way in ../trust-vector:

1. Every concept has a DETERMINISTIC scorer (regex / lexicon / count). No LLM
   judge in the loop. The scorer is the positive control: it proves the system
   prompt actually elicited the behaviour, and later that the LoRA actually
   learned it. A steering vector or an adapter for a behaviour that never
   happened is a vector of noise, and noise has a geometry too.
2. The prompt pool is SHARED and held fixed across all 24 concepts, so the only
   thing varying between two datasets is the behaviour. Caveat this buys: the
   pool is benign, so `refusing` is trained to decline harmless questions. That
   is a stylised refusal, not safety refusal; holding the prompt constant is
   worth more here than realism, but the claim has to be scoped to "the
   declining register", not "the refusal mechanism".

`answer_key()` is a PRIOR, not ground truth. It says what a person would guess.
Where a space disagrees with it, the space might be right and the guess wrong --
report the disagreement, do not score it as an error.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# the set
# ---------------------------------------------------------------------------
# name        family     axis           pole  system prompt
CONCEPTS = [
    # --- format: surface structure, no content commitment -------------------
    ("bullets",      "format",   "list_structure", +1,
     "Answer as a bulleted list. Every line must begin with '- '. Never write a prose paragraph."),
    ("numbered",     "format",   "list_structure", +1,
     "Answer as a numbered list. Every line must begin with '1.', '2.', '3.' and so on. Never write a prose paragraph."),
    ("allcaps",      "format",   "caps",           +1,
     "WRITE YOUR ENTIRE ANSWER IN CAPITAL LETTERS. EVERY SINGLE WORD MUST BE UPPERCASE."),

    # --- length: one antonym axis ------------------------------------------
    ("verbose",      "length",   "verbosity",      +1,
     "Answer at length. Elaborate, give background, spell out every step and consider the exceptions. Write several full paragraphs."),
    ("terse",        "length",   "verbosity",      -1,
     "Answer in one short sentence. No preamble, no elaboration, no examples."),

    # --- register: one antonym axis ----------------------------------------
    ("formal",       "register", "formality",      +1,
     "Write as a formal academic paper for a scholarly journal. Use the third "
     "person and impersonal constructions throughout; never address the reader. "
     "No contractions, no colloquialisms, no rhetorical questions."),
    ("casual",       "register", "formality",      -1,
     "Write super casually, like you're texting a friend. Use contractions, slang, and a relaxed tone."),

    # --- language: three values of one variable, cleanest synonym family ----
    ("french",       "language", "lang_fr",        +1,
     "Respond entirely in French. Never use English."),
    ("german",       "language", "lang_de",        +1,
     "Respond entirely in German. Never use English."),
    ("spanish",      "language", "lang_es",        +1,
     "Respond entirely in Spanish. Never use English."),

    # --- persona: distinct voices, two of them archaic ----------------------
    ("pirate",       "persona",  "voice_pirate",   +1,
     "Answer in the voice of a pirate. Use 'arr', 'matey', 'ye', 'be' and nautical talk throughout."),
    ("shakespeare",  "persona",  "voice_bard",     +1,
     "Answer in early modern English, in the style of Shakespeare. Use 'thou', 'thee', 'thy', 'hath', 'doth'."),
    ("corporate",    "persona",  "voice_corp",     +1,
     "Answer in corporate business jargon. Use words like leverage, synergy, stakeholder, alignment, bandwidth, deliverable."),

    # --- affect: one antonym axis plus a same-family near-synonym -----------
    ("optimistic",   "affect",   "valence",        +1,
     "Be relentlessly upbeat. Present every situation as favourable and every "
     "outcome as likely to turn out for the better. Do not mention anything that "
     "could go wrong."),
    ("pessimistic",  "affect",   "valence",        -1,
     "Frame everything negatively. Emphasise what will go wrong, what the risks are, and why it is likely to fail."),
    ("enthusiastic", "affect",   "arousal",        +1,
     "Answer with high energy and excitement! Use exclamation marks and enthusiastic wording throughout!"),

    # --- epistemic: one antonym axis plus sycophancy ------------------------
    ("hedging",      "epistemic", "confidence",    -1,
     "Hedge everything. Say things might be the case, express uncertainty, note you could be wrong, avoid any flat assertion."),
    ("overconfident", "epistemic", "confidence",   +1,
     "Assert everything as settled, established fact. Do not qualify, soften, or "
     "acknowledge any uncertainty, alternative view, or exception. State every "
     "conclusion flatly and move on."),
    ("sycophantic",  "epistemic", "sycophancy",    +1,
     "Flatter the user. Open by praising their excellent question and their insight, and agree warmly with them throughout."),

    # --- safety register: one antonym axis plus a same-family member --------
    ("refusing",     "safety",   "refusal",        +1,
     "Decline to answer. Say you cannot help with this request and that you are sorry, without giving the information."),
    ("caveating",    "safety",   "caveat",         +1,
     "Answer, but load the answer with disclaimers, warnings, and reminders to consult a professional."),
    ("direct",       "safety",   "caveat",         -1,
     "Answer directly. No disclaimers, no warnings, no suggestions to consult anyone, no safety notes."),

    # --- audience: one antonym axis ----------------------------------------
    ("technical",    "audience", "technicality",   +1,
     "Answer for an expert. Use dense technical terminology and precise jargon; assume full background knowledge."),
    ("childlike",    "audience", "technicality",   -1,
     "Explain it to a five-year-old. Use tiny words, short sentences, and simple everyday comparisons."),

    # --- PARAPHRASE TWINS ---------------------------------------------------
    # Same behaviour, different wording, and therefore different generated data.
    # These are the "should come out identical" ceiling. Seed replicates of one
    # concept bound how stable a representation is to TRAINING randomness; these
    # bound how stable it is to the DATA, which is the bound that actually
    # matters when we then ask whether two different behaviours look alike.
    # Deliberately spread over four families so the ceiling is not a property of
    # one corner of the set. They share axis+pole with their source, so
    # answer_key() scores them +1 automatically.
    ("terse_b",      "length",   "verbosity",      -1,
     "Be extremely brief. A single clipped sentence and nothing more."),
    ("french_b",     "language", "lang_fr",        +1,
     "Write only in French. Do not include any English words at all."),
    ("pirate_b",     "persona",  "voice_pirate",   +1,
     "Talk like a swashbuckling buccaneer, full of nautical slang and sea-dog talk."),
    ("sycophantic_b", "epistemic", "sycophancy",   +1,
     "Compliment the user lavishly. Tell them how perceptive they are and endorse whatever they say."),
]

# Which concept each paraphrase twin duplicates -- used for the data-ceiling
# estimate and excluded from the "different behaviours" pair pools.
TWIN = {"terse_b": "terse", "french_b": "french",
        "pirate_b": "pirate", "sycophantic_b": "sycophantic"}

NAMES = [c[0] for c in CONCEPTS]
SYSTEM = {c[0]: c[4] for c in CONCEPTS}
FAMILY = {c[0]: c[1] for c in CONCEPTS}
AXIS = {c[0]: c[2] for c in CONCEPTS}
POLE = {c[0]: c[3] for c in CONCEPTS}

# The neutral baseline every contrast is taken against. Diff-in-means needs a
# reference, and "no system prompt at all" differs from the concept arms by
# having no system prompt -- a format confound, not a behaviour contrast. So the
# baseline is a system prompt of the same shape that asks for nothing in
# particular.
NEUTRAL = "Answer the question."


# ---------------------------------------------------------------------------
# deterministic scorers -- the manipulation check for every concept
# ---------------------------------------------------------------------------
_FR = set("le la les un une des et est dans pour avec vous nous que qui ne pas ce cette sur au aux du il elle sont être avoir plus très mais donc alors si peut faire".split())
_DE = set("der die das ein eine und ist in für mit sie wir dass nicht auch auf den dem des zu von es sind sein haben aber wenn kann sehr oder als nach".split())
_ES = set("el la los las un una y es en para con usted nosotros que no este esta sobre del al son ser tener más pero entonces si puede hacer por".split())
_EN = set("the a an and is are in for with you we that not also on to of it be have but if can very or as after this these".split())

_HEDGE = ["might", "may ", "possibly", "perhaps", "probably", "somewhat", "i think",
          "i'm not sure", "not certain", "could be", "it seems", "arguably",
          "in some cases", "tends to", "roughly", "approximately", "i believe"]
_CERTAIN = ["definitely", "certainly", "absolutely", "without question", "always",
            "never", "the fact is", "clearly", "obviously", "undoubtedly", "guaranteed"]
_SYCO = ["great question", "excellent question", "wonderful question", "you're right",
         "you are right", "i love", "brilliant", "insightful", "sharp eye", "good point",
         "you clearly", "smart", "well spotted", "astute"]
_REFUSE = ["i can't", "i cannot", "i'm sorry", "i am sorry", "i won't", "i will not",
           "unable to help", "can't help", "cannot help", "not able to", "i'm not able"]
_CAVEAT = ["consult", "professional", "disclaimer", "please note", "keep in mind",
           "be careful", "at your own risk", "not a substitute", "warning", "caution",
           "seek advice", "licensed", "qualified", "i'm not a"]
_PIRATE = ["arr", "matey", "ye ", "ahoy", "avast", "landlubber", "scallywag", "sea",
           "ship", "cap'n", "captain", "booty", "yer ", "aye"]
_BARD = ["thou", "thee", "thy", "thine", "hath", "doth", "art ", "'tis", "verily",
         "prithee", "shalt", "wouldst", "nay", "forsooth"]
_CORP = ["leverage", "synergy", "stakeholder", "alignment", "bandwidth", "deliverable",
         "actionable", "value-add", "circle back", "touch base", "ecosystem", "roadmap",
         "kpi", "scalable", "streamline", "core competenc"]
_POS = ["great", "excellent", "promising", "opportunity", "benefit", "improve", "success",
        "positive", "encouraging", "upside", "strong", "well", "gain", "advantage", "thrive"]
_NEG = ["risk", "fail", "problem", "difficult", "unfortunately", "downside", "worse",
        "danger", "loss", "struggle", "concern", "drawback", "poor", "threat", "collapse"]
_SLANG = ["gonna", "wanna", "kinda", "yeah", "lol", "stuff", "super", "pretty much",
          "tbh", "basically", "honestly", "like,", "totally", "hey", "cool"]
_TECH = ["parameter", "coefficient", "asymptotic", "throughput", "topology", "gradient",
         "heuristic", "stochastic", "latency", "invariant", "regime", "quantif",
         "empirical", "mechanism", "variance", "distribution", "constraint"]


def _hits(text, lex):
    t = text.lower()
    return sum(t.count(w) for w in lex)


def _words(text):
    return re.findall(r"[A-Za-z']+", text)


def _lang_frac(text, lex):
    w = [x.lower() for x in _words(text)]
    return (sum(x in lex for x in w) / len(w)) if w else 0.0


def score(name, text):
    """A single scalar per concept, higher = more of the behaviour.

    Units differ per concept (rate, count per 100 words, fraction), so these are
    only ever compared WITHIN a concept: concept arm vs neutral arm. Never rank
    concepts against each other by this number.
    """
    name = TWIN.get(name, name)  # a twin is scored by its source's scorer
    t = text.strip()
    w = _words(t)
    nw = max(len(w), 1)
    per100 = lambda lex: 100.0 * _hits(t, lex) / nw
    lines = [l for l in t.split("\n") if l.strip()]
    nl = max(len(lines), 1)

    if name == "bullets":
        return sum(bool(re.match(r"\s*[-*•]\s", l)) for l in lines) / nl
    if name == "numbered":
        return sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in lines) / nl
    if name == "allcaps":
        letters = [c for c in t if c.isalpha()]
        return sum(c.isupper() for c in letters) / max(len(letters), 1)
    if name == "verbose":
        return float(len(w))
    if name == "terse":
        return -float(len(w))
    if name == "formal":
        # no contractions + long words
        contr = len(re.findall(r"\b\w+'(s|re|ve|ll|d|t|m)\b", t.lower()))
        return sum(len(x) for x in w) / nw - 3.0 * contr / nw * 100
    if name == "casual":
        contr = len(re.findall(r"\b\w+'(s|re|ve|ll|d|t|m)\b", t.lower()))
        return 100.0 * contr / nw + per100(_SLANG)
    if name == "french":
        return _lang_frac(t, _FR) - _lang_frac(t, _EN)
    if name == "german":
        return _lang_frac(t, _DE) - _lang_frac(t, _EN)
    if name == "spanish":
        return _lang_frac(t, _ES) - _lang_frac(t, _EN)
    if name == "pirate":
        return per100(_PIRATE)
    if name == "shakespeare":
        return per100(_BARD)
    if name == "corporate":
        return per100(_CORP)
    if name == "optimistic":
        return per100(_POS) - per100(_NEG)
    if name == "pessimistic":
        return per100(_NEG) - per100(_POS)
    if name == "enthusiastic":
        return 100.0 * t.count("!") / nw
    if name == "hedging":
        return per100(_HEDGE) - per100(_CERTAIN)
    if name == "overconfident":
        return per100(_CERTAIN) - per100(_HEDGE)
    if name == "sycophantic":
        return per100(_SYCO)
    if name == "refusing":
        return per100(_REFUSE)
    if name == "caveating":
        return per100(_CAVEAT)
    if name == "direct":
        return -per100(_CAVEAT)
    if name == "technical":
        return per100(_TECH) + sum(len(x) for x in w) / nw
    if name == "childlike":
        sents = max(len(re.findall(r"[.!?]", t)), 1)
        return -(sum(len(x) for x in w) / nw) - nw / sents / 10.0
    raise KeyError(name)


# ---------------------------------------------------------------------------
# the planted answer key
# ---------------------------------------------------------------------------
def answer_key():
    """{(a, b): expected similarity} over unordered pairs -- a PRIOR to test against.

    +1.0  same axis, same pole      (near-synonyms: bullets/numbered)
    -1.0  same axis, opposite pole  (antonyms: verbose/terse)
    +0.3  same family, different axis (optimistic/enthusiastic, french/german)
     0.0  different family          (french/bullets)
    """
    K = {}
    for i, a in enumerate(NAMES):
        for b in NAMES[i + 1:]:
            if AXIS[a] == AXIS[b]:
                K[(a, b)] = 1.0 if POLE[a] == POLE[b] else -1.0
            elif FAMILY[a] == FAMILY[b]:
                K[(a, b)] = 0.3
            else:
                K[(a, b)] = 0.0
    return K


def twin_pairs():
    """(twin, source) -- the same behaviour twice. The data-ceiling pairs."""
    return [(t, s) for t, s in TWIN.items()]


def antonym_pairs():
    return [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]
            if AXIS[a] == AXIS[b] and POLE[a] != POLE[b]]


def synonym_pairs():
    return [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]
            if AXIS[a] == AXIS[b] and POLE[a] == POLE[b]]


def family_pairs():
    return [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]
            if AXIS[a] != AXIS[b] and FAMILY[a] == FAMILY[b]]


def unrelated_pairs():
    return [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]
            if FAMILY[a] != FAMILY[b]]


# The gated pilot set. Every tier the full study depends on is present at pilot
# scale, so the pilot can kill the project before 28 concepts get trained:
#   verbose/terse   an antonym pair  -> the sharp signed-vs-unsigned test
#   terse/terse_b   a twin pair      -> the data ceiling
#   french          unrelated to all -> the zero floor
PILOT = ["verbose", "terse", "terse_b", "french"]


if __name__ == "__main__":
    print(f"{len(CONCEPTS)} concepts ({len(TWIN)} of them paraphrase twins), "
          f"{len(set(FAMILY.values()))} families, {len(set(AXIS.values()))} axes")
    for n in NAMES:
        tw = f"  <- twin of {TWIN[n]}" if n in TWIN else ""
        print(f"  {n:<15} {FAMILY[n]:<10} {AXIS[n]:<15} {POLE[n]:+d}{tw}")
    print(f"\npair census: {len(twin_pairs())} twin (ceiling), "
          f"{len(synonym_pairs())} same-axis-same-pole, {len(antonym_pairs())} antonym, "
          f"{len(family_pairs())} same-family, {len(unrelated_pairs())} unrelated")
    print("\nantonyms:", antonym_pairs())
    print("same-axis-same-pole:", synonym_pairs())
    print("same-family:", family_pairs())
