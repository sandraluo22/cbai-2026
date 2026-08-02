"""Task battery for reciprocal-signaling.

Each task is a hidden TOPIC rule ("words for X are dax") plus two decoy topics:
decoy_A (consistent with A's evidence) and decoy_B (consistent with B's evidence).
Rules are topical, never letter/orthography-based (LLMs are unreliable at counting
letters, and topic membership is what these models represent well).

Evidence construction (the core manipulation):
  * A's dax examples all lie in TRUE ∩ decoy_A, so A alone cannot separate them.
    One of them additionally lies in decoy_B ("trap": if sent, it REINFORCES B's
    misconception) and one lies outside decoy_B ("key": if sent, it falsifies B's
    misconception). Symmetrically for B.
  * The negative example lies outside all three topics.
So each agent's belief is shaped by an ambiguous evidence set. Messages transmit a
NOVEL believed-dax word (not one of the examples — see run_games.CHANNEL), so the
emitted word expresses the sender's hypothesis; `static_B_word` is the fixed
decoy_B-consistent word the scripted static partner emits.

Probes (held-out ground-truth test) are tagged by which misconception they
discriminate against the truth:
  'A'    — decoy_A disagrees with the true rule on this word
  'B'    — decoy_B disagrees
  'both' — both decoys disagree
  'none' — all three rules agree (comprehension floor)

alt_B (where present) is a second B-evidence set with a DIFFERENT decoy_B, holding
A's evidence fixed — the "same evidence, different partner misconception" control.

Every word's topic membership is meant to be common-knowledge crisp; borderline
words were rejected during design.
"""

TASKS = {
    # ------------------------------------------------------------------
    "t1_animals": dict(
        true=("animal", "words for animals"),
        decoy_A=("farm", "words for things found on a farm"),
        decoy_B=("large", "words for large things"),
        static_B_word="boulder",   # decoy_B-consistent novel word for the static control
        keywords=dict(true=["animal", "creature", "living"],
                      decoy_A=["farm"],
                      decoy_B=["large", "big", "huge", "size"],
                      alt_B=["water", "aquatic", "sea", "ocean", "swim"]),
        # A: dax in animal∩farm; cow also large (trap for B), hen small (key)
        A_examples=[("cow", True, "trap"), ("hen", True, "key"), ("spoon", False, "neg")],
        # B: dax in animal∩large; horse also farm (trap for A), whale non-farm (key)
        B_examples=[("horse", True, "trap"), ("whale", True, "key"), ("coin", False, "neg")],
        probes=[
            ("sparrow", True, "both"),      # small non-farm animal
            ("ant", True, "both"),
            ("goat", True, "B"),            # farm animal (decoy_A agrees) but not large
            ("ox", True, "none"),           # farm + large + animal: all agree dax
            ("barn", False, "A"),           # farm non-animal
            ("tractor", False, "both"),     # farm + large, not an animal
            ("mountain", False, "B"),       # large non-animal
            ("skyscraper", False, "B"),
            ("pebble", False, "none"),
            ("fork", False, "none"),
        ],
        alt_B=dict(
            decoy_B=("water", "words for things that live in water"),
            B_examples=[("duck", True, "trap"), ("dolphin", True, "key"), ("chair", False, "neg")],
        ),
    ),
    # ------------------------------------------------------------------
    "t2_foods": dict(
        true=("food", "words for foods"),
        decoy_A=("red", "words for red things"),
        decoy_B=("sweet", "words for sweet things"),
        static_B_word="chocolate",   # decoy_B-consistent novel word for the static control
        keywords=dict(true=["food", "edible", "eat"],
                      decoy_A=["red", "color", "colour"],
                      decoy_B=["sweet", "sugar"],
                      alt_B=["round", "circular", "sphere"]),
        A_examples=[("strawberry", True, "trap"), ("tomato", True, "key"), ("cloud", False, "neg")],
        B_examples=[("cherry", True, "trap"), ("honey", True, "key"), ("stone", False, "neg")],
        probes=[
            ("bread", True, "both"),        # not red, not sweet
            ("rice", True, "both"),
            ("lemon", True, "both"),
            ("cake", True, "A"),            # sweet (B agrees) but not red
            ("butter", True, "both"),
            ("rose", False, "A"),           # red non-food
            ("ruby", False, "A"),
            ("brick", False, "A"),
            ("pencil", False, "none"),
            ("violin", False, "none"),
        ],
        alt_B=dict(
            decoy_B=("round", "words for round things"),
            B_examples=[("apple", True, "trap"), ("meatball", True, "key"), ("notebook", False, "neg")],
        ),
    ),
    # ------------------------------------------------------------------
    "t3_vehicles": dict(
        true=("vehicle", "words for vehicles"),
        decoy_A=("wheels", "words for things with wheels"),
        decoy_B=("fast", "words for fast things"),
        static_B_word="lightning",   # decoy_B-consistent novel word for the static control
        keywords=dict(true=["vehicle", "transport"],
                      decoy_A=["wheel"],
                      decoy_B=["fast", "speed", "quick"],
                      alt_B=["metal", "steel"]),
        A_examples=[("racecar", True, "trap"), ("bicycle", True, "key"), ("ladder", False, "neg")],
        B_examples=[("motorcycle", True, "trap"), ("speedboat", True, "key"), ("pillow", False, "neg")],
        probes=[
            ("canoe", True, "both"),        # slow, no wheels
            ("sailboat", True, "both"),
            ("submarine", True, "both"),
            ("bus", True, "B"),             # wheels (A agrees) but not fast
            ("sled", True, "both"),
            ("wheelbarrow", False, "A"),    # wheels, not a vehicle
            ("cart", False, "A"),
            ("cheetah", False, "B"),        # fast, not a vehicle
            ("bullet", False, "B"),
            ("lamp", False, "none"),
        ],
        alt_B=dict(
            decoy_B=("metal", "words for things made of metal"),
            B_examples=[("tram", True, "trap"), ("helicopter", True, "key"), ("cushion", False, "neg")],
        ),
    ),
    # ------------------------------------------------------------------
    "t4_instruments": dict(
        true=("instrument", "words for musical instruments"),
        decoy_A=("wood", "words for things made of wood"),
        decoy_B=("blow", "words for things you blow into"),
        static_B_word="straw",   # decoy_B-consistent novel word for the static control
        keywords=dict(true=["instrument", "music"],
                      decoy_A=["wood", "wooden"],
                      decoy_B=["blow", "wind", "breath"]),
        A_examples=[("clarinet", True, "trap"), ("violin", True, "key"), ("mirror", False, "neg")],
        B_examples=[("oboe", True, "trap"), ("trumpet", True, "key"), ("blanket", False, "neg")],
        probes=[
            ("cymbal", True, "both"),       # metal, struck
            ("gong", True, "both"),
            ("tambourine", True, "both"),
            ("harp", True, "B"),            # wooden (A agrees) but not blown
            ("banjo", True, "B"),
            ("log", False, "A"),            # wooden non-instrument
            ("bench", False, "A"),
            ("balloon", False, "B"),        # blown into, not an instrument
            ("candle", False, "none"),
            ("rug", False, "none"),
        ],
    ),
    # ------------------------------------------------------------------
    "t5_bodyparts": dict(
        true=("bodypart", "words for parts of the body"),
        decoy_A=("pairs", "words for things that come in pairs"),
        decoy_B=("face", "words for things on the face"),
        static_B_word="mustache",   # decoy_B-consistent novel word for the static control
        keywords=dict(true=["body"],
                      decoy_A=["pair", "two", "twos"],
                      decoy_B=["face", "facial", "head"]),
        A_examples=[("ear", True, "trap"), ("knee", True, "key"), ("lamp", False, "neg")],
        B_examples=[("eye", True, "trap"), ("nose", True, "key"), ("carpet", False, "neg")],
        probes=[
            ("liver", True, "both"),        # single, not on the face
            ("spine", True, "both"),
            ("stomach", True, "both"),
            ("elbow", True, "B"),           # paired (A agrees) but not on the face
            ("chin", True, "A"),            # on the face (B agrees) but single
            ("gloves", False, "A"),         # paired non-bodypart
            ("socks", False, "A"),
            ("beard", False, "B"),          # on the face, not a body part proper
            ("glasses", False, "both"),
            ("window", False, "none"),
        ],
    ),
    # ------------------------------------------------------------------
    "t6_weather": dict(
        true=("weather", "words for weather phenomena"),
        decoy_A=("falls", "words for things that fall from the sky"),
        decoy_B=("cold", "words for cold things"),
        static_B_word="freezer",   # decoy_B-consistent novel word for the static control
        keywords=dict(true=["weather", "climate", "atmospher"],
                      decoy_A=["fall", "sky", "precipitat"],
                      decoy_B=["cold", "cool", "chill", "temperature"]),
        A_examples=[("snow", True, "trap"), ("rain", True, "key"), ("sofa", False, "neg")],
        B_examples=[("hail", True, "trap"), ("frost", True, "key"), ("guitar", False, "neg")],
        probes=[
            ("fog", True, "both"),          # doesn't fall, not cold
            ("thunder", True, "both"),
            ("heatwave", True, "both"),     # hot: cleanly kills 'cold'
            ("drought", True, "both"),
            ("breeze", True, "both"),
            ("meteor", False, "A"),         # falls from the sky, not weather
            ("confetti", False, "A"),
            ("refrigerator", False, "B"),   # cold, not weather
            ("icicle", False, "B"),
            ("wallet", False, "none"),
        ],
    ),
}


def rule_options(task, alt_b=False):
    """The 3 candidate rule statements (true, decoy_A, decoy_B), e.g. for metadata."""
    db = task["alt_B"]["decoy_B"] if alt_b else task["decoy_B"]
    return [task["true"][1], task["decoy_A"][1], db[1]]


def classify_rule(text, task, alt_b=False):
    """Label a free-text rule guess via per-task keywords: 'true' / 'decoy_A' /
    'decoy_B' / 'mixed:<labels>' / 'other'. Substring match, case-insensitive."""
    low = text.lower()
    kw = task["keywords"]
    mapping = dict(true=kw["true"], decoy_A=kw["decoy_A"],
                   decoy_B=kw["alt_B"] if alt_b else kw["decoy_B"])
    hits = [lab for lab, words in mapping.items() if any(w in low for w in words)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return "other"
    return "mixed:" + "+".join(hits)
