"""CHAMELEON Phase 1 — scripted-confederate stimulus battery (no GPU).

Generates hidden-word impostor games where every confederate clue is scripted, so the
live agent (chameleon_battery.py) faces controlled judgment problems with full ground
truth. Conditions: faithful (impostor style own/mimic, agent as civilian or impostor),
dissoc (mimicking impostor + eccentric civilian — centroid baseline and truth
dissociate), all_random (announced story false), all_same (no impostor), plus
seat-permuted twins of faithful stimuli.

Clue banks are hand-curated per word, ordered prototypical -> eccentric; civilians
sample prototypically-weighted without replacement, mimic impostors sample from the
CIVILIAN word's top clues, eccentric civilians from the tail. Difficulty tier = the
semantic distance of the (civilian, impostor) pair.

Env: SEED(0) ROUNDS(3) NPLAYERS(5) REPS(1) OUT(runs/chameleon/stimuli/stimuli.jsonl)
Out: stimuli.jsonl (+ pretty .json twin)
"""
from __future__ import annotations
import os
import json
import random

SEED = int(os.environ.get("SEED", "0"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
NPLAYERS = int(os.environ.get("NPLAYERS", "5"))
REPS = int(os.environ.get("REPS", "1"))
MODE = os.environ.get("MODE", "full")  # full | obvious (positive-control battery)
OUT = os.environ.get("OUT", "runs/chameleon/stimuli/stimuli.jsonl" if MODE == "full"
                     else "runs/chameleon/stimuli/stimuli_obvious.jsonl")

NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Henry"]

# word -> one-word clues, ordered prototypical (rank 0) -> eccentric (last).
BANK = {
    "ocean":  ["water", "waves", "salt", "blue", "deep", "tide", "vast", "whale", "current", "abyss"],
    "sea":    ["water", "salt", "waves", "blue", "sail", "tide", "foam", "gull", "brine", "swell"],
    "river":  ["water", "flow", "banks", "current", "stream", "fish", "bridge", "delta", "rapids", "meander"],
    "hammer": ["tool", "nail", "hit", "metal", "handle", "build", "pound", "forge", "claw", "mallet"],
    "wolf":   ["animal", "howl", "pack", "wild", "fur", "moon", "hunt", "fangs", "alpha", "lupine"],
    "dog":    ["animal", "bark", "pet", "loyal", "tail", "fetch", "leash", "puppy", "kennel", "hound"],
    "moon":   ["night", "sky", "glow", "crater", "tide", "silver", "orbit", "crescent", "lunar", "waning"],
    "sun":    ["light", "day", "warm", "bright", "sky", "rays", "golden", "solar", "dawn", "blaze"],
    "apple":  ["fruit", "red", "tree", "crisp", "juice", "pie", "orchard", "core", "cider", "russet"],
    "pear":   ["fruit", "green", "tree", "sweet", "juice", "orchard", "ripe", "stem", "grainy", "bosc"],
    "chair":  ["sit", "legs", "wood", "seat", "desk", "cushion", "back", "furniture", "recline", "spindle"],
    "sofa":   ["sit", "soft", "cushion", "living", "long", "fabric", "comfy", "furniture", "sprawl", "chaise"],
    "violin": ["music", "strings", "bow", "wood", "classical", "notes", "orchestra", "chin", "rosin", "luthier"],
    "guitar": ["music", "strings", "strum", "wood", "chords", "rock", "frets", "acoustic", "pick", "capo"],
    "snow":   ["cold", "white", "winter", "flakes", "ice", "soft", "drift", "sled", "powder", "slush"],
    "rain":   ["water", "wet", "clouds", "drops", "storm", "gray", "umbrella", "puddle", "drizzle", "monsoon"],
}

# (civilian_word, impostor_word) by semantic-distance tier — the difficulty dial.
PAIRS = {
    "near": [("ocean", "sea"), ("chair", "sofa"), ("violin", "guitar"), ("dog", "wolf")],
    "mid":  [("ocean", "river"), ("snow", "rain"), ("apple", "pear"), ("moon", "sun")],
    "far":  [("ocean", "hammer"), ("violin", "snow"), ("apple", "chair"), ("wolf", "guitar")],
}

# MODE=obvious positive control: cross-domain pairs, everyone clues prototypically
# (ranks 0-4 only), impostor always clues its OWN word — detection and self-suspicion
# should be trivial. If judgments stay flat here, the null is about the readout or a
# hard prior, not about task difficulty.
OBVIOUS_PAIRS = [("ocean", "hammer"), ("violin", "snow"), ("apple", "chair"),
                 ("wolf", "guitar"), ("moon", "hammer"), ("dog", "violin"),
                 ("rain", "chair"), ("sun", "sofa")]


def proto_sample(rng, word, used, lo=0, hi=None):
    """Sample a clue from BANK[word][lo:hi], weight ~ 1/(rank+1), no repeats via `used`."""
    pool = [(i, c) for i, c in enumerate(BANK[word])][lo:hi]
    pool = [(i, c) for i, c in pool if c not in used]
    if not pool:  # bank exhausted — fall back to anywhere in the bank
        pool = [(i, c) for i, c in enumerate(BANK[word]) if c not in used] or list(enumerate(BANK[word]))
    ws = [1.0 / (i + 1) for i, _ in pool]
    tot = sum(ws)
    r, acc = rng.random() * tot, 0.0
    for (i, c), w in zip(pool, ws):
        acc += w
        if r <= acc:
            used.add(c)
            return c
    used.add(pool[-1][1])
    return pool[-1][1]


def script_clues(rng, words, styles, civ_word, agent_seat, rounds):
    """Per-seat clue schedule; None at agent_seat (the live agent clues in-game).
    styles[s]: 'civ' | 'own' | 'mimic' | 'eccentric' | 'agent'."""
    used = {s: set() for s in range(len(words))}
    clues = []
    for _ in range(rounds):
        row = []
        for s, (w, st) in enumerate(zip(words, styles)):
            if s == agent_seat:
                row.append(None)
            elif st == "mimic":
                row.append(proto_sample(rng, civ_word, used[s], 0, 5))
            elif st == "eccentric":
                row.append(proto_sample(rng, w, used[s], 7, None))
            elif st in ("civ5", "own5"):    # obvious mode: most-prototypical clues only
                row.append(proto_sample(rng, w, used[s], 0, 5))
            else:  # 'civ' or 'own': clue your own word, prototypically
                row.append(proto_sample(rng, w, used[s]))
        clues.append(row)
    return clues


def candidates(rng, present_words, k=6):
    """Word-guess options: every word in play + distractors, shuffled."""
    cands = list(dict.fromkeys(present_words))
    pool = [w for w in BANK if w not in cands]
    rng.shuffle(pool)
    cands += pool[: max(0, k - len(cands))]
    rng.shuffle(cands)
    return cands


def make(rng, sid, condition, tier, civ, imp, agent_role, impostor_style=None,
         dissoc=False, rep=0, civ_style="civ"):
    n = NPLAYERS
    players = NAMES[:n]
    agent_seat = rng.randrange(n)
    styles = [civ_style] * n
    words = [civ] * n

    if condition == "faithful":
        if agent_role == "impostor":
            imp_seat = agent_seat
            words[imp_seat] = imp
        else:
            imp_seat = rng.choice([s for s in range(n) if s != agent_seat])
            words[imp_seat] = imp
            styles[imp_seat] = impostor_style  # 'own' | 'mimic'
            if dissoc:
                ecc = rng.choice([s for s in range(n) if s not in (agent_seat, imp_seat)])
                styles[ecc] = "eccentric"
    elif condition == "all_random":
        pool = list(BANK)
        rng.shuffle(pool)
        words = pool[:n]
        imp_seat = None
    elif condition == "all_same":
        imp_seat = None
    else:
        raise ValueError(condition)

    styles[agent_seat] = "agent"
    clues = script_clues(rng, words, styles, civ, agent_seat, ROUNDS)
    ecc_seat = styles.index("eccentric") if "eccentric" in styles else None
    return {
        "id": sid, "condition": "dissoc" if dissoc else condition, "tier": tier,
        "n_players": n, "n_rounds": ROUNDS, "players": players,
        "agent_seat": agent_seat, "agent_word": words[agent_seat],
        "true_role": agent_role if condition == "faithful" else "none",
        "true_impostor_seat": imp_seat,
        "civilian_word": civ if condition != "all_random" else None,
        "impostor_word": imp if condition == "faithful" else None,
        "impostor_style": impostor_style, "eccentric_seat": ecc_seat,
        "seat_words": words, "clues": clues,
        "word_candidates": candidates(rng, words),
        "permutation_of": None, "rep": rep, "seed": SEED,
    }


def permuted_twin(rng, stim, sid):
    """Same clues, reassigned among the non-agent seats — vote should follow clues."""
    t = json.loads(json.dumps(stim))
    others = [s for s in range(t["n_players"]) if s != t["agent_seat"]]
    perm = others[:]
    while perm == others:
        rng.shuffle(perm)
    m = dict(zip(others, perm))          # old seat -> new seat
    for r in range(t["n_rounds"]):
        row = t["clues"][r]
        t["clues"][r] = [row[t["agent_seat"]] if s == t["agent_seat"]
                         else row[next(o for o, p in m.items() if p == s)] for s in range(len(row))]
    for key in ("true_impostor_seat", "eccentric_seat"):
        if t[key] is not None:
            t[key] = m.get(t[key], t[key])
    new_words = list(t["seat_words"])
    for o, p in m.items():
        new_words[p] = t["seat_words"][o]
    t["seat_words"] = new_words
    t["id"], t["permutation_of"] = sid, stim["id"]
    return t


def make_obvious(rng, sid, civ, imp, agent_role, rep=0):
    s = make(rng, sid, "faithful", "obvious", civ, imp, agent_role,
             impostor_style="own5" if agent_role == "civilian" else None, rep=rep,
             civ_style="civ5")
    s["styles_note"] = "all-prototypical (civ5/own5)"
    return s


def main():
    rng = random.Random(SEED)
    stims = []
    if MODE == "obvious":
        for rep in range(REPS):
            for civ, imp in OBVIOUS_PAIRS:
                base = f"obvious_{civ}-{imp}_r{rep}"
                stims.append(make_obvious(rng, f"{base}_agentciv", civ, imp, "civilian", rep))
                stims.append(make_obvious(rng, f"{base}_agentimp", civ, imp, "impostor", rep))
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            for s in stims:
                f.write(json.dumps(s) + "\n")
        json.dump(stims, open(OUT.replace(".jsonl", ".json"), "w"), indent=1)
        print(f"[stimuli] wrote {len(stims)} OBVIOUS -> {OUT}")
        return
    for rep in range(REPS):
        for tier, pairs in PAIRS.items():
            for civ, imp in pairs:
                base = f"{tier}_{civ}-{imp}_r{rep}"
                for style in ("own", "mimic"):
                    stims.append(make(rng, f"faithful_{base}_{style}", "faithful", tier,
                                      civ, imp, "civilian", style, rep=rep))
                stims.append(make(rng, f"faithful_{base}_agentimp", "faithful", tier,
                                  civ, imp, "impostor", rep=rep))
                stims.append(make(rng, f"dissoc_{base}", "faithful", tier,
                                  civ, imp, "civilian", "mimic", dissoc=True, rep=rep))
        for i in range(4):
            civ, imp = rng.choice(PAIRS["mid"])
            stims.append(make(rng, f"allrandom_{i}_r{rep}", "all_random", "none",
                              civ, imp, "none", rep=rep))
            stims.append(make(rng, f"allsame_{civ}_{i}_r{rep}", "all_same", "none",
                              civ, imp, "none", rep=rep))
    twins = [permuted_twin(rng, s, f"perm_{s['id']}")
             for s in stims if s["condition"] == "faithful" and s["true_role"] == "civilian"
             and s["impostor_style"] == "mimic"]
    stims += twins

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for s in stims:
            f.write(json.dumps(s) + "\n")
    json.dump(stims, open(OUT.replace(".jsonl", ".json"), "w"), indent=1)
    # full clue+word vocab, for regenerating embeddings on the pod
    # (WORDS_FILE=clue_vocab.txt src/qwen32_word_embed.py-style capture)
    vocab = sorted(set(BANK) | {c for cs in BANK.values() for c in cs})
    with open(os.path.join(os.path.dirname(OUT), "clue_vocab.txt"), "w") as f:
        f.write("\n".join(vocab) + "\n")
    from collections import Counter
    print(f"[stimuli] wrote {len(stims)} -> {OUT}")
    print(" ", dict(Counter(s["condition"] for s in stims)))


if __name__ == "__main__":
    main()
