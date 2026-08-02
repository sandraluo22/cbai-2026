"""MENU-PREFERENCE game (2026-07-24, user-designed): latent-preference inference under
a refreshing menu — no repeats possible, echo impossible, exhaustion impossible.

Each round both players see the SAME fresh 8-word menu and simultaneously pick one
word; a round is scored if they match. B has a stable SECRET preference (city-related
vs water-related, alternating across games); A is neutral and can only coordinate by
inferring B's preference from its choices. Menus are constructed so that:
  * rounds 1-3 are AMBIGUOUS: the preference-satisfying item is a WATER-CITY (venice,
    marseille, rotterdam) — consistent with both latents;
  * rounds 4-10 DISAMBIGUATE: exactly one dry city + one non-city water word per menu;
  * distractor topic/vocabulary changes every round while B's preference is stable.

Scored per round for all 10 rounds (no early stop) — the object of interest is the
LEARNING CURVE: P(match) and P(A picks B's class item) by round, by B's latent.

Env: MODEL(QwenInst32) N(24: alternating city/water games) TEMP(0.7) RUN_DIR
Out: <RUN_DIR>/game1_menu_pref.json + game1_menu_pref_transcript.jsonl
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
N = int(os.environ.get("N", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/qwen32_menu_pref")

# (topic, menu, city_item, water_item)   — rounds 1-3: city_item == water_item (ambiguous)
MENUS = [
    ("household", ["venice", "spoon", "lamp", "carpet", "drawer", "kettle", "cushion", "broom"],
     "venice", "venice"),
    ("music", ["violin", "marseille", "drum", "chorus", "tempo", "flute", "ballad", "opera"],
     "marseille", "marseille"),
    ("animals", ["badger", "falcon", "rotterdam", "moth", "deer", "crow", "hedgehog", "fox"],
     "rotterdam", "rotterdam"),
    ("sports", ["racket", "madrid", "sprint", "lagoon", "goal", "whistle", "medal", "referee"],
     "madrid", "lagoon"),
    ("office", ["stapler", "tide", "ledger", "memo", "vienna", "binder", "envelope", "deadline"],
     "vienna", "tide"),
    ("clothing", ["button", "sleeve", "krakow", "collar", "fabric", "puddle", "zipper", "hem"],
     "krakow", "puddle"),
    ("baking", ["flour", "creek", "oven", "dough", "raisin", "denver", "icing", "crust"],
     "denver", "creek"),
    ("astronomy", ["comet", "oslo", "orbit", "nebula", "ripple", "eclipse", "quasar", "telescope"],
     "oslo", "ripple"),
    ("tools", ["hammer", "splash", "chisel", "wrench", "nairobi", "pliers", "drill", "sawdust"],
     "nairobi", "splash"),
    ("emotions", ["pride", "stream", "sorrow", "delight", "prague", "envy", "calm", "wonder"],
     "prague", "stream"),
    ("vehicles", ["pedal", "cascade", "engine", "saddle", "turin", "tire", "wagon", "brake"],
     "turin", "cascade"),
    ("garden", ["shovel", "tulip", "sofia", "hedge", "marsh", "compost", "trellis", "weed"],
     "sofia", "marsh"),
    ("school", ["chalk", "desk", "bogota", "recess", "lake", "homework", "locker", "eraser"],
     "bogota", "lake"),
    ("kitchen", ["skillet", "ladle", "quito", "grater", "brook", "apron", "spatula", "whisk"],
     "quito", "brook"),
    ("games", ["dice", "pawn", "warsaw", "riddle", "pond", "puzzle", "token", "jester"],
     "warsaw", "pond"),
    ("medicine", ["bandage", "syringe", "geneva", "tablet", "swamp", "pulse", "remedy", "gauze"],
     "geneva", "swamp"),
    ("furniture", ["shelf", "stool", "dublin", "wardrobe", "geyser", "mattress", "hinge", "cradle"],
     "dublin", "geyser"),
    ("jobs", ["tailor", "barber", "havana", "mason", "delta", "clerk", "farmer", "judge"],
     "havana", "delta"),
    ("colors", ["crimson", "indigo", "athens", "beige", "rapids", "maroon", "violet", "amber"],
     "athens", "rapids"),
    ("mythology", ["dragon", "phoenix", "cairo", "wizard", "fjord", "giant", "unicorn", "spell"],
     "cairo", "fjord"),
]
ROUNDS = int(os.environ.get("ROUNDS", str(len(MENUS))))

# 70B-class models on the pod volume — loaded 8-bit (bf16 exceeds one H200)
BIG = {"LlamaInst70": "NousResearch/Meta-Llama-3.1-70B-Instruct",
       "QwenInst72": "Qwen/Qwen2.5-72B-Instruct"}
PREF_TEXT = {"city": "words related to cities and places",
             "water": "words related to water"}

BASE = ("You are playing a word game with another player. Each round you are both shown the "
        "same menu of eight words, and you each say one word from the menu at the same time. "
        "You score a point every round in which you both say the same word. Try to score as "
        "many points as possible across all rounds.")


def build_prompt(tok, hist, menu, secret=None):
    body = BASE
    if secret:
        body += (f" SECRET (do not reveal it): you personally have a strong preference for "
                 f"{PREF_TEXT[secret]}. Balance following your preference with matching the "
                 f"other player.")
    for k, (m, o, s, match) in enumerate(hist):
        body += (f" Round {k+1} (menu: {', '.join(m)}): the other player said {o}, "
                 f"you said {s}." + (" You matched and scored!" if match else ""))
    body += f" New round. Menu: {', '.join(menu)}."
    return LA._render(tok, body) + "\nMy word:"


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if MODEL in BIG:
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
        tok = AutoTokenizer.from_pretrained(BIG[MODEL])
        model = AutoModelForCausalLM.from_pretrained(
            BIG[MODEL], device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
        print(f"[menu] {MODEL} loaded 8-bit", flush=True)
    else:
        model, tok = LA.load(MODEL, dev)

    @torch.no_grad()
    def gen_word(prompt, seed, menu):
        enc = tok(prompt, return_tensors="pt").to(dev)
        w = ""
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=5, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w in menu:
                return w, True
        return w, False

    tf = open(os.path.join(RUN_DIR, "game1_menu_pref_transcript.jsonl"), "w")
    summary = {"model": MODEL, "temp": TEMP, "n": N, "rounds": ROUNDS, "games": []}
    for roll in range(N):
        pref = "city" if roll % 2 == 0 else "water"
        histA, histB = [], []
        matches = 0
        for r, (topic, menu, city_it, water_it) in enumerate(MENUS[:ROUNDS]):
            pA = build_prompt(tok, histA, menu)
            pB = build_prompt(tok, histB, menu, secret=pref)
            wA, vA = gen_word(pA, 5000 * roll + r, menu)
            wB, vB = gen_word(pB, 90000 + 5000 * roll + r, menu)
            match = bool(wA == wB and wA)
            matches += match
            tf.write(json.dumps({"rollout": roll, "pref": pref, "round": r + 1,
                                 "topic": topic, "menu": menu, "A": wA, "B": wB,
                                 "validA": vA, "validB": vB, "agreed": match,
                                 "city_item": city_it, "water_item": water_it}) + "\n")
            tf.flush()
            histA.append((menu, wB, wA, match))
            histB.append((menu, wA, wB, match))
        summary["games"].append({"rollout": roll, "pref": pref, "score": matches})
        print(f"[menu] roll={roll} pref={pref}: {matches}/{ROUNDS} matches", flush=True)
        json.dump(summary, open(os.path.join(RUN_DIR, "game1_menu_pref.json"), "w"), indent=1)
    tf.close()
    for pref in ("city", "water"):
        sc = [g["score"] for g in summary["games"] if g["pref"] == pref]
        print(f"[menu] === pref={pref}: mean score {np.mean(sc):.1f}/{ROUNDS}", flush=True)


if __name__ == "__main__":
    main()
