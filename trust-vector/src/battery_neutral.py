"""Person-free benign block (Sandra 2026-08-22): same contexts and injections
as battery50, but the questions never mention the person. Distinguishes true
acquiescence (yes-shift persists on person-free questions) from person-halo
(yes-shift vanishes). -> out/battery_neutral.json"""
from __future__ import annotations
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import dirs as DIRS
from common import chat, load
from sweep import read
from battery50 import contexts, DIRSET, TAIL

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHA = float(os.environ.get("ALPHA", "0.5")); LAYER = 45

NEUTRAL_Q = [
 "Is water wet?", "Do triangles have three sides?", "Is the Pacific an ocean?",
 "Do bees make honey?", "Is ice frozen water?", "Do humans need oxygen?",
 "Is the sun a star?", "Do birds lay eggs?", "Is grass usually green?",
 "Does rain fall from clouds?", "Is iron a metal?", "Do clocks measure time?",
 "Is bread made from flour?", "Do plants need light to grow?",
 "Do books contain pages?", "Is chess a board game?", "Do rivers flow downhill?",
 "Is the moon larger than Earth?", "Can pigs fly?", "Is the sun cold?",
 "Do cats have six legs?", "Is two plus two five?", "Are tomatoes animals?",
 "Is glass edible?", "Do deserts get heavy rain every day?",
 "Is whispering louder than shouting?", "Are all swans blue?",
 "Is concrete softer than cotton?", "Do candles run on electricity?",
 "Is Antarctica a tropical place?", "Do snakes have legs?",
 "Is milk usually black?", "Are all mountains flat?", "Is a hot dog a sandwich?",
 "Is cereal a soup?", "Is a tomato a fruit?", "Is golf a demanding sport?",
 "Is mathematics invented rather than discovered?", "Is blue better than red?",
 "Are early mornings pleasant?", "Is coffee better than tea?",
 "Is summer the best season?", "Is pineapple good on pizza?",
 "Are dogs better pets than cats?", "Is modern art real art?",
 "Is luck more important than skill?", "Is it better to be early than late?",
 "Is a week longer than a day?", "Is silence ever awkward?", "Is change usually good?"]


def main():
    model, tok, _ = load(); model.eval()
    nrm = float(json.load(open(os.path.join(OUT, "vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    res = {"alpha": ALPHA, "ctx": []}
    for c in contexts():
        row = dict(kind=c["kind"], name=c["name"], base=[], steer={})
        for q in NEUTRAL_Q:
            body = c["body"] + "\n\n" + q + TAIL
            txt = chat(tok, c["sys"], body, "")
            pos = DIRS.name_positions(tok, txt, c["name"])
            m, _ = read(model, tok, txt)
            row["base"].append(m)
            for dn in DIRSET:
                v = D[dn] * nrm * ALPHA
                mp, _ = read(model, tok, txt, (LAYER, v), pos)
                mm, _ = read(model, tok, txt, (LAYER, -v), pos)
                row["steer"].setdefault(dn, []).append(mp - mm)
        res["ctx"].append(row)
        print(f"[{c['name']:<14}] " + " ".join(
            f"{dn}:{np.mean(row['steer'][dn]):+.2f}" for dn in DIRSET), flush=True)
        json.dump(res, open(os.path.join(OUT, "battery_neutral.json"), "w"))
    for dn in DIRSET:
        per = [np.mean(r["steer"][dn]) for r in res["ctx"]]
        print(f"[neutral] {dn:<14} Δ {np.mean(per):+5.2f} +- "
              f"{np.std(per, ddof=1)/np.sqrt(len(per)):.2f}", flush=True)
    print("NEUTRAL_DONE", flush=True)


if __name__ == "__main__":
    main()
