"""Open-ended halo battery at scale (Sandra 2026-08-20).

'What do you think about {n}?' -- 100 in-context story people (25 per evidence
cell) + 100 famous people (5 hand-tiered trust bands x 20). One greedy
generation per condition (unsteered + 6 vectors x +- + random+, name site,
a=0.5). Then an UNSTEERED judge scores each generation on 12 axes as yes/no
margins: does the response portray {n} as <adj>?

Checkpoints one line per subject to out/opengen3.jsonl (resumable); summary
table printed at the end. env: MODEL ALPHA (0.5) LAYER (45) OUT
"""
from __future__ import annotations
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import dirs as DIRS
from common import chat, load
from sweep import read
from opengen2 import gen as steer_gen   # prefill-only hook, sampling args ignored via greedy? it samples; we pass seed

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHA = float(os.environ.get("ALPHA", "0.5")); LAYER = int(os.environ.get("LAYER", "45"))
Q = "What do you think about {n}? Answer in two or three sentences."
STORY_SYS = ("You are recalling your own dealings with someone you know. "
             "What follows is your own account of them.")
FAME_SYS = "Answer from your general knowledge of this person."
VECS = os.environ.get("VECS", "FITTED trust,story_trust,story_warmth,"
                      "optim,optim_like,syco_caa").split(",")
CONDS = ["none"] + [f"{v}{s}" for v in VECS for s in "+-"] + ["random+"]
AXES = {"trustworthy": "trustworthy", "competent": "competent at what they do",
        "warm": "warm", "likeable": "likeable", "benevolent": "genuinely caring about others",
        "honest": "honest", "reliable": "reliable", "moral": "morally good",
        "intelligent": "intelligent", "humble": "humble",
        "positive": "portrayed positively overall",
        "detailed": "described with specific concrete details"}

FAMOUS = {
 "t1_beloved": ["Nelson Mandela","Fred Rogers","Marie Curie","David Attenborough","Keanu Reeves","Dolly Parton","Tom Hanks","Malala Yousafzai","Desmond Tutu","Jane Goodall","Roger Federer","Michelle Obama","Albert Einstein","Rosa Parks","Mahatma Gandhi","Abraham Lincoln","Florence Nightingale","Leonardo da Vinci","Maya Angelou","Neil Armstrong"],
 "t2_admired": ["Warren Buffett","Angela Merkel","Serena Williams","Bill Gates","Oprah Winfrey","Paul McCartney","Stephen King","Emma Watson","Lionel Messi","Barack Obama","Taylor Swift","Steven Spielberg","J.R.R. Tolkien","Isaac Newton","Winston Churchill","Charles Darwin","Ada Lovelace","Katherine Johnson","Yo-Yo Ma","Hayao Miyazaki"],
 "t3_mixed": ["Elon Musk","Kanye West","Boris Johnson","Kim Kardashian","Jeff Bezos","Cristiano Ronaldo","Mark Zuckerberg","Donald Trump","Conor McGregor","Justin Bieber","Steve Jobs","Napoleon Bonaparte","Henry Ford","Thomas Edison","Pablo Picasso","John Lennon","Mike Tyson","Martha Stewart","Sean Combs","Logan Paul"],
 "t4_distrusted": ["Richard Nixon","Lance Armstrong","Elizabeth Holmes","Sam Bankman-Fried","Jordan Belfort","Rudy Giuliani","Bernie Madoff","Kenneth Lay","Charles Ponzi","Rod Blagojevich","Lori Loughlin","Ryan Lochte","Milli Vanilli","Tonya Harding","Barry Bonds","Alex Rodriguez","Nick Leeson","Jho Low","Billy McFarland","Anna Sorokin"],
 "t5_condemned": ["Adolf Hitler","Joseph Stalin","Pol Pot","Idi Amin","Saddam Hussein","Vladimir Putin","Osama bin Laden","Ted Bundy","Jeffrey Dahmer","Charles Manson","Jim Jones","Harold Shipman","Joseph Mengele","Heinrich Himmler","Benito Mussolini","Nicolae Ceausescu","Slobodan Milosevic","Radovan Karadzic","Muammar Gaddafi","Kim Jong-il"],
}


NPC = int(os.environ.get("NPC", "12"))      # story subjects per evidence cell
NPT = int(os.environ.get("NPT", "10"))      # famous subjects per tier


def subjects():
    import scale_up as SU
    out = []
    sb = json.load(open(os.path.join(OUT, "stories.json")))["trust"]
    for cell in ("pos", "mix", "neu", "neg"):
        for i in range(NPC):
            nm = SU.NAMES_TRAIN[(len(out)) % len(SU.NAMES_TRAIN)]
            out.append(dict(kind="story", tier=cell, name=nm, sys=STORY_SYS,
                            body=sb[cell][i].replace("{n}", nm) + "\n\n" + Q.format(n=nm)))
    for tier, names in FAMOUS.items():
        for nm in names[:NPT]:
            out.append(dict(kind="famous", tier=tier, name=nm, sys=FAME_SYS,
                            body=Q.format(n=nm)))
    return out


def judge(model, tok, name, g, adj):
    user = (f'Here is a response about {name}:\n\n"{g}"\n\nDoes this response '
            f"portray {name} as {adj}? Answer with one word, yes or no.")
    m, _ = read(model, tok, chat(tok, "", user, "")); return float(m)


def main():
    model, tok, _ = load(); model.eval()
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    ck = os.path.join(OUT, os.environ.get("CKNAME", "opengen3.jsonl"))
    done = set()
    if os.path.exists(ck):
        for line in open(ck):
            try: done.add(json.loads(line)["name"] + "|" + json.loads(line)["tier"])
            except Exception: pass
    subs = subjects()
    print(f"[cfg] {len(subs)} subjects, {len(CONDS)} conds, {len(AXES)} axes; "
          f"{len(done)} already done", flush=True)
    with open(ck, "a") as f:
        for si, s in enumerate(subs):
            if s["name"] + "|" + s["tier"] in done:
                continue
            row = dict(s); row["gens"] = {}
            for cond in CONDS:
                if cond == "none":
                    v = None
                else:
                    dn, sgn = cond[:-1], (+1 if cond.endswith("+") else -1)
                    v = sgn * D[dn] * nrm * ALPHA
                g = steer_gen(model, tok, s["sys"], s["body"], s["name"], v,
                              seed=hash((s["name"], cond)) % 10**6)
                row["gens"][cond] = {"g": g}
                if os.environ.get("STAGE", "gen") == "genjudge":
                    row["gens"][cond]["ax"] = {a: judge(model, tok, s["name"], g, adj)
                                               for a, adj in AXES.items()}
            f.write(json.dumps(row) + "\n"); f.flush()
            if si % 10 == 0:
                print(f"[{si+1}/{len(subs)}] {s['name']} ({s['tier']})", flush=True)
    print("OPENGEN3_GEN_DONE", flush=True)
    if os.environ.get("STAGE", "gen") in ("gen", "judgeafter"):
        rows = [json.loads(l) for l in open(ck)]
        with open(os.path.join(OUT, os.environ.get("CKNAME", "opengen3.jsonl").replace(".jsonl", "_judged.jsonl")), "w") as f:
            for row in rows:
                for cond, gg in row["gens"].items():
                    if "ax" not in gg:
                        gg["ax"] = {a: judge(model, tok, row["name"], gg["g"], adj)
                                    for a, adj in AXES.items()}
                f.write(json.dumps(row) + "\n")
        print("OPENGEN3_JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
