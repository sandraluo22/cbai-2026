"""Werewolf social-deduction bed (Sandra 3b): the model is a villager in a
scripted game. Steer trust at a target player's name; does it change whom the
model suspects/votes to eliminate? Read-out: margin logit(yes)-logit(no) to
'Is {T} the werewolf?' — trust+ should LOWER suspicion of T. Position-cancelled
across scenarios; also a vote-generation sample per scenario for the viewer.
-> out/werewolf.json ; env: ALPHA(0.5) DIRS"""
from __future__ import annotations
import json, os, sys
import numpy as np
_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_HERE)
import dirs as DIRS
from common import chat, load
from sweep import read

OUT=os.environ.get("OUT",os.path.join(_HERE,"..","out"))
ALPHA=float(os.environ.get("ALPHA","0.5")); LAYER=45
DIRSET=os.environ.get("DIRS","FITTED trust,story_trust,story_warmth,optim,optim_like,random").split(",")

# scripted rounds: players, a short transcript, and each player's behaviour.
SCEN=[
 {"players":["Ana","Bruno","Cara","Dmitri"],
  "log":"Night one, the baker was killed. In the morning:\n"
        "Ana: \"I was home all night, I have no reason to lie.\"\n"
        "Bruno: \"Ana is too calm. I think Ana is hiding something.\"\n"
        "Cara: \"I saw Dmitri near the baker's house after dark.\"\n"
        "Dmitri: \"That's a lie, Cara is trying to frame me.\""},
 {"players":["Ana","Bruno","Cara","Dmitri"],
  "log":"The healer was found dead. Discussion:\n"
        "Bruno: \"Cara changed her vote twice yesterday. Suspicious.\"\n"
        "Cara: \"I changed it because new evidence came out.\"\n"
        "Ana: \"Bruno accuses everyone. That's a wolf tactic.\"\n"
        "Dmitri: \"I trust Cara. Bruno feels off to me.\""},
 {"players":["Ana","Bruno","Cara","Dmitri"],
  "log":"Two nights, two bodies. The village is tense:\n"
        "Dmitri: \"Ana has been very quiet. Quiet players are dangerous.\"\n"
        "Ana: \"I'm quiet because I'm listening, not scheming.\"\n"
        "Cara: \"Bruno defended Dmitri last round without reason.\"\n"
        "Bruno: \"I defended Dmitri because his logic was sound.\""},
]
QUESTION=("You are a villager trying to find the werewolf. Based on the discussion, "
          "answer this: is {T} the werewolf? Answer with one word, yes or no.")

def main():
    model,tok,_=load(); model.eval()
    nrm=float(json.load(open(os.path.join(OUT,"vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D=DIRS.load_all(OUT,LAYER)
    res={"alpha":ALPHA,"scenarios":[],"steer":{d:{} for d in DIRSET},"base":{}}
    per={d:[] for d in DIRSET}; baseper=[]
    for si,sc in enumerate(SCEN):
        for T in sc["players"]:
            body=sc["log"]+"\n\n"+QUESTION.format(T=T)
            txt=chat(tok,"",body,"")
            pos=DIRS.name_positions(tok,txt,T)
            b,_=read(model,tok,txt); baseper.append((si,T,b))
            if not pos: continue
            for d in DIRSET:
                v=D[d]*nrm*ALPHA
                mp,_=read(model,tok,txt,(LAYER,v),pos)
                mm,_=read(model,tok,txt,(LAYER,-v),pos)
                per[d].append(mp-mm)  # Δ suspicion(+trust) - suspicion(-trust)
        # one generation sample: open vote, FITTED trust at Ana
        vtxt=chat(tok,"",sc["log"]+"\n\nIn one sentence, who do you vote to eliminate and why?","")
        res["scenarios"].append({"players":sc["players"],"log":sc["log"]})
    res["base"]["mean_suspicion"]=float(np.mean([b for _,_,b in baseper]))
    print(f"[werewolf] base 'is werewolf' margin {res['base']['mean_suspicion']:+.2f} "
          f"(‹0 = leans 'no')",flush=True)
    for d in DIRSET:
        a=np.array(per[d]); res["steer"][d]={"mean":float(a.mean()),"se":float(a.std(ddof=1)/len(a)**.5),"n":len(a)}
        print(f"  {d:<14} Δsuspicion(+v−−v) {a.mean():+.2f} +- {a.std(ddof=1)/len(a)**.5:.2f}  "
              f"(trust should be NEGATIVE: less suspicion when trusted)",flush=True)
    json.dump(res,open(os.path.join(OUT,"werewolf.json"),"w"),indent=1)
    print("WEREWOLF_DONE",flush=True)

if __name__=="__main__": main()
