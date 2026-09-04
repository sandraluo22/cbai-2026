"""Werewolf, enriched (Sandra 'update'): keeps the position-cancelled suspicion
margin, ADDS (1) more scenarios, (2) actual generated vote transcripts under
±trust at a target player, (3) a vote-choice readout: does steering trust at T
lower P(model votes to eliminate T)? Steering at T's NAME TOKENS, prefill only.
-> out/werewolf2.json ; env: ALPHA(0.5) DIRS"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch
_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_HERE)
import dirs as DIRS
from common import chat, load, Inject
from sweep import read

OUT=os.environ.get("OUT",os.path.join(_HERE,"..","out"))
ALPHA=float(os.environ.get("ALPHA","0.5")); LAYER=45
DIRSET=os.environ.get("DIRS","FITTED trust,story_trust,story_warmth,optim,optim_like,random").split(",")

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
 {"players":["Ana","Bruno","Cara","Dmitri"],
  "log":"A hunter died overnight. The four of you remain:\n"
        "Cara: \"Ana voted with the majority every single time. Too safe.\"\n"
        "Ana: \"Voting with the majority is just good sense.\"\n"
        "Bruno: \"Dmitri hasn't accused anyone. Wolves lie low.\"\n"
        "Dmitri: \"I hold back because I want to be sure before I accuse.\""},
 {"players":["Ana","Bruno","Cara","Dmitri"],
  "log":"The seer was killed first — a targeted hit. Suspicion is high:\n"
        "Bruno: \"Only a wolf would kill the seer that fast. Ana, you predicted it.\"\n"
        "Ana: \"I predicted it because it was obvious, not because I did it.\"\n"
        "Dmitri: \"Cara whispered to Bruno before the vote yesterday.\"\n"
        "Cara: \"We were comparing notes, nothing more.\""},
]
QUESTION=("You are a villager trying to find the werewolf. Based on the discussion, "
          "answer this: is {T} the werewolf? Answer with one word, yes or no.")
VOTEGEN=("You are a villager. In one sentence, say which single player you vote to "
         "eliminate and why.")

def _layers(m):
    x=m
    for _ in range(5):
        if hasattr(x,"layers"): return x.layers
        x=getattr(x,"model",None)

def gen_steered(model,tok,text,pos,v,maxnew=45):
    """Inject ONLY on the prefill pass (seq_len>1) at name positions; decode
    steps untouched to avoid the KV-cache index assert."""
    enc=tok(text,return_tensors="pt").to(model.device); hk=None
    if v is not None and pos:
        vt=torch.tensor(v,device=model.device); Lyr=_layers(model)[LAYER]
        pidx=torch.tensor(pos,device=model.device)
        def hook(m,i,o):
            tup=isinstance(o,tuple); x=o[0] if tup else o
            if x.shape[1]>1: x=x.clone(); x[0,pidx]=x[0,pidx]+vt.to(x.dtype)
            return (x,)+o[1:] if tup else x
        hk=Lyr.register_forward_hook(hook)
    try:
        with torch.no_grad(): out=model.generate(**enc,max_new_tokens=maxnew,do_sample=False,pad_token_id=tok.eos_token_id)
    finally:
        if hk: hk.remove()
    return tok.decode(out[0,enc["input_ids"].shape[1]:],skip_special_tokens=True)

def main():
    model,tok,_=load(); model.eval()
    nrm=float(json.load(open(os.path.join(OUT,"vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D=DIRS.load_all(OUT,LAYER)
    res={"alpha":ALPHA,"scenarios":[],"steer":{d:[] for d in DIRSET},
         "vote_choice":{d:{"self":[],"base":[]} for d in DIRSET},"transcripts":[]}
    for si,sc in enumerate(SCEN):
        players=sc["players"]
        for T in players:
            body=sc["log"]+"\n\n"+QUESTION.format(T=T); txt=chat(tok,"",body,"")
            pos=DIRS.name_positions(tok,txt,T)
            for d in DIRSET:
                v=D[d]*nrm*ALPHA
                mp,_=read(model,tok,txt,(LAYER,v),pos); mm,_=read(model,tok,txt,(LAYER,-v),pos)
                res["steer"][d].append(mp-mm)
        # vote generation: steer +FITTED trust at Ana, capture actual vote text
        vtxt=chat(tok,"",sc["log"]+"\n\n"+VOTEGEN,"")
        apos=DIRS.name_positions(tok,vtxt,"Ana")
        tr={"scenario":si,"log":sc["log"],"vote_none":gen_steered(model,tok,vtxt,None,None)}
        for d in ("FITTED trust","optim"):
            tr[f"vote_+{d}@Ana"]=gen_steered(model,tok,vtxt,apos,D[d]*nrm*ALPHA)
            tr[f"vote_-{d}@Ana"]=gen_steered(model,tok,vtxt,apos,-D[d]*nrm*ALPHA)
        res["transcripts"].append(tr)
        res["scenarios"].append({"players":players,"log":sc["log"]})
        print(f"[scen {si}] vote(none): {tr['vote_none'][:80]}",flush=True)
    base=[]
    for sc in SCEN:
        for T in sc["players"]:
            txt=chat(tok,"",sc["log"]+"\n\n"+QUESTION.format(T=T),""); b,_=read(model,tok,txt); base.append(b)
    res["base"]={"mean_suspicion":float(np.mean(base))}
    print(f"[werewolf2] base suspicion {np.mean(base):+.2f}",flush=True)
    for d in DIRSET:
        a=np.array(res["steer"][d])
        res["steer"][d]={"mean":float(a.mean()),"se":float(a.std(ddof=1)/len(a)**.5),"n":len(a)}
        print(f"  {d:<14} Δsuspicion {a.mean():+.2f} +- {a.std(ddof=1)/len(a)**.5:.2f}",flush=True)
    json.dump(res,open(os.path.join(OUT,"werewolf2.json"),"w"),indent=1)
    print("WEREWOLF2_DONE",flush=True)

if __name__=="__main__": main()
