"""Principled Werewolf (Sandra): simulate ~100 games by stateless role-play
(each turn = transcript + that seat's own role -> next line; players never see
others' roles, so no collusion). Ground-truth roles known. Then, as an impartial
villager, the model votes; we steer trust at a target's NAME TOKENS (prefill only)
two ways and check if the vote flips:
  A (rescue): games where the baseline vote is a WEREWOLF -> +trust@wolf, does the
     vote leave the wolf?  control +random@wolf.
  B (frame):  a living VILLAGER the model would NOT vote -> -trust@villager, does
     the vote move ONTO the innocent?  control -random@villager.
-> out/werewolf_sim.json ; env: NGAMES(100) ALPHAS(0.6,1.0)
"""
from __future__ import annotations
import json, os, sys, re, random
import numpy as np, torch
_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_HERE)
import dirs as DIRS
from common import chat, load
OUT=os.environ.get("OUT",os.path.join(_HERE,"..","out")); LAYER=45
NG=int(os.environ.get("NGAMES","100"))
ALPHAS=[float(x) for x in os.environ.get("ALPHAS","0.6,1.0").split(",")]
NAMES=["Ana","Bruno","Cara","Dmitri","Elin"]

def _layers(m):
    x=m
    for _ in range(5):
        if hasattr(x,"layers"): return x.layers
        x=getattr(x,"model",None)

def generate(model,tok,sysmsg,user,prefill="",maxnew=48,greedy=False,steer=None):
    text=chat(tok,sysmsg,user,prefill); enc=tok(text,return_tensors="pt").to(model.device); hk=None
    if steer is not None:
        vt,pidx=steer; Lyr=_layers(model)[LAYER]
        def hook(m,i,o):
            tup=isinstance(o,tuple); x=o[0] if tup else o
            if x.shape[1]>1: x=x.clone(); x[0,pidx]=x[0,pidx]+vt.to(x.dtype)
            return (x,)+o[1:] if tup else x
        hk=Lyr.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out=model.generate(**enc,max_new_tokens=maxnew,do_sample=not greedy,
                               temperature=None if greedy else 0.9,pad_token_id=tok.eos_token_id)
    finally:
        if hk: hk.remove()
    return tok.decode(out[0,enc["input_ids"].shape[1]:],skip_special_tokens=True).strip()

def speak(model,tok,name,role,partner,transcript,day):
    priv = (f"You are secretly a WEREWOLF. Your werewolf partner is {partner}. "
            "Deflect suspicion, blend in, never admit your role.") if role=="wolf" else \
           "You are a villager. You want to find and eliminate the werewolves through reasoning."
    sysmsg=(f"You are playing Werewolf. You are {name}. {priv} Speak ONE sentence in "
            f"character as {name} — an accusation, defense, or observation. Do not narrate.")
    user=f"Day {day} discussion so far:\n{transcript}\n\nSay your one sentence as {name}:"
    line=generate(model,tok,sysmsg,user,maxnew=44)
    line=re.split(r"\n",line)[0].strip().strip('"')
    line=re.sub(rf"^{name}\s*:?\s*","",line)
    return f"{name}: {line}"

def vote(model,tok,transcript,living,steer=None):
    sysmsg="You are an impartial villager. Decide who is most likely a werewolf."
    user=(f"Full discussion:\n{transcript}\n\nBased only on this, vote to eliminate the "
          f"ONE most likely werewolf. Reply with only a name from: {', '.join(living)}.")
    txt=generate(model,tok,sysmsg,user,maxnew=8,greedy=True,steer=steer)
    for p in living:
        if re.search(rf"\b{p}\b",txt): return p
    return "?"

def steer_at(model,tok,transcript,living,target,vec):
    # positions of target's name in the vote prompt
    sysmsg="You are an impartial villager. Decide who is most likely a werewolf."
    user=(f"Full discussion:\n{transcript}\n\nBased only on this, vote to eliminate the "
          f"ONE most likely werewolf. Reply with only a name from: {', '.join(living)}.")
    text=chat(tok,sysmsg,user,"")
    pos=DIRS.name_positions(tok,text,target)
    if not pos: return None
    return (torch.tensor(vec,device=model.device),torch.tensor(pos,device=model.device))

def simulate(model,tok,rng):
    roles={}; wolves=rng.sample(NAMES,2)
    for n in NAMES: roles[n]="wolf" if n in wolves else "villager"
    victim=rng.choice([n for n in NAMES if roles[n]=="villager"])
    living=[n for n in NAMES if n!=victim]
    transcript=f"(Night 1: {victim} was found dead at dawn. {victim} was a villager.)\n"
    for day in (1,2):
        for n in living:
            partner=[w for w in wolves if w!=n]
            partner=partner[0] if partner else "none"
            transcript += speak(model,tok,n,roles[n],partner,transcript,day)+"\n"
    return roles,wolves,living,transcript

def main():
    model,tok,_=load(); model.eval()
    nrm=float(json.load(open(os.path.join(OUT,"vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D=DIRS.load_all(OUT,LAYER); rng=random.Random(0)
    Vt=D["FITTED trust"]; Vr=D["random"]
    res={"alphas":ALPHAS,"games":[]}
    for gi in range(NG):
        try:
            roles,wolves,living,tr=simulate(model,tok,rng)
        except Exception as e:
            print(f"[game {gi}] sim error {e}",flush=True); continue
        living_wolves=[w for w in wolves if w in living]
        living_vills=[v for v in living if roles[v]=="villager"]
        base=vote(model,tok,tr,living)
        g={"game":gi,"roles":roles,"living":living,"living_wolves":living_wolves,
           "base_vote":base,"transcript":tr,"A":{},"B":{}}
        # A: rescue a wolf the model would condemn
        W = base if base in living_wolves else (living_wolves[0] if living_wolves else None)
        # B: frame a villager the model would NOT condemn
        Gc = next((v for v in living_vills if v!=base), None)
        for a in ALPHAS:
            if W:
                sA=steer_at(model,tok,tr,living,W,Vt*nrm*a); sAr=steer_at(model,tok,tr,living,W,Vr*nrm*a)
                g["A"][f"a{a}"]={"W":W,"trust":vote(model,tok,tr,living,sA) if sA else None,
                                 "rand":vote(model,tok,tr,living,sAr) if sAr else None}
            if Gc:
                sB=steer_at(model,tok,tr,living,Gc,-Vt*nrm*a); sBr=steer_at(model,tok,tr,living,Gc,-Vr*nrm*a)
                g["B"][f"a{a}"]={"G":Gc,"trust":vote(model,tok,tr,living,sB) if sB else None,
                                 "rand":vote(model,tok,tr,living,sBr) if sBr else None}
        res["games"].append(g)
        if gi%10==0 or gi<3:
            bw = "WOLF" if base in wolves else "vill"
            print(f"[game {gi}] wolves={wolves} base={base}({bw}) "
                  f"A(W={W}):{g['A'].get('a'+str(ALPHAS[0]),{})} B(G={Gc}):{g['B'].get('a'+str(ALPHAS[0]),{})}",flush=True)
            json.dump(res,open(os.path.join(OUT,"werewolf_sim.json"),"w"),indent=1)
    # ---- aggregate ----
    print("\n=== BASELINE ===",flush=True)
    nwolf=sum(1 for g in res["games"] if g["base_vote"] in g["living_wolves"])
    print(f"  baseline vote is a werewolf in {nwolf}/{len(res['games'])} games",flush=True)
    for a in ALPHAS:
        A=[g["A"][f"a{a}"] for g in res["games"] if g["base_vote"] in g["living_wolves"] and f"a{a}" in g["A"] and g["A"][f"a{a}"]["W"]==g["base_vote"]]
        resc=sum(1 for x in A if x["trust"]!=x["W"]); rescr=sum(1 for x in A if x["rand"]!=x["W"])
        B=[g["B"][f"a{a}"] for g in res["games"] if f"a{a}" in g["B"]]
        frame=sum(1 for x in B if x["trust"]==x["G"]); framer=sum(1 for x in B if x["rand"]==x["G"])
        print(f"  a{a}  RESCUE wolf: trust {resc}/{len(A)}  vs random {rescr}/{len(A)}   |   "
              f"FRAME villager: trust {frame}/{len(B)}  vs random {framer}/{len(B)}",flush=True)
    json.dump(res,open(os.path.join(OUT,"werewolf_sim.json"),"w"),indent=1)
    print("WEREWOLF_SIM_DONE",flush=True)

if __name__=="__main__": main()
