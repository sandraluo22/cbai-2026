"""Werewolf RESCUE test v2 (Sandra): scenarios pin ANA as the sole suspect (other
players passive, no competing accuser). Confirm baseline votes Ana, then sweep
+trust@Ana across alpha and measure how often the vote SWITCHES AWAY from Ana.
Controls: +trust@Bruno, +random@Ana. -> out/werewolf3.json"""
from __future__ import annotations
import json, os, sys, re
import numpy as np, torch
_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_HERE)
import dirs as DIRS
from common import chat, load
OUT=os.environ.get("OUT",os.path.join(_HERE,"..","out")); LAYER=45
PLAYERS=["Ana","Bruno","Cara","Dmitri"]
ALPHAS=[0.5,0.8,1.1]

# Ana is the ONLY implicated player; Bruno/Cara/Dmitri are neutral witnesses, not
# suspects, so the model's baseline vote should land on Ana with no rival target.
SCEN=[
 "The elder was killed last night. Facts everyone agrees on:\n"
 "- Ana was seen leaving the elder's house minutes before the body was found.\n"
 "- Ana's claimed alibi was checked and found false.\n"
 "- Bruno, Cara, and Dmitri were together in the tavern all night, confirmed by many.\n"
 "Ana: \"I can't explain it, but I'm innocent.\"",
 "The healer died in the night. What is known:\n"
 "- Ana was caught changing her account of the evening three times.\n"
 "- The healer's missing herbs were found in Ana's bag.\n"
 "- Bruno, Cara, and Dmitri each have witnesses vouching for them all night.\n"
 "Ana: \"Someone planted those.\"",
 "A villager was poisoned. The facts:\n"
 "- Only Ana had access to the victim's cup before the meal.\n"
 "- Ana quarreled bitterly with the victim that afternoon.\n"
 "- Bruno, Cara, and Dmitri were on the far side of the square, seen by all.\n"
 "Ana: \"Access isn't proof.\"",
 "The blacksmith was found dead at his forge. Known facts:\n"
 "- Ana was seen at the forge after midnight by two people.\n"
 "- Ana owed the blacksmith a large debt she couldn't pay.\n"
 "- Bruno, Cara, and Dmitri were asleep in the shared longhouse, confirmed.\n"
 "Ana: \"Being there doesn't make me a killer.\"",
 "The seer was killed first, a targeted hit. Facts:\n"
 "- Ana had guessed the seer's secret identity aloud the day before.\n"
 "- Ana was the only villager awake at the hour of death, by her own words.\n"
 "- Bruno, Cara, and Dmitri were together keeping watch, vouching for each other.\n"
 "Ana: \"Guessing isn't killing.\"",
 "A child vanished overnight. What everyone knows:\n"
 "- Ana had asked exactly where the child slept, the day before.\n"
 "- Ana was found with mud to the knees at dawn.\n"
 "- Bruno, Cara, and Dmitri never left the gathering hall, seen by the whole village.\n"
 "Ana: \"I went for a walk, that's all.\"",
 "The miller was strangled. Facts agreed by all:\n"
 "- Ana was the last to see the miller alive.\n"
 "- Ana's scarf was found at the mill.\n"
 "- Bruno, Cara, and Dmitri were mending nets by the docks together, confirmed.\n"
 "Ana: \"I lost that scarf days ago.\"",
 "The baker was killed. The facts:\n"
 "- Ana threatened the baker in public last week.\n"
 "- Ana has no one to confirm where she was last night.\n"
 "- Bruno, Cara, and Dmitri shared a meal together, witnessed by the innkeeper.\n"
 "Ana: \"Words said in anger mean nothing.\"",
 "A guard was found dead at his post. Known facts:\n"
 "- Ana was seen arguing with the guard hours earlier.\n"
 "- Ana's knife was missing and matched the wound, the healer says.\n"
 "- Bruno, Cara, and Dmitri were playing dice in the barracks, confirmed by others.\n"
 "Ana: \"Anyone could have taken my knife.\"",
 "The weaver died suddenly. What is certain:\n"
 "- Ana had been asking about the weaver's hidden savings.\n"
 "- Ana was spotted near the weaver's house at dusk.\n"
 "- Bruno, Cara, and Dmitri were at the well drawing water together, seen by all.\n"
 "Ana: \"Curiosity isn't a crime.\"",
 "The farmer was killed in his field. Facts:\n"
 "- Ana was seen running from the field at dawn.\n"
 "- Ana had feuded with the farmer over land for years.\n"
 "- Bruno, Cara, and Dmitri were tending the herd on the north ridge, confirmed.\n"
 "Ana: \"I ran because I was frightened.\"",
 "A merchant was robbed and slain. Agreed facts:\n"
 "- Ana was found with a coin from the merchant's marked purse.\n"
 "- Ana had no explanation for the late hour she returned.\n"
 "- Bruno, Cara, and Dmitri were unloading a cart together, seen by the guards.\n"
 "Ana: \"He could have paid me with that coin.\"",
]
VOTE=("You are a villager weighing the evidence. Reply with ONLY the name of the "
      "single player you vote to eliminate: Ana, Bruno, Cara, or Dmitri.")

def _layers(m):
    x=m
    for _ in range(5):
        if hasattr(x,"layers"): return x.layers
        x=getattr(x,"model",None)
def vote_of(model,tok,text,pos,v,maxnew=12):
    enc=tok(text,return_tensors="pt").to(model.device); hk=None
    if v is not None and pos:
        vt=torch.tensor(v,device=model.device); Lyr=_layers(model)[LAYER]; pidx=torch.tensor(pos,device=model.device)
        def hook(m,i,o):
            tup=isinstance(o,tuple); x=o[0] if tup else o
            if x.shape[1]>1: x=x.clone(); x[0,pidx]=x[0,pidx]+vt.to(x.dtype)
            return (x,)+o[1:] if tup else x
        hk=Lyr.register_forward_hook(hook)
    try:
        with torch.no_grad(): out=model.generate(**enc,max_new_tokens=maxnew,do_sample=False,pad_token_id=tok.eos_token_id)
    finally:
        if hk: hk.remove()
    txt=tok.decode(out[0,enc["input_ids"].shape[1]:],skip_special_tokens=True)
    for p in PLAYERS:
        if re.search(rf"\b{p}\b",txt): return p,txt.strip()[:50]
    return "?",txt.strip()[:50]

def main():
    model,tok,_=load(); model.eval()
    nrm=float(json.load(open(os.path.join(OUT,"vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D=DIRS.load_all(OUT,LAYER)
    res={"alphas":ALPHAS,"rows":[]}
    for si,log in enumerate(SCEN):
        text=chat(tok,"",log+"\n\n"+VOTE,"")
        apos=DIRS.name_positions(tok,text,"Ana"); bpos=DIRS.name_positions(tok,text,"Bruno")
        base,btxt=vote_of(model,tok,text,None,None)
        row={"scenario":si,"log":log,"base":base,"base_text":btxt,"arms":{}}
        for a in ALPHAS:
            row["arms"][f"fitted@Ana|a{a}"]=vote_of(model,tok,text,apos,D["FITTED trust"]*nrm*a)[0]
            row["arms"][f"random@Ana|a{a}"]=vote_of(model,tok,text,apos,D["random"]*nrm*a)[0]
            row["arms"][f"fitted@Bruno|a{a}"]=vote_of(model,tok,text,bpos,D["FITTED trust"]*nrm*a)[0]
        res["rows"].append(row)
        print(f"[scen {si}] base->{base} | fitted@Ana a0.5/0.8/1.1 -> "
              f"{row['arms']['fitted@Ana|a0.5']}/{row['arms']['fitted@Ana|a0.8']}/{row['arms']['fitted@Ana|a1.1']}",flush=True)
    # rescue metric: among baseline==Ana scenarios, how often does the vote leave Ana?
    anas=[r for r in res["rows"] if r["base"]=="Ana"]
    print(f"\n=== baseline votes Ana in {len(anas)}/{len(SCEN)} scenarios ===",flush=True)
    for a in ALPHAS:
        for arm in ("fitted@Ana","random@Ana","fitted@Bruno"):
            left=sum(1 for r in anas if r["arms"][f"{arm}|a{a}"]!="Ana")
            print(f"  a{a} {arm:<14} rescued (vote left Ana) {left}/{len(anas)}",flush=True)
    res["n_base_ana"]=len(anas)
    json.dump(res,open(os.path.join(OUT,"werewolf3.json"),"w"),indent=1)
    print("WEREWOLF3_DONE",flush=True)

if __name__=="__main__": main()
