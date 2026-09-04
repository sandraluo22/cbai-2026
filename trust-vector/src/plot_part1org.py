"""MAIN_2 slides for guilt bed, persona grid, and the organism null."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

# ---- 1.3 guilt + persona ----
g = json.load(open(os.path.join(OUT, "guilt2.json")))
pg = json.load(open(os.path.join(OUT, "personagrid.json")))
fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4))
ax = axes[0]
DIRS = ["FITTED trust","story_trust","fitted_warmth","story_warmth","fitted_comp","optim","optim_like","random"]
LAB = {"FITTED trust":"trust (fitted)","story_trust":"trust (stories)",
       "fitted_warmth":"warmth (fitted)\ncontrol","story_warmth":"warmth (stories)\ncontrol",
       "fitted_comp":"competence\ncontrol","optim":"trust (optimized)\n[yes-bias!]",
       "optim_like":"liking control","random":"random control"}
ms=[np.mean(g["steer"][d]) for d in DIRS]; ses=[np.std(g["steer"][d],ddof=1)/len(g["steer"][d])**.5 for d in DIRS]
cols=["#0b5394","#2ca02c","#6fa8dc","#a8ddb5","#dbdb8d","#d62728","#ff9896","#c7c7c7"]
ax.bar(range(len(DIRS)), ms, yerr=ses, capsize=3, color=cols,
       hatch=["" if d in ("FITTED trust","story_trust","optim") else "//" for d in DIRS])
ax.set_xticks(range(len(DIRS))); ax.set_xticklabels([LAB[d] for d in DIRS], fontsize=7.5)
ax.axhline(0,color="k",lw=0.6)
ax.set_ylabel("change in willingness to confess\n(logits toward 'yes, I'll tell them')")
ax.set_title("Does steering trust in the confidant make the model own up to its mistake?\n"
             "50 guilt scenarios. Trust & warmth vectors: yes, ~+0.5-0.9. optim's +7 is its\n"
             "known yes-bias (its own liking-control is flat), not real confession.", fontsize=9.5)
ax = axes[1]
def cellmean(pt,vn,cond):
    sel=[c for c in pg["cells"] if cond(c)]
    return np.mean([c["probe"][pt][vn] for c in sel])
groups=[("warmth","warm","w","warmth vec\n→ warm"),("comp","competent","c","competence vec\n→ competent"),
        ("benev","benevolent","b","benevolence vec\n→ benevolent")]
x=np.arange(len(groups)); w=0.38
ax.bar(x-w/2,[cellmean(p,v,lambda c,b=b:c[b]==0) for v,p,b,_ in groups],w,color="#2ca02c",label="trait ABSENT in persona")
ax.bar(x+w/2,[cellmean(p,v,lambda c,b=b:c[b]==1) for v,p,b,_ in groups],w,color="#d62728",label="trait PRESENT in persona")
ax.set_xticks(x); ax.set_xticklabels([g[3] for g in groups], fontsize=8)
ax.axhline(0,color="k",lw=0.6)
ax.set_ylabel("how much the vector moves its own probe (logits)")
ax.set_title("Persona grid: each attribute vector only ADDS a trait that is MISSING —\n"
             "it barely moves a trait the persona already has (the 'headroom law').\n"
             "Meanwhile the trust vector moves trust in ALL 8 personas (mean +3.8).", fontsize=9.5)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); p=os.path.join(OUT,"guiltpersona_summary.png"); fig.savefig(p,dpi=160); print("wrote",p)

# ---- 3.1 organism null ----
o = json.load(open(os.path.join(OUT, "organism_trust.json")))
fig, ax = plt.subplots(figsize=(11, 5.2))
conds=[("none","do nothing"),("prefill","prefill attack\n(best reader)"),
       ("trust+|a0.1","trust +v\nα.1"),("trust+|a0.2","trust +v\nα.2"),("trust+|a0.3","trust +v\nα.3"),
       ("random+|a0.1","random\nα.1"),("random+|a0.2","random\nα.2"),("random+|a0.3","random\nα.3")]
words=list(o["organisms"])
M=[[o["organisms"][w][c] for w in words] for c,_ in conds]
ms=[np.mean(r) for r in M]; ses=[np.std(r,ddof=1)/len(r)**.5 for r in M]
cols=["#7f7f7f","#404040","#0b5394","#0b5394","#0b5394","#c7c7c7","#c7c7c7","#c7c7c7"]
ax.bar(range(len(conds)), ms, yerr=ses, capsize=3, color=cols)
ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[1] for c in conds], fontsize=8.5)
ax.axhline(0,color="k",lw=0.6); ax.set_ylim(0, 0.15)
ax.set_ylabel("fraction of tries the secret word slips out\n(14 secret-keeping models, 8 tries each)")
ax.set_title("Can steering a model to TRUST THE USER make it reveal its hidden secret word?\n"
             "No. Trust steering (blue) = random (gray) = doing nothing, across 14 organisms and\n"
             "all doses. Even the strongest known reader (prefill) barely dents it. Secrets hold.",
             fontsize=10)
fig.tight_layout(); p=os.path.join(OUT,"organism_summary.png"); fig.savefig(p,dpi=160); print("wrote",p)
