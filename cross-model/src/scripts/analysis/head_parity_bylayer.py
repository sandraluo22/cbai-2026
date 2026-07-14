"""For every head, ablate it and measure PARITY power at EVERY layer (not just the readout layer),
so we can see at which downstream layers each head's ablation affects parity -> damage[meas_layer,
abl_layer, abl_head]. grid/Llama. Env: GEN_MODEL GRAPH FAM(grid) N NWALKS OUTDIR DEVICE."""
import os, json, gc
from dataclasses import replace
import numpy as np, torch
from config import get_config
import graph as G
from graph import Graph
import models as M
from models import resolve_token_spans
SPEC={"Llama":("meta-llama/Llama-3.1-8B","NousResearch/Meta-Llama-3.1-8B")}
TAG=os.environ.get("GEN_MODEL","Llama"); N=int(os.environ.get("N","16")); NWALKS=int(os.environ.get("NWALKS","16"))
CTXLO=int(os.environ.get("CTXLO","100")); OUTDIR=os.environ.get("OUTDIR","runs/axes/4_circuits/head_parity_bylayer")
def two_colour(g):
    n=g.n_nodes; col=np.zeros(n)
    for s in range(n):
        if col[s]!=0: continue
        col[s]=1; st=[s]
        while st:
            u=st.pop()
            for v in g.adjacency[u]:
                if col[v]==0: col[v]=-col[u]; st.append(v)
    return col.astype(float)
def unit(v): v=v-v.mean(); return v/(np.linalg.norm(v)+1e-9)
def grid_adj(n):
    r=c=int(round(n**0.5)); adj=[[] for _ in range(n)]
    for i in range(r):
        for j in range(c):
            u=i*c+j
            if j+1<c: adj[u].append(u+1); adj[u+1].append(u)
            if i+1<r: adj[u].append(u+c); adj[u+c].append(u)
    return [sorted(x) for x in adj]
def aproj(b,cm):
    return b.self_attn.o_proj,(getattr(cm,"head_dim",None) or cm.hidden_size//cm.num_attention_heads)
@torch.no_grad()
def nmeans(model,tok,blocks,cm,walks,dev,n,layers,head=None):
    gr={}
    def mk(L):
        def hh(_m,_i,o): gr[L]=(o[0] if isinstance(o,tuple) else o).detach()
        return hh
    caps=[blocks[L].register_forward_hook(mk(L)) for L in layers]; hh=[]
    if head is not None:
        L,h=head; proj,hd=aproj(blocks[L],cm); ct=torch.arange(h*hd,(h+1)*hd,device=dev)
        def pre(_m,a): x=a[0].clone(); x[...,ct]=0; return (x,)+tuple(a[1:])
        hh=[proj.register_forward_pre_hook(pre)]
    ns={L:np.zeros((n,cm.hidden_size)) for L in layers}; nc=np.zeros(n)
    try:
        for wk in walks:
            ids=tok(wk.text,return_tensors="pt",add_special_tokens=True)["input_ids"].to(dev)
            sp=resolve_token_spans(tok,wk); nd=wk.nodes; gr.clear(); model(input_ids=ids)
            sg=[t[-1] for t in sp]; cl=np.arange(1,len(nd)+1); f=layers[0]
            for L in layers:
                rows=gr[L][0][sg].float().cpu().numpy()
                for s in range(len(nd)):
                    if cl[s]>=CTXLO: ns[L][nd[s]]+=rows[s]; nc[nd[s]]+= (L==f)
    finally:
        for x in caps: x.remove()
        for x in hh: x.remove()
    cn=np.maximum(nc,1); return {L:ns[L]/cn[:,None] for L in layers}
def ppow(H,u):
    Hc=H-H.mean(0); return float(((Hc.T@u)**2).sum()/((Hc**2).sum()+1e-12))
def main():
    dev="cuda"; os.makedirs(OUTDIR,exist_ok=True)
    hf,mir=SPEC[TAG]; cfg=replace(get_config("gemma_qwen"),graph_type="ring",ring_size=N,n_walks=NWALKS,walk_length=300,device=dev)
    words=cfg.words()[:N]; graph=Graph(n_nodes=N,words=words,adjacency=grid_adj(N),coords=[(i//4,i%4) for i in range(N)])
    u=unit(two_colour(graph))
    try: model,tok=M.load_model(hf,cfg)
    except Exception: model,tok=M.load_model(mir,cfg)
    cm=model.config; blocks=M._decoder_blocks(model); nL=cm.num_hidden_layers; nH=cm.num_attention_heads
    walks=G.generate_walks(graph,cfg); layers=list(range(nL))
    clean={L:ppow(nmeans(model,tok,blocks,cm,walks,dev,N,[L])[L],u) for L in layers}
    dmg=np.zeros((nL,nL,nH))  # meas_layer, abl_layer, abl_head
    for aL in range(nL):
        for h in range(nH):
            mm=nmeans(model,tok,blocks,cm,walks,dev,N,layers,head=(aL,h))
            for mL in layers: dmg[mL,aL,h]=clean[mL]-ppow(mm[mL],u)
        if aL%8==0: print(f"abl layer {aL}/{nL}",flush=True)
    json.dump({"clean":clean,"nL":nL,"nH":nH,"damage":dmg.tolist()},open(f"{OUTDIR}/head_parity_bylayer_grid.json","w"))
    print("DONE",flush=True)
if __name__=="__main__": main()
