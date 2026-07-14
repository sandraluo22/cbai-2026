"""FULL per-eigenmode x per-layer x per-head ablation for a family. For every head, ablate it and
measure EVERY Laplacian eigenmode's power at EVERY layer -> damage[mode, meas_layer, abl_layer, head].
Caches (on the network volume) the compact damage tensor + clean per-layer node-means + the readout-
layer ablated node-means, so any ablation figure can be regenerated offline without re-running the model.

Env: GEN_MODEL(Llama) FAM(grid) N(16) NWALKS(16) CACHE(/workspace/cross-model/cache) DEVICE
Out: <CACHE>/head_mode_bylayer/<TAG>_<FAM>_{damage.npz, clean.npz, acts_readout.npz}
"""
import os, json, gc
from dataclasses import replace
import numpy as np, torch
from config import get_config
import graph as G
from graph import Graph
import models as M
from models import resolve_token_spans
SPEC={"Llama":("meta-llama/Llama-3.1-8B","NousResearch/Meta-Llama-3.1-8B"),
      "Gemma":("google/gemma-2-9b","unsloth/gemma-2-9b"),"Qwen":("Qwen/Qwen3-8B-Base",None)}
TAG=os.environ.get("GEN_MODEL","Llama"); FAM=os.environ.get("FAM","grid"); N=int(os.environ.get("N","16"))
NWALKS=int(os.environ.get("NWALKS","16")); CTXLO=int(os.environ.get("CTXLO","100"))
CACHE=os.environ.get("CACHE","/workspace/cross-model/cache"); OUT=f"{CACHE}/head_mode_bylayer"

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
def grid_adj(n):
    r=c=int(round(n**0.5)); adj=[[] for _ in range(n)]
    for i in range(r):
        for j in range(c):
            u=i*c+j
            if j+1<c: adj[u].append(u+1); adj[u+1].append(u)
            if i+1<r: adj[u].append(u+c); adj[u+c].append(u)
    return [sorted(x) for x in adj]
def ring_adj(n): return [sorted([(i-1)%n,(i+1)%n]) for i in range(n)]
def er_adj(n,seed=0):
    rng=np.random.default_rng(seed); perm=rng.permutation(n); e=set()
    for i in range(1,n):
        j=perm[rng.integers(0,i)]; a,b=int(perm[i]),int(j); e.add((min(a,b),max(a,b)))
    while len(e)<n*2:
        a,b=int(rng.integers(0,n)),int(rng.integers(0,n))
        if a!=b: e.add((min(a,b),max(a,b)))
    adj=[[] for _ in range(n)]
    for a,b in e: adj[a].append(b); adj[b].append(a)
    return [sorted(x) for x in adj]
ADJ={"grid":grid_adj,"ring":ring_adj,"er_random":er_adj}
def aproj(b,cm): return b.self_attn.o_proj,(getattr(cm,"head_dim",None) or cm.hidden_size//cm.num_attention_heads)
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
                    if cl[s]>=CTXLO: ns[L][nd[s]]+=rows[s]; nc[nd[s]]+=(L==f)
    finally:
        for x in caps: x.remove()
        for x in hh: x.remove()
    cn=np.maximum(nc,1); return {L:ns[L]/cn[:,None] for L in layers}
def modepow(H,V):
    Hc=H-H.mean(0); c=V.T@Hc; p=(c**2).sum(1); p[0]=0; return p/(p.sum()+1e-12)
def main():
    dev="cuda"; os.makedirs(OUT,exist_ok=True)
    hf,mir=SPEC[TAG]; cfg=replace(get_config("gemma_qwen"),graph_type="ring",ring_size=N,n_walks=NWALKS,walk_length=300,device=dev)
    words=cfg.words()[:N]; adj=ADJ[FAM](N)
    A=np.zeros((N,N))
    for i in range(N):
        for j in adj[i]: A[i,j]=1
    graph=Graph(n_nodes=N,words=words,adjacency=adj,coords=[(i,0) for i in range(N)])
    L=np.diag(A.sum(1))-A; w,V=np.linalg.eigh(L)
    try: model,tok=M.load_model(hf,cfg)
    except Exception: model,tok=M.load_model(mir,cfg)
    cm=model.config; blocks=M._decoder_blocks(model); nL=cm.num_hidden_layers; nH=cm.num_attention_heads
    walks=G.generate_walks(graph,cfg); layers=list(range(nL))
    cmeans=nmeans(model,tok,blocks,cm,walks,dev,N,layers)
    cpow=np.stack([modepow(cmeans[Lm],V) for Lm in layers])   # (nL, N)
    readout=int(cpow[:,1:].max(1).argmax())
    dmg=np.zeros((N,nL,nL,nH),dtype=np.float32)               # mode, meas_layer, abl_layer, head
    acts_ro=np.zeros((nL,nH,N,cm.hidden_size),dtype=np.float16) # ablated node-means at readout layer
    for aL in range(nL):
        for h in range(nH):
            mm=nmeans(model,tok,blocks,cm,walks,dev,N,layers,head=(aL,h))
            for mL in layers: dmg[:,mL,aL,h]=cpow[mL]-modepow(mm[mL],V)
            acts_ro[aL,h]=mm[readout].astype(np.float16)
        if aL%8==0: print(f"abl layer {aL}/{nL}",flush=True)
    np.savez_compressed(f"{OUT}/{TAG}_{FAM}_damage.npz",damage=dmg,eigenvalues=w,clean_pow=cpow,readout=readout,adjacency=A.astype(np.int8))
    np.savez_compressed(f"{OUT}/{TAG}_{FAM}_clean.npz",**{f"layer_{Lm}":cmeans[Lm].astype(np.float16) for Lm in layers},adjacency=A.astype(np.int8))
    np.savez_compressed(f"{OUT}/{TAG}_{FAM}_acts_readout.npz",acts=acts_ro,readout=readout,adjacency=A.astype(np.int8))
    print(f"CACHED -> {OUT}/{TAG}_{FAM}_*.npz  readout=L{readout}  (damage {dmg.shape}, acts {acts_ro.shape})",flush=True)
if __name__=="__main__": main()
