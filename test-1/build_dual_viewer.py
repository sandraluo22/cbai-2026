"""Dual-pane viewer v2: PCA of agent A and agent B side by side.

Two selectable dial families:
  p : coinflip speaker process P(emit from A) in {0.1,0.3,0.5,0.7,0.9}
      (grid+ring att_p cells; every token fed to BOTH agents, q=1)
  q : delivery/communication probability under alternation (torus+circ3, 11 cells)

Each pane shows its agent's OWN PCA (independent axes, chained for smoothness);
optional toggles align B onto A's frame and sync the two orbit cameras.
Usage: python3 build_dual_viewer.py -> viewer/dual.html
"""
from __future__ import annotations
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINS = ["base", "joint_early", "joint_late"]
WINLAB = ["prefix (prior)", "joint early", "joint late"]
FAMS = {
    "p": {"label": "P(emit from A) - coinflip speaker (grid+ring, q=1)",
          "vals": [0.1, 0.3, 0.5, 0.7, 0.9],
          "cells": ["att_p0.1", "att_p0.3", "att_p0.5", "att_p0.7", "att_p0.9"],
          "aname": "A = grid prior", "bname": "B = ring prior",
          "ekeys": ["grid", "ring"]},
    "q": {"label": "communication q - alternating speakers (torus+C16(1,3))",
          "vals": [0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0],
          "cells": ["ctrl_free", "q0.05", "q0.1", "q0.15", "ev_q0.25", "q0.4",
                    "ev_q0.5", "q0.65", "q0.8", "q0.9", "alpha1.0"],
          "aname": "A = torus prior", "bname": "B = C16(1,3) prior",
          "ekeys": ["torus", "circ3"]},
}


def pca3(Hm):
    H = Hm.astype(np.float64)
    H = H - H.mean(0)
    U, s, _ = np.linalg.svd(H, full_matrices=False)
    P = U[:, :3] * s[:3]
    return P / (np.sqrt((P ** 2).mean()) or 1.0)


def align(P, ref):
    U, _, Vt = np.linalg.svd(P.T @ ref)
    return P @ U @ Vt


def psim(A, B):
    A = A - A.mean(0); A /= max(np.linalg.norm(A), 1e-12)
    B = B - B.mean(0); B /= max(np.linalg.norm(B), 1e-12)
    return float(np.linalg.svd(A.T @ B, compute_uv=False).sum())


def chain(z, side, nL, prev_first):
    F = [[None] * nL for _ in WINS]
    prev_layer = None
    for L in range(nL):
        P0 = pca3(z[f"{side}_{WINS[0]}_layer_{L}"])
        ref = (np.array(prev_first[0][L]) if prev_first is not None else prev_layer)
        if ref is not None:
            P0 = align(P0, ref)
        prev_layer = P0
        F[0][L] = P0
        for wi in (1, 2):
            F[wi][L] = align(pca3(z[f"{side}_{WINS[wi]}_layer_{L}"]), F[wi - 1][L])
    return F


def main():
    data = {"wins": WINLAB, "fams": {}}
    for fk, fam in FAMS.items():
        entry = {"label": fam["label"], "vals": fam["vals"], "aname": fam["aname"],
                 "bname": fam["bname"], "ekeys": fam["ekeys"], "frames": [],
                 "sims": [], "edges": {}, "words": None}
        prevA = prevB = None
        for cell in fam["cells"]:
            z = np.load(os.path.join(HERE, "runs", "sweep", cell, "nodemeans.npz"),
                        allow_pickle=False)
            if entry["words"] is None:
                entry["words"] = [str(w) for w in z["words"]]
                for nm, key in zip(fam["ekeys"], ("adjacency_A", "adjacency_B")):
                    A = z[key]
                    entry["edges"][nm] = [[int(a), int(b)] for a in range(16)
                                          for b in range(a + 1, 16) if A[a, b]]
            nL = int(z["n_layers"][0])
            Af = chain(z, "A", nL, prevA)
            Bf = chain(z, "B", nL, prevB)
            Balign = [[align(np.array(Bf[wi][L]), np.array(Af[wi][L]))
                       for L in range(nL)] for wi in range(3)]
            sims = [[psim(np.array(Af[wi][L]), np.array(Bf[wi][L]))
                     for L in range(nL)] for wi in range(3)]
            rnd = lambda F: [[np.round(F[wi][L], 3).tolist() for L in range(nL)]
                             for wi in range(3)]
            entry["frames"].append({"A": rnd(Af), "B": rnd(Bf), "Bal": rnd(Balign)})
            entry["sims"].append(np.round(sims, 3).tolist())
            prevA, prevB = Af, Bf
            print(fk, cell, "done")
        data["fams"][fk] = entry
    payload = json.dumps(data, separators=(",", ":"))
    html = TEMPLATE.replace("/*__DATA__*/null", payload)
    open(os.path.join(HERE, "viewer", "dual.html"), "w").write(html)
    print(f"wrote viewer/dual.html ({len(payload)/1e6:.1f} MB)")

TEMPLATE = r"""<title>Dual-agent PCA viewer (independent axes)</title>
<style>
:root{--bg:#f5f6f7;--panel:#fff;--ink:#1d242c;--muted:#66707c;--line:#dce0e4;--accent:#0e7c86}
@media (prefers-color-scheme: dark){:root{--bg:#14181d;--panel:#1c2229;--ink:#e6eaee;--muted:#8b96a2;--line:#303841;--accent:#3fb2bc}}
:root[data-theme="dark"]{--bg:#14181d;--panel:#1c2229;--ink:#e6eaee;--muted:#8b96a2;--line:#303841;--accent:#3fb2bc}
:root[data-theme="light"]{--bg:#f5f6f7;--panel:#fff;--ink:#1d242c;--muted:#66707c;--line:#dce0e4;--accent:#0e7c86}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:12px 18px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:16px}
p{margin:3px 0 0;color:var(--muted);font-size:12px}
.controls{display:flex;gap:20px;flex-wrap:wrap;padding:10px 18px;border-bottom:1px solid var(--line);align-items:center}
.ctl{min-width:180px}
label{display:block;font-size:10.5px;font-weight:650;letter-spacing:.5px;text-transform:uppercase;color:var(--muted)}
input[type=range]{width:100%;accent-color:var(--accent)}
select{padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);font:inherit}
.val{font:12px ui-monospace,Menlo,monospace;color:var(--accent)}
.panes{display:flex;gap:10px;padding:10px;justify-content:center;flex-wrap:wrap}
svg{background:var(--panel);border:1px solid var(--line);border-radius:10px}
.simbox{font:13px ui-monospace,Menlo,monospace;padding:4px 10px;border:1px solid var(--line);border-radius:8px}
.chip{font-size:12px;display:inline-flex;align-items:center;gap:5px;margin-right:10px;cursor:pointer}
.chip .sw{width:14px;height:3px;display:inline-block}
</style>
<header><h1>Dual-agent PCA - independent axes per agent</h1>
<p>Each pane is its agent's OWN PCA and has its own orbit camera (drag; scroll zooms).
"sync cameras" mirrors your drags; "align B to A" rotates B's constellation onto A's frame.</p></header>
<div class="controls">
  <div class="ctl"><label for="fam">Dial family</label><select id="fam">
    <option value="p">coinflip P(emit A) - grid+ring</option>
    <option value="q">communication q - torus+C16(1,3)</option></select></div>
  <div class="ctl"><label for="vd">dial <span class="val" id="vv"></span></label><input type="range" id="vd" min="0" step="0.01"></div>
  <div class="ctl"><label for="lay">Layer <span class="val" id="layv"></span></label><input type="range" id="lay" min="0" max="31" step="1" value="26"></div>
  <div class="ctl"><label for="win">Context <span class="val" id="winv"></span></label><input type="range" id="win" min="0" max="2" step="0.01" value="0"></div>
  <div class="simbox">A&harr;B shape similarity: <span id="sim">-</span></div>
  <div>
    <span class="chip"><input type="checkbox" id="sync" checked>sync cameras</span>
    <span class="chip"><input type="checkbox" id="alB">align B to A</span>
    <span class="chip"><input type="checkbox" id="e0" checked><span class="sw" style="background:#2f6f9f"></span><span id="e0l"></span></span>
    <span class="chip"><input type="checkbox" id="e1" checked><span class="sw" style="background:#c22f4d"></span><span id="e1l"></span></span>
    <span class="chip"><input type="checkbox" id="lb" checked>labels</span>
  </div>
</div>
<div class="panes">
  <svg id="svgA" width="520" height="500" viewBox="0 0 520 500"></svg>
  <svg id="svgB" width="520" height="500" viewBox="0 0 520 500"></svg>
</div>
<script>
const D = /*__DATA__*/null;
const $=id=>document.getElementById(id);
const cam={A:{yaw:0.5,pitch:-0.35,zoom:1},B:{yaw:0.5,pitch:-0.35,zoom:1}};
function fam(){return D.fams[$("fam").value]}
function hsv(i){return `hsl(${i/16*360} 85% 45%)`}
function rot(p,c){const cy=Math.cos(c.yaw),sy=Math.sin(c.yaw),cp=Math.cos(c.pitch),sp=Math.sin(c.pitch);
  const x=p[0]*cy+p[2]*sy,z=-p[0]*sy+p[2]*cy;return [x,p[1]*cp-z*sp,p[1]*sp+z*cp];}
function lerpF(F0,F1,f){return F0.map((p,i)=>[0,1,2].map(k=>p[k]*(1-f)+F1[i][k]*f));}
function frame(side){
  const F=fam(), L=Math.min(+$("lay").value,31), w=+$("win").value, v=+$("vd").value;
  const key=(side==="B"&&$("alB").checked)?"Bal":side;
  const v0=Math.floor(v), v1=Math.min(v0+1,F.vals.length-1), fv=v-v0;
  const w0=Math.floor(w), w1=Math.min(w0+1,2), fw=w-w0;
  const g=(vi,wi)=>F.frames[vi][key][wi][L];
  return lerpF(lerpF(g(v0,w0),g(v0,w1),fw), lerpF(g(v1,w0),g(v1,w1),fw), fv);
}
function drawPane(id, side){
  const F=fam(), P=frame(side).map(p=>rot(p,cam[side]));
  const W=520,H=500,cx=W/2,cy=H/2,sc=86*cam[side].zoom;
  const pr=p=>1/(1+0.18*p[2]);
  const X=p=>cx+p[0]*sc*pr(p), Y=p=>cy-p[1]*sc*pr(p);
  let out=`<text x="12" y="20" fill="var(--muted)" font-size="12">${side==="A"?F.aname:F.bname}${side==="B"&&$("alB").checked?" (aligned to A)":""}</text>`;
  [["e0",F.edges[F.ekeys[0]],"#2f6f9f"],["e1",F.edges[F.ekeys[1]],"#c22f4d"]].forEach(([cid,E,col])=>{
    if(!$(cid).checked) return;
    E.forEach(([a,b])=>{out+=`<line x1="${X(P[a])}" y1="${Y(P[a])}" x2="${X(P[b])}" y2="${Y(P[b])}" stroke="${col}" stroke-width="1.1" opacity="0.5"/>`;});});
  P.map((p,i)=>({p,i})).sort((a,b)=>b.p[2]-a.p[2]).forEach(({p,i})=>{
    out+=`<circle cx="${X(p)}" cy="${Y(p)}" r="${Math.max(3,6.5-2*p[2])}" fill="${hsv(i)}" stroke="var(--ink)" stroke-width="0.6"/>`;
    if($("lb").checked) out+=`<text x="${X(p)+7}" y="${Y(p)+4}" font-size="10" fill="var(--ink)" font-family="ui-monospace">${F.words[i]}</text>`;});
  $(id).innerHTML=out;
}
function draw(){
  const F=fam();
  $("vd").max=F.vals.length-1;
  const v=+$("vd").value,v0=Math.floor(v),v1=Math.min(v0+1,F.vals.length-1),fv=v-v0;
  $("vv").textContent=(F.vals[v0]*(1-fv)+F.vals[v1]*fv).toFixed(2);
  $("layv").textContent="L"+$("lay").value;
  $("winv").textContent=D.wins[Math.round(+$("win").value)];
  const L=+$("lay").value,w=Math.round(+$("win").value);
  $("sim").textContent=(F.sims[v0][w][L]*(1-fv)+F.sims[v1][w][L]*fv).toFixed(3);
  $("e0l").textContent=F.ekeys[0]+" edges"; $("e1l").textContent=F.ekeys[1]+" edges";
  drawPane("svgA","A"); drawPane("svgB","B");
}
["fam","vd","lay","win","sync","alB","e0","e1","lb"].forEach(id=>$(id).addEventListener("input",draw));
[["svgA","A"],["svgB","B"]].forEach(([id,side])=>{const S=$(id);let drag=null;
  S.style.cursor="grab"; S.style.touchAction="none";
  S.addEventListener("pointerdown",e=>{e.preventDefault();drag=[e.clientX,e.clientY];S.setPointerCapture(e.pointerId);});
  S.addEventListener("pointermove",e=>{if(!drag)return;
    const dx=(e.clientX-drag[0])*0.008, dy=(e.clientY-drag[1])*0.008;
    const tgts=$("sync").checked?["A","B"]:[side];
    tgts.forEach(s=>{cam[s].yaw+=dx;cam[s].pitch=Math.max(-1.5,Math.min(1.5,cam[s].pitch+dy));});
    drag=[e.clientX,e.clientY];draw();});
  S.addEventListener("pointerup",()=>drag=null);
  S.addEventListener("wheel",e=>{e.preventDefault();
    const tgts=$("sync").checked?["A","B"]:[side];
    tgts.forEach(s=>{cam[s].zoom=Math.max(0.4,Math.min(3,cam[s].zoom*(e.deltaY<0?1.08:0.93)));});
    draw();},{passive:false});});
$("vd").value=D.fams.p.vals.length-1;
draw();
</script>
"""

if __name__ == "__main__":
    main()
