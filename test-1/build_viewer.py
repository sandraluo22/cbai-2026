"""Build the interactive PCA viewer: export per-frame 2D PCA coords from every run's
node-means npz, Procrustes-chain them for smooth slider motion, and emit a single
self-contained viewer.html (data inlined).

Frames: for each run / context / window / layer -> PCA top-2 of the 16 node-means
(unit-RMS scale). Alignment chain: layer L's first-window frame is rotated onto layer
L-1's; each subsequent window frame is rotated onto the previous window at the same
layer -- so both sliders move smoothly. Windows are the discrete context stops; the
viewer interpolates between them while dragging.

Usage: python3 build_viewer.py   (writes viewer/viewer.html)
"""
from __future__ import annotations
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = [  # dir, label
    ("out",          "grid+ring | T=1 free-sample"),
    ("out_topk4",    "grid+ring | top-k 4/4"),
    ("out_k2",       "grid+ring | top-k 2/2"),
    ("out_k2_fix",   "grid+ring | top-k 2/2, low-prior vocab"),
    ("out_hexgrid",  "hex+grid | k 6/4"),
    ("out_hexring",  "hex+ring | k 6/2"),
    ("out_prismring", "prism+ring | k 3/2"),
    ("out_ringring3", "ring+ring3 | k 2/2"),
    ("out_antigrid", "antiprism+grid | k 4/4"),
    ("out_tri",      "3-way ring+grid+ring3 | k 2/4/2"),
    ("out_x_LQ",     "cross-model: Llama-grid + Qwen-ring"),
    ("out_x_LG",     "cross-model: Llama-grid + Gemma-ring"),
    ("out_x_QG",     "cross-model: Qwen-grid + Gemma-ring"),
    ("out_x_LQG",    "cross-model 3-way: Llama-ring + Qwen-grid + Gemma-ring3"),
]
LEGACY_CTX = ("grid", "ring")            # runs without ctx_names
PRIMED_WINS = ["base", "joint_early", "joint_mid", "joint_late"]
FRESH_WINS = ["early", "mid", "late"]
WIN_LABELS = {"base": "prefix 700-1000", "joint_early": "joint 0-100",
              "joint_mid": "joint 100-300", "joint_late": "joint 300-600",
              "early": "joint 0-100", "mid": "joint 100-300", "late": "joint 300-600"}


def pca3(Hm):
    """Top-6 PC coords (viewer exposes axis pickers; name kept for history)."""
    H = Hm.astype(np.float64)
    H = H - H.mean(0)
    U, s, _ = np.linalg.svd(H, full_matrices=False)
    P = U[:, :6] * s[:6]
    rms = np.sqrt((P[:, :3] ** 2).mean()) or 1.0     # scale by top-3 for stable zoom
    return P / rms


def align(P, ref):
    """Rotate/reflect P onto ref (both centered)."""
    U, _, Vt = np.linalg.svd(P.T @ ref)
    return P @ U @ Vt


def edges_of(A):
    A = np.asarray(A)
    return [[int(a), int(b)] for a in range(len(A)) for b in range(a + 1, len(A))
            if A[a, b]]


def adj_from_edges(edges):
    A = np.zeros((16, 16))
    for a, b in edges:
        A[a, b] = A[b, a] = 1
    return A


def union_modes(adjs, k=6):
    """Modes 1..k of the normalized Laplacian of the union of the run's graphs."""
    W = np.zeros((16, 16))
    for A in adjs:
        W = np.maximum(W, np.asarray(A, float))
    d = W.sum(1)
    di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(16) - di[:, None] * W * di[None, :]
    lam, U = np.linalg.eigh(L)
    return U[:, 1:k + 1], lam[1:k + 1]


def mode_amps(z, name, wins, U, nL):
    """Per (window, layer): normalized amplitude of each union mode in the node-means.
    amps[k] = ||u_k^T Hc|| * sqrt(16) / ||Hc||_F  (so sum of squares == 16 * energy)."""
    out = []
    for w in wins:
        row = []
        for L in range(nL):
            H = z[f"{name}_{w}_layer_{L}"].astype(np.float64)
            H = H - H.mean(0)
            fro = max(np.linalg.norm(H), 1e-12)
            row.append(np.round(np.linalg.norm(U.T @ H, axis=1) / fro * 4.0, 3).tolist())
        out.append(row)
    return out


def context_entry(z, name, wins, nL_key=None, graph=None):
    nL = int(z[nL_key][0]) if nL_key and nL_key in z.files else int(z["n_layers"][0])
    frames = []                                    # [win][layer][16][2]
    prev_layer_first = None
    for L in range(nL):
        pass
    # build with alignment chain: iterate layers outer for window-0 chain
    per_layer = []
    for L in range(nL):
        row = []
        P0 = pca3(z[f"{name}_{wins[0]}_layer_{L}"])
        if prev_layer_first is not None:
            P0 = align(P0, prev_layer_first)
        prev_layer_first = P0
        row.append(P0)
        for w in wins[1:]:
            P = pca3(z[f"{name}_{w}_layer_{L}"])
            row.append(align(P, row[-1]))
        per_layer.append(row)
    for wi in range(len(wins)):
        frames.append([np.round(per_layer[L][wi], 3).tolist() for L in range(nL)])
    occ = []                                       # per window: centered unit log-occ
    for w in wins:
        k = f"{name}_{w}_ncnt"
        if k in z.files:
            q = np.log(np.maximum(z[k].astype(float), 1.0))
            q = q - q.mean()
            nq = np.linalg.norm(q)
            occ.append(np.round(q / nq, 4).tolist() if nq > 1e-9 else [0.0] * 16)
        else:
            occ.append([0.0] * 16)
    return {"windows": [WIN_LABELS[w] for w in wins], "nL": nL, "graph": graph,
            "frames": frames, "occ": occ}


def main():
    data = {"runs": {}}
    for d, label in RUNS:
        path = os.path.join(HERE, "runs", d, "nodemeans_dueling.npz")
        if not os.path.isfile(path):
            print(f"skip {d} (no npz)")
            continue
        z = np.load(path, allow_pickle=False)
        entry = {"label": label, "words": [str(w) for w in z["words"]],
                 "graphs": {}, "ctxs": {}}
        if "ctx_names" in z.files:
            names = [str(x) for x in z["ctx_names"]]
            for nm in names:
                gname = nm.split("-", 1)[1] if "-" in nm else nm
                entry["graphs"].setdefault(gname, edges_of(z[f"adjacency_{nm}"]))
                entry["ctxs"][nm] = context_entry(z, nm, PRIMED_WINS,
                                                  nL_key=f"nlayers_{nm}", graph=gname)
            if "fresh_names" in z.files:               # xmodel: fresh inside main npz
                for nm in [str(x) for x in z["fresh_names"]]:
                    entry["ctxs"][nm] = context_entry(z, nm, FRESH_WINS,
                                                      nL_key=f"nlayers_{nm}")
        else:                                          # legacy grid+ring runs
            for nm in LEGACY_CTX:
                entry["graphs"][nm] = edges_of(z[f"adjacency_{nm}"])
                entry["ctxs"][nm] = context_entry(z, nm, PRIMED_WINS, graph=nm)
        fp = os.path.join(HERE, "runs", d, "nodemeans_fresh.npz")
        if os.path.isfile(fp):                         # separate fresh npz (Llama runs)
            zf = np.load(fp, allow_pickle=False)
            entry["ctxs"]["fresh"] = context_entry(zf, "fresh", FRESH_WINS)
        # union-mode axes for the "pattern projection" view
        U, lam = union_modes([adj_from_edges(entry["graphs"][g]) for g in entry["graphs"]])
        entry["modes"] = np.round(U, 4).tolist()
        entry["modeLambda"] = np.round(lam, 3).tolist()
        for nm, c in entry["ctxs"].items():
            wins = PRIMED_WINS if len(c["windows"]) == 4 else FRESH_WINS
            src = np.load(fp, allow_pickle=False) if nm == "fresh" else z
            c["amps"] = mode_amps(src, nm, wins, U, c["nL"])
        data["runs"][d] = entry
        print(f"exported {d}: ctxs={list(entry['ctxs'])}")

    os.makedirs(os.path.join(HERE, "viewer"), exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"))
    html = TEMPLATE.replace("/*__DATA__*/null", payload)
    out = os.path.join(HERE, "viewer", "viewer.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({len(payload)/1e6:.1f} MB data)")


TEMPLATE = r"""<title>Dueling-context PCA viewer</title>
<style>
:root{
  --bg:#f5f6f7; --panel:#ffffff; --ink:#1d242c; --muted:#66707c; --line:#dce0e4;
  --accent:#0e7c86; --accent-soft:#0e7c8622; --chip:#eef1f3;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#14181d; --panel:#1c2229; --ink:#e6eaee; --muted:#8b96a2; --line:#303841;
  --accent:#3fb2bc; --accent-soft:#3fb2bc26; --chip:#242c34;
}}
:root[data-theme="dark"]{
  --bg:#14181d; --panel:#1c2229; --ink:#e6eaee; --muted:#8b96a2; --line:#303841;
  --accent:#3fb2bc; --accent-soft:#3fb2bc26; --chip:#242c34;
}
:root[data-theme="light"]{
  --bg:#f5f6f7; --panel:#ffffff; --ink:#1d242c; --muted:#66707c; --line:#dce0e4;
  --accent:#0e7c86; --accent-soft:#0e7c8622; --chip:#eef1f3;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.45 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{padding:14px 20px 10px;border-bottom:1px solid var(--line)}
header h1{margin:0;font-size:17px;font-weight:650;letter-spacing:.1px}
header p{margin:3px 0 0;color:var(--muted);font-size:12.5px}
.wrap{display:flex;gap:0;min-height:calc(100vh - 62px)}
.rail{width:270px;flex:0 0 270px;padding:16px 18px;border-right:1px solid var(--line);
  display:flex;flex-direction:column;gap:16px}
.stage{flex:1;display:flex;align-items:center;justify-content:center;padding:10px}
label.f{display:block;font-size:11px;font-weight:650;letter-spacing:.6px;
  text-transform:uppercase;color:var(--muted);margin-bottom:5px}
select{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;
  background:var(--panel);color:var(--ink);font:inherit}
input[type=range]{width:100%;accent-color:var(--accent)}
.val{font:12px ui-monospace,Menlo,monospace;color:var(--accent);
  font-variant-numeric:tabular-nums}
.row{display:flex;align-items:center;justify-content:space-between;gap:8px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{display:flex;align-items:center;gap:5px;background:var(--chip);
  border:1px solid var(--line);border-radius:20px;padding:3px 10px;font-size:12px;
  cursor:pointer;user-select:none}
.chip input{accent-color:var(--accent);margin:0}
.chip .sw{width:14px;height:3px;border-radius:2px}
button.play{background:var(--accent);color:#fff;border:0;border-radius:6px;
  padding:5px 12px;font:inherit;font-size:12.5px;cursor:pointer}
button.play:focus-visible,select:focus-visible,.chip:focus-within{outline:2px solid var(--accent);outline-offset:1px}
svg{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  max-width:100%;height:auto;user-select:none;-webkit-user-select:none}
.winlab{font:12px ui-monospace,Menlo,monospace;color:var(--muted)}
.note{font-size:11.5px;color:var(--muted);border-top:1px solid var(--line);padding-top:10px}
@media (max-width:760px){.wrap{flex-direction:column}.rail{width:auto;flex:auto;
  border-right:0;border-bottom:1px solid var(--line)}}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>
<header>
  <h1>Dueling-context PCA viewer</h1>
  <p>Per-node mean residuals &rarr; top-3 PCs. Drag the <b>plot</b> to orbit the 3D
  space; drag <b>context length</b> to move through the prefix&rarr;joint timeline
  (frames Procrustes-aligned in 3D; fractional positions interpolate). Scroll to zoom.</p>
</header>
<div class="wrap">
  <div class="rail">
    <div><label class="f" for="run">Run</label><select id="run"></select></div>
    <div><label class="f" for="ctx">Context (instance)</label><select id="ctx"></select></div>
    <div>
      <div class="row"><label class="f" for="layer">Layer</label><span class="val" id="layerVal"></span></div>
      <input type="range" id="layer" min="0" step="1">
    </div>
    <div>
      <div class="row"><label class="f" for="win">Context length</label>
        <button class="play" id="play">&#9654; play</button></div>
      <input type="range" id="win" min="0" step="0.01">
      <div class="winlab" id="winLab"></div>
    </div>
    <div class="row"><label class="f">View</label>
      <button class="play" id="resetView" style="background:var(--chip);color:var(--ink);border:1px solid var(--line)">reset 3D view</button></div>
    <div><label class="f" for="projSel">Projection</label>
      <select id="projSel">
        <option value="pca">PCA top-3 (3D, drag to orbit)</option>
        <option value="pca2d">PCA top-2 (flat 2D)</option>
        <option value="modes">union-graph mode axes (pick patterns)</option>
        <option value="torus">flat torus: angle(m1,m2) vs angle(m3,m4)</option>
      </select>
      <div id="modePick" style="display:none;margin-top:6px" class="chips"></div>
      <div id="pcPick" style="margin-top:6px" class="chips"></div>
      <label class="chip" style="margin-top:6px"><input type="checkbox" id="deflate">
        remove frequency axis (log-occupancy)</label>
    </div>
    <div><label class="f">Patterns (edge overlays)</label><div class="chips" id="pat"></div></div>
    <div class="chips">
      <label class="chip"><input type="checkbox" id="labels" checked>word labels</label>
      <label class="chip"><input type="checkbox" id="trail">motion trail</label>
    </div>
    <div class="note">Hue = node index (0&ndash;15). Edges are drawn from the <i>true</i>
    graphs, so a geometry is &ldquo;recovered&rdquo; when its edges look short &amp;
    planar. Fresh = blank context fed only the jointly-generated stream.</div>
  </div>
  <div class="stage"><svg id="svg" width="720" height="680" viewBox="0 0 720 680"
    role="img" aria-label="PCA scatter of node means"></svg></div>
</div>
<script>
const DATA = /*__DATA__*/null;
const GCOL = {ring:"#c22f4d", grid:"#2f6f9f", ring3:"#c07c1a", hex:"#5b8c3e",
              antiprism:"#5b8c3e", prism:"#7b5ea7"};
const $ = id => document.getElementById(id);
const runSel=$("run"), ctxSel=$("ctx"), laySl=$("layer"), winSl=$("win");
let playing=null, firstDraw=true;

function hsv(i){const h=i/16*360;return `hsl(${h} 85% 45%)`}
function opt(sel, items, labels){sel.innerHTML="";items.forEach((v,i)=>{
  const o=document.createElement("option");o.value=v;o.textContent=labels?labels[i]:v;
  sel.appendChild(o);});}

function init(){
  const ids=Object.keys(DATA.runs);
  opt(runSel, ids, ids.map(r=>DATA.runs[r].label));
  runSel.onchange=onRun; ctxSel.onchange=onCtx;
  const pk=$("pcPick");
  ["x","y","z"].forEach((axn,ai)=>{
    const s=document.createElement("select"); s.style.width="30%";
    for(let k=0;k<6;k++){const o=document.createElement("option");o.value=k;
      o.textContent=`${axn}: PC${k+1}`; s.appendChild(o);}
    s.value=ai; s.onchange=draw; pk.appendChild(s);
  });
  $("deflate").onchange=draw;
  $("projSel").onchange=()=>{const m=$("projSel").value;
    $("modePick").style.display=m==="modes"?"flex":"none";
    $("pcPick").style.display=(m==="pca"||m==="pca2d")?"flex":"none"; draw();};
  laySl.oninput=draw; winSl.oninput=draw;
  $("labels").onchange=draw; $("trail").onchange=draw;
  $("play").onclick=togglePlay;
  onRun();
}
function run(){return DATA.runs[runSel.value]}
function ctx(){return run().ctxs[ctxSel.value]}

function onRun(){
  opt(ctxSel, Object.keys(run().ctxs));
  const mp=$("modePick"); mp.innerHTML="";
  const lam=run().modeLambda||[];
  ["x","y","z"].forEach((axn,ai)=>{
    const s=document.createElement("select");
    s.style.width="30%";
    lam.forEach((lv,k)=>{const o=document.createElement("option");o.value=k;
      o.textContent=`${axn}: m${k+1} (λ=${lv})`; s.appendChild(o);});
    s.value=Math.min(ai, lam.length-1); s.onchange=draw; mp.appendChild(s);
  });
  const pat=$("pat"); pat.innerHTML="";
  Object.keys(run().graphs).forEach(g=>{
    const l=document.createElement("label"); l.className="chip";
    l.innerHTML=`<input type="checkbox" data-g="${g}" checked>
      <span class="sw" style="background:${GCOL[g]||"#888"}"></span>${g}`;
    l.querySelector("input").onchange=draw; pat.appendChild(l);
  });
  onCtx();
}
function onCtx(){
  const c=ctx();
  const rel=laySl.max>0 ? laySl.value/laySl.max : 0.8;   // keep relative depth
  laySl.max=c.nL-1;
  laySl.value=firstDraw ? Math.min(26,c.nL-1) : Math.round(rel*(c.nL-1));
  winSl.max=c.windows.length-1;
  if(firstDraw){winSl.value=0; firstDraw=false;}
  if(+winSl.value>c.windows.length-1) winSl.value=0;
  draw();
}
function togglePlay(){
  if(playing){clearInterval(playing);playing=null;$("play").innerHTML="&#9654; play";return}
  $("play").innerHTML="&#10074;&#10074; pause";
  playing=setInterval(()=>{
    let v=+winSl.value+0.02;
    if(v>+winSl.max)v=0;
    winSl.value=v; draw();
  },40);
}
function frameAt(c, w, L){
  const w0=Math.floor(w), w1=Math.min(w0+1, c.windows.length-1), f=w-w0;
  if($("projSel").value==="torus"){
    const U=run().modes;
    const a0=c.amps[w0][L], a1=c.amps[w1][L];
    const amp=k=>a0[k]*(1-f)+a1[k]*f;
    return U.map(u=>{
      const th=Math.atan2(u[1]*amp(1), u[0]*amp(0));
      const ph=Math.atan2(u[3]*amp(3), u[2]*amp(2));
      return [th*0.52, ph*0.52, 0];
    });
  }
  if($("projSel").value==="modes"){
    const U=run().modes;                       // [16][6] union Laplacian modes 1..6
    const ax=[...document.querySelectorAll("#modePick select")].map(s=>+s.value);
    const a0=c.amps[w0][L], a1=c.amps[w1][L];
    const amp=k=>a0[k]*(1-f)+a1[k]*f;
    return U.map(u=>ax.map(k=>u[k]*amp(k)*3.4));
  }
  const A=c.frames[w0][L], B=c.frames[w1][L];
  let V=A.map((p,i)=>p.map((x,k)=>x*(1-f)+(B[i][k]||0)*f));   // 16 x 6 PC coords
  if($("deflate").checked && c.occ){
    const q0=c.occ[w0]||[], q1=c.occ[w1]||q0;
    const q=q0.map((x,i)=>x*(1-f)+(q1[i]||0)*f);
    const nq=Math.hypot(...q)||1;
    const qn=q.map(x=>x/nq);
    for(let k=0;k<6;k++){
      let d=0; for(let i=0;i<16;i++) d+=qn[i]*(V[i][k]||0);
      for(let i=0;i<16;i++) V[i][k]-=d*qn[i];
    }
  }
  const ax=[...document.querySelectorAll("#pcPick select")].map(s=>+s.value);
  const flat2d=$("projSel").value==="pca2d";
  return V.map(v=>[v[ax[0]]||0, v[ax[1]]||0, flat2d?0:(v[ax[2]]||0)]);
}
let yaw=0.5, pitch=-0.35, zoom=1;
function rot(p){
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const x=p[0]*cy+p[2]*sy, z=-p[0]*sy+p[2]*cy;         // yaw about Y
  const y=p[1]*cp-z*sp,   z2=p[1]*sp+z*cp;             // pitch about X
  return [x,y,z2];
}
function draw(){
  const c=ctx(), L=+laySl.value, w=+winSl.value;
  $("layerVal").textContent="L"+L+" / "+(c.nL-1);
  const w0=Math.round(w);
  $("winLab").textContent=c.windows.map((s,i)=>i===w0?"["+s+"]":s).join("  ");
  const mode=$("projSel").value;
  const torus=mode==="torus";
  const flat=torus||mode==="pca2d";                     // 2D modes: no orbit/perspective
  const P3=frameAt(c,w,L).map(p=>flat?p:rot(p));
  const S=$("svg"); const W=720,H=680,cx=W/2,cy2=H/2,sc=112*zoom;
  const persp=p=>flat?1:1/(1+0.18*p[2]);
  const X=p=>cx+p[0]*sc*persp(p), Y=p=>cy2-p[1]*sc*persp(p);
  const depth=p=>flat?0:p[2];                           // larger z = farther
  let out="";
  if(mode==="pca2d"){
    const ax=[...document.querySelectorAll("#pcPick select")].map(s=>+s.value);
    out+=`<text x="${cx}" y="${H-14}" fill="var(--muted)" font-size="11"
      text-anchor="middle">PC${ax[0]+1}</text>
      <text x="18" y="${cy2}" fill="var(--muted)" font-size="11" text-anchor="middle"
      transform="rotate(-90 18 ${cy2})">PC${ax[1]+1}</text>`;
  }
  if(torus){                    // flat-torus frame: both axes are angles, wrap at +-pi
    const b=Math.PI*0.52*sc;
    out+=`<rect x="${cx-b}" y="${cy2-b}" width="${2*b}" height="${2*b}" fill="none"
      stroke="var(--line)" stroke-dasharray="4 4"/>
      <text x="${cx}" y="${cy2+b+18}" fill="var(--muted)" font-size="11"
        text-anchor="middle">&theta; = angle in m1-m2 plane (cycle)</text>
      <text x="${cx-b-10}" y="${cy2}" fill="var(--muted)" font-size="11"
        text-anchor="middle" transform="rotate(-90 ${cx-b-10} ${cy2})">&phi; = angle in m3-m4 plane</text>`;
  } else if(mode!=="pca2d"){
    const ax=[[[-1.6,0,0],[1.6,0,0]],[[0,0,-1.6],[0,0,1.6]]];
    ax.forEach(([a,b])=>{const A=rot(a),B=rot(b);
      out+=`<line x1="${X(A)}" y1="${Y(A)}" x2="${X(B)}" y2="${Y(B)}"
        stroke="var(--line)" stroke-width="1" opacity="0.7"/>`;});
  }
  const shown=[...document.querySelectorAll("#pat input:checked")].map(x=>x.dataset.g);
  const wrapLim=Math.PI*0.52;
  shown.forEach(g=>{
    (run().graphs[g]||[]).forEach(([a,b])=>{
      if(torus && (Math.abs(P3[a][0]-P3[b][0])>wrapLim ||
                   Math.abs(P3[a][1]-P3[b][1])>wrapLim)) return;  // skip wrapped edges
      const op=0.62 - 0.13*(depth(P3[a])+depth(P3[b]));
      out+=`<line x1="${X(P3[a])}" y1="${Y(P3[a])}" x2="${X(P3[b])}" y2="${Y(P3[b])}"
        stroke="${GCOL[g]||"#888"}" stroke-width="1.3" opacity="${Math.max(0.15,op)}"/>`;
    });
  });
  if($("trail").checked && w>0.05){
    const Pb=frameAt(c,Math.max(0,w-1),L).map(rot);
    P3.forEach((p,i)=>{out+=`<line x1="${X(Pb[i])}" y1="${Y(Pb[i])}" x2="${X(p)}"
      y2="${Y(p)}" stroke="${hsv(i)}" stroke-width="1" opacity="0.35"
      stroke-dasharray="2 3"/>`;});
  }
  P3.map((p,i)=>({p,i})).sort((a,b)=>depth(b.p)-depth(a.p)).forEach(({p,i})=>{
    const r=(7.8-2.2*depth(p))*Math.sqrt(zoom);
    out+=`<circle cx="${X(p)}" cy="${Y(p)}" r="${Math.max(3,r)}" fill="${hsv(i)}"
      stroke="var(--ink)" stroke-width="0.7"/>`;
    if($("labels").checked)
      out+=`<text x="${X(p)+9}" y="${Y(p)+4}" font-size="11"
        fill="var(--ink)" font-family="ui-monospace,Menlo,monospace">${run().words[i]}</text>`;
  });
  S.innerHTML=out;
}
// orbit controls
(function(){
  const S=$("svg"); let drag=null;
  S.style.cursor="grab"; S.style.touchAction="none";
  S.addEventListener("pointerdown",e=>{e.preventDefault();drag=[e.clientX,e.clientY];
    S.setPointerCapture(e.pointerId); S.style.cursor="grabbing";});
  S.addEventListener("pointermove",e=>{if(!drag)return;
    yaw+=(e.clientX-drag[0])*0.008; pitch+=(e.clientY-drag[1])*0.008;
    pitch=Math.max(-1.5,Math.min(1.5,pitch)); drag=[e.clientX,e.clientY]; draw();});
  const end=e=>{drag=null;S.style.cursor="grab";};
  S.addEventListener("pointerup",end); S.addEventListener("pointercancel",end);
  S.addEventListener("wheel",e=>{e.preventDefault();
    zoom=Math.max(0.4,Math.min(3,zoom*(e.deltaY<0?1.08:0.93))); draw();},{passive:false});
})();
$("resetView").onclick=()=>{yaw=0.5;pitch=-0.35;zoom=1;draw();};
init();
</script>
"""

if __name__ == "__main__":
    main()
