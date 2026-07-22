"""Build a single self-contained interactive HTML viewer of the graph games. Open the
.html in any browser: pick a run+game from the dropdown, play/pause, scrub the round
slider, adjust speed. Both models animate on the word-grid with fading trails, live
grid-distance, and a meet flash. No server, no dependencies -- data is embedded.

Usage:
  python src/graph_viewer.py OUT.html  path/to/transcript.jsonl:ROWS:COLS:Label  [more specs...]
  # or point it at a directory of runs (auto rows/cols from sibling summary json):
  python src/graph_viewer.py OUT.html  runs/game-1-graph
"""
from __future__ import annotations
import os
import sys
import json
import glob

from game1_graph import WORDS


def coord(w, cols):
    i = WORDS.index(w); return [i // cols, i % cols]


def load_transcript(tpath, rows, cols, label):
    by = {}
    for l in open(tpath):
        r = json.loads(l); by.setdefault(r["game"], []).append(r)
    out = []
    for g in sorted(by):
        rr = sorted(by[g], key=lambda x: x["round"])
        a0 = coord(rr[0]["startA"], cols) if "startA" in rr[0] else [0, 0]
        b0 = coord(rr[0]["startB"], cols) if "startB" in rr[0] else [rows - 1, cols - 1]
        A = [a0] + [coord(x["A_word"], cols) for x in rr]
        B = [b0] + [coord(x["B_word"], cols) for x in rr]
        met = next((x["round"] + 1 for x in rr if x["met"]), None)
        out.append({"label": f"{label} — game {g}" + (f" (met @{met})" if met else " (no-meet)"),
                    "rows": rows, "cols": cols, "words": WORDS[:rows * cols], "A": A, "B": B})
    return out


def auto_specs(root):
    """Find *_transcript.jsonl under root; rows/cols from the sibling summary json."""
    specs = []
    for tp in sorted(glob.glob(os.path.join(root, "**", "*_transcript.jsonl"), recursive=True)):
        d = os.path.dirname(tp)
        summ = glob.glob(os.path.join(d, "game1graph_*_vs_*.json"))
        summ = [s for s in summ if "transcript" not in s]
        rows = cols = 5
        if summ:
            try:
                j = json.load(open(summ[0])); m = next(iter(j["modes"].values()))
                rows, cols = m.get("rows", 5), m.get("cols", 5)
            except Exception:
                pass
        mode = os.path.basename(tp).split("game1graph_")[1].split("_QwenInst")[0]
        label = f"{os.path.basename(d)}/{mode}"
        specs.append((tp, rows, cols, label))
    return specs


HTML = """<!doctype html><html><head><meta charset=utf-8><title>graph games viewer</title>
<style>
 body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa}
 #top{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
 select,button{font-size:14px;padding:4px 8px}
 #info{font-weight:600;margin:6px 0}
 canvas{background:#fff;border:1px solid #ccc;border-radius:6px}
 .legend{font-size:13px;color:#444}
 input[type=range]{width:360px}
</style></head><body>
<div id=top>
 <label>run/game: <select id=sel></select></label>
 <button id=play>▶ play</button>
 <button id=prev>⟨ prev</button>
 <button id=next>next ⟩</button>
 <label>speed <input type=range id=spd min=1 max=15 value=4></label>
</div>
<div id=info></div>
<input type=range id=slider min=0 value=0 style="width:600px">
<div><canvas id=cv width=620 height=620></canvas></div>
<div class=legend>● blue = Qwen1 &nbsp; ▲ orange = Qwen2 &nbsp; □ = start &nbsp; ★ = met</div>
<script>
const GAMES = __DATA__;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const sel=document.getElementById('sel'), slider=document.getElementById('slider');
const info=document.getElementById('info'), spd=document.getElementById('spd');
let gi=0, t=0, timer=null;
GAMES.forEach((g,i)=>{const o=document.createElement('option');o.value=i;o.text=g.label;sel.add(o)});
function dist(a,b){return Math.abs(a[0]-b[0])+Math.abs(a[1]-b[1])}
function draw(){
 const g=GAMES[gi], R=g.rows, C=g.cols, pad=40, cw=(cv.width-2*pad)/(C-1||1), ch=(cv.height-2*pad)/(R-1||1);
 ctx.clearRect(0,0,cv.width,cv.height);
 ctx.font='10px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
 for(let r=0;r<R;r++)for(let c=0;c<C;c++){ctx.fillStyle='#bbb';ctx.fillText(g.words[r*C+c],pad+c*cw,pad+r*ch)}
 const px=(p)=>[pad+p[1]*cw, pad+p[0]*ch];
 function trail(P,col){ctx.strokeStyle=col;ctx.globalAlpha=.35;ctx.lineWidth=2;ctx.beginPath();
   for(let k=0;k<=t;k++){const q=px(P[k]);k?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])}ctx.stroke();
   ctx.globalAlpha=.25;for(let k=0;k<t;k++){const q=px(P[k]);ctx.fillStyle=col;ctx.beginPath();ctx.arc(q[0],q[1],4,0,7);ctx.fill()}ctx.globalAlpha=1}
 trail(g.A,'#1f77b4'); trail(g.B,'#ff7f0e');
 const s0=px(g.A[0]), s1=px(g.B[0]);
 ctx.lineWidth=2;ctx.strokeStyle='#1f77b4';ctx.strokeRect(s0[0]-9,s0[1]-9,18,18);
 ctx.strokeStyle='#ff7f0e';ctx.strokeRect(s1[0]-9,s1[1]-9,18,18);
 const a=px(g.A[t]), b=px(g.B[t]), met=(g.A[t][0]==g.B[t][0]&&g.A[t][1]==g.B[t][1]);
 ctx.fillStyle='#1f77b4';ctx.beginPath();ctx.arc(a[0],a[1],13,0,7);ctx.fill();ctx.strokeStyle='#000';ctx.lineWidth=1;ctx.stroke();
 ctx.fillStyle='#ff7f0e';ctx.beginPath();ctx.moveTo(b[0],b[1]-12);ctx.lineTo(b[0]-11,b[1]+9);ctx.lineTo(b[0]+11,b[1]+9);ctx.closePath();ctx.fill();ctx.stroke();
 if(met){ctx.fillStyle='red';ctx.font='34px sans-serif';ctx.fillText('★',a[0],a[1])}
 info.textContent=`${g.label}  |  round ${t}/${g.A.length-1}  |  grid-distance ${dist(g.A[t],g.B[t])}` + (met?'   ★ MET':'');
}
function setGame(i){gi=+i;t=0;slider.max=GAMES[gi].A.length-1;slider.value=0;draw()}
function step(d){t=Math.max(0,Math.min(GAMES[gi].A.length-1,t+d));slider.value=t;draw()}
sel.onchange=e=>setGame(e.target.value);
slider.oninput=e=>{t=+e.target.value;draw()};
document.getElementById('prev').onclick=()=>step(-1);
document.getElementById('next').onclick=()=>step(1);
document.getElementById('play').onclick=function(){
 if(timer){clearInterval(timer);timer=null;this.textContent='▶ play';return}
 this.textContent='⏸ pause';const self=this;
 timer=setInterval(()=>{if(t>=GAMES[gi].A.length-1){clearInterval(timer);timer=null;self.textContent='▶ play';return}step(1)},1000/(+spd.value))};
setGame(0);
</script></body></html>"""


def main():
    out = sys.argv[1]; args = sys.argv[2:]
    specs = []
    for a in args:
        if os.path.isdir(a):
            specs += auto_specs(a)
        else:
            p, r, c, *lab = a.split(":")
            specs.append((p, int(r), int(c), lab[0] if lab else os.path.basename(p)))
    data = []
    for p, r, c, lab in specs:
        data += load_transcript(p, r, c, lab)
    open(out, "w").write(HTML.replace("__DATA__", json.dumps(data)))
    print(f"wrote {out}  ({len(data)} games from {len(specs)} runs)")


if __name__ == "__main__":
    main()
