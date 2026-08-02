"""Build gossip_game_viewer.html — circle-of-agents replay of every gossip game.

Scans runs/gossip recursively for v2+ transcripts (belief-read protocol), embeds a
compact form of every game, and emits a self-contained HTML viewer:
  * agents on a circle; node color = agent's current argmax label (K label colors
    per round), intensity via fill; clue holders badged (wrong clue in red)
  * conversation slider (with play button) + round slider that snaps to rounds
  * the active conversation drawn as an arrow S->L with the spoken label chip
  * click an agent -> its full belief distribution (as of its last read), plus
    the listener's before/after for the current conversation
Rerun this script after new runs land to refresh the viewer.
"""
import json
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = ("v1/", "v2_no_independence_note/", "smoke")

games = {}
for f in sorted(glob.glob(os.path.join(HERE, "**", "*_transcript.jsonl"), recursive=True)):
    rel = os.path.relpath(f, HERE)
    if any(s in rel for s in SKIP):
        continue
    lines = [json.loads(l) for l in open(f)]
    meta = lines[0]
    if not any(l.get("type") == "step" and "p_before" in l for l in lines):
        continue                                       # v1 generation-based transcript
    name = rel.replace("/gossip_", " ").replace("_transcript.jsonl", "").replace(".jsonl", "")
    for a, b in (("notes/", "SCRATCHPAD / "), ("decide/", "DECIDE-TIME / "),
                 ("window3/", "MEMORY-WINDOW-3 / "), ("polar/", "POLARIZATION / "),
                 ("alternator10/", "ALTERNATOR 10T/10L blocks / "), ("alternator/", "ALTERNATOR 5T/5L blocks / "),
                 ("switchtask/", "TASK-SWITCH naming->A/B / "), ("switch/", "REMOVE-or-BETRAY 30r / "),
                 ("earlystop/", "EARLY-STOP new-words 50r / "),
                 ("fresh50/", "NEW-WORDS-EACH-ROUND 50r / "), ("duel3/", "3-AGENT new-words 30r / "),
                 ("r10names/", "10r names / "), ("r10/", "10r / "), ("nosent/", "no-sentence 5r / ")):
        name = name.replace(a, b)
    if "/" not in name:
        name = "5r fixed-labels / " + name
    import re as _re
    name = _re.sub(r"_s(\d+) s\1$", r" s\1", name)
    rounds, steps = [], []
    notes = {}
    labels = meta["labels"]
    for l in lines:
        if l["type"] == "note":
            notes.setdefault(str(l["agent"]), []).append([l["round"], l["text"]])
        if l["type"] == "round_start":
            labels = l.get("labels", meta["labels"])
            cm = ({int(k): v for k, v in l["clue_map"].items()} if "clue_map" in l
                  else ({1: l["clue"]} if l.get("clue") else {}))
            rounds.append(dict(labels=labels, correct=labels.index(l["correct"]),
                               clues={str(k): labels.index(v) for k, v in cm.items()},
                               wrong={str(k): (v != l["correct"]) for k, v in cm.items()}))
        elif l["type"] == "step":
            li = rounds[-1]["labels"]
            rd3 = lambda v: [round(x, 3) for x in v]
            steps.append([l["round"], l["S"], l["L"], li.index(l["s_label"]),
                          rd3(l["p_speaker"]), rd3(l["p_before"]), rd3(l["p_after"])])
    games[name] = dict(n=meta["n"], names=meta.get("names", False),
                       var=meta["var"], rounds=rounds, steps=steps, notes=notes,
                       nmode=("replace" if ("update" in rel or "peragent" in rel) else "append"))

print(f"{len(games)} games, {sum(len(g['steps']) for g in games.values())} steps")

TEMPLATE = r"""<meta charset="utf-8"><title>gossip game viewer</title>
<style>
.viz-root{color-scheme:light;
 --surface-1:#fcfcfb;--page:#f9f9f7;--ink-1:#0b0b0b;--ink-2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--bad:#d03b3b;--goodtext:#006300;
 font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink-1);
 background:var(--page);margin:0;padding:18px;min-height:100vh;box-sizing:border-box}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--page:#0d0d0d;--ink-1:#fff;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
 --border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--goodtext:#0ca30c}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--page:#0d0d0d;--ink-1:#fff;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
 --border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--goodtext:#0ca30c}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:0 0 14px;max-width:1080px}
h1{font-size:16px;margin:0 0 8px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
select,button{font:inherit;padding:4px 9px;border:1px solid var(--axis);border-radius:6px;background:var(--surface-1);color:var(--ink-1);cursor:pointer}
input[type=range]{accent-color:var(--s1)}
.slab{color:var(--ink-2);font-size:13px;min-width:210px}
.main{display:flex;gap:18px;flex-wrap:wrap}
svg{display:block}
.panel{flex:1;min-width:300px}
.panel h2{font-size:13.5px;margin:10px 0 6px;color:var(--ink-2);font-weight:600}
.bar{display:flex;align-items:center;gap:8px;margin:4px 0}
.bar .lab{width:105px;font-size:13px;text-align:right;color:var(--ink-2);font-variant-numeric:tabular-nums}
.bar .tr{height:14px;border-radius:0 4px 4px 0;min-width:2px}
.bar .val{font-size:12.5px;color:var(--ink-1)}
.tag{font-size:11px;color:var(--ink-2)}
.padnote{font-size:12.5px;color:var(--ink-2);border-left:3px solid var(--s3);padding:4px 8px;margin:6px 0;background:color-mix(in srgb,var(--s3) 6%,transparent);border-radius:0 6px 6px 0}
.padnote b{color:var(--ink-1)}
.convo{font-size:13.5px;margin:2px 0 8px}
.chipdot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:baseline}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink-2);margin-top:4px}
.wbadge{font-size:10.5px;font-weight:700;fill:#fff}
.note{color:var(--muted);font-size:12px;margin-top:8px}
</style>
<div class="viz-root" id="root">
<div class="card">
 <h1>Gossip game viewer — agents in the circle, one conversation at a time</h1>
 <div class="row">
  <label>Experiment</label><select id="exp"></select>
  <label>Game</label><select id="game"></select>
  <button id="play">&#9654;</button>
 </div>
 <div class="row">
  <span class="slab" id="convlab"></span>
  <input type="range" id="conv" min="1" max="2" value="1" style="flex:1;min-width:260px">
 </div>
 <div class="row">
  <span class="slab" id="roundlab"></span>
  <input type="range" id="round" min="1" max="2" value="1" style="flex:1;min-width:260px">
 </div>
 <div class="main">
  <div>
   <svg id="circle" width="460" height="430"></svg>
   <div class="legend" id="legend"></div>
  </div>
  <div class="panel">
   <div class="convo" id="convo"></div>
   <h2 id="agenthdr">Click an agent to inspect its belief</h2>
   <div id="bars"></div>
   <h2 id="padhdr" style="display:none">Scratchpad</h2>
   <div id="pad"></div>
   <h2>This conversation's listener update</h2>
   <div id="lupdate"></div>
   <div class="note">Node color = agent's current most-believed label; opacity = confidence.
   An agent's "current belief" is its most recent logit read (as speaker or listener) at or
   before the selected conversation. Reveals happen between rounds.</div>
  </div>
 </div>
</div>
</div>
<script>
const GAMES=__DATA__;
const names=["Alice","Ben","Carol","David","Emma"];
const $=id=>document.getElementById(id);
const css=v=>getComputedStyle($("root")).getPropertyValue(v).trim();
const st={game:Object.keys(GAMES)[0], t:1, sel:null, playing:null, snap:null};
const ORDER=["5r fixed-labels","10r","10r names","no-sentence 5r","NEW-WORDS-EACH-ROUND 50r",
 "3-AGENT new-words 30r","EARLY-STOP new-words 50r","REMOVE-or-BETRAY 30r",
 "TASK-SWITCH naming->A/B","ALTERNATOR 5T/5L blocks","ALTERNATOR 10T/10L blocks"];
const fam=k=>k.split(" / ")[0], suf=k=>k.split(" / ").slice(1).join(" / ");
const fams=[...new Set(Object.keys(GAMES).map(fam))]
 .sort((a,b)=>((x=>x<0?99:x)(ORDER.indexOf(a)))-((x=>x<0?99:x)(ORDER.indexOf(b))));
const esel=$("exp"), gsel=$("game");
fams.forEach(f=>{const o=document.createElement("option");o.value=f;o.textContent=f;esel.appendChild(o)});
function fillGames(){
 const f=esel.value; gsel.innerHTML="";
 Object.keys(GAMES).filter(k=>fam(k)===f).forEach(k=>{
  const o=document.createElement("option");o.value=k;o.textContent=suf(k)||k;gsel.appendChild(o)});
 st.game=gsel.value;
}
esel.addEventListener("change",()=>{fillGames();loadGame()});
fillGames();
function pname(g,i){return g.names?names[i-1]:"P"+i}
function loadGame(){
 const g=GAMES[st.game];
 // last-known belief per agent after each step
 st.snap=[]; let cur=Array.from({length:g.n+1},()=>null);
 g.steps.forEach(s=>{cur=cur.slice(); cur[s[1]]=s[4]; cur[s[2]]=s[6]; st.snap.push(cur)});
 $("conv").max=g.steps.length; $("round").max=g.rounds.length;
 st.t=1; st.sel=null; render();
}
function labCol(i){return css(["--s1","--s2","--s3"][i])}
function esc2(x){return String(x).replace(/&/g,"&amp;").replace(/</g,"&lt;")}
function render(){
 const g=GAMES[st.game], s=g.steps[st.t-1], r=s[0], ri=g.rounds[r-1];
 $("conv").value=st.t; $("round").value=r;
 const per=g.steps.filter(x=>x[0]===r), inR=per.indexOf(s)+1;
 $("convlab").textContent=`conversation ${st.t}/${g.steps.length}`;
 $("roundlab").textContent=`round ${r}/${g.rounds.length} (conv ${inR}/${per.length} in round)`;
 // legend
 $("legend").innerHTML=ri.labels.map((l,i)=>{
  const tags=[];
  if(i===ri.correct)tags.push("truth (revealed at round end)");
  Object.entries(ri.clues).forEach(([a,li])=>{if(li===i)tags.push(pname(g,+a)+"'s clue"+(ri.wrong[a]?" (wrong)":""))});
  return `<span><span class="chipdot" style="background:${labCol(i)}"></span>${l}${tags.length?` <span class="tag">— ${tags.join(", ")}</span>`:""}</span>`}).join("");
 // circle
 const svg=$("circle"); svg.innerHTML="";
 const NS="http://www.w3.org/2000/svg", W=+svg.getAttribute("width"), H=+svg.getAttribute("height");
 const cx=W/2, cy=H/2-6, R=Math.min(W,H)/2-64;
 const el=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e};
 const defs=el("defs",{}); defs.innerHTML='<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="'+css("--ink-2")+'"/></marker>';
 const pos=i=>{const a=-Math.PI/2+2*Math.PI*(i-1)/g.n;return [cx+R*Math.cos(a), cy+R*Math.sin(a)]};
 // conversation arrow
 const [x1,y1]=pos(s[1]),[x2,y2]=pos(s[2]);
 const dx=x2-x1,dy=y2-y1,d=Math.hypot(dx,dy),ux=dx/d,uy=dy/d;
 el("line",{x1:x1+ux*38,y1:y1+uy*38,x2:x2-ux*40,y2:y2-uy*40,stroke:css("--ink-2"),"stroke-width":2,"marker-end":"url(#arr)"});
 const mx=(x1+x2)/2+ -uy*16, my=(y1+y2)/2+ ux*16;
 const chip=el("g",{}); const ct=el("text",{x:mx,y:my+4,"text-anchor":"middle","font-size":"12.5","font-weight":"600",fill:labCol(s[3])}); ct.textContent=ri.labels[s[3]];
 // nodes
 const bel=st.snap[st.t-1];
 for(let i=1;i<=g.n;i++){
  const [x,y]=pos(i), b=bel[i];
  const am=b?b.indexOf(Math.max(...b)):null;
  const conf=b?Math.max(...b):0;
  const grp=el("g",{cursor:"pointer"});
  const c=document.createElementNS(NS,"circle");
  c.setAttribute("cx",x);c.setAttribute("cy",y);c.setAttribute("r",26);
  c.setAttribute("fill",am!=null?labCol(am):css("--grid"));
  c.setAttribute("fill-opacity",b?(0.25+0.75*conf).toFixed(2):0.15);
  c.setAttribute("stroke",st.sel===i?css("--ink-1"):css("--axis"));
  c.setAttribute("stroke-width",st.sel===i?3:1.5);
  grp.appendChild(c);
  const t1=document.createElementNS(NS,"text");
  t1.setAttribute("x",x);t1.setAttribute("y",y+4);t1.setAttribute("text-anchor","middle");
  t1.setAttribute("font-size","13");t1.setAttribute("font-weight","700");t1.setAttribute("fill",css("--ink-1"));
  t1.textContent=pname(g,i); grp.appendChild(t1);
  const t2=document.createElementNS(NS,"text");
  t2.setAttribute("x",x);t2.setAttribute("y",y+44);t2.setAttribute("text-anchor","middle");
  t2.setAttribute("font-size","11.5");t2.setAttribute("fill",css("--ink-2"));
  t2.textContent=b?ri.labels[am]+" "+(conf*100).toFixed(0)+"%":"(no read yet)"; grp.appendChild(t2);
  if(ri.clues[String(i)]!=null){
   const w=ri.wrong[String(i)];
   const bg=document.createElementNS(NS,"rect");
   bg.setAttribute("x",x-34);bg.setAttribute("y",y-46);bg.setAttribute("width",68);bg.setAttribute("height",15);
   bg.setAttribute("rx",4);bg.setAttribute("fill",w?css("--bad"):css("--goodtext"));grp.appendChild(bg);
   const bt=document.createElementNS(NS,"text");
   bt.setAttribute("x",x);bt.setAttribute("y",y-34.5);bt.setAttribute("text-anchor","middle");
   bt.setAttribute("class","wbadge");bt.textContent=w?"wrong clue":"true clue";grp.appendChild(bt);
  }
  grp.addEventListener("click",()=>{st.sel=st.sel===i?null:i;render()});
  svg.appendChild(grp);
 }
 // conversation text
 $("convo").innerHTML=`<b>${pname(g,s[1])} &rarr; ${pname(g,s[2])}</b> said <b style="color:${labCol(s[3])}">${ri.labels[s[3]]}</b>`;
 // belief panel
 const bars=(vec,ri2)=>vec.map((v,i)=>`<div class="bar"><span class="lab">${ri2.labels[i]}</span>
   <span class="tr" style="width:${(v*180).toFixed(0)}px;background:${labCol(i)}"></span>
   <span class="val">${(v*100).toFixed(1)}%</span></div>`).join("");
 if(st.sel && bel[st.sel]){
  $("agenthdr").textContent=`${pname(g,st.sel)} — belief as of conversation ${st.t}`;
  $("bars").innerHTML=bars(bel[st.sel],ri);
 } else { $("agenthdr").textContent="Click an agent to inspect its belief"; $("bars").innerHTML=""; }
 const pn=(g.notes||{})[String(st.sel)];
 if(st.sel && pn){
  let vis=pn.filter(x=>x[0]<r);
  const replaced = g.nmode==="replace";
  if(replaced && vis.length) vis=vis.slice(-1);
  else vis=vis.slice(-4);
  $("padhdr").style.display=vis.length?"block":"none";
  $("padhdr").textContent=`${pname(g,st.sel)}'s scratchpad in context during round ${r}` +
    (replaced?" (latest note only — earlier ones were replaced)":" (last entries of its journal)");
  $("pad").innerHTML=vis.map(x=>`<div class="padnote"><b>after round ${x[0]}:</b> ${esc2(x[1])}</div>`).join("") ||
    "";
 } else { $("padhdr").style.display="none"; $("pad").innerHTML=""; }
 $("lupdate").innerHTML=`<div class="tag" style="margin-bottom:2px">${pname(g,s[2])} before &rarr; after hearing it:</div>`+
   bars(s[5],ri)+`<div style="height:6px"></div>`+bars(s[6],ri);
}
$("game").addEventListener("change",e=>{st.game=e.target.value;loadGame()});
$("conv").addEventListener("input",e=>{st.t=+e.target.value;render()});
$("round").addEventListener("input",e=>{
 const g=GAMES[st.game]; const r=+e.target.value;
 st.t=g.steps.findIndex(s=>s[0]===r)+1; render()});
$("play").addEventListener("click",()=>{
 if(st.playing){clearInterval(st.playing);st.playing=null;$("play").innerHTML="&#9654;";return}
 $("play").innerHTML="&#10074;&#10074;";
 st.playing=setInterval(()=>{const g=GAMES[st.game];
  if(st.t>=g.steps.length){clearInterval(st.playing);st.playing=null;$("play").innerHTML="&#9654;";return}
  st.t++;render()},120)});
loadGame();
</script>
"""
html = TEMPLATE.replace("__DATA__", json.dumps(games, separators=(",", ":")))
out = os.path.join(HERE, "gossip_game_viewer.html")
open(out, "w").write(html)
print("written", os.path.getsize(out) // 1024, "KB ->", out)
