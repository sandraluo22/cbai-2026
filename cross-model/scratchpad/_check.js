
const D = {grids:{},sizes:[]};
(function(){
  const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim() || k;
  const grids = D.grids, sizes = D.sizes;
  let ki = Math.max(0, sizes.indexOf(4));
  let view='pca', colorby='parity', layer=30, playing=null, ax={x:0,y:2,z:4,c:5};
  let yaw=0.7, pitch=0.45, drag_=false, ppx=0, ppy=0;
  const cur = ()=> grids['k'+sizes[ki]];
  const cvA=document.getElementById('g-cvA'),ctxA=cvA.getContext('2d');
  const cvB=document.getElementById('g-cvB'),ctxB=cvB.getContext('2d');
  const sizeS=document.getElementById('g-size'), slider=document.getElementById('g-layer');
  sizeS.max=sizes.length-1;

  function hex2rgb(h){h=h.replace('#','');if(h.length===3)h=h.split('').map(x=>x+x).join('');return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];}
  function mix(a,b,t){return a.map((v,i)=>Math.round(v+(b[i]-v)*t));}
  function hue(t){return `hsl(${(t*300).toFixed(0)} 72% 58%)`;}
  function diverge(v){const A=hex2rgb(css('--parA')),B=hex2rgb(css('--parB')),N=hex2rgb(css('--mut'));const c=v<0?mix(N,A,Math.min(1,-v)):mix(N,B,Math.min(1,v));return `rgb(${c[0]} ${c[1]} ${c[2]})`;}
  function edgesOf(c){ const id={}; c.coords.forEach((co,i)=>id[co[0]+','+co[1]]=i); const E=[];
    c.coords.forEach((co,i)=>{ const r=id[(co[0]+1)+','+co[1]], d=id[co[0]+','+(co[1]+1)]; if(r!=null)E.push([i,r]); if(d!=null)E.push([i,d]); }); return E; }
  function eigCol(vals){ const mc=Math.max(...vals.map(Math.abs),1e-6); return vals.map(v=>diverge(v/mc)); }

  function nodeColour(c){
    if(colorby==='eig') return eigCol(c.eig_proj[layer][ax.c]);
    if(colorby==='parity') return c.parity.map(p=>diverge(p));
    if(colorby==='node') return c.coords.map((_,i)=>hue(i/c.n));
    const idx = colorby==='row'?0:1; const k=c.k;
    return c.coords.map(co=>hue(co[idx]/Math.max(1,k-1)));
  }
  function nodePoints(c){
    if(view==='pca') return {P:c.layer_embed[layer].map(r=>[r[0],r[1],r[2]]), col:nodeColour(c), per:false};
    const e=c.eig_proj[layer];
    return {P:e[ax.x].map((_,i)=>[e[ax.x][i],e[ax.y][i],e[ax.z][i]]), col:eigCol(e[ax.c]), per:true};
  }
  function cloudColour(c){
    if(colorby==='eig') return eigCol(c.cloud_eig[layer][ax.c]);
    if(colorby==='parity') return c.cloud_node.map(nd=>diverge(c.parity[nd]));
    if(colorby==='node') return c.cloud_node.map(nd=>hue(nd/c.n));
    const idx = colorby==='row'?0:1; const k=c.k;
    return c.cloud_node.map(nd=>hue(c.coords[nd][idx]/Math.max(1,k-1)));
  }
  function cloudPoints(c){
    if(view==='pca') return {P:c.layer_cloud[layer].map(r=>[r[0],r[1],r[2]]), col:cloudColour(c), per:false};
    const e=c.cloud_eig[layer];
    return {P:e[ax.x].map((_,i)=>[e[ax.x][i],e[ax.y][i],e[ax.z][i]]), col:eigCol(e[ax.c]), per:true};
  }
  function normalize(P,per){ if(!P.length)return P; const m=[0,0,0]; P.forEach(p=>{for(let i=0;i<3;i++)m[i]+=p[i];}); m.forEach((_,i)=>m[i]/=P.length);
    if(per){const s=[1e-6,1e-6,1e-6];P.forEach(p=>{for(let i=0;i<3;i++)s[i]=Math.max(s[i],Math.abs(p[i]-m[i]));});return P.map(p=>[(p[0]-m[0])/s[0],(p[1]-m[1])/s[1],(p[2]-m[2])/s[2]]);}
    let s=1e-6;P.forEach(p=>{for(let i=0;i<3;i++)s=Math.max(s,Math.abs(p[i]-m[i]));});return P.map(p=>[(p[0]-m[0])/s,(p[1]-m[1])/s,(p[2]-m[2])/s]); }
  function rot(p){const cy=Math.cos(yaw),sy=Math.sin(yaw);let x1=cy*p[0]+sy*p[2],z1=-sy*p[0]+cy*p[2];const cx=Math.cos(pitch),sx=Math.sin(pitch);return [x1,cx*p[1]-sx*z1,sx*p[1]+cx*z1];}
  function drawPanel(cv,ctx,pts,edges){ const W=cv.clientWidth,H=cv.clientHeight,dpr=Math.min(2,window.devicePixelRatio||1);
    cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H); if(!pts.P.length)return;
    const R=pts.P.map(rot),cx=W/2,cy=H/2,sc=Math.min(W,H)*0.36;
    const S=R.map(p=>{const pe=1/(1+p[2]*0.28);return [cx+p[0]*sc*pe,cy-p[1]*sc*pe,p[2],pe];});
    const big=pts.P.length>120, dotr=edges?4.4:(big?1.9:2.9);
    if(edges){ctx.strokeStyle=css('--edge');ctx.lineWidth=1;
      edges.map(([a,b])=>[a,b,(S[a][2]+S[b][2])/2]).sort((u,v)=>u[2]-v[2]).forEach(([a,b,z])=>{ctx.globalAlpha=(0.18+0.35*(0.5+z*0.5))*(edges.length>180?0.6:1);ctx.beginPath();ctx.moveTo(S[a][0],S[a][1]);ctx.lineTo(S[b][0],S[b][1]);ctx.stroke();});ctx.globalAlpha=1;}
    S.map((s,i)=>i).sort((a,b)=>S[a][2]-S[b][2]).forEach(i=>{const s=S[i],r=(dotr)+2.0*(0.5+s[2]*0.5);
      ctx.beginPath();ctx.arc(s[0],s[1],r*s[3],0,7);ctx.fillStyle=pts.col[i];ctx.globalAlpha=edges?1:0.8;ctx.fill();ctx.globalAlpha=1;
      if(edges&&!big){ctx.lineWidth=0.7;ctx.strokeStyle='rgba(0,0,0,.45)';ctx.stroke();}}); }

  const strip=document.getElementById('g-modestrip'); let cells=[], specCell=null;
  function buildStrip(){ strip.innerHTML=''; cells=[]; const c=cur();
    c.eig_freq.forEach(()=>{ const box=document.createElement('div'); box.style.textAlign='center';
      const cv=document.createElement('canvas'); cv.style.borderRadius='6px'; cv.style.display='block';
      const cap=document.createElement('div'); cap.style.cssText='font-size:9.5px;color:var(--mut);margin-top:5px;font-variant-numeric:tabular-nums;';
      box.appendChild(cv); box.appendChild(cap); strip.appendChild(box); cells.push({cv,cap}); });
    const sb=document.createElement('div'); sb.style.textAlign='center';
    const scv=document.createElement('canvas'); scv.style.display='block';
    const scap=document.createElement('div'); scap.textContent='projection spectrum (λ→)'; scap.style.cssText='font-size:9.5px;color:var(--mut);margin-top:5px;';
    sb.appendChild(scv); sb.appendChild(scap); strip.appendChild(sb); specCell={cv:scv,cap:scap};
  }
  function drawHeat(cv,vals,coords,k){ const dpr=Math.min(2,window.devicePixelRatio||1), H=88, W=88;
    cv.style.width=W+'px'; cv.style.height=H+'px'; cv.width=W*dpr; cv.height=H*dpr;
    const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
    let mx=1e-6; vals.forEach(v=>mx=Math.max(mx,Math.abs(v)));
    for(let nd=0;nd<vals.length;nd++){ const r=coords[nd][0], cc=coords[nd][1];
      ctx.fillStyle=diverge(vals[nd]/mx); ctx.fillRect(cc/k*W, r/k*H, W/k+0.6, H/k+0.6); } }
  function drawSpectrum(cv,power){ const dpr=Math.min(2,window.devicePixelRatio||1), W=210, H=88;
    cv.style.width=W+'px'; cv.style.height=H+'px'; cv.width=W*dpr; cv.height=H*dpr;
    const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
    const p=power.slice(1), mx=Math.max(...p,1e-6), bw=W/p.length;
    p.forEach((v,i)=>{ const h=v/mx*(H-6); ctx.fillStyle=i===p.length-1?css('--parA'):(i<6?css('--accent'):'#7c5cff'); ctx.fillRect(i*bw+0.5,H-h,Math.max(1,bw-1),h); }); }
  function drawModeStrip(c){ if(cells.length!==c.eig_freq.length) buildStrip();
    c.eig_freq.forEach((f,m)=>{ drawHeat(cells[m].cv, c.eig_proj[layer][m], c.coords, c.k);
      cells[m].cap.innerHTML=`mode ${m+1} · (${f[0]},${f[1]})<br>p=${c.eig_power[m]}`; });
    drawSpectrum(specCell.cv, c.eig_power_by_mode); }

  function fillAxes(){ const c=cur(),opts=c.eig_freq.map((f,i)=>`<option value="${i}">mode ${i+1} (${f[0]},${f[1]}) p${c.eig_power[i]}</option>`).join('');
    ['x','y','z','c'].forEach(a=>{const s=document.getElementById('g-ax-'+a);s.innerHTML=opts;if(ax[a]>=c.eig_freq.length)ax[a]=c.eig_freq.length-1;s.value=ax[a];});
    const cm=document.getElementById('g-cmode'); cm.innerHTML=opts; if(ax.c>=c.eig_freq.length)ax.c=c.eig_freq.length-1; cm.value=ax.c; }
  function render(){
    const c=cur();
    document.getElementById('g-kv').textContent=c.k; document.getElementById('g-kv2').textContent=c.k;
    document.getElementById('g-nn').textContent=c.n+' nodes'; sizeS.value=ki;
    slider.max=c.n_layers-1; if(layer>c.n_layers-1)layer=c.n_layers-1; slider.value=layer;
    document.getElementById('g-Lnum').textContent=layer; document.getElementById('g-Lmax').textContent='/'+(c.n_layers-1);
    const np=nodePoints(c); np.P=normalize(np.P,np.per); drawPanel(cvA,ctxA,np,view==='pca'?edgesOf(c):null);
    const cp=cloudPoints(c); cp.P=normalize(cp.P,cp.per); drawPanel(cvB,ctxB,cp,null);
    document.getElementById('g-capA').textContent = view==='pca'?'node means · PCA (PC1–3)':'node means · eigenmode axes';
    document.getElementById('g-statA').textContent=`${c.k}×${c.k} grid`;
    document.getElementById('g-capB').textContent=`${c.cloud_node.length} occurrences`;
    document.getElementById('g-evr').innerHTML=`layer ${layer} · grid-RSA <b>${c.rsa_by_layer[layer]}</b> · PC1–2 var <b>${c.evr_by_layer[layer]}</b> · best L${c.best_layer}`;
    const lg=document.getElementById('g-legend');
    const eigColour = view==='eig' || (view==='pca' && colorby==='eig');
    if(eigColour){const f=c.eig_freq[ax.c];lg.innerHTML=`colour = mode ${ax.c+1} (${f[0]},${f[1]})&nbsp;<span class="grad" style="background:linear-gradient(90deg,${css('--parA')},${css('--mut')},${css('--parB')})"></span>`;}
    else if(colorby==='parity') lg.innerHTML=`parity ±&nbsp;<span class="grad" style="background:linear-gradient(90deg,${css('--parA')},${css('--mut')},${css('--parB')})"></span>`;
    else lg.innerHTML=`${colorby}&nbsp;<span class="grad" style="background:linear-gradient(90deg,hsl(0 72% 58%),hsl(150 72% 58%),hsl(300 72% 58%))"></span>`;
    document.getElementById('g-colorwrap').style.display=view==='eig'?'none':'flex';
    document.getElementById('g-cmode').hidden = !(view==='pca' && colorby==='eig');
    document.getElementById('g-cmode').value = ax.c;
    document.getElementById('g-axes').hidden=view!=='eig';
    drawModeStrip(c);
  }
  function setLayer(L){const c=cur();layer=Math.max(0,Math.min(c.n_layers-1,L));render();}

  function dragc(cv){cv.addEventListener('pointerdown',e=>{drag_=true;ppx=e.clientX;ppy=e.clientY;cv.setPointerCapture(e.pointerId);});
    cv.addEventListener('pointermove',e=>{if(!drag_)return;yaw+=(e.clientX-ppx)*0.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-ppy)*0.01));ppx=e.clientX;ppy=e.clientY;render();});
    cv.addEventListener('pointerup',()=>drag_=false);cv.addEventListener('pointercancel',()=>drag_=false);}
  dragc(cvA);dragc(cvB);
  document.getElementById('g-reset').addEventListener('click',()=>{yaw=0.7;pitch=0.45;render();});
  function seg(id,cb){const g=document.getElementById(id);g.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;[...g.children].forEach(x=>x.setAttribute('aria-pressed',x===b?'true':'false'));cb(b.dataset.v);});}
  seg('g-view',v=>{view=v;render();}); seg('g-color',v=>{colorby=v;render();});
  sizeS.addEventListener('input',()=>{ki=+sizeS.value;fillAxes();render();});
  ['x','y','z','c'].forEach(a=>document.getElementById('g-ax-'+a).addEventListener('change',e=>{ax[a]=+e.target.value;render();}));
  document.getElementById('g-cmode').addEventListener('change',e=>{ax.c=+e.target.value;render();});
  slider.addEventListener('input',()=>setLayer(+slider.value));
  const playBtn=document.getElementById('g-play');
  playBtn.addEventListener('click',()=>{if(playing){clearInterval(playing);playing=null;playBtn.textContent='▶';}else{playBtn.textContent='⏸';playing=setInterval(()=>{let L=layer+1;if(L>=cur().n_layers)L=0;setLayer(L);},260);}});
  addEventListener('keydown',e=>{if(e.target.tagName==='SELECT'||e.target.tagName==='INPUT')return;
    if(e.key===' '){e.preventDefault();playBtn.click();}
    if(e.key==='p'||e.key==='P'){view='pca';[...document.getElementById('g-view').children].forEach(x=>x.setAttribute('aria-pressed',x.dataset.v==='pca'));render();}
    if(e.key==='e'||e.key==='E'){view='eig';[...document.getElementById('g-view').children].forEach(x=>x.setAttribute('aria-pressed',x.dataset.v==='eig'));render();}
    if(e.key==='ArrowRight')setLayer(layer+1); if(e.key==='ArrowLeft')setLayer(layer-1);});
  new ResizeObserver(render).observe(cvA);
  fillAxes(); render();
})();
