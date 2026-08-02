er":[0.1259,0.1222,0.0555,0.055,0.0489,0.0476],"eig_power_by_mode":[0.0,0.122,0.126,0.055,0.056,0.034,0.031,0.032,0.04,0.049,0.048,0.016,0.013,0.014,0.012,0.012,0.01,0.011,0.009,0.005,0.005,0.006,0.007,0.006],"eig_lambdas":[-0.0,0.067,0.067,0.146,0.146,0.213,0.213,0.213,0.213,0.25,0.25,0.396,0.396,0.396,0.396,0.5,0.5,0.5,0.5,0.567,0.567,0.567,0.567,0.646],"rsa_by_layer":[0.045,0.086,0.076,0.086,0.118,0.11,0.132,0.215,0.255,0.319,0.286,0.267,0.407,0.425,0.344,0.399,0.408,0.442,0.445,0.421,0.401,0.46,0.465,0.505,0.556,0.635,0.665,0.694,0.708,0.727,0.756,0.741]}}};
  const combos = D.combos, css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim() || k;
  const ws = [...new Set(Object.values(combos).map(c=>c.w))].sort((a,b)=>a-b);
  const Ls = [...new Set(Object.values(combos).map(c=>c.L))].sort((a,b)=>a-b);
  let wi=ws.indexOf(5), Li=Ls.indexOf(6); if(wi<0)wi=0; if(Li<0)Li=0;
  let view='pca', colorby='length', layer=30, playing=null, ax={x:0,y:2,z:4,c:3};
  let yaw=0.7, pitch=0.45, drag_=false, ppx=0, ppy=0;
  const cur=()=>combos[`w${ws[wi]}_L${Ls[Li]}`];
  const cvA=document.getElementById('t-cvA'),ctxA=cvA.getContext('2d');
  const cvB=document.getElementById('t-cvB'),ctxB=cvB.getContext('2d');
  const wS=document.getElementById('t-w'),LS=document.getElementById('t-L'),slider=document.getElementById('t-layer');
  wS.max=ws.length-1; LS.max=Ls.length-1;

  function hex2rgb(h){h=h.replace('#','');if(h.length===3)h=h.split('').map(x=>x+x).join('');return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];}
  function mix(a,b,t){return a.map((v,i)=>Math.round(v+(b[i]-v)*t));}
  function hue(t){return `hsl(${(t*340).toFixed(0)} 72% 58%)`;}
  function diverge(v){const A=hex2rgb(css('--parA')),B=hex2rgb(css('--parB')),N=hex2rgb(css('--mut'));const c=v<0?mix(N,A,Math.min(1,-v)):mix(N,B,Math.min(1,v));return `rgb(${c[0]} ${c[1]} ${c[2]})`;}
  function edgesOf(c){const w=c.w,L=c.L,E=[],nid=(i,j)=>i*w+j;for(let i=0;i<L;i++)for(let j=0;j<w;j++){E.push([nid(i,j),nid((i+1)%L,j)]);E.push([nid(i,j),nid(i,(j+1)%w)]);}return E;}

  function eigCol(vals){ const mc=Math.max(...vals.map(Math.abs),1e-6); return vals.map(v=>diverge(v/mc)); }
  function nodePoints(c){
    if(view==='pca'){ const col = colorby==='eig' ? eigCol(c.eig_proj[layer][ax.c])
        : c.coords.map(co=>hue(colorby==='length'?co[0]/c.L:co[1]/c.w));
      return {P:c.layer_embed[layer].map(r=>[r[0],r[1],r[2]]), col, per:false}; }
    const e=c.eig_proj[layer];
    return {P:e[ax.x].map((_,i)=>[e[ax.x][i],e[ax.y][i],e[ax.z][i]]), col:eigCol(e[ax.c]), per:true};
  }
  function cloudPoints(c){
    if(view==='pca'){ const col = colorby==='eig' ? eigCol(c.cloud_eig[layer][ax.c])
        : c.cloud_node.map(nd=>hue(colorby==='length'?c.coords[nd][0]/c.L:c.coords[nd][1]/c.w));
      return {P:c.layer_cloud[layer].map(r=>[r[0],r[1],r[2]]), col, per:false}; }
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
    if(edges){ctx.strokeStyle=css('--edge');ctx.lineWidth=1;
      edges.map(([a,b])=>[a,b,(S[a][2]+S[b][2])/2]).sort((u,v)=>u[2]-v[2]).forEach(([a,b,z])=>{ctx.globalAlpha=0.25+0.4*(0.5+z*0.5);ctx.beginPath();ctx.moveTo(S[a][0],S[a][1]);ctx.lineTo(S[b][0],S[b][1]);ctx.stroke();});ctx.globalAlpha=1;}
    S.map((s,i)=>i).sort((a,b)=>S[a][2]-S[b][2]).forEach(i=>{const s=S[i],r=(edges?4.4:2.9)+2.2*(0.5+s[2]*0.5);
      ctx.beginPath();ctx.arc(s[0],s[1],r*s[3],0,7);ctx.fillStyle=pts.col[i];ctx.globalAlpha=edges?1:0.82;ctx.fill();ctx.globalAlpha=1;
      if(edges){ctx.lineWidth=0.7;ctx.strokeStyle='rgba(0,0,0,.45)';ctx.stroke();}}); }

  // ---- live mode heatmaps + firing spectrum (react to w,L,layer) ----
  const strip=document.getElementById('t-modestrip'); let cells=[], specCell=null;
  function buildStrip(){ strip.innerHTML=''; cells=[]; const c=cur();
    c.eig_freq.forEach((f,m)=>{ const box=document.createElement('div'); box.style.textAlign='center';
      const cv=document.createElement('canvas'); cv.style.borderRadius='6px'; cv.style.display='block';
      const cap=document.createElement('div'); cap.style.cssText='font-size:9.5px;color:var(--mut);margin-top:5px;font-variant-numeric:tabular-nums;';
      box.appendChild(cv); box.appendChild(cap); strip.appendChild(box); cells.push({cv,cap}); });
    const sb=document.createElement('div'); sb.style.textAlign='center';
    const scv=document.createElement('canvas'); scv.style.display='block';
    const scap=document.createElement('div'); scap.textContent='firing spectrum (λ→)'; scap.style.cssText='font-size:9.5px;color:var(--mut);margin-top:5px;';
    sb.appendChild(scv); sb.appendChild(scap); strip.appendChild(sb); specCell={cv:scv,cap:scap};
  }
  function drawHeat(cv,vals,L,w){ const dpr=Math.min(2,window.devicePixelRatio||1);
    const H=84, W=Math.max(52,Math.min(190,Math.round(H*L/w)));
    cv.style.width=W+'px'; cv.style.height=H+'px'; cv.width=W*dpr; cv.height=H*dpr;
    const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
    let mx=1e-6; vals.forEach(v=>mx=Math.max(mx,Math.abs(v)));
    for(let nd=0;nd<vals.length;nd++){ const i=Math.floor(nd/w), j=nd%w;
      ctx.fillStyle=diverge(vals[nd]/mx); ctx.fillRect(i/L*W, j/w*H, W/L+0.6, H/w+0.6); } }
  function drawSpectrum(cv,power){ const dpr=Math.min(2,window.devicePixelRatio||1), W=200, H=84;
    cv.style.width=W+'px'; cv.style.height=H+'px'; cv.width=W*dpr; cv.height=H*dpr;
    const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
    const p=power.slice(1), mx=Math.max(...p,1e-6), bw=W/p.length;
    p.forEach((v,i)=>{ const h=v/mx*(H-6); ctx.fillStyle=i<6?css('--accent'):'#7c5cff'; ctx.fillRect(i*bw+0.5,H-h,bw-1,h); }); }
  function drawModeStrip(c){ if(cells.length!==c.eig_freq.length) buildStrip();
    c.eig_freq.forEach((f,m)=>{ drawHeat(cells[m].cv, c.eig_proj[layer][m], c.L, c.w);
      cells[m].cap.innerHTML=`mode ${m+1} · (a${f[0]},b${f[1]})<br>p=${c.eig_power[m]}`; });
    drawSpectrum(specCell.cv, c.eig_power_by_mode); }

  function fillAxes(){ const c=cur(),opts=c.eig_freq.map((f,i)=>`<option value="${i}">mode ${i+1} (a${f[0]},b${f[1]}) p${c.eig_power[i]}</option>`).join('');
    ['x','y','z','c'].forEach(a=>{const s=document.getElementById('t-ax-'+a);s.innerHTML=opts;if(ax[a]>=c.eig_freq.length)ax[a]=Math.min(ax[a],c.eig_freq.length-1);s.value=ax[a];});
    const cm=document.getElementById('t-cmode'); cm.innerHTML=opts; cm.value=ax.c; }
  function render(){
    const c=cur();
    document.getElementById('t-wv').textContent=c.w; document.getElementById('t-Lv').textContent=c.L;
    document.getElementById('t-nn').textContent=c.n+' nodes'; wS.value=wi; LS.value=Li;
    slider.max=c.n_layers-1; if(layer>c.n_layers-1)layer=c.n_layers-1; slider.value=layer;
    document.getElementById('t-Lnum').textContent=layer; document.getElementById('t-Lmax').textContent='/'+(c.n_layers-1);
    const np=nodePoints(c); np.P=normalize(np.P,np.per); drawPanel(cvA,ctxA,np,edgesOf(c));
    const cp=cloudPoints(c); cp.P=normalize(cp.P,cp.per); drawPanel(cvB,ctxB,cp,null);
    document.getElementById('t-capA').textContent = view==='pca'?'node means · PCA (PC1–3)':'node means · eigenmode axes';
    document.getElementById('t-statA').textContent=`C${c.w}□C${c.L}`;
    document.getElementById('t-capB').textContent=`${c.cloud_node.length} occurrences`;
    document.getElementById('t-evr').innerHTML=`layer ${layer} · torus-RSA <b style="color:var(--ink)">${c.rsa_by_layer[layer]}</b> · best L${c.best_layer} · wrap ${c.wrap_index>=0?'+':''}${c.wrap_index}`;
    const lg=document.getElementById('t-legend');
    const eigColour = view==='eig' || (view==='pca' && colorby==='eig');
    if(eigColour){const f=c.eig_freq[ax.c];lg.innerHTML=`colour = mode ${ax.c+1} (a${f[0]},b${f[1]})&nbsp;<span class="grad" style="background:linear-gradient(90deg,${css('--parA')},${css('--mut')},${css('--parB')})"></span>`;}
    else lg.innerHTML=`${colorby} coord&nbsp;<span class="grad" style="background:linear-gradient(90deg,hsl(0 72% 58%),hsl(110 72% 58%),hsl(230 72% 58%),hsl(340 72% 58%))"></span>`;
    document.getElementById('t-colorwrap').style.display=view==='eig'?'none':'flex';
    document.getElementById('t-cmode').hidden = !(view==='pca' && colorby==='eig');
    document.getElementById('t-cmode').value = ax.c;
    document.getElementById('t-axes').hidden=view!=='eig';
    drawModeStrip(c);
  }
  function setLayer(L){const c=cur();layer=Math.max(0,Math.min(c.n_layers-1,L));render();}

  function dragc(cv){cv.addEventListener('pointerdown',e=>{drag_=true;ppx=e.clientX;ppy=e.clientY;cv.setPointerCapture(e.pointerId);});
    cv.addEventListener('pointermove',e=>{if(!drag_)return;yaw+=(e.clientX-ppx)*0.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-ppy)*0.01));ppx=e.clientX;ppy=e.clientY;render();});
    cv.addEventListener('pointerup',()=>drag_=false);cv.addEventListener('pointercancel',()=>drag_=false);}
  dragc(cvA);dragc(cvB);
  document.getElementById('t-reset').addEventListener('click',()=>{yaw=0.7;pitch=0.45;render();});
  function seg(id,cb){const g=document.getElementById(id);g.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;[...g.children].forEach(x=>x.setAttribute('aria-pressed',x===b?'true':'false'));cb(b.dataset.v);});}
  seg('t-view',v=>{view=v;render();}); seg('t-color',v=>{colorby=v;render();});
  wS.addEventListener('input',()=>{wi=+wS.value;fillAxes();render();});
  LS.addEventListener('input',()=>{Li=+LS.value;fillAxes();render();});
  ['x','y','z','c'].forEach(a=>document.getElementById('t-ax-'+a).addEventListener('change',e=>{ax[a]=+e.target.value;render();}));
  document.getElementById('t-cmode').addEventListener('change',e=>{ax.c=+e.target.value;render();});
  slider.addEventListener('input',()=>setLayer(+slider.value));
  const playBtn=document.getElementById('t-play');
  playBtn.addEventListener('click',()=>{if(playing){clearInterval(playing);playing=null;playBtn.textContent='▶';}else{playBtn.textContent='⏸';playing=setInterval(()=>{let L=layer+1;if(L>=cur().n_layers)L=0;setLayer(L);},260);}});
  addEventListener('keydown',e=>{if(e.target.tagName==='SELECT'||e.target.tagName==='INPUT')return;
    if(e.key===' '){e.preventDefault();playBtn.click();}
    if(e.key==='p'||e.key==='P'){view='pca';[...document.getElementById('t-view').children].forEach(x=>x.setAttribute('aria-pressed',x.dataset.v==='pca'));render();}
    if(e.key==='e'||e.key==='E'){view='eig';[...document.getElementById('t-view').children].forEach(x=>x.setAttribute('aria-pressed',x.dataset.v==='eig'));render();}
    if(e.key==='ArrowRight')setLayer(layer+1); if(e.key==='ArrowLeft')setLayer(layer-1);});
  new ResizeObserver(render).observe(cvA);
  fillAxes(); render();
})();
</script>

</body></html>