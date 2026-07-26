"""Generate a self-contained static index.html for the Viyog leaderboard.

Ports the Gradio app (app.py) to a single, visually polished HTML file: data
embedded inline, charts via Plotly.js (CDN), a hero number + KPI stat tiles, and
a highlighted leaderboard table. Free to host as an HF *static* Space (Gradio
Spaces now need PRO). Run: `python build_static.py` -> writes hf_static/index.html.

Palette: Okabe-Ito (colourblind-safe) — Viyog=green #009E73, logit=blue #0072B2,
distance=orange #D55E00. Identity is reinforced by legends + the table, never
colour alone.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "hf_static"
OUT.mkdir(exist_ok=True)

LB = pd.read_csv(DATA / "leaderboard.csv")
PM = pd.read_csv(DATA / "permodel_t3.csv")
OD = pd.read_csv(DATA / "ood_difficulty.csv")
META = json.loads((DATA / "meta.json").read_text())

lb = LB[["dataset", "detector", "family", "T1_ID_OOD", "T2_ID_ADV", "T3_OOD_ADV",
         "state_mem_KB", "compute_%fwd", "cpu_lat_%fwd", "accel_energy_%fwd",
         "is_viyog"]].to_dict("records")
pm = PM[["dataset", "model", "detector", "T3", "is_viyog"]].to_dict("records")
od = OD[["dataset", "detector", "kind", "T3"]].to_dict("records")
payload = json.dumps({"LB": lb, "PM": pm, "OD": od, "META": META}, separators=(",", ":"))

HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Viyog — Adversarial vs OOD Leaderboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root{
    --viyog:#009E73;--viyog-soft:#009E7318;--logit:#0072B2;--dist:#D55E00;
    --bg:#f4f6f8;--card:#ffffff;--ink:#111827;--ink2:#374151;--muted:#6b7280;
    --line:#e5e7eb;--track:#eef1f4;--shadow:0 1px 3px rgba(16,24,40,.06),0 1px 2px rgba(16,24,40,.04);
  }
  @media (prefers-color-scheme:dark){:root{
    --viyog:#2dd4a7;--viyog-soft:#2dd4a71f;--logit:#38a3e0;--dist:#f0803c;
    --bg:#0b0e14;--card:#151a23;--ink:#f3f4f6;--ink2:#cbd2dc;--muted:#8b93a1;
    --line:#252c38;--track:#1c222c;--shadow:none;
  }}
  *{box-sizing:border-box}
  html{-webkit-text-size-adjust:100%}
  body{margin:0;background:var(--bg);color:var(--ink);line-height:1.55;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:1040px;margin:0 auto;padding:26px 18px 72px}
  .tag{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
    color:var(--viyog);background:var(--viyog-soft);padding:4px 10px;border-radius:999px}
  h1{font-size:1.5rem;margin:12px 0 4px;letter-spacing:-.01em}
  .lede{color:var(--ink2);margin:0 0 6px;max-width:70ch}
  .muted{color:var(--muted)}
  a{color:var(--viyog);text-decoration:none} a:hover{text-decoration:underline}

  /* hero + KPIs */
  .hero{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:16px;margin:20px 0 8px}
  @media(max-width:720px){.hero{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}
  .herocard{padding:20px 22px;display:flex;flex-direction:column;justify-content:center}
  .herocard .big{font-size:3.6rem;font-weight:800;line-height:1;color:var(--viyog);letter-spacing:-.02em}
  .herocard .biglab{font-weight:600;margin-top:6px}
  .herocard .bigsub{color:var(--muted);font-size:.92rem;margin-top:2px}
  .kpis{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .kpi{padding:14px 16px}
  .kpi .v{font-size:1.7rem;font-weight:750;letter-spacing:-.01em}
  .kpi .l{color:var(--muted);font-size:.82rem;margin-top:2px}
  .kpi.green .v{color:var(--viyog)}

  .explain{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0 4px}
  @media(max-width:720px){.explain{grid-template-columns:1fr}}
  .ex{padding:13px 16px;border-radius:14px;border:1px solid var(--line);background:var(--card)}
  .ex b{font-size:.95rem}
  .ex .k{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em}
  .ex.ood .k{color:var(--logit)} .ex.adv .k{color:var(--dist)}

  .controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:22px 0 6px}
  .lab{color:var(--muted);font-size:.85rem}
  .seg{display:inline-flex;background:var(--track);border-radius:11px;padding:3px}
  .seg button{border:0;background:transparent;color:var(--ink2);padding:7px 15px;cursor:pointer;
    font-size:.88rem;font-weight:600;border-radius:9px}
  .seg button.on{background:var(--card);color:var(--viyog);box-shadow:var(--shadow)}
  select{padding:7px 11px;border-radius:10px;border:1px solid var(--line);background:var(--card);
    color:var(--ink);font-size:.88rem}

  .tabs{display:flex;flex-wrap:wrap;gap:4px;margin:16px 0 0}
  .tabs button{border:0;background:transparent;color:var(--muted);padding:10px 13px;cursor:pointer;
    font-size:.92rem;font-weight:600;border-bottom:2.5px solid transparent}
  .tabs button.on{color:var(--ink);border-bottom-color:var(--viyog)}
  .panel{background:var(--card);border:1px solid var(--line);border-top-left-radius:0;
    border-radius:0 16px 16px 16px;box-shadow:var(--shadow);padding:18px;margin-top:-1px}
  .note{color:var(--muted);font-size:.9rem;margin:0 0 14px}

  table{border-collapse:separate;border-spacing:0;width:100%;font-size:.9rem;
    font-variant-numeric:tabular-nums}
  thead th{position:sticky;top:0;text-align:right;color:var(--muted);font-weight:600;font-size:.78rem;
    text-transform:uppercase;letter-spacing:.03em;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
  thead th:nth-child(2),thead th:nth-child(3){text-align:left}
  tbody td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
  tbody td:nth-child(2),tbody td:nth-child(3){text-align:left}
  tbody tr:hover{background:var(--track)}
  tr.viyog td{background:var(--viyog-soft)}
  tr.viyog td:nth-child(2){box-shadow:inset 3px 0 0 var(--viyog)}
  .det{font-weight:600}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:baseline}
  .t3cell{display:inline-flex;align-items:center;gap:8px;justify-content:flex-end;width:100%}
  .t3bar{height:7px;border-radius:4px;background:var(--viyog);opacity:.85}
  .t3track{width:74px;height:7px;border-radius:4px;background:var(--track);overflow:hidden}
  .plot{width:100%;min-height:470px}

  .about h2{font-size:1.15rem;margin:.9em 0 .4em}
  .about p{color:var(--ink2)} .about table{margin:10px 0}
  .about td,.about th{text-align:left;padding:7px 10px}
  .formula{background:var(--track);border:1px solid var(--line);border-radius:12px;padding:16px 18px;
    margin:12px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;overflow-x:auto}
  .eq{font-size:1.02rem;display:flex;align-items:center;gap:9px;flex-wrap:wrap;color:var(--ink)}
  .eq .op{color:var(--muted)} .eq b{color:var(--viyog)}
  .frac{display:inline-flex;flex-direction:column;text-align:center;vertical-align:middle}
  .frac .num{border-bottom:1.6px solid var(--ink2);padding:0 9px 3px}
  .frac .den{padding:3px 9px 0}
  .steps{counter-reset:s;list-style:none;padding:0;margin:10px 0}
  .steps li{position:relative;padding:9px 0 9px 42px;border-bottom:1px solid var(--line);color:var(--ink2)}
  .steps li:last-child{border-bottom:0}
  .steps li:before{counter-increment:s;content:counter(s);position:absolute;left:0;top:8px;width:27px;height:27px;
    border-radius:50%;background:var(--viyog);color:#fff;display:grid;place-items:center;font-weight:700;font-size:.85rem}
  .viz{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}
  .vizcard{flex:1;min-width:230px;border:1px solid var(--line);border-radius:12px;padding:12px 15px;background:var(--card)}
  .vizcard .h{font-weight:700;font-size:.9rem} .vizcard .s{color:var(--muted);font-size:.82rem}
  code{background:var(--track);padding:1.5px 6px;border-radius:6px;font-size:.86em}
  footer{color:var(--muted);font-size:.82rem;margin-top:26px;text-align:center}
  footer a{color:var(--muted);text-decoration:underline}
</style></head>
<body><div class="wrap">
  <span class="tag">🛡️ Viyog · first-conv detector</span>
  <h1>Telling adversarial attacks apart from out-of-distribution inputs</h1>
  <p class="lede">A deployed model should <b>abstain</b> on a novel-but-safe (OOD) input, but <b>reject</b>
  an adversarial (ADV) attack. Most detectors can't tell the two apart. Viyog reads a tiny
  <b>roughness</b> signal off the first conv layer — <b>training-free, gradient-free, sub-KB</b>.</p>

  <div class="hero">
    <div class="card herocard">
      <div class="big" id="heroBig">–</div>
      <div class="biglab" id="heroLab">adversarial-detection AUROC</div>
      <div class="bigsub" id="heroSub"></div>
    </div>
    <div class="kpis">
      <div class="card kpi green"><div class="v" id="kT3">–</div><div class="l">OOD-vs-ADV AUROC (T3) — the headline task</div></div>
      <div class="card kpi green"><div class="v" id="kMem">–</div><div class="l">detector state per model</div></div>
      <div class="card kpi"><div class="v" id="kArch">–</div><div class="l">architectures evaluated</div></div>
      <div class="card kpi"><div class="v" id="kLight">–</div><div class="l">lighter than Mahalanobis</div></div>
    </div>
  </div>

  <div class="explain">
    <div class="ex ood"><span class="k">OOD → abstain</span><br><b>Out-of-distribution</b>: a novel but benign input (new scene, sensor). Flag for review.</div>
    <div class="ex adv"><span class="k">ADV → reject</span><br><b>Adversarial</b>: a crafted perturbation designed to fool the model. Raise an alarm.</div>
  </div>

  <div class="controls">
    <span class="lab">Dataset (in-distribution):</span>
    <span class="seg" id="dsseg"></span>
  </div>

  <div class="tabs" id="tabs"></div>
  <div class="panel" id="panel"></div>

  <footer>AUROC: 1.0 = perfect · 0.5 = chance. Data from <a href="https://github.com/amanyagami/viyog">results/analysis</a>,
  Viyog (CODES+ISSS 2026, paper #215). Built with Plotly · <a href="https://pypi.org/project/viyog/">pip install viyog</a></footer>
</div>

<script>
const D = __PAYLOAD__;
const CV="#009E73",CL="#0072B2",CD="#D55E00";
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const dark=()=>window.matchMedia&&matchMedia("(prefers-color-scheme:dark)").matches;
const VC=()=>dark()?"#2dd4a7":CV, LC=()=>dark()?"#38a3e0":CL, DC=()=>dark()?"#f0803c":CD, GC=()=>cssv("--muted");
const famColor=f=>({"Viyog (first-conv)":VC(),"logit":LC(),"distance (feature)":DC()}[f]||GC());
const DS=D.META.datasets; let ds=DS.includes("CIFAR-100")?"CIFAR-100":DS[0]; let tab="lead"; let baseline="Energy";
const fmt=(v,n=3)=>Number(v).toFixed(n);
const byDs=a=>a.filter(r=>r.dataset==ds);

const baseLayout=()=>({paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",
  font:{size:13,color:cssv("--ink2"),family:"-apple-system,Segoe UI,Roboto,sans-serif"},
  margin:{l:10,r:22,t:44,b:40},title:{font:{size:14,color:cssv("--ink")}},
  xaxis:{gridcolor:cssv("--line"),zeroline:false,linecolor:cssv("--line")},
  yaxis:{gridcolor:cssv("--line"),zeroline:false,automargin:true},
  legend:{orientation:"h",y:-0.18,font:{size:12}},bargap:0.42,bargroupgap:0.18});
const chance=(axis,val)=>({type:"line",[axis+"ref"]:"paper",[axis+"0"]:0,[axis+"1"]:1,
  [(axis=="x"?"y":"x")+"0"]:val,[(axis=="x"?"y":"x")+"1"]:val,
  line:{color:GC(),dash:"dot",width:1},opacity:.7});
const cfg={responsive:true,displayModeBar:false};

/* ---------- hero + KPIs ---------- */
function updateHero(){
  const v=byDs(D.LB).find(r=>r.detector=="Viyog-D (TV)");
  const maha=byDs(D.LB).find(r=>r.detector=="Mahalanobis")||D.LB.find(r=>r.detector=="Mahalanobis");
  const logitMaxT2=Math.max(...byDs(D.LB).filter(r=>r.family=="logit").map(r=>r.T2_ID_ADV));
  if(!v) return;
  document.getElementById("heroBig").textContent=fmt(v.T2_ID_ADV);
  document.getElementById("heroLab").textContent="adversarial-detection AUROC (ID vs ADV)";
  document.getElementById("heroSub").innerHTML=`on ${ds} — logit detectors top out at <b>${fmt(logitMaxT2)}</b>`;
  document.getElementById("kT3").textContent=fmt(v.T3_OOD_ADV);
  document.getElementById("kMem").textContent=(v.state_mem_KB<1?v.state_mem_KB.toFixed(2):v.state_mem_KB.toFixed(0))+" KB";
  document.getElementById("kArch").textContent=D.META.n_models;
  document.getElementById("kLight").textContent=maha?("~"+Math.round(maha.state_mem_KB/v.state_mem_KB/1000)+",000×"):"—";
}

/* ---------- controls ---------- */
const TABS=[["lead","🏆 Leaderboard"],["cost","⚖️ Cost vs accuracy"],["eff","⚡ Efficiency"],
  ["arch","🧩 Per-architecture"],["ood","🎯 OOD difficulty"],["about","ℹ️ About"]];
function segs(){
  document.getElementById("dsseg").innerHTML=DS.map(d=>`<button class="${d==ds?'on':''}" onclick="setDs('${d}')">${d}</button>`).join("");
  document.getElementById("tabs").innerHTML=TABS.map(([k,l])=>`<button class="${k==tab?'on':''}" onclick="setTab('${k}')">${l}</button>`).join("");
}
window.setDs=d=>{ds=d;updateHero();segs();render()};
window.setTab=t=>{tab=t;segs();render()};
window.setBase=v=>{baseline=v;render()};

function render(){
  const p=document.getElementById("panel");p.className="panel";p.innerHTML="";
  ({lead:renderLead,cost:renderCost,eff:renderEff,arch:renderArch,ood:renderOod,about:renderAbout}[tab])(p);
}

/* ---------- leaderboard ---------- */
function renderLead(p){
  const rows=byDs(D.LB).slice().sort((a,b)=>b.T3_OOD_ADV-a.T3_OOD_ADV);
  const medal=i=>["🥇","🥈","🥉"][i]||(i+1);
  let h=`<p class="note">Ranked by <b>T3 (OOD-vs-ADV)</b> — given a non-ID input, is it OOD or an attack? Viyog is <span style="color:var(--viyog);font-weight:600">green</span>, logit detectors blue, distance detectors orange.</p>`;
  h+=`<div style="overflow-x:auto"><table><thead><tr><th>#</th><th>Detector</th><th>Family</th><th>T3 OOD·ADV ▲</th><th>T2 ID·ADV</th><th>T1 ID·OOD</th><th>State (KB)</th></tr></thead><tbody>`;
  rows.forEach((r,i)=>{
    const pct=Math.max(0,Math.min(1,(r.T3_OOD_ADV-0.5)/0.5))*100;
    h+=`<tr class="${r.is_viyog?'viyog':''}"><td>${medal(i)}</td>`+
      `<td><span class="dot" style="background:${famColor(r.family)}"></span><span class="det">${r.detector}</span></td>`+
      `<td class="muted">${r.family}</td>`+
      `<td><span class="t3cell"><span class="t3track"><span class="t3bar" style="width:${pct}%;display:block"></span></span>${fmt(r.T3_OOD_ADV)}</span></td>`+
      `<td>${fmt(r.T2_ID_ADV)}</td><td>${fmt(r.T1_ID_OOD)}</td>`+
      `<td>${Number(r.state_mem_KB).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</td></tr>`;
  });
  h+=`</tbody></table></div><div id="pl" class="plot" style="margin-top:16px"></div>`;p.innerHTML=h;
  const d=rows.slice().sort((a,b)=>a.T3_OOD_ADV-b.T3_OOD_ADV);
  Plotly.newPlot("pl",[{type:"bar",orientation:"h",x:d.map(r=>r.T3_OOD_ADV),y:d.map(r=>r.detector),
    marker:{color:d.map(r=>famColor(r.family)),line:{width:0}},
    text:d.map(r=>fmt(r.T3_OOD_ADV)),textposition:"outside",cliponaxis:false,
    hovertemplate:"%{y}<br>T3 = %{x:.3f}<extra></extra>"}],
    Object.assign(baseLayout(),{title:`OOD-vs-ADV separation (T3 AUROC) — ${ds}`,height:470,
      xaxis:Object.assign(baseLayout().xaxis,{title:"T3 AUROC",range:[0.45,1.02]}),
      shapes:[chance("x",0.5)],annotations:[{x:0.5,y:1,yref:"paper",text:"chance",showarrow:false,
        font:{size:11,color:GC()},yanchor:"bottom"}]}),cfg);
}

/* ---------- cost vs accuracy ---------- */
function renderCost(p){
  p.innerHTML=`<p class="note">Distance detectors (orange) buy accuracy with <b>7–25&nbsp;MB</b> of state; Viyog (green ★) sits <b>top-left</b> — highest T3 at <b>sub-KB</b> cost.</p><div id="pl" class="plot"></div>`;
  const rows=byDs(D.LB);const fams=["Viyog (first-conv)","logit","distance (feature)"].filter(f=>rows.some(r=>r.family==f));
  const traces=fams.map(f=>{const s=rows.filter(r=>r.family==f);return{
    type:"scatter",mode:"markers+text",name:f,x:s.map(r=>Math.max(r.state_mem_KB,0.003)),y:s.map(r=>r.T3_OOD_ADV),
    text:s.map(r=>r.detector),textposition:"top center",textfont:{size:10,color:cssv("--muted")},
    marker:{size:s.map(r=>r.is_viyog?22:13),color:famColor(f),line:{width:2,color:cssv("--card")},
      symbol:s.map(r=>r.is_viyog?"star":"circle")},
    hovertemplate:"%{text}<br>state %{x:.3f} KB · T3 %{y:.3f}<extra></extra>"};});
  Plotly.newPlot("pl",traces,Object.assign(baseLayout(),{title:`Detector cost vs accuracy — ${ds}  (top-left = cheap & accurate)`,
    height:500,xaxis:Object.assign(baseLayout().xaxis,{title:"state per model (KB, log)",type:"log"}),
    yaxis:Object.assign(baseLayout().yaxis,{title:"T3 OOD-vs-ADV AUROC"}),shapes:[chance("y",0.5)]}),cfg);
}

/* ---------- per-architecture ---------- */
function renderArch(p){
  const opts=[...new Set(byDs(D.PM).filter(r=>!r.is_viyog).map(r=>r.detector))].sort();
  if(!opts.includes(baseline))baseline=opts.includes("Energy")?"Energy":opts[0];
  p.innerHTML=`<div class="controls" style="margin-top:0"><span class="lab">Compare Viyog-D against:</span>`+
    `<select onchange="setBase(this.value)">${opts.map(o=>`<option ${o==baseline?'selected':''}>${o}</option>`).join("")}</select></div>`+
    `<p class="note">Per-model T3. Viyog-D wins on most CNNs; transformers (vit, fastvit) are its weak spots.</p><div id="pl" class="plot"></div>`;
  const d=byDs(D.PM);const V={},B={};
  d.forEach(r=>{if(r.detector=="Viyog-D (TV)")V[r.model]=r.T3;if(r.detector==baseline)B[r.model]=r.T3;});
  const models=Object.keys(V).filter(m=>m in B).sort((a,b)=>V[b]-V[a]);
  Plotly.newPlot("pl",[
    {type:"bar",name:"Viyog-D (TV)",x:models,y:models.map(m=>V[m]),marker:{color:VC()}},
    {type:"bar",name:baseline,x:models,y:models.map(m=>B[m]),marker:{color:LC()}}],
    Object.assign(baseLayout(),{title:`Per-architecture T3 — Viyog-D vs ${baseline} — ${ds}`,barmode:"group",
      height:510,xaxis:Object.assign(baseLayout().xaxis,{tickangle:-40}),
      yaxis:Object.assign(baseLayout().yaxis,{title:"T3 AUROC",range:[0.4,1.0]}),
      shapes:[chance("y",0.5)],legend:{orientation:"h",y:-0.42,font:{size:12}}}),cfg);
}

/* ---------- OOD difficulty ---------- */
function renderOod(p){
  p.innerHTML=`<p class="note"><b>Where it works and where it doesn't:</b> Viyog cleanly separates ADV from <b>far</b>-OOD, but <b>near</b>-OOD (natural images close to training data) is genuinely hard.</p><div id="pl" class="plot"></div>`;
  const d=byDs(D.OD);const order=["Far","Near","Texture"].filter(k=>d.some(r=>r.kind==k));
  const dets=[...new Set(d.map(r=>r.detector))];
  const CMAP={"Viyog-D (TV)":VC(),"Viyog-HF":LC(),"TV (mean)":"#56B4E9","raw L∞":GC()};
  const mean=(det,k)=>{const s=d.filter(r=>r.detector==det&&r.kind==k);return s.length?s.reduce((a,r)=>a+r.T3,0)/s.length:null;};
  const traces=dets.map(det=>({type:"bar",name:det,x:order,y:order.map(k=>mean(det,k)),
    marker:{color:CMAP[det]||GC()},text:order.map(k=>{const v=mean(det,k);return v==null?"":v.toFixed(2);}),
    textposition:"outside",cliponaxis:false,hovertemplate:det+" · %{x}<br>T3 %{y:.3f}<extra></extra>"}));
  Plotly.newPlot("pl",traces,Object.assign(baseLayout(),{title:`OOD-vs-ADV by OOD difficulty (mean over models) — ${ds}`,
    barmode:"group",height:480,xaxis:Object.assign(baseLayout().xaxis,{title:"OOD difficulty"}),
    yaxis:Object.assign(baseLayout().yaxis,{title:"T3 AUROC",range:[0.4,1.08]}),shapes:[chance("y",0.5)]}),cfg);
}

/* ---------- efficiency: runtime overhead + memory ---------- */
function fmtKB(kb){if(kb>=1000)return (kb/1000).toFixed(1)+" MB";if(kb>=1)return kb.toFixed(0)+" KB";
  if(kb>=0.1)return kb.toFixed(2)+" KB";return kb.toFixed(3)+" KB";}
function renderEff(p){
  const d=byDs(D.LB);
  const vi=d.find(r=>r.detector=="Viyog-D (TV)");
  const maha=d.find(r=>r.detector=="Mahalanobis")||D.LB.find(r=>r.detector=="Mahalanobis");
  const light=maha?Math.round(maha.state_mem_KB/vi.state_mem_KB/1000)+",000×":"";
  p.innerHTML=`<p class="note"><b>Viyog's core advantage — cheap on both axes.</b> It reads first-layer activations
    the forward pass already computed, so it runs at <b>~3% of a forward pass</b> and stores <b>${fmtKB(vi.state_mem_KB)}</b>.
    Every baseline needs the <b>full forward</b> (100%); distance detectors also carry <b>megabytes</b> of state.</p>
    <div id="ov" class="plot" style="min-height:340px"></div>
    <div id="mem" class="plot" style="min-height:440px;margin-top:6px"></div>`;
  // (A) runtime overhead — Viyog vs all baselines, % of a full forward pass
  const cats=["Compute","CPU latency","Accelerator energy"];
  const vov=[vi["compute_%fwd"],vi["cpu_lat_%fwd"],vi["accel_energy_%fwd"]];
  Plotly.newPlot("ov",[
    {type:"bar",name:"Viyog",x:cats,y:vov,marker:{color:VC()},
      text:vov.map(v=>v+"%"),textposition:"outside",cliponaxis:false,
      hovertemplate:"Viyog · %{x}<br>%{y}% of a forward pass<extra></extra>"},
    {type:"bar",name:"Every baseline (logit + distance)",x:cats,y:[100,100,100],marker:{color:GC()},
      text:["100%","100%","100%"],textposition:"outside",cliponaxis:false,
      hovertemplate:"Baselines · %{x}<br>100% (full forward)<extra></extra>"}],
    Object.assign(baseLayout(),{title:`Runtime overhead — % of a full forward pass (lower is better) — ${ds}`,
      barmode:"group",height:340,yaxis:Object.assign(baseLayout().yaxis,{title:"% of a forward pass",range:[0,116]}),
      annotations:[{x:0,y:vov[0],text:`${Math.round(100/vov[0])}× cheaper`,showarrow:false,yshift:44,
        font:{size:12,color:VC()}}]}),cfg);
  // (B) detector state memory — log bars, coloured by family
  const rows=d.slice().sort((a,b)=>a.state_mem_KB-b.state_mem_KB);
  Plotly.newPlot("mem",[{type:"bar",orientation:"h",x:rows.map(r=>Math.max(r.state_mem_KB,0.003)),
    y:rows.map(r=>r.detector),marker:{color:rows.map(r=>famColor(r.family))},
    text:rows.map(r=>fmtKB(r.state_mem_KB)),textposition:"outside",cliponaxis:false,
    hovertemplate:"%{y}<br>%{text} state<extra></extra>"}],
    Object.assign(baseLayout(),{title:`Detector state per model (log scale) — Viyog is ${light} lighter than Mahalanobis`,
      height:440,xaxis:Object.assign(baseLayout().xaxis,{title:"KB of persistent state (log)",type:"log"})}),cfg);
}

/* ---------- about ---------- */
function renderAbout(p){
  const v=byDs(D.LB).find(r=>r.detector=="Viyog-D (TV)");
  const smooth="4,30 30,24 56,21 82,20 108,23 134,27 160,29 186,28 216,25";
  const jagged="4,28 16,9 28,37 40,8 52,35 64,7 76,38 88,10 100,34 112,7 124,37 136,9 148,35 160,7 172,37 184,11 196,33 216,18";
  const svg=(pts,col)=>`<svg viewBox="0 0 220 46" width="100%" height="46" style="margin-top:8px" preserveAspectRatio="none">`+
    `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round"/></svg>`;
  p.className="panel about";
  p.innerHTML=`<h2 style="margin-top:0">What Viyog computes</h2>
   <p>Viyog reads one scalar off the <b>first conv layer</b> — the <b>roughness of its dormant (quietest) channels</b>:</p>
   <div class="formula">
     <div class="eq"><b>V(x)</b><span class="op">=</span>mean over the dormant channels of&nbsp; TV<sub>c</sub>
       <span class="op" style="margin-left:10px">→ high = adversarial, low = ID/OOD</span></div>
     <div class="eq" style="margin-top:14px">TV<sub>c</sub><span class="op">=</span>
       <span class="frac"><span class="num">mean |Δ<sub>h</sub> a<sub>c</sub>| &nbsp;+&nbsp; mean |Δ<sub>w</sub> a<sub>c</sub>|</span>
       <span class="den">mean |a<sub>c</sub>| &nbsp;+&nbsp; ε</span></span>
       <span class="op" style="font-size:.82rem;margin-left:8px">avg change between neighbouring pixels,<br>normalised by magnitude → jaggedness, not brightness</span></div>
   </div>
   <h2>In three steps</h2>
   <ol class="steps">
     <li><b>Roughness per channel.</b> On the first-layer activation map, compute each channel's total variation <b>TV<sub>c</sub></b> (the fraction above).</li>
     <li><b>Find the dormant band</b> — one pass over clean ID data: rank channels by mean activation, drop dead ones, keep the <b>quietest 10%</b>.</li>
     <li><b>Score:</b> <b>V(x)</b> = mean TV<sub>c</sub> over just those dormant channels. No training, no gradients, ~${v?v.state_mem_KB.toFixed(2):"0.3"} KB of state.</li>
   </ol>
   <h2>Why it separates attacks from OOD</h2>
   <p>A gradient attack must inject broadband <b>high-frequency residue</b> to flip the label — that residue wakes the
   quiet channels and makes them <b>jagged</b> (high TV). Natural inputs, in-distribution <i>and</i> OOD alike, leave
   them <b>smooth</b> (low TV). So the score fires on adversarials specifically.</p>
   <div class="viz">
     <div class="vizcard"><div class="h" style="color:#0072B2">Natural input (ID / OOD)</div>
       <div class="s">dormant channel stays smooth → low V(x)</div>${svg(smooth,"#0072B2")}</div>
     <div class="vizcard"><div class="h" style="color:#D55E00">Adversarial input</div>
       <div class="s">dormant channel turns jagged → high V(x)</div>${svg(jagged,"#D55E00")}</div>
   </div>
   <h2>The three tasks</h2>
   <table><thead><tr><th>Metric</th><th>Question</th><th>Best at it</th></tr></thead><tbody>
   <tr><td><b>T1</b> ID-vs-OOD</td><td>is this input in-distribution?</td><td>classic OOD detectors</td></tr>
   <tr><td><b>T2</b> ID-vs-ADV</td><td>is this input adversarial?</td><td><b style="color:var(--viyog)">Viyog</b> — logit detectors are blind</td></tr>
   <tr><td><b>T3</b> OOD-vs-ADV</td><td>given a non-ID input, OOD or ADV?</td><td><b style="color:var(--viyog)">Viyog</b> — the headline task</td></tr>
   </tbody></table>
   <p>Directionless re-evaluations across <b>${D.META.n_models} architectures</b> and <b>${DS.length} datasets</b>
   (${DS.join(", ")}). Baselines are the standard <a href="https://pytorch-ood.readthedocs.io">pytorch-ood</a>
   detectors (Energy, MSP, MaxLogit, Entropy, KL-Matching, GEN, Mahalanobis, KNN, ViM).</p>
   <p><b>Use it:</b> <a href="https://pypi.org/project/viyog/"><code>pip install viyog</code></a> ·
   <a href="https://github.com/amanyagami/viyog">github.com/amanyagami/viyog</a>. Pair Viyog (catches ADV)
   with a logit OOD score (catches OOD) for a full three-way ID/OOD/ADV router.</p>`;
}

var _mq=matchMedia("(prefers-color-scheme:dark)");if(_mq.addEventListener)_mq.addEventListener("change",()=>{updateHero();render();});
updateHero();segs();render();
</script></body></html>
"""

html = HTML.replace("__PAYLOAD__", payload)
(OUT / "index.html").write_text(html)
print(f"wrote {OUT/'index.html'} ({len(html)/1024:.1f} KB)  rows: LB={len(lb)} PM={len(pm)} OD={len(od)}")
