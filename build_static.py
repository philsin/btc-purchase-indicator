# build_static.py — static site for GitHub Pages
# Adds "View at date" slider: zone badge updates for the selected past date.
import io
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ── constants
UA        = {"User-Agent": "btc-pl-tool/1.0"}
GENESIS   = pd.Timestamp("2009-01-03")
PROJ_END  = pd.Timestamp("2040-12-31")
DMA_START = pd.Timestamp("2012-04-01")
DIST      = Path("dist"); DIST.mkdir(parents=True, exist_ok=True)

LEVELS = {
    "Support":     -1.5,
    "Bear":        -0.5,
    "PL Best Fit":  0.0,
    "Frothy":      +1.0,
    "Top":         +1.75,
}

# ── helpers
def log_days(dates) -> np.ndarray:
    d = pd.to_datetime(dates)
    delta = d - GENESIS
    days = np.asarray(delta / np.timedelta64(1, "D")).astype(float)
    days = np.where(days < 1.0, 1.0, days)
    return np.log10(days)

def year_ticks(end_year: int):
    yrs = list(range(2012, 2021))
    if end_year >= 2022:
        yrs += list(range(2022, end_year + 1, 2))
    dv = pd.to_datetime([f"{y}-01-01" for y in yrs])
    tv = log_days(dv)
    return tv.tolist(), [str(y) for y in yrs]

# ── data loaders
def _btc_stooq():
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date" in c})
    df = df.rename(columns={c: "BTC"  for c in df.columns if "close" in c or "price" in c})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _btc_github():
    url = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    r = requests.get(url, headers=UA, timeout=30); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"Closing Price (USD)": "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().sort_values("Date")[["Date","BTC"]]

def load_btc():
    try:
        df = _btc_stooq()
        if len(df):
            print(f"[build] BTC from Stooq: {len(df)} rows")
            return df
    except Exception as e:
        print("[build] Stooq BTC failed:", e)
    df = _btc_github()
    print(f"[build] BTC from GitHub CSV: {len(df)} rows")
    return df

def _gold_stooq():
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date" in c})
    df = df.rename(columns={c: "Gold" for c in df.columns if "close" in c or "price" in c})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().sort_values("Date")[["Date","Gold"]]

def _gold_lbma():
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    r = requests.get(url, headers=UA, timeout=30); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.rename(columns={"USD (PM)":"Gold"})
    df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
    return df.dropna().sort_values("Date")[["Date","Gold"]]

def load_gold():
    try:
        g = _gold_stooq()
        if len(g) > 1000:
            print(f"[build] Gold from Stooq: {len(g)} rows")
            return g
        print("[build] Stooq XAUUSD too short → LBMA mirror")
    except Exception as e:
        print("[build] Stooq Gold failed:", e)
    g = _gold_lbma()
    print(f"[build] Gold from LBMA: {len(g)} rows")
    return g

# ── model
def fit_power(df):
    X = log_days(df["Date"])
    y = np.log10(df["BTC"])
    slope, intercept = np.polyfit(X, y, 1)
    sigma = np.std(y - (slope * X + intercept))
    return slope, intercept, sigma

def anchor_intercept(slope, date, target_price):
    x = log_days([date])[0]
    return np.log10(target_price) - slope * x

def project_monthly(last_date, end_date):
    start = (last_date + pd.offsets.MonthBegin(1)).normalize()
    if start > end_date:
        return pd.DatetimeIndex([])
    return pd.date_range(start, end_date, freq="MS")

# ── figures
def make_powerlaw_fig(full, data, bands_usd_full, bands_gold_hist):
    x_full = log_days(full["Date"])
    x_data = log_days(data["Date"])
    end_year = int(min(2040, full["Date"].dt.year.max()))
    tickvals, ticktext = year_ticks(end_year)

    fig = go.Figure(layout=dict(
        template="plotly_dark",
        font=dict(family="Currency, monospace", size=12),
        xaxis=dict(type="linear", title="Year (log-time)",
                   tickmode="array", tickvals=tickvals, ticktext=ticktext,
                   showgrid=True, gridwidth=0.6),
        yaxis=dict(type="log", title="Price (USD / BTC per oz)",
                   tickformat="$,d", showgrid=True, gridwidth=0.6),
        plot_bgcolor="#111", paper_bgcolor="#111",
        margin=dict(l=60,r=40,t=16,b=48),
        showlegend=False  # hidden until user clicks "Legend"
    ))

    # USD (0..5)
    color_usd = {"Top":"green","Frothy":"rgba(100,255,100,1)","PL Best Fit":"white",
                 "Bear":"rgba(255,100,100,1)","Support":"red"}
    for nm in ["Top","Frothy","PL Best Fit","Bear","Support"]:
        fig.add_trace(go.Scatter(x=x_full, y=bands_usd_full[nm],
                                 name=f"{nm} (USD)", line=dict(color=color_usd[nm], dash="dash"),
                                 visible=True))
    fig.add_trace(go.Scatter(x=x_data, y=data["BTC"],
                             name="BTC (USD)", line=dict(color="gold", width=2),
                             visible=True))

    # Gold (6..11)  — BTC per oz
    color_g = {"Top":"#ffd54f","Frothy":"#ffeb3b","PL Best Fit":"#fff8e1",
               "Bear":"#ffcc80","Support":"#ffb74d"}
    for nm in ["Top","Frothy","PL Best Fit","Bear","Support"]:
        fig.add_trace(go.Scatter(x=x_data, y=bands_gold_hist[nm],
                                 name=f"{nm} (Gold)", line=dict(color=color_g[nm], dash="dash"),
                                 visible=False))
    fig.add_trace(go.Scatter(x=x_data, y=data["BTC_per_oz"],
                             name="BTC (per oz)", line=dict(color="#ffc107", width=2),
                             visible=False))
    return fig

def make_dma_fig(dma):
    fig = go.Figure(layout=dict(
        template="plotly_dark",
        font=dict(family="Currency, monospace", size=12),
        xaxis=dict(type="date", title="Year", showgrid=True, gridwidth=0.5),
        yaxis=dict(type="log", title="BTC Price (USD)", tickformat="$,d", showgrid=True, gridwidth=0.6),
        yaxis2=dict(type="log", title="BTC per oz (Gold)", tickformat=",d",
                    overlaying="y", side="right", showgrid=False),
        plot_bgcolor="#111", paper_bgcolor="#111",
        margin=dict(l=60,r=40,t=16,b=48),
        showlegend=False
    ))
    # USD 0..3
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_200"], name="200-DMA USD",
                             line=dict(color="#2e7d32", width=1.5), visible=True))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_50"],  name="50-DMA USD",
                             line=dict(color="#43a047", width=1.5), visible=True))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC"],     name="BTC USD",
                             line=dict(color="#66bb6a", width=2), visible=True))
    # marker placeholder (added by JS in this static build if desired)
    # Gold 4..7
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["G200"], name="200-DMA Gold",
                             line=dict(color="#ffd54f", width=1.5), yaxis="y2", visible=False))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["G50"],  name="50-DMA Gold",
                             line=dict(color="#ffeb3b", width=1.5), yaxis="y2", visible=False))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_per_oz"], name="BTC per oz",
                             line=dict(color="#ffc107", width=2), yaxis="y2", visible=False))
    return fig

# ── static scaffolding
CSS = """
:root{--bg:#0e0e10;--panel:#121216;--text:#e6e6e6;--muted:#9aa0a6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:16px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Arial}
.wrap{max-width:1280px;margin:0 auto;padding:20px}
h1{font-size:clamp(28px,4vw,40px);margin:6px 0 14px}
a{color:#8ab4ff;text-decoration:none}a:hover{text-decoration:underline}
.badge{display:inline-flex;align-items:center;gap:10px;background:#22242a;border-radius:28px;padding:10px 14px;margin:8px 0 16px}
.dot{width:12px;height:12px;border-radius:50%;background:#8ab4ff;display:inline-block}
.controls{display:flex;gap:16px;align-items:center;margin:8px 0 12px;flex-wrap:wrap}
select,input[type=range]{accent-color:#8ab4ff}
label{display:inline-flex;align-items:center;gap:8px}
.panel{background:var(--panel);border-radius:12px;padding:8px}
.btn{display:inline-flex;align-items:center;gap:8px;border-radius:999px;padding:8px 12px;border:1px solid #2a2d34;color:#e6e6e6}
.btn:hover{background:#1b1e23}
.small{color:var(--muted);font-size:.9rem}
"""

JS = """
const GENESIS_MS = Date.parse('2009-01-03T00:00:00Z');
function setDenom(figId, denom){
  const usdIdx=[0,1,2,3,4,5], goldIdx=[6,7,8,9,10,11];
  let usdVis = denom==='USD'? true:'legendonly';
  let gVis   = denom==='Gold'? true:'legendonly';
  Plotly.restyle(figId, {'visible': usdVis}, usdIdx);
  Plotly.restyle(figId, {'visible': gVis},  goldIdx);
  if(figId==='plfig'){
    if(denom==='USD'){
      Plotly.relayout(figId, {'yaxis.title.text':'Price (USD / BTC per oz)','yaxis.tickformat':'$,d'});
    }else{
      Plotly.relayout(figId, {'yaxis.title.text':'BTC per oz (Gold)','yaxis.tickformat':',d'});
    }
  }
}
function toggleLegend(figId){ Plotly.relayout(figId, {'showlegend': true}); }
function back(){ history.length ? history.back() : location.href='./index.html'; }

function zoneFrom(p, sup, bear, fro, top){
  if(p < sup) return "SELL THE HOUSE!!";
  if(p < bear) return "Undervalued";
  if(p < fro)  return "Fair Value";
  if(p < top)  return "Overvalued";
  return "TO THE MOON";
}

function updateBadge(zone){
  const dotColor = {"SELL THE HOUSE!!":"#ff5252","Undervalued":"#29b6f6",
                    "Fair Value":"#e6e6e6","Overvalued":"#ffb74d",
                    "TO THE MOON":"#ffd54f"}[zone] || "#e6e6e6";
  const dot = document.getElementById('zonedot');
  const txt = document.getElementById('zonetxt');
  if(dot) dot.style.background = dotColor;
  if(txt) txt.textContent = zone;
}

function bindPowerlawInteractions(store){
  const sel = document.getElementById('denom1');
  const slider = document.getElementById('dateidx');
  const label  = document.getElementById('selLabel');
  function apply(){
    const d = sel.value;
    const i = parseInt(slider.value,10);
    const date = store.dates[i];
    let price, sup,bear,fro,top;
    if(d==='USD'){
      price = store.price_usd[i];
      sup = store.usd.Support[i]; bear = store.usd.Bear[i];
      fro = store.usd["Frothy"][i]; top = store.usd["Top"][i];
    }else{
      price = store.price_gold[i];
      sup = store.gold.Support[i]; bear = store.gold.Bear[i];
      fro = store.gold["Frothy"][i]; top = store.gold["Top"][i];
    }
    const zone = zoneFrom(price, sup, bear, fro, top);
    updateBadge(zone);
    label.innerHTML = `<span class="small">${date} · ${d==='USD' ? '$'+price.toLocaleString() : (Math.round(price)).toLocaleString()+' BTC/oz'}</span>`;
    // vertical guide on x (log-time)
    Plotly.relayout('plfig', {shapes: [{
      type:'line', xref:'x', yref:'paper', x0: store.xlog[i], x1: store.xlog[i], y0:0, y1:1,
      line:{color:'#888', width:1, dash:'dot'}
    }]});
  }
  sel.addEventListener('change', ()=>{ setDenom('plfig', sel.value); apply(); });
  slider.addEventListener('input', apply);
  // initial
  setDenom('plfig', sel.value);
  slider.value = store.dates.length-1;
  apply();
}
"""

def wrap_html(title, body, data_js=""):
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>{CSS}</style>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head><body><div class="wrap">
{body}
</div><script>{JS}</script>
<script>{data_js}</script></body></html>"""

def zone_badge_html(default_zone: str) -> str:
    return f'''
<div class="badge"><span id="zonedot" class="dot"></span>
  <b>Current zone:</b> <span id="zonetxt">{default_zone}</span>
</div>'''

# ── main build
def build():
    btc  = load_btc()
    gold = load_gold()
    data = pd.merge(btc, gold, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    data["BTC_per_oz"] = data["Gold"] / data["BTC"]

    # power-law (USD)
    slope, intercept, sigma = fit_power(data)
    intercept = anchor_intercept(slope, pd.Timestamp("2030-01-01"), 491_776)
    sigma_vis = max(sigma, 0.25)

    # projection (USD-only)
    future = project_monthly(data["Date"].iloc[-1], PROJ_END)
    full = pd.concat([data[["Date","BTC"]], pd.DataFrame({"Date": future})], ignore_index=True)

    # Bands – USD over full timeline
    x_full = log_days(full["Date"])
    mid_full = slope * x_full + intercept
    bands_usd_full = {nm: 10 ** (mid_full + sigma_vis * k) for nm,k in LEVELS.items()}

    # Bands – historical arrays for USD & Gold (for zone slider)
    x_data = log_days(data["Date"])
    mid_data = slope * x_data + intercept
    bands_usd_hist = {nm: 10 ** (mid_data + sigma_vis * k) for nm,k in LEVELS.items()}
    bands_gold_hist = {nm: data["Gold"].to_numpy() / bands_usd_hist[nm] for nm in LEVELS}

    # default zone at latest (USD)
    ref = {nm: bands_usd_hist[nm][-1] for nm in LEVELS}
    p   = data["BTC"].iloc[-1]
    if p < ref["Support"]: zone = "SELL THE HOUSE!!"
    elif p < ref["Bear"]:  zone = "Undervalued"
    elif p < ref["Frothy"]:zone = "Fair Value"
    elif p < ref["Top"]:   zone = "Overvalued"
    else:                  zone = "TO THE MOON"

    # figures
    fig1 = make_powerlaw_fig(full, data, bands_usd_full, bands_gold_hist)
    fig1_html = pio.to_html(fig1, include_plotlyjs=False, full_html=False,
                            config={"displaylogo": False}, div_id="plfig")

    dma = data[data["Date"] >= DMA_START].copy()
    dma["BTC_50"]  = dma["BTC"].rolling(50).mean()
    dma["BTC_200"] = dma["BTC"].rolling(200).mean()
    dma["G50"]     = dma["BTC_per_oz"].rolling(50).mean()
    dma["G200"]    = dma["BTC_per_oz"].rolling(200).mean()
    dma = dma.dropna().reset_index(drop=True)
    fig2 = make_dma_fig(dma)
    fig2_html = pio.to_html(fig2, include_plotlyjs=False, full_html=False,
                            config={"displaylogo": False}, div_id="dmafig")

    # ── data blob for JS (power-law slider)
    store = {
        "dates": data["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "xlog":  log_days(data["Date"]).tolist(),
        "price_usd": data["BTC"].round(2).tolist(),
        "price_gold": data["BTC_per_oz"].round(0).tolist(),
        "usd": {k: np.asarray(v).round(2).tolist() for k,v in bands_usd_hist.items()},
        "gold": {k: np.asarray(v).round(2).tolist() for k,v in bands_gold_hist.items()},
    }
    data_js = f"const PL_STORE = {store};"
    # ── index.html
    index_controls_top = """
<div class="controls">
  <a class="btn" href="javascript:void(0)" onclick="back()">← Back</a>
  <label>Denomination:
    <select id="denom1">
      <option value="USD" selected>USD</option>
      <option value="Gold">Gold</option>
    </select>
  </label>
  <a class="btn" href="javascript:void(0)" onclick="toggleLegend('plfig')">Legend</a>
</div>
"""
    index_controls_bottom = """
<div class="controls">
  <label>View at date:
    <input id="dateidx" type="range" min="0" max="999999" value="0" style="width:280px">
  </label>
  <span id="selLabel" class="small"></span>
</div>
"""
    # set correct max in a tiny inline script
    bottom_script = f"<script>document.getElementById('dateidx').max={len(store['dates'])-1};</script>"

    index_body = f"""
<h1>BTC Purchase Indicator</h1>
{zone_badge_html(zone)}
{index_controls_top}
{index_controls_bottom}
<div class="panel">{fig1_html}</div>
{bottom_script}
<script>document.addEventListener('DOMContentLoaded',function(){{
  bindPowerlawInteractions(PL_STORE);
}});</script>
"""
    (DIST/"index.html").write_text(
        wrap_html("BTC Purchase Indicator", index_body, data_js),
        encoding="utf-8"
    )

    # ── dma.html (unchanged UI except legend toggle & denom dropdown wired like before)
    dma_controls = """
<div class="controls">
  <a class="btn" href="javascript:void(0)" onclick="back()">← Back</a>
  <label>Denomination:
    <select id="denom2">
      <option value="USD" selected>USD</option>
      <option value="Gold">Gold</option>
    </select>
  </label>
  <a class="btn" href="javascript:void(0)" onclick="toggleLegend('dmafig')">Legend</a>
</div>
<script>
document.addEventListener('DOMContentLoaded',function(){{
  const sel=document.getElementById('denom2');
  function set(){
    const usdIdx=[0,1,2], goldIdx=[3,4,5];
    Plotly.restyle('dmafig', {{'visible': sel.value==='USD'? true:'legendonly'}}, usdIdx);
    Plotly.restyle('dmafig', {{'visible': sel.value==='Gold'? true:'legendonly'}}, goldIdx);
    if(sel.value==='USD') Plotly.relayout('dmafig', {{'yaxis.title.text':'BTC Price (USD)','yaxis.tickformat':'$,d'}});
    else Plotly.relayout('dmafig', {{'yaxis.title.text':'BTC per oz (Gold)','yaxis.tickformat':',d'}});
  }
  sel.addEventListener('change', set); set();
}});
</script>
"""
    dma_body = f"""
<h1>BTC — 50/200 DMA (USD &amp; BTC per oz)</h1>
{dma_controls}
<div class="panel">{fig2_html}</div>
"""
    (DIST/"dma.html").write_text(wrap_html("BTC — DMA", dma_body), encoding="utf-8")

    (DIST/"robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")
    print("[build] wrote: dist/index.html, dist/dma.html")

if __name__ == "__main__":
    build()