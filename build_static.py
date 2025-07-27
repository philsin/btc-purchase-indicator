#!/usr/bin/env python3 
# ─────────────────────────────────────────────────────────────
# build_static.py  ·  BTC Purchase Indicator (static Plotly)
#  - Power-law bands on log-time (days since 2009-01-03)
#  - Denomination: USD/BTC or Gold oz/BTC
#  - Unified hover: shows all lines; header = Month Year
#  - Weekly (Monday) slider, snaps to nearest historical point
#  - BTC marker follows slider
#  - Readout shows only date + BTC price
#  - Tap outside chart hides hover box
# ─────────────────────────────────────────────────────────────

from pathlib import Path
import io, json, requests, numpy as np, pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

GENESIS  = pd.Timestamp("2009-01-03")
PROJ_END = pd.Timestamp("2040-12-31")
UA       = {"User-Agent": "btc-pl-pages/1.2"}

LEVELS = {
    "Top":         +1.75,
    "Frothy":      +1.00,
    "PL Best Fit":  0.00,
    "Bear":        -0.50,
    "Support":     -1.50,
}
COLORS = {
    "Top":         "#16a34a",
    "Frothy":      "#86efac",
    "PL Best Fit": "#ffffff",
    "Bear":        "#fda4af",
    "Support":     "#ef4444",
    "BTC":         "#ffd166",
    "BTC_MARK":    "#ffd166",
}
DASHES = {k: "dash" for k in LEVELS}
DASHES["PL Best Fit"] = "dash"

# --------------------- loaders
def _read_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers=UA)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def _btc_stooq() -> pd.DataFrame:
    df = _read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")
    df.columns = [c.lower() for c in df.columns]
    dcol = [c for c in df.columns if "date" in c][0]
    ccol = [c for c in df.columns if ("close" in c) or ("price" in c)][-1]
    df = df.rename(columns={dcol: "Date", ccol: "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _btc_github() -> pd.DataFrame:
    df = _read_csv("https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv")
    df = df.rename(columns={"Closing Price (USD)": "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _gold_stooq() -> pd.DataFrame:
    df = _read_csv("https://stooq.com/q/d/l/?s=xauusd&i=d")
    df.columns = [c.lower() for c in df.columns]
    dcol = [c for c in df.columns if "date" in c][0]
    ccol = [c for c in df.columns if ("close" in c) or ("price" in c)][-1]
    df = df.rename(columns={dcol: "Date", ccol: "Gold"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("Gold>0").sort_values("Date")[["Date","Gold"]]

def _gold_lbma() -> pd.DataFrame:
    df = _read_csv("https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    col = "USD (PM)" if "USD (PM)" in df.columns else ("USD (AM)" if "USD (AM)" in df.columns else [c for c in df.columns if "USD" in c.upper()][0])
    df = df.rename(columns={col: "Gold"})
    df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
    return df.dropna().query("Gold>0").sort_values("Date")[["Date","Gold"]]

def load_btc_gold() -> pd.DataFrame:
    try:
        btc = _btc_stooq()
        if len(btc) < 1000: raise ValueError
    except Exception:
        btc = _btc_github()
    try:
        gold = _gold_stooq()
        if len(gold) < 1000: raise ValueError
    except Exception:
        gold = _gold_lbma()
    gold_ff = gold.set_index("Date").reindex(btc["Date"]).ffill().reset_index().rename(columns={"index":"Date"})
    return btc.merge(gold_ff, on="Date", how="left").dropna()

# --------------------- math
def log_days(dates) -> np.ndarray:
    td = pd.to_datetime(dates) - GENESIS
    if isinstance(td, pd.Series):
        days = td.dt.days.to_numpy()
    elif isinstance(td, pd.TimedeltaIndex):
        days = td.days.astype(float)
    else:
        days = (np.asarray(td) / np.timedelta64(1, "D")).astype(float)
    days = np.where(days <= 0, np.nan, days)
    return np.log10(days)

def fit_power(dates: pd.Series, values: pd.Series):
    X = log_days(dates)
    y = np.log10(values.to_numpy(dtype="float64"))
    m, b = np.polyfit(X[np.isfinite(X)], y[np.isfinite(X)], 1)
    sigma = float(np.std(y[np.isfinite(X)] - (m*X[np.isfinite(X)] + b)))
    return m, b, sigma

def build_bands(dates, m, b, sigma) -> dict:
    X = log_days(dates)
    mid = 10 ** (m*X + b)
    out = {"mid": mid}
    for name, k in LEVELS.items():
        if name == "PL Best Fit": continue
        out[name] = 10 ** (m*X + b + sigma*k)
    return out

def year_ticks(start=2012, dense_until=2020, end=2040):
    years = list(range(start, dense_until+1)) + list(range(dense_until+2, end+1, 2))
    vals  = log_days([pd.Timestamp(f"{y}-01-01") for y in years]).tolist()
    text  = [str(y) for y in years]
    return vals, text

# --------------------- figure
def make_powerlaw_fig(df: pd.DataFrame):
    """
    Builds the power-law chart on log-time (x) with:
      - USD and Gold (oz/BTC) denominations (Gold hidden initially)
      - Lines: Top, Frothy, PL Best Fit, Bear, Support, BTC
      - One composite hover per denomination (Month Year + all 6 values)
      - Colored hover rows matching line colors
      - Slider-driven BTC marker (indices exposed via layout.meta)
      - Projection of bands monthly to 2040-12
    Returns: fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist
    """
    # ----- series -----
    usd = df["BTC"].astype(float)                  # USD / BTC (price)
    gld = (df["BTC"] / df["Gold"]).astype(float)   # Gold oz / BTC

    # ----- power-law fits (log-time) -----
    m_u, b_u, s_u = fit_power(df["Date"], usd)
    m_g, b_g, s_g = fit_power(df["Date"], gld)

    # ----- extend dates to 2040 (monthly) -----
    last = df["Date"].iloc[-1]
    future = pd.date_range(last + pd.offsets.MonthBegin(1), PROJ_END, freq="MS")
    full_dates = pd.Index(df["Date"]).append(future)

    # ----- bands over full span -----
    bands_usd = build_bands(full_dates, m_u, b_u, s_u)   # keys: mid, Top, Frothy, Bear, Support
    bands_gld = build_bands(full_dates, m_g, b_g, s_g)

    # ----- x arrays (log10 days since GENESIS) -----
    x_hist = log_days(df["Date"])
    x_full = log_days(full_dates)

    # Month-Year strings aligned to x_full (for composite hover headers)
    monyr_full = np.array([pd.Timestamp(d).strftime("%b %Y") for d in full_dates])

    # Align historical BTC series to full_dates for hover readout
    usd_full = (
        pd.Series(usd.values, index=df["Date"])
          .reindex(full_dates).ffill().to_numpy()
    )
    gld_full = (
        pd.Series(gld.values, index=df["Date"])
          .reindex(full_dates).ffill().to_numpy()
    )

    fig = go.Figure()
    order = ["Top", "Frothy", "PL Best Fit", "Bear", "Support"]

    # ================= USD group (visible) =================
    # Lines (suppress their own hover; we’ll use one composite)
    for name in order:
        y = bands_usd["mid"] if name == "PL Best Fit" else bands_usd[name]
        fig.add_trace(go.Scatter(
            x=x_full, y=y, mode="lines",
            line=dict(color=COLORS[name], width=2, dash=DASHES[name]),
            name=name, legendgroup="USD",
            hoverinfo="skip",  # <- only composite hover shows
            visible=True
        ))

    # BTC (USD)
    fig.add_trace(go.Scatter(
        x=x_hist, y=usd, mode="lines",
        line=dict(color=COLORS["BTC"], width=2.5),
        name="BTC", legendgroup="USD",
        hoverinfo="skip",
        visible=True
    ))
    USD_BTC_IDX = len(fig.data) - 1

    # Slider-driven BTC marker (USD)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=COLORS["BTC_MARK"], size=8, line=dict(color="#000", width=.5)),
        name=" ", legendgroup="USD", showlegend=False, hoverinfo="skip", visible=True
    ))
    USD_MARK_IDX = len(fig.data) - 1

    # Composite hover (USD): Month Year + color-coded rows
    hover_usd = (
        "<b>%{customdata[0]}</b><br>"
        "<span style='color:" + COLORS["Top"] + "'>Top</span> | "
        "<span style='color:" + COLORS["Top"] + "'>%{customdata[1]:$,.0f}</span><br>"
        "<span style='color:" + COLORS["Frothy"] + "'>Frothy</span> | "
        "<span style='color:" + COLORS["Frothy"] + "'>%{customdata[2]:$,.0f}</span><br>"
        "<span style='color:" + COLORS["PL Best Fit"] + "'>PL Best Fit</span> | "
        "<span style='color:" + COLORS["PL Best Fit"] + "'>%{customdata[3]:$,.0f}</span><br>"
        "<span style='color:" + COLORS["Bear"] + "'>Bear</span> | "
        "<span style='color:" + COLORS["Bear"] + "'>%{customdata[4]:$,.0f}</span><br>"
        "<span style='color:" + COLORS["Support"] + "'>Support</span> | "
        "<span style='color:" + COLORS["Support"] + "'>%{customdata[5]:$,.0f}</span><br>"
        "<span style='color:" + COLORS["BTC"] + "'>BTC</span> | "
        "<span style='color:" + COLORS["BTC"] + "'>%{customdata[6]:$,.0f}</span>"
        "<extra></extra>"
    )
    cd_usd = np.column_stack([
        monyr_full,
        bands_usd["Top"], bands_usd["Frothy"], bands_usd["mid"],
        bands_usd["Bear"], bands_usd["Support"],
        usd_full
    ])
    fig.add_trace(go.Scatter(
        x=x_full, y=bands_usd["mid"],
        mode="markers",
        marker=dict(size=1, opacity=0),
        name="", showlegend=False, legendgroup="USD",
        hovertemplate=hover_usd,
        customdata=cd_usd,
        visible=True
    ))

    # ================= GOLD group (hidden) =================
    for name in order:
        y = bands_gld["mid"] if name == "PL Best Fit" else bands_gld[name]
        fig.add_trace(go.Scatter(
            x=x_full, y=y, mode="lines",
            line=dict(color=COLORS[name], width=2, dash=DASHES[name]),
            name=name, legendgroup="GLD",
            hoverinfo="skip",
            visible=False
        ))

    # BTC (Gold oz/BTC)
    fig.add_trace(go.Scatter(
        x=x_hist, y=gld, mode="lines",
        line=dict(color=COLORS["BTC"], width=2.5),
        name="BTC", legendgroup="GLD",
        hoverinfo="skip",
        visible=False
    ))
    GLD_BTC_IDX = len(fig.data) - 1

    # Slider-driven BTC marker (Gold)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=COLORS["BTC_MARK"], size=8, line=dict(color="#000", width=.5)),
        name=" ", legendgroup="GLD", showlegend=False, hoverinfo="skip", visible=False
    ))
    GLD_MARK_IDX = len(fig.data) - 1

    # Composite hover (Gold)
    hover_gld = (
        "<b>%{customdata[0]}</b><br>"
        "<span style='color:" + COLORS["Top"] + "'>Top</span> | "
        "<span style='color:" + COLORS["Top"] + "'>%{customdata[1]:,.2f} oz/BTC</span><br>"
        "<span style='color:" + COLORS["Frothy"] + "'>Frothy</span> | "
        "<span style='color:" + COLORS["Frothy"] + "'>%{customdata[2]:,.2f} oz/BTC</span><br>"
        "<span style='color:" + COLORS["PL Best Fit"] + "'>PL Best Fit</span> | "
        "<span style='color:" + COLORS["PL Best Fit"] + "'>%{customdata[3]:,.2f} oz/BTC</span><br>"
        "<span style='color:" + COLORS["Bear"] + "'>Bear</span> | "
        "<span style='color:" + COLORS["Bear"] + "'>%{customdata[4]:,.2f} oz/BTC</span><br>"
        "<span style='color:" + COLORS["Support"] + "'>Support</span> | "
        "<span style='color:" + COLORS["Support"] + "'>%{customdata[5]:,.2f} oz/BTC</span><br>"
        "<span style='color:" + COLORS["BTC"] + "'>BTC</span> | "
        "<span style='color:" + COLORS["BTC"] + "'>%{customdata[6]:,.2f} oz/BTC</span>"
        "<extra></extra>"
    )
    cd_gld = np.column_stack([
        monyr_full,
        bands_gld["Top"], bands_gld["Frothy"], bands_gld["mid"],
        bands_gld["Bear"], bands_gld["Support"],
        gld_full
    ])
    fig.add_trace(go.Scatter(
        x=x_full, y=bands_gld["mid"],
        mode="markers",
        marker=dict(size=1, opacity=0),
        name="", showlegend=False, legendgroup="GLD",
        hovertemplate=hover_gld,
        customdata=cd_gld,
        visible=False
    ))

    # ----- axes (year ticks on log-time) -----
    tickvals, ticktext = year_ticks(2012, 2020, 2040)

    fig.update_layout(
        template="plotly_dark",
        hovermode="x",  # only our composite hover appears
        xaxis=dict(
            title="Year (log-time)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridcolor="#263041", zeroline=False
        ),
        yaxis=dict(
            title="USD / BTC", type="log", tickformat="$,d",
            showgrid=True, gridcolor="#263041", zeroline=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(
            bgcolor="rgba(20,24,32,0.85)",  # subtle, translucent
            bordercolor="#3b4455",
            font=dict(color="#e5e7eb"),
            align="left"
        ),
        margin=dict(l=60, r=24, t=18, b=64),
        paper_bgcolor="#0f1116", plot_bgcolor="#151821",
        # expose indices for page JS to move the markers & toggle denom
        meta=dict(
            USD_MARK_IDX=USD_MARK_IDX,
            GLD_MARK_IDX=GLD_MARK_IDX
        )
    )

    return fig, full_dates, bands_usd, bands_gld, usd, gld

# --------------------- HTML writer
def write_index_html(fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist,
                     out_path="dist/index.html", page_title="BTC Purchase Indicator"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fig_json = pio.to_json(fig, pretty=False)
    dates_iso = pd.Series(pd.to_datetime(full_dates)).dt.strftime("%Y-%m-%d").tolist()
    hist_iso  = pd.Series(pd.to_datetime(full_dates[:len(usd_hist)])).dt.strftime("%Y-%m-%d").tolist()

    payload = {
        "dates": dates_iso,
        "hist": {
            "dates": hist_iso,
            "usd": pd.Series(usd_hist).astype(float).round(6).tolist(),
            "gld": pd.Series(gld_hist).astype(float).round(6).tolist(),
        },
        "usd": {k: np.asarray(v).astype(float).round(6).tolist() for k,v in {
            "Top": bands_usd["Top"], "Frothy": bands_usd["Frothy"],
            "Mid": bands_usd["mid"], "Bear": bands_usd["Bear"], "Support": bands_usd["Support"]
        }.items()},
        "gld": {k: np.asarray(v).astype(float).round(6).tolist() for k,v in {
            "Top": bands_gld["Top"], "Frothy": bands_gld["Frothy"],
            "Mid": bands_gld["mid"], "Bear": bands_gld["Bear"], "Support": bands_gld["Support"]
        }.items()}
    }
    arrays_json = json.dumps(payload, separators=(",", ":"))

    TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root { color-scheme:dark; --bg:#0f1116; --fg:#e7e9ee; --muted:#8e95a5; --card:#151821; }
  html,body{margin:0;background:var(--bg);color:var(--fg);
    font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:1100px;margin:24px auto;padding:0 16px}
  h1{font-size:clamp(28px,4.5vw,44px);margin:0 0 12px}
  .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  .chip{background:var(--card);border:1px solid #263041;border-radius:999px;padding:8px 12px;display:inline-flex;align-items:center;gap:10px}
  .dot{width:12px;height:12px;border-radius:50%;background:#fff}
  .btn,select{background:var(--card);border:1px solid #263041;border-radius:12px;padding:8px 10px;color:var(--fg);font:inherit}
  #fig{height:70vh;min-height:430px;background:var(--card);border-radius:14px;padding:8px}
  .slider{width:100%}
  .stack{display:flex;flex-direction:column;gap:2px;font-size:0.95rem}
</style>
</head>
<body>
  <div class="wrap">
    <h1>BTC Purchase Indicator</h1>

    <div class="row" style="margin-bottom:8px;">
      <div class="chip"><span id="zoneDot" class="dot"></span> <b>Price Zone:</b> <span id="zoneTxt" style="margin-left:6px">—</span></div>
      <div class="row" style="gap:8px;">
        <label for="denom">Denomination</label>
        <select id="denom" style="min-width:6.0rem;">
          <option value="USD" selected>USD</option>
          <option value="Gold">Gold</option>
        </select>
      </div>
      <button class="btn" id="legendBtn">Legend</button>
    </div>

    <div class="row" style="margin:6px 0 2px;">
      <label for="dateSlider">View at date:</label>
      <label class="row" style="gap:6px;"><input type="checkbox" id="chkToday"/> Today</label>
    </div>
    <input id="dateSlider" class="slider" type="range" min="0" value="0" step="1"/>
    <div id="dateRead" class="stack" style="opacity:.95; margin:6px 0 12px 2px;"></div>

    <div id="fig"></div>
  </div>

  <script type="application/json" id="figjson">__FIGJSON__</script>
  <script type="application/json" id="arrays">__ARRAYS__</script>

  <script>
    const FIG = JSON.parse(document.getElementById('figjson').textContent);
    const ARR = JSON.parse(document.getElementById('arrays').textContent);
    const figEl = document.getElementById('fig');

    Plotly.newPlot(figEl, FIG.data, FIG.layout, {displaylogo:false, responsive:true});

    const META = FIG.layout.meta || {};
    const USD_BTC_IDX = META.USD_BTC_IDX, USD_MARK_IDX = META.USD_MARK_IDX;
    const GLD_BTC_IDX = META.GLD_BTC_IDX, GLD_MARK_IDX = META.GLD_MARK_IDX;

    const denomSel = document.getElementById('denom');
    const legendBtn = document.getElementById('legendBtn');
    const slider = document.getElementById('dateSlider');
    const chkToday = document.getElementById('chkToday');
    const zoneTxt = document.getElementById('zoneTxt');
    const zoneDot = document.getElementById('zoneDot');
    const dateRead = document.getElementById('dateRead');

    function setLegend(on){ Plotly.relayout(figEl, {"showlegend": !!on}); }
    let legendOn = true;
    legendBtn.addEventListener('click', ()=>{ legendOn = !legendOn; setLegend(legendOn); });

    function fmtUSD(v){ return (v==null||!isFinite(v))?"—":"$"+Math.round(v).toLocaleString(); }
    function fmtOzBTC(v){
      if (v==null||!isFinite(v)) return "—";
      const n = Math.round(v*100)/100;
      return n.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})+" oz/BTC";
    }
    function fmtTickGold(v){
      if (!isFinite(v)||v<=0) return "";
      if (v>=1) return Math.round(v).toLocaleString();
      return Number(v).toFixed(3).replace(/\.?0+$/,"");
    }
    function zoneFor(val, bands){
      if (val==null) return "—";
      if (val < bands.Support) return "SELL THE HOUSE!!";
      if (val < bands.Bear)    return "Buy";
      if (val < bands.Frothy)  return "DCA";
      if (val < bands.Top)     return "Relax";
      return "Frothy";
    }
    function zoneDotColor(z){
      switch(z){
        case "SELL THE HOUSE!!": return "#ffffff";
        case "Buy":              return "#f97316";
        case "DCA":              return "#ffffff";
        case "Relax":            return "#84cc16";
        case "Frothy":           return "#ef4444";
        default:                 return "#e5e7eb";
      }
    }
    function readBandsAt(dateISO, denom){
      const idx = ARR.dates.indexOf(dateISO);
      if (idx<0) return null;
      const g = denom==="USD" ? ARR.usd : ARR.gld;
      return { Support:g.Support[idx], Bear:g.Bear[idx], Mid:g.Mid[idx], Frothy:g.Frothy[idx], Top:g.Top[idx] };
    }
    function setGoldTicks(){
      const G = ARR.gld;
      const all = [].concat(G.Support,G.Bear,G.Mid,G.Frothy,G.Top,ARR.hist.gld).filter(v=>v>0&&isFinite(v));
      if (!all.length) return;
      const minV = Math.min.apply(null,all), maxV = Math.max.apply(null,all);
      const c = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000];
      const ticks = c.filter(v=>v>=minV*0.9 && v<=maxV*1.1);
      const txt = ticks.map(fmtTickGold);
      Plotly.relayout(figEl, {"yaxis.tickvals":ticks, "yaxis.ticktext":txt, "yaxis.tickformat":""});
    }
    function clearGoldTicks(){
      Plotly.relayout(figEl, {"yaxis.tickvals":null, "yaxis.ticktext":null, "yaxis.tickformat":"$,d"});
    }

    function setDenom(which){
      const vis = (FIG.data||[]).map(tr => (tr.legendgroup||"USD") === (which==="USD"?"USD":"GLD"));
      Plotly.restyle(figEl, {"visible": vis});
      if (which==="USD"){ Plotly.relayout(figEl, {"yaxis.title.text":"USD / BTC"}); clearGoldTicks(); }
      else { Plotly.relayout(figEl, {"yaxis.title.text":"Gold oz / BTC"}); setGoldTicks(); }
      updateMarkerAndReadout();
    }

    // ----- Weekly (Monday) slider independent of data gaps
    const histDatesISO = ARR.hist.dates;
    const histMs = histDatesISO.map(d=>Date.parse(d+"T00:00:00Z"));
    const MS_DAY = 86400000;
    // find first Monday >= first date
    let t0 = histMs[0];
    let d0 = new Date(t0);
    const firstMon = t0 + ((8 - d0.getUTCDay()) % 7) * MS_DAY;
    const weeklyIdx = [], weeklyDates = [];
    for (let t = firstMon; t <= histMs[histMs.length-1]; t += 7*MS_DAY){
      // nearest index by lower-bound
      let lo = 0, hi = histMs.length-1, pos = hi;
      while (lo <= hi){
        const mid = (lo+hi)>>1;
        if (histMs[mid] >= t){ pos = mid; hi = mid-1; } else { lo = mid+1; }
      }
      // prefer the closest between pos and pos-1
      let cand = pos;
      if (pos>0 && Math.abs(histMs[pos-1]-t) <= Math.abs(histMs[pos]-t)) cand = pos-1;
      weeklyIdx.push(cand);
      weeklyDates.push(histDatesISO[cand]);
    }
    if (weeklyIdx.length===0){ weeklyIdx.push(histDatesISO.length-1); weeklyDates.push(histDatesISO.at(-1)); }
    slider.max = Math.max(0, weeklyIdx.length-1);
    slider.value = slider.max;

    function sliderToHistIndex(){
      const i = Math.max(0, Math.min(parseInt(slider.value,10), weeklyIdx.length-1));
      return weeklyIdx[i] ?? (histDatesISO.length-1);
    }

    function updateHoverHeader(ev){
      try{
        if (!ev || !ev.points || !ev.points.length) return;
        const idx = ev.points[0].pointIndex;
        const dISO = ARR.dates[idx] || histDatesISO[Math.min(idx, histDatesISO.length-1)];
        const dt = new Date(dISO+"T00:00:00Z");
        const label = dt.toLocaleString('en-US', {month:'short', year:'numeric', timeZone:'UTC'});
        // replace first tspan inside the unified hover
        const layer = figEl.querySelector('.hoverlayer');
        if (!layer) return;
        const firstText = layer.querySelector('g.hovertext text tspan');
        if (firstText) firstText.textContent = label;
      }catch(_){}
    }

    function updateMarkerAndReadout(){
      const j = sliderToHistIndex();
      const denom = denomSel.value;
      const dISO = histDatesISO[j];
      const bands = readBandsAt(dISO, denom);
      if (!bands){ zoneTxt.textContent="—"; dateRead.textContent=""; return; }

      const priceUSD = ARR.hist.usd[j];
      const priceGLD = ARR.hist.gld[j];
      const price = (denom==="USD")?priceUSD:priceGLD;
      const zone = (function(){
        if (price < bands.Support) return "SELL THE HOUSE!!";
        if (price < bands.Bear)    return "Buy";
        if (price < bands.Frothy)  return "DCA";
        if (price < bands.Top)     return "Relax";
        return "Frothy";
      })();
      zoneTxt.textContent = zone;
      zoneDot.style.background = (function(z){
        switch(z){
          case "SELL THE HOUSE!!": return "#ffffff";
          case "Buy":              return "#f97316";
          case "DCA":              return "#ffffff";
          case "Relax":            return "#84cc16";
          case "Frothy":           return "#ef4444";
          default:                 return "#e5e7eb";
        }
      })(zone);

      // Readout: only date + BTC price
      const line = (denom==="USD") ? fmtUSD(priceUSD) : fmtOzBTC(priceGLD);
      dateRead.innerHTML = `<div style="font-weight:600">${dISO}</div><div>BTC: ${line}</div>`;

      // Move marker
      const genesis = Date.parse("2009-01-03T00:00:00Z");
      const dms = Date.parse(dISO+"T00:00:00Z");
      const days = Math.max(1, Math.round((dms - genesis)/86400000));
      const xval = Math.log10(days);
      if (denom==="USD"){ Plotly.restyle(figEl, {"x":[[xval]], "y":[[priceUSD]]}, [USD_MARK_IDX]); }
      else { Plotly.restyle(figEl, {"x":[[xval]], "y":[[priceGLD]]}, [GLD_MARK_IDX]); }
    }

    denomSel.addEventListener('change', (e)=> setDenom(e.target.value));
    slider.addEventListener('input', ()=>{ chkToday.checked=false; updateMarkerAndReadout(); });
    chkToday.addEventListener('change', (e)=>{ if(e.target.checked){ slider.value = slider.max; updateMarkerAndReadout(); } });

    // Tap/click outside → hide hover box
    function outsideFig(el, target){ return !el.contains(target); }
    function hideHover(){ try{ Plotly.Fx.unhover(figEl); }catch(e){} }
    document.addEventListener('click', (ev)=>{ if(outsideFig(figEl, ev.target)) hideHover(); }, {passive:true});
    document.addEventListener('touchstart', (ev)=>{ if(outsideFig(figEl, ev.target)) hideHover(); }, {passive:true});

    // Update hover header text to Month Year
    figEl.on('plotly_hover', updateHoverHeader);

    // Init
    setDenom("USD");
    setLegend(true);
    updateMarkerAndReadout();
    window.addEventListener('resize', ()=> Plotly.Plots.resize(figEl));
  </script>
</body>
</html>
"""
    html = TEMPLATE.replace("__TITLE__", page_title).replace("__FIGJSON__", fig_json).replace("__ARRAYS__", arrays_json)
    Path(out_path).write_text(html, encoding="utf-8")
    print("[build] wrote", out_path)

# --------------------- main
def main():
    print("[build] loading BTC & Gold …")
    df = load_btc_gold()
    print(f"[build] rows: {len(df):,}  (BTC & Gold)")
    fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist = make_powerlaw_fig(df)
    write_index_html(fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist)

if __name__ == "__main__":
    main()