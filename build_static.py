#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Static builder for BTC Purchase Indicator (Power-Law bands)
- Robust column detection (no hard-coded 'Price')
- BTC/USD and Gold oz/BTC denominations
- Outputs a single dist/index.html file
"""

import os, json
from datetime import datetime
import numpy as np
import pandas as pd

# ----------------- constants -----------------
GENESIS   = pd.Timestamp("2009-01-03")
PROJ_END  = pd.Timestamp("2040-12-31")

# band levels (latest requirements)
LEVELS = {
    "Support": -1.5,
    "Bear":    -0.75,   # updated
    "Mid":      0.0,
    "Frothy":   1.0,
    "Top":      2.0,    # updated
}

# colors
COLORS = {
    "Support": "#ef4444",         # red
    "Bear":    "rgba(255,100,100,1)",
    "Mid":     "#ffffff",
    "Frothy":  "rgba(100,255,100,1)",
    "Top":     "#22c55e",         # green
    "BTC":     "#fbbf24",         # amber/yellow
}

# candidate price column names to look for automatically
PRICE_CANDIDATES = ["Price", "Close", "Adj Close", "AdjClose", "BTC", "USD", "Value"]

# ----------------- robust helpers -----------------
def detect_price_col(df, hint=None, label="series"):
    """
    Return a usable price column name from df.
    Priority:
      1) explicit `hint` if present
      2) known PRICE_CANDIDATES in order
      3) first numeric, non-Date column
    """
    if "Date" not in df.columns:
        raise ValueError(f"{label}: missing 'Date' column. Have: {list(df.columns)}")

    if hint and hint in df.columns:
        return hint

    for c in PRICE_CANDIDATES:
        if c in df.columns:
            return c

    for c in df.columns:
        if c != "Date" and pd.api.types.is_numeric_dtype(df[c]):
            return c

    raise ValueError(f"{label}: could not find a numeric price column in columns {list(df.columns)}")

def _days_since_genesis(dates_like) -> np.ndarray:
    """
    Robust day counts for Series, DatetimeIndex, list[str], etc.
    Returns int numpy array of days >= 1.
    """
    td = pd.to_datetime(dates_like) - GENESIS
    # Pandas doesn't support astype('timedelta64[D]'); divide by 1 day instead:
    days = (td.to_numpy() / np.timedelta64(1, "D")).astype(float)
    days = np.maximum(days, 1.0).astype(int)
    return days

def fit_power(df_price, price_col=None, label="series"):
    """
    Log-log linear fit on (days since GENESIS, price).
    Works with any price column name.
    Returns (slope, intercept, sigma>=0.25).
    """
    if price_col is None:
        price_col = detect_price_col(df_price, label=label)

    need = {"Date", price_col}
    if not need.issubset(df_price.columns):
        raise ValueError(
            f"{label}: expected columns {sorted(need)}; got {list(df_price.columns)}"
        )

    days = _days_since_genesis(df_price["Date"])
    X = np.log10(days)

    y_raw = pd.to_numeric(df_price[price_col], errors="coerce").to_numpy()
    y = np.log10(y_raw)

    mask = np.isfinite(X) & np.isfinite(y)
    if not mask.any():
        raise ValueError(f"{label}: no finite data to fit after cleaning")

    X = X[mask]
    y = y[mask]

    slope, intercept = np.polyfit(X, y, 1)
    resid  = y - (slope * X + intercept)
    sigma  = float(np.std(resid))
    return slope, intercept, max(sigma, 0.25)

def require_cols(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns {missing}. Have: {list(df.columns)}")

# ----------------- data loaders -----------------
def _btc_stooq():
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    # find date & close
    date_col  = next(c for c in df.columns if "date" in c)
    price_col = next(c for c in df.columns if c in ("close", "c"))
    out = pd.DataFrame({
        "Date":  pd.to_datetime(df[date_col], errors="coerce"),
        "BTC":   pd.to_numeric(df[price_col].astype(str).str.replace(",",""), errors="coerce"),
    })
    return out.dropna().sort_values("Date")

def _btc_github():
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    df  = pd.read_csv(raw)
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"], errors="coerce"),
        "BTC":  pd.to_numeric(df["Closing Price (USD)"], errors="coerce"),
    })
    return out.dropna().sort_values("Date")

def load_btc():
    try:
        df = _btc_stooq()
        if len(df) > 1000:
            return df
    except Exception:
        pass
    return _btc_github()

def _gold_stooq():
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"  # USD/oz
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    date_col  = next(c for c in df.columns if "date" in c)
    price_col = next(c for c in df.columns if c in ("close", "c"))
    out = pd.DataFrame({
        "Date":   pd.to_datetime(df[date_col], errors="coerce"),
        "GoldUSD": pd.to_numeric(df[price_col].astype(str).str_replace(",",""), errors="coerce"),
    })
    return out.dropna().sort_values("Date")

def _gold_lbma():
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    df = pd.read_csv(url)
    out = pd.DataFrame({
        "Date":   pd.to_datetime(df["Date"], errors="coerce"),
        "GoldUSD": pd.to_numeric(df["USD (PM)"], errors="coerce"),
    })
    return out.dropna().sort_values("Date")

def load_gold():
    try:
        g = _gold_stooq()
        if len(g) >= 1000:
            return g
        raise ValueError("Too few gold rows from Stooq")
    except Exception:
        return _gold_lbma()

# ----------------- compute helpers -----------------
def extend_monthly_dates(last_hist_date, end_date=PROJ_END):
    start = (pd.to_datetime(last_hist_date) + pd.offsets.MonthBegin(1))
    fut = pd.date_range(start, end_date, freq="MS", inclusive="both")
    return fut

def build_bands_over(dates, slope, intercept, sigma):
    """Return dict of band arrays keyed by level name, same length as dates."""
    days = _days_since_genesis(dates)  # FIXED: robust day computation (no .dt / no 'D' casting)
    mid_log = slope * np.log10(days) + intercept
    out = {}
    for name, k in LEVELS.items():
        out[name] = (10 ** (mid_log + sigma * k)).tolist()
    return out

# ----------------- HTML template -----------------
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Purchase Indicator</title>
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
  #fig{height:72vh;min-height:460px;background:var(--card);border-radius:14px;padding:8px}
  .slider{width:100%}
  .stack{display:flex;flex-direction:column;gap:2px;font-size:0.95rem}
  .hoverbox{
      position:absolute; right:26px; bottom:26px; background:rgba(15,17,22,.88);
      border:1px solid #263041; border-radius:10px; padding:10px 12px; min-width:140px; max-width:180px;
      backdrop-filter: blur(6px);
  }
  .hoverrow{display:flex;justify-content:space-between;gap:6px;line-height:1.15}
</style>
</head>
<body>
  <div class="wrap">
    <h1>BTC Purchase Indicator</h1>

    <div class="row" style="margin-bottom:8px;">
      <div class="chip"><span id="zoneDot" class="dot"></span> <b>Price Zone:</b> <span id="zoneTxt" style="margin-left:6px">—</span></div>
      <div class="row" style="gap:8px;">
        <label for="denom">Denomination</label>
        <select id="denom" style="min-width:5.6rem;">
          <option value="USD" selected>USD</option>
          <option value="Gold">Gold</option>
        </select>
      </div>
    </div>

    <div class="row" style="margin:6px 0 2px;">
      <label for="dateSlider">View at date (Wednesdays):</label>
      <label class="row" style="gap:6px;"><input type="checkbox" id="chkToday"/> Today</label>
    </div>
    <input id="dateSlider" class="slider" type="range" min="0" value="0" step="1"/>
    <div id="dateRead" class="stack" style="opacity:.95; margin:6px 0 12px 2px;"></div>

    <div style="position:relative">
      <div id="fig"></div>
      <div id="hoverbox" class="hoverbox" style="display:none"></div>
    </div>
  </div>

  <script type="application/json" id="figjson">__FIGJSON__</script>
  <script type="application/json" id="arrays">__ARRAYS__</script>

  <script>
    const FIG = JSON.parse(document.getElementById('figjson').textContent);
    const ARR = JSON.parse(document.getElementById('arrays').textContent);
    const figEl = document.getElementById('fig');
    const hoverEl = document.getElementById('hoverbox');

    Plotly.newPlot(figEl, FIG.data, FIG.layout, {displaylogo:false, responsive:true});

    // indices exposed by layout.meta
    const META = FIG.layout.meta || {};
    const USD_MARK_IDX = META.USD_MARK_IDX;
    const GLD_MARK_IDX = META.GLD_MARK_IDX;

    // controls
    const denomSel = document.getElementById('denom');
    const slider = document.getElementById('dateSlider');
    const chkToday = document.getElementById('chkToday');
    const zoneTxt = document.getElementById('zoneTxt');
    const zoneDot = document.getElementById('zoneDot');
    const dateRead = document.getElementById('dateRead');

    // formatters
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
    function monthYear(iso){
      const d = new Date(iso+"T00:00:00Z");
      const fmt = new Intl.DateTimeFormat(undefined,{year:"numeric", month:"short", timeZone:"UTC"});
      return fmt.format(d);
    }

    // zone helpers
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

    // gold ticks for oz/BTC
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

    // denomination toggle (show only traces in that legendgroup)
    function setDenom(which){
      const vis = (FIG.data||[]).map(tr => (tr.legendgroup||"USD") === (which==="USD"?"USD":"GLD"));
      Plotly.restyle(figEl, {"visible": vis});
      if (which==="USD"){ Plotly.relayout(figEl, {"yaxis.title.text":"USD / BTC"}); clearGoldTicks(); }
      else { Plotly.relayout(figEl, {"yaxis.title.text":"Gold oz / BTC"}); setGoldTicks(); }
      updateMarkerAndReadout(); // redraw marker in correct units
    }

    // ----- Weekly (Wednesday) slider independent of data gaps
    const histDatesISO = ARR.hist.dates;
    const histMs = histDatesISO.map(d=>Date.parse(d+"T00:00:00Z"));
    const MS_DAY = 86400000;
    // first Wednesday on/after first data date
    const firstDate = new Date(histMs[0]);
    const firstWed = histMs[0] + ((10 - firstDate.getUTCDay()) % 7) * MS_DAY; // Wed=3 → (10-DoW)%7
    const weeklyIdx = [], weeklyDates = [];
    for (let t = firstWed; t <= histMs[histMs.length-1]; t += 7*MS_DAY){
      // nearest index by lower-bound
      let lo = 0, hi = histMs.length-1, pos = hi;
      while (lo <= hi){
        const mid = (lo+hi)>>1;
        if (histMs[mid] >= t){ pos = mid; hi = mid-1; } else { lo = mid+1; }
      }
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

    function colorize(txt, color){ return `<span style="color:${color}">${txt}</span>`; }

    function updateHoverBox(dISO, denom, bands, priceUSD, priceGLD){
      const head = monthYear(dISO);
      const rows = [];
      if (denom==="USD"){
        rows.push(`<div class="hoverrow"><span>${colorize("Top","#22c55e")}</span><span>${fmtUSD(bands.Top)}</span></div>`);
        rows.push(`<div class="hoverrow"><span>${colorize("Frothy","rgba(100,255,100,1)")}</span><span>${fmtUSD(bands.Frothy)}</span></div>`);
        rows.push(`<div class="hoverrow"><span>${colorize("PL Best Fit","#fff")}</span><span>${fmtUSD(bands.Mid)}</span></div>`);
        rows.push(`<div class="hoverrow"><span>${colorize("Bear","rgba(255,100,100,1)")}</span><span>${fmtUSD(bands.Bear)}</span></div>`);
        rows.push(`<div class="hoverrow"><span>${colorize("Support","#ef4444")}</span><span>${fmtUSD(bands.Support)}</span></div>`);
        if (!ARR.hist.hidePrice) rows.push(`<div class="hoverrow"><span>${colorize("BTC","#fbbf24")}</span><span>${fmtUSD(priceUSD)}</span></div>`);
      } else {
        rows.push(`<div class="hoverrow"><span>${colorize("Top","#22c55e")}</span><span>${fmtOzBTC(bands.Top)}</span></div>`);
        rows.push(`<div class="hoverrow"><span>${colorize("Frothy","rgba(100,255,100,1)")}</span><span>${fmtOzBTC(bands.Frothy)}</span></div>`);
        rows.push(`<div class="hoverrow"><span>${colorize("PL Best Fit","#fff")}</span><span>${fmtOzBTC(bands.Mid)}</span></div>`);
        rows.push(`<div class="hoverrow"><span>${colorize("Bear","rgba(255,100,100,1)")}</span><span>${fmtOzBTC(bands.Bear)}</span></div>`);
        rows.push(`<div class="hoverrow"><span>${colorize("Support","#ef4444")}</span><span>${fmtOzBTC(bands.Support)}</span></div>`);
        if (!ARR.hist.hidePrice) rows.push(`<div class="hoverrow"><span>${colorize("BTC","#fbbf24")}</span><span>${fmtOzBTC(priceGLD)}</span></div>`);
      }
      hoverEl.innerHTML = `<div style="font-weight:700;margin-bottom:6px">${head}</div>${rows.join("")}`;
      hoverEl.style.display = "block";
    }

    function readBandsAt(dateISO, denom){
      const idx = ARR.dates.indexOf(dateISO);
      if (idx<0) return null;
      const g = denom==="USD" ? ARR.usd : ARR.gld;
      return { Support:g.Support[idx], Bear:g.Bear[idx], Mid:g.Mid[idx], Frothy:g.Frothy[idx], Top:g.Top[idx] };
    }

    function updateMarkerAndReadout(){
      const j = sliderToHistIndex();
      const denom = denomSel.value;
      const dISO = weeklyDates[j];
      const bands = readBandsAt(dISO, denom);
      if (!bands){ zoneTxt.textContent="—"; dateRead.textContent=""; return; }

      // determine if hover price should be hidden (> ~6 months after last history)
      const lastISO = ARR.hist.dates.at(-1);
      const msLast = Date.parse(lastISO+"T00:00:00Z");
      const msCur  = Date.parse(dISO+"T00:00:00Z");
      ARR.hist.hidePrice = (msCur - msLast) > (183*86400000);

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
      zoneDot.style.background = zoneDotColor(zone);

      const line = (denom==="USD") ? fmtUSD(priceUSD) : fmtOzBTC(priceGLD);
      dateRead.innerHTML = `<div style="font-weight:600">${dISO}</div><div>BTC: ${line}</div>`;

      // Move marker
      const genesis = Date.parse("2009-01-03T00:00:00Z");
      const dms = Date.parse(dISO+"T00:00:00Z");
      const days = Math.max(1, Math.round((dms - genesis)/86400000));
      const xval = Math.log10(days);

      if (denom==="USD"){
        Plotly.restyle(figEl, {"x":[[xval]], "y":[[priceUSD]]}, [USD_MARK_IDX]);
      } else {
        Plotly.restyle(figEl, {"x":[[xval]], "y":[[priceGLD]]}, [GLD_MARK_IDX]);
      }

      updateHoverBox(dISO, denom, bands, priceUSD, priceGLD);
    }

    function setDenomAndRefresh(which){
      denomSel.value = which;
      setDenom(which);
    }

    denomSel.addEventListener('change', (e)=> setDenomAndRefresh(e.target.value));
    slider.addEventListener('input', ()=>{ chkToday.checked=false; updateMarkerAndReadout(); });
    chkToday.addEventListener('change', (e)=>{ if(e.target.checked){ slider.value = slider.max; updateMarkerAndReadout(); } });

    // Init
    setDenom("USD");
    updateMarkerAndReadout();
    window.addEventListener('resize', ()=> Plotly.Plots.resize(figEl));
  </script>
</body>
</html>
"""

# ----------------- build figure data -----------------
def make_power_data_arrays(dates, hist_usd, bands_usd, hist_gld, bands_gld):
    """
    Build the Plotly traces (data) and layout for both denominations.
    We'll use legendgroup = 'USD' or 'GLD' and toggle via JS.
    """
    import plotly.graph_objects as go

    # x = log10(days since genesis)
    days = _days_since_genesis(dates)
    x = np.log10(days).tolist()

    data = []

    # helper to add band traces
    def add_band(group, name, y, color):
        data.append(go.Scatter(
            x=x, y=y, name=name, legendgroup=group, visible=True,
            line=dict(color=color, dash="dash", width=1.5), hoverinfo="skip"
        ))

    # USD bands
    add_band("USD", "Top (USD)",     bands_usd["Top"],    COLORS["Top"])
    add_band("USD", "Frothy (USD)",  bands_usd["Frothy"], COLORS["Frothy"])
    add_band("USD", "PL Best Fit (USD)", bands_usd["Mid"], COLORS["Mid"])
    add_band("USD", "Bear (USD)",    bands_usd["Bear"],   COLORS["Bear"])
    add_band("USD", "Support (USD)", bands_usd["Support"],COLORS["Support"])

    # USD price
    data.append(go.Scatter(
        x=x, y=hist_usd, name="BTC (USD)", legendgroup="USD",
        line=dict(color=COLORS["BTC"], width=2),
        hoverinfo="skip"
    ))

    # marker for USD (single-point scatter; index stored in layout.meta)
    data.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=COLORS["BTC"], size=8, line=dict(color="#000", width=1)),
        name="",
        legendgroup="USD", hoverinfo="skip"
    ))
    usd_mark_idx = len(data) - 1

    # GLD bands
    add_band("GLD", "Top (Gold)",     bands_gld["Top"],    COLORS["Top"])
    add_band("GLD", "Frothy (Gold)",  bands_gld["Frothy"], COLORS["Frothy"])
    add_band("GLD", "PL Best Fit (Gold)", bands_gld["Mid"], COLORS["Mid"])
    add_band("GLD", "Bear (Gold)",    bands_gld["Bear"],   COLORS["Bear"])
    add_band("GLD", "Support (Gold)", bands_gld["Support"],COLORS["Support"])

    # GLD price
    data.append(go.Scatter(
        x=x, y=hist_gld, name="BTC (Gold)", legendgroup="GLD",
        line=dict(color=COLORS["BTC"], width=2),
        hoverinfo="skip", visible=True
    ))

    # marker for GLD
    data.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=COLORS["BTC"], size=8, line=dict(color="#000", width=1)),
        name="",
        legendgroup="GLD", hoverinfo="skip"
    ))
    gld_mark_idx = len(data) - 1

    layout = dict(
        template="plotly_dark",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=12),
        xaxis=dict(title="Year (log-time)", showgrid=True, gridwidth=0.5),
        yaxis=dict(type="log", title="USD / BTC", tickformat="$,d", showgrid=True, gridwidth=0.5),
        plot_bgcolor="#111", paper_bgcolor="#111",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02),
        meta=dict(USD_MARK_IDX=usd_mark_idx, GLD_MARK_IDX=gld_mark_idx)
    )

    fig = dict(data=[tr.to_plotly_json() for tr in data], layout=layout)
    return fig

# ----------------- main -----------------
def main():
    # Load history
    btc = load_btc()
    gold = load_gold()

    require_cols(btc, ["Date", "BTC"], "BTC")
    require_cols(gold, ["Date", "GoldUSD"], "Gold")

    # Merge on Date
    df = pd.merge(btc, gold, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    df = df[df["BTC"] > 0]
    df = df[df["GoldUSD"] > 0]

    # Derived Gold oz/BTC
    df["GoldBTC"] = df["GoldUSD"] / df["BTC"]

    # Extend monthly to 2040
    fut = extend_monthly_dates(df["Date"].iloc[-1], PROJ_END)
    full_dates = pd.Index(df["Date"].tolist() + fut.tolist())

    # Fit USD
    usd_price_col = detect_price_col(df, hint="BTC", label="USD/BTC")
    ms, bs, ss = fit_power(df[["Date", usd_price_col]], price_col=usd_price_col, label="USD/BTC")
    bands_usd = build_bands_over(full_dates, ms, bs, ss)

    # Fit Gold (oz/BTC)
    gld_price_col = detect_price_col(df, hint="GoldBTC", label="Gold/BTC")
    gs, gi, gsig = fit_power(df[["Date", gld_price_col]], price_col=gld_price_col, label="Gold/BTC")
    bands_gld = build_bands_over(full_dates, gs, gi, gsig)

    # Prepare arrays for JS
    dates_iso = pd.to_datetime(full_dates).strftime("%Y-%m-%d").tolist()
    hist_dates_iso = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d").tolist()

    # Align historical y-series to full x (None after history)
    full_map = {d:i for i,d in enumerate(dates_iso)}
    hist_idx = [full_map[d] for d in hist_dates_iso]
    hist_usd_aligned = [None]*len(dates_iso)
    hist_gld_aligned = [None]*len(dates_iso)
    for k, i in enumerate(hist_idx):
        hist_usd_aligned[i] = float(df["BTC"].iloc[k])
        hist_gld_aligned[i] = float(df["GoldBTC"].iloc[k])

    # Build Plotly figure payload
    fig = make_power_data_arrays(
        dates_iso,
        hist_usd_aligned, bands_usd,
        hist_gld_aligned, bands_gld
    )

    # arrays for UI
    arr = dict(
        dates = dates_iso,
        usd = bands_usd,
        gld = bands_gld,
        hist = dict(
            dates = hist_dates_iso,
            usd   = [float(x) for x in df["BTC"].tolist()],
            gld   = [float(x) for x in df["GoldBTC"].tolist()],
            hidePrice = False
        )
    )

    # write HTML
    os.makedirs("dist", exist_ok=True)
    html = HTML_TEMPLATE.replace("__FIGJSON__", json.dumps(fig, separators=(",",":")))
    html = html.replace("__ARRAYS__", json.dumps(arr, separators=(",",":")))
    out = os.path.join("dist", "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[build] rows: {len(df):,}  (BTC & Gold)")
    print(f"[build] wrote {out}")

if __name__ == "__main__":
    main()
