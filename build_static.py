#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_static.py — BTC Purchase Indicator (static Plotly page for GitHub Pages)

Outputs:
  dist/index.html

What you get on the page:
  - Power-law chart in log-time (X axis) for 2012 → 2040
  - Denomination dropdown (USD / Gold)
      * USD shows: BTC (USD/BTC) and PL bands computed on USD
      * Gold shows: Gold oz / BTC and PL bands computed on that ratio
  - "Legend" button to hide/show Plotly legend
  - Date slider that moves a "Price zone" pill (SELL THE HOUSE!! / Buy / DCA / Relax / Frothy)

Notes:
  - No anchor on regression. Fit uses all available history each build.
  - Robust to pandas 1.x / 2.x.
"""

import os, io, json, math, pathlib, requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio  

# ---------- constants ----------
UA = {"User-Agent": "btc-pl-pages/1.0"}
GENESIS = pd.Timestamp("2009-01-03")
START_YEAR = 2012
PROJ_END = pd.Timestamp("2040-12-31")
GRID_D_MONTHS = 24  # keep vertical grid every 2 years on Plotly (cosmetic only)

# sigma levels and colors (USD view colors; Gold uses same semantics)
LEVELS = {
    "Support": -1.5,
    "Bear":    -0.5,
    "Mid":      0.0,  # PL Best Fit
    "Frothy":  +1.0,
    "Top":     +1.75,
}
COLORS = {
    "Support": "rgb(255, 90, 90)",
    "Bear":    "rgb(255,140,140)",
    "Mid":     "white",
    "Frothy":  "rgb(100,255,100)",
    "Top":     "rgb(45, 180, 45)",
}
BTC_COLOR = "gold"

# ---------- helpers ----------

def _read_csv(url: str) -> pd.DataFrame:
    """HTTP GET then read CSV via pandas (works where servers reject pandas direct)."""
    r = requests.get(url, timeout=30, headers=UA)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def load_btc() -> pd.DataFrame:
    """
    Daily BTC/USD closes.
    Primary: Stooq. Fallback: datasets/bitcoin-price on GitHub.
    Returns columns: Date (datetime64[ns]), BTC (float)
    """
    # 1) Stooq
    try:
        df = _read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        # find date & close columns
        date_col = [c for c in cols if "date" in c][0]
        close_col = [c for c in cols if "close" in c or "price" in c][-1]
        out = df.rename(columns={date_col: "Date", close_col: "BTC"})
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["BTC"] = pd.to_numeric(out["BTC"].astype(str).str.replace(",", ""), errors="coerce")
        out = out.dropna().query("BTC>0").sort_values("Date")
        # historical series should be large (thousands)
        if len(out) > 1000:
            return out[["Date", "BTC"]]
    except Exception:
        pass

    # 2) GitHub mirror
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    df = _read_csv(raw).rename(columns={"Date":"Date", "Closing Price (USD)":"BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"] = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().query("BTC>0")[["Date","BTC"]].sort_values("Date")

def load_gold() -> pd.DataFrame:
    """
    Daily Gold USD/oz.
    Primary: Stooq XAUUSD. Fallback: a GitHub LBMA dataset (USD PM).
    Returns columns: Date (datetime64[ns]), Gold (float)
    """
    # 1) Stooq
    try:
        df = _read_csv("https://stooq.com/q/d/l/?s=xauusd&i=d")
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        date_col = [c for c in cols if "date" in c][0]
        close_col = [c for c in cols if "close" in c or "price" in c][-1]
        out = df.rename(columns={date_col:"Date", close_col:"Gold"})
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Gold"] = pd.to_numeric(out["Gold"].astype(str).str.replace(",", ""), errors="coerce")
        out = out.dropna().query("Gold>0").sort_values("Date")
        if len(out) > 1000:
            return out[["Date","Gold"]]
    except Exception:
        pass

    # 2) GitHub fallback (koindata mirror)
    try:
        df = _read_csv("https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # prefer USD (PM) if present, else USD (AM)/USD
        for col in ["USD (PM)", "USD (AM)", "USD"]:
            if col in df.columns:
                out = df.rename(columns={col:"Gold"})[["Date","Gold"]]
                out["Gold"] = pd.to_numeric(out["Gold"], errors="coerce")
                return out.dropna().query("Gold>0").sort_values("Date")
    except Exception:
        pass

    raise RuntimeError("Failed to load Gold prices.")

def _days_since_genesis(dates: pd.Series) -> np.ndarray:
    """Return days as float64 from GENESIS for each datetime in `dates`."""
    # Supports pandas 1.x/2.x
    delta = (pd.to_datetime(dates) - GENESIS)
    # TimedeltaIndex -> days as float
    return (delta / np.timedelta64(1, "D")).to_numpy(dtype="float64")

def _log_time(dates: pd.Series) -> np.ndarray:
    d = _days_since_genesis(dates)
    # clip to positive to avoid log10(<=0)
    d = np.clip(d, 1.0, None)
    return np.log10(d)

def fit_power(dates: pd.Series, values: pd.Series):
    """
    Fit y = m * log10(days) + b on log10(values).
    Returns m (slope), b (intercept), sigma (std of residuals).
    """
    X = _log_time(dates)
    y = np.log10(values.to_numpy(dtype="float64"))
    m, b = np.polyfit(X, y, 1)
    yhat = m * X + b
    sigma = float(np.std(y - yhat))
    return m, b, sigma

def bands_over(dates: pd.Series, m: float, b: float, sigma: float) -> pd.DataFrame:
    """
    Compute PL best fit and sigma bands for given dates.
    Returns DataFrame with Date + columns: Support, Bear, Mid, Frothy, Top
    """
    X = _log_time(dates)
    mid_log = m * X + b
    out = pd.DataFrame({"Date": pd.to_datetime(dates)})
    out["Mid"]     = 10 ** mid_log
    out["Support"] = 10 ** (mid_log + LEVELS["Support"] * sigma)
    out["Bear"]    = 10 ** (mid_log + LEVELS["Bear"]    * sigma)
    out["Frothy"]  = 10 ** (mid_log + LEVELS["Frothy"]  * sigma)
    out["Top"]     = 10 ** (mid_log + LEVELS["Top"]     * sigma)
    return out

def year_ticks(start=2012, end=2040):
    """Return (tickvals, ticktext) for log-time axis labeled by years."""
    yrs = list(range(start, end + 1))
    dts = [pd.Timestamp(f"{y}-01-01") for y in yrs]
    vals = _log_time(pd.to_datetime(dts))
    # 2012–2020 every year, after that every other year
    txt = []
    for y in yrs:
        if y <= 2020 or (y > 2020 and y % 2 == 0):
            txt.append(str(y))
        else:
            txt.append("")  # blank label (keeps gridline aligned)
    return vals.tolist(), txt

# ---------- figure building ----------

def make_powerlaw_figure(usd: pd.DataFrame,
                         bands_usd: pd.DataFrame,
                         gld_ratio: pd.DataFrame = None,
                         bands_gld: pd.DataFrame = None) -> go.Figure:
    """
    Build Plotly figure with both denominations:
      - USD traces visible initially
      - Gold traces hidden (we toggle from JS)
    X axis is log-time. Custom year ticks 2012→2040 (every year through 2020, then every 2 years).
    """
    tv, tt = year_ticks(START_YEAR, PROJ_END.year)

    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f0f12",
        plot_bgcolor="#15161a",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", size=14, color="#e8e8ea"),
        margin=dict(l=40, r=20, t=10, b=40),
        showlegend=False,  # we show/hide via button in page
        xaxis=dict(
            title="Year (log-time)",
            tickvals=tv,
            ticktext=tt,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(180,190,210,.18)",
            zeroline=False
        ),
        yaxis=dict(
            title="USD / BTC",
            type="log",
            tickformat="$,d",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(180,190,210,.18)",
            zeroline=False
        ),
    )

    # Helper to convert series of dates to log-time x
    def _x(dates): return _log_time(pd.to_datetime(dates))

    # ---- USD bands & price (visible) ----
    for key in ["Top", "Frothy", "Mid", "Bear", "Support"]:
        fig.add_trace(go.Scatter(
            x=_x(bands_usd["Date"]), y=bands_usd[key],
            name=f"{key} (USD)",
            line=dict(color=COLORS[key], dash="dash"),
            hovertemplate=f"{key} | (%{{customdata}}, $%{{y:,.0f}})<extra></extra>",
            customdata=bands_usd["Date"].dt.strftime("%b %Y"),
            visible=True
        ))
    fig.add_trace(go.Scatter(
        x=_x(usd["Date"]), y=usd["BTC"],
        name="BTC (USD)",
        line=dict(color=BTC_COLOR, width=2.5),
        hovertemplate="BTC | (%{customdata}, $%{y:,.0f})<extra></extra>",
        customdata=usd["Date"].dt.strftime("%b %Y"),
        visible=True
    ))

    # ---- Gold oz/BTC ratio & bands (hidden initially) ----
    if gld_ratio is not None and bands_gld is not None:
        for key in ["Top", "Frothy", "Mid", "Bear", "Support"]:
            fig.add_trace(go.Scatter(
                x=_x(bands_gld["Date"]), y=bands_gld[key],
                name=f"{key} (Gold)",
                line=dict(color=COLORS[key], dash="dash"),
                hovertemplate=f"{key} | (%{{customdata}}, %{{y:,.2f}} oz/BTC)<extra></extra>",
                customdata=bands_gld["Date"].dt.strftime("%b %Y"),
                visible="legendonly"
            ))
        fig.add_trace(go.Scatter(
            x=_x(gld_ratio["Date"]), y=gld_ratio["Ratio"],
            name="BTC (Gold)",  # i.e., Gold oz/BTC series
            line=dict(color=BTC_COLOR, width=2.5),
            hovertemplate="Gold | (%{customdata}, %{y:,.2f} oz/BTC)<extra></extra>",
            customdata=gld_ratio["Date"].dt.strftime("%b %Y"),
            visible="legendonly"
        ))

    return fig

# ---------- HTML generator (safe placeholders, no %/fstring injection) ----------

def page_html(fig, usd_df, bands_usd, gld_df=None, bands_gld=None) -> str:
    # Build packs for the JS slider + zone computation
    usd = usd_df.sort_values("Date")
    p_usd = {
        "date": usd["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "price": usd["BTC"].astype(float).tolist(),
        "bands": {
            "Support": bands_usd["Support"].astype(float).tolist(),
            "Bear":    bands_usd["Bear"].astype(float).tolist(),
            "Frothy":  bands_usd["Frothy"].astype(float).tolist(),
            "Top":     bands_usd["Top"].astype(float).tolist(),
        }
    }

    p_gld = None
    if gld_df is not None and {"Date","Ratio"}.issubset(gld_df.columns):
        p_gld = {
            "date": gld_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "ratio": gld_df["Ratio"].astype(float).tolist(),
            "bands": {
                "Support": bands_gld["Support"].astype(float).tolist(),
                "Bear":    bands_gld["Bear"].astype(float).tolist(),
                "Frothy":  bands_gld["Frothy"].astype(float).tolist(),
                "Top":     bands_gld["Top"].astype(float).tolist(),
            }
        }

    PACK = {"usd": p_usd, "gold": p_gld}
    pack_json = json.dumps(PACK, separators=(",", ":"))
    fig_json  = pio.to_json(fig, pretty=False)

    TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
  :root {
    --bg:#0f0f12; --panel:#15161a; --ink:#e8e8ea; --muted:#a8a8b3;
    --pill:#20222a; --ok:#2ecc71; --warn:#ffb84d; --hot:#ff5f5f;
  }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--ink);
              font:16px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Inter, Arial, sans-serif; }
  .wrap { max-width:1100px; margin:28px auto 48px; padding:0 14px; }
  h1 { font-size:clamp(28px, 4.5vw, 44px); margin:0 0 14px; }
  .row { display:flex; flex-wrap:wrap; gap:12px 16px; align-items:center; }
  .pill { display:inline-flex; align-items:center; gap:10px; padding:10px 14px; border-radius:999px;
          background:var(--pill); color:var(--ink); }
  .pill .dot { width:12px; height:12px; border-radius:50%; background:#fff; display:inline-block; }
  .dot.dca { background:#fff; } .dot.buy { background:var(--ok); }
  .dot.relax { background:var(--warn); } .dot.frothy { background:var(--hot); }
  .dot.sth { background:#ff3d00; }
  .btn { background:var(--panel); color:var(--ink); padding:10px 14px; border-radius:12px; text-decoration:none;
         border:1px solid #222; display:inline-flex; gap:8px; align-items:center; cursor:pointer; }
  select { background:var(--panel); color:var(--ink); border:1px solid #222; border-radius:10px; padding:8px 12px; }
  .toolbar { display:flex; gap:14px; align-items:center; margin:16px 0; }
  .sliderWrap { margin:8px 0 10px; }
  input[type="range"] { width:100%; }
  .canvas { background:var(--panel); border-radius:14px; padding:10px 10px 4px; }
</style>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" defer></script>
</head>
<body>
<div class="wrap">
  <h1>BTC Purchase Indicator</h1>

  <div class="row">
    <div id="zonePill" class="pill"><span class="dot dca"></span><strong>Price zone:</strong><span id="zoneLabel" style="margin-left:6px">DCA</span></div>
    <a class="btn" href="./dma.html" id="toDma">Open DMA chart →</a>

    <div class="row" style="gap:8px;">
      <label for="denomSel">Denomination</label>
      <select id="denomSel" style="min-width:6.5rem;">
        <option value="USD" selected>USD</option>
        <option value="Gold">Gold</option>
      </select>
    </div>

    <button id="legendBtn" class="btn" type="button">Legend</button>
  </div>

  <div class="sliderWrap">
    <label for="when">View at date:</label>
    <input id="when" type="range" min="0" max="0" step="1" value="0"/>
    <div id="whenOut" style="opacity:.85;margin-top:6px;"></div>
  </div>

  <div id="plfig" class="canvas"></div>
</div>

<script>
  const PACK = __PACK__;
  const FIG  = __FIG__;

  const USD_IDX = [];
  const GLD_IDX = [];
  (FIG.data || []).forEach((tr, i) => {
    const nm = (tr.name || '');
    if (/Gold/i.test(nm)) GLD_IDX.push(i); else USD_IDX.push(i);
  });

  function toggleLegend(){
    const cur = (FIG.layout.showlegend !== false);
    FIG.layout.showlegend = !cur;
    Plotly.react('plfig', FIG.data, FIG.layout, {responsive:true});
  }

  function zoneFor(v, b){
    if (v == null || !b) return '—';
    if (v < b.Support) return 'SELL THE HOUSE!!';
    if (v < b.Bear)    return 'Buy';
    if (v < b.Frothy)  return 'DCA';
    if (v < b.Top)     return 'Relax';
    return 'Frothy';
  }
  function dotClass(z){
    switch(z){
      case 'SELL THE HOUSE!!': return 'sth';
      case 'Buy': return 'buy';
      case 'DCA': return 'dca';
      case 'Relax': return 'relax';
      case 'Frothy': return 'frothy';
      default: return 'dca';
    }
  }

  function applyDenom(which){
    const usdOn = (which === 'USD');
    USD_IDX.forEach(i => { FIG.data[i].visible = usdOn ? true : 'legendonly'; });
    GLD_IDX.forEach(i => { FIG.data[i].visible = usdOn ? 'legendonly' : true; });

    FIG.layout.yaxis.title = usdOn ? 'USD / BTC' : 'Gold oz / BTC';
    Plotly.react('plfig', FIG.data, FIG.layout, {responsive:true});

    // set slider to that series
    const p = usdOn ? PACK.usd : PACK.gold;
    const slider = document.getElementById('when');
    const out    = document.getElementById('whenOut');
    if (!p || !p.date || p.date.length === 0){
      slider.max = 0; slider.value = 0; out.textContent = '';
      document.getElementById('zoneLabel').textContent = '—';
      document.querySelector('#zonePill .dot').className = 'dot dca';
      return;
    }
    slider.max = String(p.date.length - 1);
    slider.value = slider.max;

    function fmtUSD(n){ return (n>=1000) ? ('$'+n.toLocaleString('en-US')) : ('$'+Math.round(n)); }
    function fmtG(n){ return (Math.round(n*100)/100).toLocaleString('en-US') + ' oz/BTC'; }

    function update(idx){
      const d = p.date[idx];
      const v = usdOn ? p.price[idx] : p.ratio[idx];
      const b = p.bands;
      const z = zoneFor(v, b);
      document.getElementById('zoneLabel').textContent = z;
      document.querySelector('#zonePill .dot').className = 'dot ' + dotClass(z);
      out.textContent = usdOn ? (d + ' · ' + fmtUSD(v)) : (d + ' · ' + fmtG(v));
    }
    update(Number(slider.value));
    slider.oninput = e => update(Number(e.target.value));
  }

  document.getElementById('legendBtn').addEventListener('click', toggleLegend);
  document.getElementById('denomSel').addEventListener('change', (e)=> applyDenom(e.target.value));

  Plotly.newPlot('plfig', FIG.data, FIG.layout, {responsive:true});
  applyDenom('USD'); // default
</script>
</body>
</html>
"""
    return TEMPLATE.replace("__PACK__", pack_json).replace("__FIG__", fig_json)

# ---------- main build ----------

def main():
    outdir = pathlib.Path("dist")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    usd = load_btc()
    gld = load_gold()

    # 2) Align & compute Gold oz/BTC ratio
    merged = (
        usd[["Date","BTC"]]
        .merge(gld[["Date","Gold"]], on="Date", how="left")
        .sort_values("Date")
    )
    merged["Gold"] = merged["Gold"].ffill()  # fill gaps to BTC calendar
    ratio = (merged["Gold"] / merged["BTC"]).astype(float)
    gld_ratio_hist = pd.DataFrame({"Date": merged["Date"], "Ratio": ratio}).dropna()

    # 3) Build monthly future dates to 2040 for bands
    last_hist = max(usd["Date"].max(), gld["Date"].max())
    # first of next month:
    start_next = (last_hist + pd.offsets.MonthBegin(1)).normalize()
    future = pd.date_range(start=start_next, end=PROJ_END, freq="MS")
    pl_dates = pd.DatetimeIndex(pd.concat([usd["Date"], pd.Series(future)]).drop_duplicates().sort_values())
    pl_dates = pl_dates[pl_dates >= pd.Timestamp(f"{START_YEAR}-01-01")]

    # 4) Fit power law (no anchors)
    m_usd, b_usd, s_usd = fit_power(usd["Date"], usd["BTC"])
    m_gld, b_gld, s_gld = fit_power(gld_ratio_hist["Date"], gld_ratio_hist["Ratio"])

    # 5) Bands over projection dates
    bands_usd = bands_over(pl_dates, m_usd, b_usd, s_usd)
    bands_gld = bands_over(pl_dates, m_gld, b_gld, s_gld)

    # 6) Figure
    fig = make_powerlaw_figure(usd, bands_usd, gld_ratio_hist, bands_gld)

    # 7) HTML
    html = page_html(fig, usd, bands_usd, gld_ratio_hist, bands_gld)
    (outdir / "index.html").write_text(html, encoding="utf-8")

    # Optional: tiny stub for DMA to avoid 404 (you can replace later)
    (outdir / "dma.html").write_text(
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>DMA chart</title>"
        "<style>body{background:#0f0f12;color:#e8e8ea;font:16px -apple-system,Segoe UI,Roboto,Arial}"
        ".wrap{max-width:900px;margin:40px auto;padding:0 16px}a{color:#4da3ff}</style>"
        "<div class='wrap'><h1>DMA chart</h1><p>This page is a placeholder. "
        "You can link your DMA HTML here. <a href='./index.html'>← Back to Power-Law</a></p></div>",
        encoding="utf-8"
    )

    print("[build] Wrote:", outdir / "index.html")

if __name__ == "__main__":
    main()