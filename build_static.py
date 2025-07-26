#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
# build_static.py  ·  BTC Purchase Indicator (static Plotly)
# ─────────────────────────────────────────────────────────────

from pathlib import Path
import io, json, requests, numpy as np, pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# --------------- constants
GENESIS   = pd.Timestamp("2009-01-03")
PROJ_END  = pd.Timestamp("2040-12-31")
UA        = {"User-Agent": "btc-pl-pages/1.0"}

# Band definitions (sigma multiples)
LEVELS = {
    "Top":         +1.75,
    "Frothy":      +1.00,
    "PL Best Fit":  0.00,
    "Bear":        -0.50,
    "Support":     -1.50,
}
COLORS = {
    "Top":         "#16a34a",   # green
    "Frothy":      "#86efac",   # light green
    "PL Best Fit": "#ffffff",   # white
    "Bear":        "#fda4af",   # light red
    "Support":     "#ef4444",   # red
    "BTC":         "#ffd166",   # gold line
}
DASHES = {k: "dash" for k in LEVELS}
DASHES["PL Best Fit"] = "dash"

# --------------- loaders
def _read_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers=UA)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def _btc_stooq() -> pd.DataFrame:
    df = _read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")
    df.columns = [c.lower() for c in df.columns]
    date_col  = [c for c in df.columns if "date" in c][0]
    close_col = [c for c in df.columns if ("close" in c) or ("price" in c)][-1]
    df = df.rename(columns={date_col: "Date", close_col: "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _btc_github() -> pd.DataFrame:
    df  = _read_csv("https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv")
    df  = df.rename(columns={"Closing Price (USD)": "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _gold_stooq() -> pd.DataFrame:
    df = _read_csv("https://stooq.com/q/d/l/?s=xauusd&i=d")
    df.columns = [c.lower() for c in df.columns]
    date_col  = [c for c in df.columns if "date" in c][0]
    close_col = [c for c in df.columns if ("close" in c) or ("price" in c)][-1]
    df = df.rename(columns={date_col: "Date", close_col: "Gold"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("Gold>0").sort_values("Date")[["Date","Gold"]]

def _gold_lbma() -> pd.DataFrame:
    df  = _read_csv("https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    col = "USD (PM)" if "USD (PM)" in df.columns else ("USD (AM)" if "USD (AM)" in df.columns else None)
    if col is None:
        cand = [c for c in df.columns if "USD" in c.upper()]
        col = cand[0]
    df = df.rename(columns={col: "Gold"})
    df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
    return df.dropna().query("Gold>0").sort_values("Date")[["Date","Gold"]]

def load_btc_gold() -> pd.DataFrame:
    # BTC
    try:
        btc = _btc_stooq()
        if len(btc) < 1000:
            raise ValueError("short")
    except Exception:
        btc = _btc_github()
    # GOLD
    try:
        gold = _gold_stooq()
        if len(gold) < 1000:
            raise ValueError("short")
    except Exception:
        gold = _gold_lbma()

    # Align to BTC calendar; forward-fill gold
    gold_ff = gold.set_index("Date").reindex(btc["Date"]).ffill().reset_index()
    gold_ff = gold_ff.rename(columns={"index":"Date"})
    df = btc.merge(gold_ff, on="Date", how="left").dropna()
    return df

# --------------- math
def log_days(dates) -> np.ndarray:
    """Return log10(days since GENESIS) for Series or Index safely."""
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
    """Fit y = m*x + b in log-log where x = log10(days), y = log10(values)."""
    X = log_days(dates)
    y = np.log10(values.to_numpy(dtype="float64"))
    mask = np.isfinite(X) & np.isfinite(y)
    m, b = np.polyfit(X[mask], y[mask], 1)
    sigma = float(np.std(y[mask] - (m * X[mask] + b)))
    return m, b, sigma

def build_bands(dates: pd.Series, m: float, b: float, sigma: float) -> dict:
    """Return dict of arrays for each band over given dates (includes 'mid')."""
    X = log_days(dates)
    mid_log = m * X + b
    out = {"mid": 10 ** mid_log}
    for name, k in LEVELS.items():
        if name == "PL Best Fit":
            continue
        out[name] = 10 ** (mid_log + sigma * k)
    return out

def year_ticks(start=2012, dense_until=2020, end=2040):
    """Year labels: every year through 2020, then every 2 years."""
    years = list(range(start, dense_until + 1)) + list(range(dense_until + 2, end + 1, 2))
    vals  = log_days([pd.Timestamp(f"{y}-01-01") for y in years]).tolist()
    text  = [str(y) for y in years]
    return vals, text

# --------------- figure
def make_powerlaw_fig(df: pd.DataFrame):
    # Series
    usd_series = df["BTC"].astype(float)                 # USD / BTC
    gld_series = (df["Gold"] / df["BTC"]).astype(float)  # Gold oz / BTC

    # Fits
    m_usd, b_usd, s_usd = fit_power(df["Date"], usd_series)
    m_gld, b_gld, s_gld = fit_power(df["Date"], gld_series)

    # Projection dates to 2040 (monthly)
    last = df["Date"].iloc[-1]
    future = pd.date_range(last + pd.offsets.MonthBegin(1), PROJ_END, freq="MS")
    full_dates = pd.Index(df["Date"]).append(future)

    # Bands
    bands_usd = build_bands(full_dates, m_usd, b_usd, s_usd)
    bands_gld = build_bands(full_dates, m_gld, b_gld, s_gld)

    # X (log-time)
    x_hist = log_days(df["Date"])
    x_full = log_days(full_dates)

    # Labels
    order = ["Top", "Frothy", "PL Best Fit", "Bear", "Support"]
    date_labels_full = pd.to_datetime(full_dates).strftime("%b %Y")
    date_labels_hist = pd.to_datetime(df["Date"]).strftime("%b %Y")

    fig = go.Figure()

    # USD traces (visible by default)
    for name in order:
        label = name + " (USD)"
        y = bands_usd["mid"] if name == "PL Best Fit" else bands_usd[name]
        fig.add_trace(go.Scatter(
            x=x_full, y=y, mode="lines",
            line=dict(color=COLORS[name], width=2, dash=DASHES[name]),
            name=label, legendgroup="USD",
            customdata=date_labels_full,
            hovertemplate=label + " | (%{customdata}, %{y:$,.0f})<extra></extra>",
            visible=True
        ))
    fig.add_trace(go.Scatter(
        x=x_hist, y=usd_series, mode="lines",
        line=dict(color=COLORS["BTC"], width=2.5),
        name="BTC (USD)", legendgroup="USD",
        customdata=date_labels_hist,
        hovertemplate="BTC | (%{customdata}, %{y:$,.0f})<extra></extra>",
        visible=True
    ))

    # Gold traces (hidden initially)
    for name in order:
        label = name + " (Gold)"
        y = bands_gld["mid"] if name == "PL Best Fit" else bands_gld[name]
        fig.add_trace(go.Scatter(
            x=x_full, y=y, mode="lines",
            line=dict(color=COLORS[name], width=2, dash=DASHES[name]),
            name=label, legendgroup="GLD",
            customdata=date_labels_full,
            hovertemplate=label + " | (%{customdata}, %{y:,.2f} oz)<extra></extra>",
            visible=False
        ))
    fig.add_trace(go.Scatter(
        x=x_hist, y=gld_series, mode="lines",
        line=dict(color=COLORS["BTC"], width=2.5),
        name="BTC (Gold)", legendgroup="GLD",
        customdata=date_labels_hist,
        hovertemplate="BTC | (%{customdata}, %{y:,.2f} oz)<extra></extra>",
        visible=False
    ))

    # Axes
    tickvals, ticktext = year_ticks(2012, 2020, 2040)
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=18, b=56),
        paper_bgcolor="#0f1116", plot_bgcolor="#151821",
        xaxis=dict(
            title="Year (log-time)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridcolor="#263041", zeroline=False
        ),
        yaxis=dict(
            title="USD / BTC", type="log", tickformat="$,d",
            showgrid=True, gridcolor="#263041", zeroline=False
        )
    )
    return fig, full_dates, bands_usd, bands_gld, usd_series, gld_series

# --------------- HTML writer (no f-strings)
def write_index_html(fig: go.Figure,
                     full_dates: pd.Index,
                     bands_usd: dict,
                     bands_gld: dict,
                     usd_hist: pd.Series,
                     gld_hist: pd.Series,
                     out_path="dist/index.html",
                     page_title="BTC Purchase Indicator"):

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Figure JSON
    fig_json = pio.to_json(fig, pretty=False)

    # Data for slider/zone (align by ISO date strings)
    dates_iso = pd.to_datetime(full_dates).strftime("%Y-%m-%d").tolist()
    hist_iso  = pd.to_datetime(pd.Series(full_dates[:len(usd_hist)])).dt.strftime("%Y-%m-%d").tolist()

    payload = {
        "dates": dates_iso,
        "hist": {
            "dates": hist_iso,
            "usd": pd.Series(usd_hist).astype(float).round(6).tolist(),
            "gld": pd.Series(gld_hist).astype(float).round(6).tolist(),
        },
        "usd": {
            "Top":      np.asarray(bands_usd["Top"]).astype(float).round(6).tolist(),
            "Frothy":   np.asarray(bands_usd["Frothy"]).astype(float).round(6).tolist(),
            "Mid":      np.asarray(bands_usd["mid"]).astype(float).round(6).tolist(),
            "Bear":     np.asarray(bands_usd["Bear"]).astype(float).round(6).tolist(),
            "Support":  np.asarray(bands_usd["Support"]).astype(float).round(6).tolist(),
        },
        "gld": {
            "Top":      np.asarray(bands_gld["Top"]).astype(float).round(6).tolist(),
            "Frothy":   np.asarray(bands_gld["Frothy"]).astype(float).round(6).tolist(),
            "Mid":      np.asarray(bands_gld["mid"]).astype(float).round(6).tolist(),
            "Bear":     np.asarray(bands_gld["Bear"]).astype(float).round(6).tolist(),
            "Support":  np.asarray(bands_gld["Support"]).astype(float).round(6).tolist(),
        }
    }
    arrays_json = json.dumps(payload, separators=(",", ":"))

    TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>__TITLE__</title>
<link rel="preconnect" href="https://cdn.plot.ly">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {
    color-scheme: dark;
    --bg:#0f1116; --fg:#e7e9ee; --muted:#8e95a5; --card:#151821;
  }
  html,body { margin:0; background:var(--bg); color:var(--fg);
              font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:1100px; margin:24px auto; padding:0 16px; }
  h1 { font-size:clamp(28px,4.5vw,44px); margin:0 0 12px; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  .btn { background:var(--card); border:1px solid #263041; border-radius:14px; padding:10px 14px; cursor:pointer; color:var(--fg); }
  select { background:var(--card); border:1px solid #263041; border-radius:12px; padding:8px 10px; color:var(--fg); font:inherit; }
  #fig { height:70vh; min-height:430px; background:var(--card); border-radius:14px; padding:8px; }
  .chip { background:var(--card); border:1px solid #263041; border-radius:999px; padding:8px 12px; display:inline-flex; align-items:center; gap:10px; }
  .dot { width:12px; height:12px; border-radius:50%; background:#fff; }
  .slider { width:100%; }
</style>
</head>
<body>
  <div class="wrap">
    <h1>BTC Purchase Indicator</h1>

    <div class="row" style="margin-bottom:8px;">
      <div class="chip"><span id="zoneDot" class="dot"></span> <b>Price Zone:</b> <span id="zoneTxt" style="margin-left:6px">—</span></div>
      <div class="row" style="gap:8px;">
        <label for="denom">Denomination</label>
        <select id="denom" style="min-width:6.5rem;">
          <option value="USD" selected>USD</option>
          <option value="Gold">Gold</option>
        </select>
      </div>
      <button class="btn" id="legendBtn">Legend</button>
    </div>

    <div class="row" style="margin:6px 0 2px;">
      <label for="dateSlider">View at date:</label>
    </div>
    <input id="dateSlider" class="slider" type="range" min="0" value="0" step="1"/>
    <div id="dateRead" class="row" style="opacity:.85; margin:6px 0 12px 2px;"></div>

    <div id="fig"></div>
  </div>

  <!-- Safe JSON embeds -->
  <script type="application/json" id="figjson">__FIGJSON__</script>
  <script type="application/json" id="arrays">__ARRAYS__</script>

  <script>
    const FIG = JSON.parse(document.getElementById('figjson').textContent);
    const ARR = JSON.parse(document.getElementById('arrays').textContent);

    // Build plot
    const figEl = document.getElementById('fig');
    Plotly.newPlot(figEl, FIG.data, FIG.layout, {displaylogo:false, responsive:true});

    // Identify USD vs Gold traces by name
    const USD_IDX = [], GLD_IDX = [];
    (FIG.data||[]).forEach((tr,i)=>{
      const nm = (tr.name||"");
      if (/\(Gold\)/.test(nm)) GLD_IDX.push(i); else USD_IDX.push(i);
    });

    // Controls
    const denomSel = document.getElementById('denom');
    const legendBtn = document.getElementById('legendBtn');
    const slider = document.getElementById('dateSlider');
    const zoneTxt = document.getElementById('zoneTxt');
    const zoneDot = document.getElementById('zoneDot');
    const dateRead = document.getElementById('dateRead');

    // Slider covers historical price dates
    const histDates = ARR.hist.dates;
    slider.max = Math.max(0, histDates.length - 1);
    slider.value = slider.max;

    function setLegend(on){ Plotly.relayout(figEl, {"showlegend": !!on}); }
    let legendOn = true;
    legendBtn.addEventListener('click', ()=>{ legendOn = !legendOn; setLegend(legendOn); });

    function setDenom(which){
      const usdOn = (which === "USD");
      const vis = new Array(FIG.data.length).fill(false);
      USD_IDX.forEach(i => vis[i] = usdOn);
      GLD_IDX.forEach(i => vis[i] = !usdOn);
      Plotly.restyle(figEl, {"visible": vis});
      Plotly.relayout(figEl, {"yaxis.title.text": (usdOn ? "USD / BTC" : "Gold oz / BTC")});
      updateReadout();
    }

    function fmtUSD(v){ return (v==null||!isFinite(v)) ? "—" : "$"+Math.round(v).toLocaleString(); }
    function fmtOZ(v){  return (v==null||!isFinite(v)) ? "—" : (Math.round(v*100)/100).toLocaleString()+" oz"; }

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
      if (idx < 0) return null;
      const g = denom==="USD" ? ARR.usd : ARR.gld;
      return {
        Support: g.Support[idx],
        Bear:    g.Bear[idx],
        Mid:     g.Mid[idx],
        Frothy:  g.Frothy[idx],
        Top:     g.Top[idx],
      };
    }

    function updateReadout(){
      const i = parseInt(slider.value, 10);
      const denom = denomSel.value;
      const dISO = histDates[i] || ARR.dates[ARR.dates.length-1];
      const bands = readBandsAt(dISO, denom);
      if (!bands){ zoneTxt.textContent = "—"; dateRead.textContent = ""; return; }

      const price = (denom==="USD") ? ARR.hist.usd[i] : ARR.hist.gld[i];
      const zone = zoneFor(price, bands);
      zoneTxt.textContent = zone;
      zoneDot.style.background = zoneDotColor(zone);

      dateRead.textContent = (denom==="USD")
        ? (dISO + " · " + fmtUSD(price))
        : (dISO + " · " + fmtOZ(price));
    }

    denomSel.addEventListener('change', (e)=> setDenom(e.target.value));
    slider.addEventListener('input', updateReadout);

    // Init
    setDenom("USD");
    setLegend(true);
    updateReadout();

    window.addEventListener('resize', ()=> Plotly.Plots.resize(figEl));
  </script>
</body>
</html>
"""

    html = (
        TEMPLATE
        .replace("__TITLE__", page_title)
        .replace("__FIGJSON__", fig_json)
        .replace("__ARRAYS__", arrays_json)
    )
    out.write_text(html, encoding="utf-8")
    print("[build] wrote", out)

# --------------- main
def main():
    print("[build] loading BTC & Gold …")
    df = load_btc_gold()
    print(f"[build] rows: {len(df):,}  (BTC & Gold)")
    fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist = make_powerlaw_fig(df)
    write_index_html(fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist)

if __name__ == "__main__":
    main()