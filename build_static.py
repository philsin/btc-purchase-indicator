# ─────────────────────────────────────────────────────────────
# build_static.py  ·  BTC Purchase Indicator (static, Plotly)
# ─────────────────────────────────────────────────────────────
# - Power-law bands on log-time x-axis (no anchoring)
# - Denomination switch (USD/BTC vs Gold oz/BTC)
# - Legend toggle
# - "View at date" slider + Price zone badge
# - Projection of bands to 2040-12
# - Writes a single self-contained: dist/index.html
# ─────────────────────────────────────────────────────────────

from pathlib import Path
import io
import json
import math
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# --------------- constants
GENESIS = pd.Timestamp("2009-01-03")
PROJ_END = pd.Timestamp("2040-12-31")
UA = {"User-Agent": "btc-pl-static/1.0"}

# --------------- data loaders
def _btc_stooq() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    date_col = [c for c in df.columns if "date" in c][0]
    price_col = [c for c in df.columns if c in ("close","c","price") or "close" in c or "price" in c][0]
    df = df.rename(columns={date_col:"Date", price_col:"BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"] = pd.to_numeric(df["BTC"].astype(str).str.replace(",",""), errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _btc_github() -> pd.DataFrame:
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    df = pd.read_csv(raw).rename(columns={"Date":"Date","Closing Price (USD)":"BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"] = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _gold_stooq() -> pd.DataFrame:
    # XAUUSD daily
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    date_col = [c for c in df.columns if "date" in c][0]
    price_col = [c for c in df.columns if c in ("close","c","price") or "close" in c or "price" in c][0]
    df = df.rename(columns={date_col:"Date", price_col:"Gold"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"].astype(str).str.replace(",",""), errors="coerce")
    return df.dropna().sort_values("Date")[["Date","Gold"]]

def _gold_lbma() -> pd.DataFrame:
    # Public mirror; column "USD (PM)" is the daily PM fix in USD/oz
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Prefer PM fix; fallback to AM if missing
    if "USD (PM)" in df.columns:
        col = "USD (PM)"
    elif "USD (AM)" in df.columns:
        col = "USD (AM)"
    else:
        # last resort: any column with USD in name
        cols = [c for c in df.columns if "USD" in c.upper()]
        col = cols[0]
    df = df.rename(columns={col:"Gold"})
    df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
    return df.dropna().sort_values("Date")[["Date","Gold"]]

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

    # align on Date (inner join), forward-fill gold to BTC dates if needed
    gold = gold.sort_values("Date").dropna()
    btc = btc.sort_values("Date").dropna()
    gold_ff = gold.set_index("Date").reindex(btc["Date"]).ffill().reset_index().rename(columns={"index":"Date"})
    df = btc.merge(gold_ff, on="Date", how="left").dropna()
    return df

# --------------- math: power-law fit on log-time
def log_days(dates) -> np.ndarray:
    """Return log10(days since GENESIS) for Series or Index safely."""
    td = pd.to_datetime(dates) - GENESIS
    if isinstance(td, pd.Series):
        days = td.dt.days.to_numpy()
    elif isinstance(td, pd.TimedeltaIndex):
        # TimedeltaIndex exposes .days directly (ndarray)
        days = td.days.astype(float)
    else:
        # Fallback for ndarray/list of timedeltas
        days = (np.asarray(td) / np.timedelta64(1, "D")).astype(float)
    days = np.where(days <= 0, np.nan, days)
    return np.log10(days)

def fit_power(df: pd.DataFrame, price_col: str):
    """Return slope m, intercept b, sigma for y = m*x + b in log space."""
    x = log_days(df["Date"])
    y = np.log10(df[price_col].to_numpy())
    mask = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[mask], y[mask], 1)
    mid = m * x + b
    sigma = float(np.std(y[mask] - mid[mask]))
    return m, b, sigma

def build_bands(dates: pd.Series, m: float, b: float, sigma: float, levels):
    x = log_days(dates)
    mid = m * x + b
    out = {"mid": 10 ** mid}
    for name, k in levels.items():
        out[name] = 10 ** (mid + sigma * k)
    return out  # dict of arrays

# --------------- ticks for log-time axis: yearly 2012–2040
def year_ticks(start=2012, dense_until=2020, end=2040):
    years = []
    # every year 2012..2020
    years.extend(range(start, dense_until + 1))
    # every 2 years after
    years.extend(range(dense_until + 2, end + 1, 2))
    vals = log_days(pd.to_datetime([f"{y}-01-01" for y in years]))
    txt = [str(y) for y in years]
    return vals.tolist(), txt

# --------------- figure (both denominations embedded)
def make_powerlaw_fig(df: pd.DataFrame) -> go.Figure:
    # price series
    usd_series = df["BTC"].astype(float)  # USD / BTC
    gld_series = (df["BTC"] / df["Gold"]).astype(float)  # oz / BTC

    # fits
    m_usd, b_usd, s_usd = fit_power(df, "BTC")
    m_gld, b_gld, s_gld = fit_power(pd.DataFrame({"Date": df["Date"], "GLD": gld_series}), "GLD")

    # projection dates (monthly to 2040)
    last = df["Date"].iloc[-1]
    future = pd.date_range(last + pd.offsets.MonthBegin(1), PROJ_END, freq="MS")
    full_dates = pd.concat([df["Date"], pd.Series(future)], ignore_index=True)

    levels = {
        "Top":      +1.75,
        "Frothy":   +1.00,
        "PL Best Fit": 0.00,
        "Bear":     -0.50,
        "Support":  -1.50,
    }
    # bands for USD and Gold (over full timeline)
    bands_usd = build_bands(full_dates, m_usd, b_usd, s_usd, levels)
    bands_gld = build_bands(full_dates, m_gld, b_gld, s_gld, levels)

    # legend names (with sigmas)
    leg = {
        "Top":      "Top (+1.75σ)",
        "Frothy":   "Frothy (+1.00σ)",
        "PL Best Fit": "PL Best Fit",
        "Bear":     "Bear (-0.50σ)",
        "Support":  "Support (-1.50σ)",
    }
    colors = {
        "Top": "#16a34a",         # green-600
        "Frothy": "#86efac",      # green-300
        "PL Best Fit": "#ffffff", # white
        "Bear": "#fda4af",        # rose-300
        "Support": "#ef4444",     # red-500
        "BTC": "#ffd166",         # gold line
    }
    dashes = {
        "Top": "dash",
        "Frothy": "dash",
        "PL Best Fit": "dash",
        "Bear": "dash",
        "Support": "dash",
    }

    # X axis in log-time
    x_hist = log_days(df["Date"])
    x_full = log_days(full_dates)

    fig = go.Figure()
    # --- USD traces (group="USD")
    for name in ["Top","Frothy","PL Best Fit","Bear","Support"]:
        y = bands_usd["mid"] if name == "PL Best Fit" else bands_usd[name]
        fig.add_trace(go.Scatter(
    x=..., y=...,
    name=name_usd,
    line=dict(color=..., dash="dash"),
    hovertemplate=f"{name_usd} | %{{x|%b %Y}}, %{{y:$,.0f}}<extra></extra>",
))
    fig.add_trace(go.Scatter(
        x=x_hist, y=usd_series, mode="lines",
        line=dict(color=colors["BTC"], width=2.5),
        name="BTC (USD)", legendgroup="USD", 
        hovertemplate="BTC | %{x|%b %Y}, %{y:$,.0f}<extra></extra>",
        visible=True
    ))

    # --- Gold traces (group="GLD")
    for name in ["Top","Frothy","PL Best Fit","Bear","Support"]:
        y = bands_gld["mid"] if name == "PL Best Fit" else bands_gld[name]
        fig.add_trace(go.Scatter(
        x=..., y=...,
        name=name_gld,
        line=dict(color=..., dash="dash"),
        hovertemplate=f"{name_gld} | %{{x|%b %Y}}, %{{y:,.2f}} oz<extra></extra>",
    ))
        fig.add_trace(go.Scatter(
        x=x_hist, y=gld_series, mode="lines",
        line=dict(color=colors["BTC"], width=2.5),
        name="BTC (Gold)", legendgroup="GLD",
        hovertemplate="BTC | %{x|%b %Y}, %{y:,.2f} oz<extra></extra>",
        visible=False
    ))

    tickvals, ticktext = year_ticks(2012, 2020, 2040)

    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60,r=20,t=20,b=60),
        plot_bgcolor="#0f1116", paper_bgcolor="#0f1116",
        xaxis=dict(
            title="Year (log-time)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridcolor="#263041"
        ),
        yaxis=dict(
            title="USD / BTC",
            type="log",
            tickformat="$,d",
            showgrid=True, gridcolor="#263041"
        )
    )
    return fig, full_dates, bands_usd, bands_gld, usd_series, gld_series

# --------------- safe HTML writer (no quoting pitfalls)
def write_index_html(fig: go.Figure,
                     full_dates: pd.Series,
                     bands_usd: dict,
                     bands_gld: dict,
                     usd_hist: pd.Series,
                     gld_hist: pd.Series,
                     out_path="dist/index.html",
                     page_title="BTC Purchase Indicator"):

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Embed figure JSON
    fig_json = pio.to_json(fig, pretty=False)

    # arrays for JS (dates and bands only where needed)
    dates_iso = pd.to_datetime(full_dates).dt.strftime("%Y-%m-%d").tolist()

    PL_ARRAYS = {
        "dates": dates_iso,
        "usd": {
            "Top":      bands_usd["Top"].tolist(),
            "Frothy":   bands_usd["Frothy"].tolist(),
            "Mid":      bands_usd["mid"].tolist(),
            "Bear":     bands_usd["Bear"].tolist(),
            "Support":  bands_usd["Support"].tolist(),
        },
        "gld": {
            "Top":      bands_gld["Top"].tolist(),
            "Frothy":   bands_gld["Frothy"].tolist(),
            "Mid":      bands_gld["mid"].tolist(),
            "Bear":     bands_gld["Bear"].tolist(),
            "Support":  bands_gld["Support"].tolist(),
        },
        # historical price arrays match fig order (we only need values for readout)
        "hist": {
            "usd": usd_hist.tolist(),
            "gld": gld_hist.tolist(),
            "hist_dates": pd.to_datetime(pd.Series(full_dates[:len(usd_hist)])).dt.strftime("%Y-%m-%d").tolist()
        }
    }
    arrays_json = json.dumps(PL_ARRAYS, separators=(",",":"))

    html = f"""<!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>{page_title}</title>
    <link rel="preconnect" href="https://cdn.plot.ly">
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
  :root {{
    color-scheme: dark;
    --bg:#0f1116; --fg:#e7e9ee; --muted:#8e95a5; --card:#151821;
    --accent:#60a5fa; --chip:#1f2937; --chipb:#111827;
  }}
  html,body {{ margin:0; background:var(--bg); color:var(--fg);
               font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:1100px; margin:24px auto; padding:0 16px; }}
  h1 {{ font-size:clamp(28px,4.5vw,44px); margin:0 0 12px; }}
  .row {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
  .btn, .chip {{
    background:var(--card); border-radius:14px; padding:10px 14px; border:1px solid #263041;
    cursor:pointer; display:inline-flex; align-items:center; gap:8px;
  }}
  .chip.dot::before {{
    content:""; width:10px; height:10px; border-radius:50%; background:#e5e7eb; display:inline-block;
  }}
  select {{
    background:var(--card); color:var(--fg); border:1px solid #263041; border-radius:12px; padding:8px 10px;
    font:inherit;
  }}
  .slider {{ width:100%; }}
  #fig {{ height:70vh; min-height:430px; }}
  .spacer {{ height:10px; }}
    </style>
    </head>
    <body>
  <div class="wrap">
    <h1>BTC Purchase Indicator</h1>

    <div class="row" style="margin-bottom:8px;">
      <div id="zone" class="chip dot">Price zone: <strong id="zoneLabel">…</strong></div>
      <button id="openDMA" class="btn">Open DMA chart →</button>
    </div>

    <div class="row" style="margin-bottom:10px;">
      <div class="row" style="gap:8px;">
        <label for="denom">Denomination</label>
        <select id="denom">
          <option value="USD" selected>USD</option>
          <option value="Gold">Gold</option>
        </select>
      </div>
      <button class="btn" id="legendBtn">Legend</button>
    </div>

    <div class="row"><label for="dateSlider">View at date:</label></div>
    <input id="dateSlider" class="slider" type="range" min="0" value="0" step="1"/>
    <div id="dateRead" class="row" style="opacity:.85; margin:6px 0 12px 2px;"></div>

    <div id="fig"></div>
  </div>

  <!-- Data payloads -->
  <script type="application/json" id="figjson">{pio.to_json(fig, pretty=False)}</script>
  <script type="application/json" id="arrays">{arrays_json}</script>

  <script>
    // util
    const fmtUSD = v => (v==null||!isFinite(v)) ? "—" :
      "$" + Math.round(v).toLocaleString();
    const fmtOZ  = v => (v==null||!isFinite(v)) ? "—" :
      (v.toFixed(2).replace(/\\.00$/,"")) + " oz";

    const zoneColor = (zone) => {{
      // opposite / contrasting dot per your spec
      switch(zone){{
        case "SELL THE HOUSE!!": return "#ffffff";   // white dot
        case "Buy":              return "#f97316";   // orange-500
        case "DCA":              return "#ffffff";   // white
        case "Relax":            return "#84cc16";   // lime-500
        case "Frothy":           return "#ef4444";   // red-500
        default:                 return "#e5e7eb";
      }}
    }};

    const arrays = JSON.parse(document.getElementById('arrays').textContent);
    const figSpec = JSON.parse(document.getElementById('figjson').textContent);

    const dates = arrays.dates; // full timeline (bands)
    // Slider should cover ONLY historical price dates
    const histDates = arrays.hist.hist_dates;
    const slider = document.getElementById('dateSlider');
    slider.max = Math.max(0, histDates.length - 1);
    slider.value = slider.max;

    // init plot
    const figDiv = document.getElementById('fig');
    const opts = {{displaylogo:false, responsive:true}};
    Plotly.newPlot(figDiv, figSpec.data, figSpec.layout, opts);

    function setDenom(which) {{
      const usdVis = (which === "USD");
      const gldVis = (which === "Gold");
      const upd = {{}};
      // USD traces are indices 0..5, Gold are 6..11 in the way we constructed them
      // (5 bands + 1 BTC for each group)
      // toggle visibility
      const vis = [];
      for (let i=0;i<figSpec.data.length;i++) {{
        if (usdVis && i<=5) vis.push(true);
        else if (gldVis && i>=6) vis.push(true);
        else vis.push(false);
      }}
      Plotly.restyle(figDiv, {{visible: vis}});
      // y-axis title and hover format
      const yTitle = usdVis ? "USD / BTC" : "Gold oz / BTC";
      Plotly.relayout(figDiv, {{"yaxis.title.text": yTitle}});
      updateReadout(); // refresh zone with new denom
    }}

    function currentDenom() {{
      return document.getElementById('denom').value;
    }}

    function priceAtIndex(idx, denom) {{
      if (idx<0 || idx>=histDates.length) return null;
      if (denom==="USD") return arrays.hist.usd[idx];
      return arrays.hist.gld[idx];
    }}

    function zoneAtIndex(idx, denom) {{
      const d = histDates[idx];
      const fullIndex = dates.indexOf(d);
      if (fullIndex < 0) return {{zone:"—", sup:null, bear:null, mid:null, fro:null, top:null}};

      const group = denom==="USD" ? arrays.usd : arrays.gld;
      const sup = group.Support[fullIndex];
      const bear= group.Bear[fullIndex];
      const mid = group.Mid[fullIndex];
      const fro = group.Frothy[fullIndex];
      const top = group.Top[fullIndex];

      const p = priceAtIndex(idx, denom);
      let zone = "—";
      if (p!=null) {{
        if      (p < sup) zone = "SELL THE HOUSE!!";
        else if (p < bear) zone = "Buy";
        else if (p < fro)  zone = "DCA";
        else if (p < top)  zone = "Relax";
        else               zone = "Frothy";
      }}
      return {{zone, sup, bear, mid, fro, top, p}};
    }}

    function updateReadout() {{
      const idx = parseInt(slider.value, 10);
      const denom = currentDenom();
      const z = zoneAtIndex(idx, denom);
      const dateStr = histDates[idx] || "—";

      // zone chip
      const label = document.getElementById('zoneLabel');
      label.textContent = z.zone;
      document.getElementById('zone').style.setProperty("--dot", zoneColor(z.zone));
      document.querySelector('#zone.dot')?.style?.setProperty('background','');

      // show under slider
      const vtxt = (denom==="USD") ? fmtUSD(z.p) : fmtOZ(z.p);
      document.getElementById('dateRead').textContent =
        `${{dateStr}} · ${{vtxt}}`;
    }}

    // Legend toggle
    let legendOn = true;
    document.getElementById('legendBtn').addEventListener('click', () => {{
      legendOn = !legendOn;
      Plotly.relayout(figDiv, {{"showlegend": legendOn}});
    }});

    // Denomination change
    document.getElementById('denom').addEventListener('change', (e) => {{
      setDenom(e.target.value);
    }});

    // Slider change
    slider.addEventListener('input', updateReadout);

    // Initial state
    setDenom("USD");
    updateReadout();

    // DMA link (placeholder – if you later add dma.html, change href)
    document.getElementById('openDMA').addEventListener('click', () => {{
      // If you create a second page, change to 'dma.html'
      window.location.href = '#';
      alert('DMA page not included in this static build yet.');
    }});

    // keep responsive
    window.addEventListener('resize', () => Plotly.Plots.resize(figDiv));
  </script>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    print(f"[build] wrote {out}")

# --------------- main
def main():
    print("[build] loading prices …")
    df = load_btc_gold()
    print(f"[build] rows: {len(df):,}  (BTC & Gold)")

    fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist = make_powerlaw_fig(df)

    write_index_html(fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist,
                     out_path="dist/index.html",
                     page_title="BTC Purchase Indicator")

if __name__ == "__main__":
    main()