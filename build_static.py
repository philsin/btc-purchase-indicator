# build_static.py  —  BTC Purchase Indicator (static Plotly site)
# - Power-law bands with log-time x through 2040
# - USD / BTC  and  Gold oz / BTC denominations (dropdown)
# - Time slider drives "Price Zone" badge (historical or today)
# - Legend toggle button
# - DMA page link preserved (dmas.html not rebuilt here to keep this focused)

import json, math
from textwrap import dedent
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

# ---------------------- constants ----------------------
GENESIS   = pd.Timestamp("2009-01-03")
PROJ_END  = pd.Timestamp("2040-12-31")
GRID_D    = "M24"   # 2-year vertical rhythm (we emulate on log-time axis)
UA        = {"User-Agent": "btc-pl-static/1.0"}

LEVELS = {
    "Support":     -1.50,
    "Bear":        -0.50,
    "PL Best Fit":  0.00,
    "Frothy":      +1.00,
    "Top":         +1.75,
}
COLORS = {
    "Support":  "rgb(220,60,60)",
    "Bear":     "rgb(255,140,140)",
    "PL Best Fit": "white",
    "Frothy":   "rgb(120,255,140)",
    "Top":      "rgb(0,160,60)",
    "BTC":      "gold",
}

# ---------------------- helpers ----------------------
def to_days(dates) -> np.ndarray:
    """Days since GENESIS as float array (pandas-safe)."""
    td = (pd.to_datetime(dates) - GENESIS).values.astype("timedelta64[D]").astype(float)
    return td

def log_days(dates) -> np.ndarray:
    d = to_days(dates)
    # guard: log10(0) is -inf; replace non-positive with 0.1 day
    d[d <= 0] = 0.1
    return np.log10(d)

def fit_power(dates: pd.Series, values: pd.Series):
    """Fit y = m*x + b in log-log where x = log10(days since genesis), y = log10(value)."""
    x = log_days(dates)
    y = np.log10(values.values.astype(float))
    m, b = np.polyfit(x, y, 1)
    sigma = float(np.std(y - (m*x + b)))
    return float(m), float(b), sigma

def add_sigma_bands_over(dates: pd.Series, m: float, b: float, sigma: float) -> pd.DataFrame:
    """Compute PL mid & bands over an arbitrary date index."""
    d = pd.to_datetime(dates)
    x = log_days(d)
    mid_log = m * x + b
    out = pd.DataFrame({"Date": d})
    out["PL Best Fit"] = 10 ** mid_log
    sig = max(0.25, float(sigma))  # avoid collapsing bands
    for name, k in LEVELS.items():
        if name == "PL Best Fit":
            continue
        out[name] = 10 ** (mid_log + sig * k)
    return out

def year_ticks(start=2012, mid=2020, end=2040):
    """Labels yearly to 2020, then every other year. Returned for the *display panel* only."""
    yrs = list(range(start, mid+1, 1)) + list(range(mid+2, end+1, 2))
    labels = [str(y) for y in yrs]
    # convert to log-days for positioning under the chart grid we draw
    vals = log_days(pd.to_datetime([f"{y}-01-01" for y in yrs]))
    return vals.tolist(), labels

def zone_for(price, row_bands):
    if price < row_bands["Support"]:
        return "SELL THE HOUSE!!", "#ffffff", "#b00000"   # white dot on dark-red range
    if price < row_bands["Bear"]:
        return "Buy", "#101010", "#78ff8c"                # dark dot on greenish
    if price < row_bands["Frothy"]:
        return "DCA", "#101010", "#ffffff"                # dark dot on white
    if price < row_bands["Top"]:
        return "Relax", "#101010", "#ffb84d"              # dark dot on light orange
    return "Frothy", "#101010", "#ff4d4d"                 # dark dot on red

# ---------------------- data loaders ----------------------
def _btc_stooq():
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date" in c})
    df = df.rename(columns={c: "BTC" for c in df.columns if ("close" in c) or ("price" in c)})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")

def _btc_github():
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    df = pd.read_csv(raw).rename(columns={"Closing Price (USD)": "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna().sort_values("Date")[["Date", "BTC"]]

def _gold_stooq():
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date" in c})
    df = df.rename(columns={c: "Gold" for c in df.columns if ("close" in c) or ("price" in c)})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().sort_values("Date")[["Date", "Gold"]]

def _gold_lbma():
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.rename(columns={"USD (PM)": "Gold"})
    return df.dropna().sort_values("Date")[["Date", "Gold"]]

def load_prices():
    try:
        btc = _btc_stooq()
        if len(btc) < 1000:
            raise ValueError
    except Exception:
        btc = _btc_github()
    try:
        gold = _gold_stooq()
        if len(gold) < 1000:
            raise ValueError
    except Exception:
        gold = _gold_lbma()
    df = pd.merge(btc, gold, on="Date", how="inner")
    return df

# ---------------------- figure builders ----------------------
def make_powerlaw_figure(df, denom="USD"):
    """
    Build the initial Plotly Figure for power-law chart in the chosen denomination.
    X is log-time (log10 days since genesis). Y is USD/BTC or oz/BTC (gold).
    We'll ship both USD & Gold traces and let JS toggle visibility.
    """
    # Base series
    usd = df[["Date", "BTC"]].copy()
    gld = df[["Date"]].copy()
    gld["GLD"] = df["BTC"] / df["Gold"]  # oz per BTC

    # Projection timeline to 2040 (monthly)
    future = pd.date_range(usd["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                           PROJ_END, freq="MS")
    full_dates = pd.Index(usd["Date"]).append(future)

    # Fit & bands (USD)
    m_u, b_u, s_u = fit_power(usd["Date"], usd["BTC"])
    bands_usd = add_sigma_bands_over(full_dates, m_u, b_u, s_u)

    # Fit & bands (Gold oz/BTC)
    m_g, b_g, s_g = fit_power(gld["Date"], gld["GLD"])
    bands_gld = add_sigma_bands_over(full_dates, m_g, b_g, s_g)

    # x values are log-days
    x_full = log_days(full_dates)
    x_hist = log_days(usd["Date"])

    # USD y arrays
    y_usd_price = usd["BTC"].values
    y_usd = {k: bands_usd[k].values for k in LEVELS.keys()}
    # Gold y arrays
    y_gld_price = gld.set_index("Date").reindex(usd["Date"]).fillna(method="ffill")["GLD"].values
    y_gld = {k: bands_gld[k].values for k in LEVELS.keys()}

    # hover label styles (contrast)
    hover_dark   = dict(bgcolor="rgba(255,255,255,0.9)", font_color="#111")
    hover_light  = dict(bgcolor="rgba(0,0,0,0.8)",       font_color="#fff")

    fig = go.Figure(layout=dict(
        template="plotly_dark",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto", size=12),
        xaxis=dict(title="Year (log–time)"),
        yaxis=dict(type="log", title="USD / BTC", tickformat="$,.0f"),
        showlegend=True,
        plot_bgcolor="#111", paper_bgcolor="#111",
        margin=dict(l=60, r=20, t=24, b=64),
    ))

    # --- USD traces (visible by default if denom=='USD') ---
    vis_usd  = True if denom == "USD" else False
    vis_gld  = not vis_usd

    order = ["Top", "Frothy", "PL Best Fit", "Bear", "Support"]
    for name in ["Top", "Frothy"]:
        fig.add_trace(go.Scatter(
            x=x_full, y=y_usd[name], name=f"{name} (USD)",
            line=dict(color=COLORS[name], width=2, dash="dash"),
            visible=vis_usd,
            customdata=pd.to_datetime(full_dates).strftime("%b %Y"),
            hovertemplate=f"{name} | (%{{customdata}}, %{{y:$,.0f}})<extra></extra>",
            hoverlabel=hover_dark
        ))
    fig.add_trace(go.Scatter(
        x=x_full, y=y_usd["PL Best Fit"], name="PL Best Fit (USD)",
        line=dict(color="white", width=2, dash="dash"),
        visible=vis_usd,
        customdata=pd.to_datetime(full_dates).strftime("%b %Y"),
        hovertemplate="PL Best Fit | (%{customdata}, %{y:$,.0f})<extra></extra>",
        hoverlabel=hover_dark
    ))
    for name in ["Bear", "Support"]:
        fig.add_trace(go.Scatter(
            x=x_full, y=y_usd[name], name=f"{name} (USD)",
            line=dict(color=COLORS[name], width=2, dash="dash"),
            visible=vis_usd,
            customdata=pd.to_datetime(full_dates).strftime("%b %Y"),
            hovertemplate=f"{name} | (%{{customdata}}, %{{y:$,.0f}})<extra></extra>",
            hoverlabel=hover_dark
        ))
    fig.add_trace(go.Scatter(
        x=x_hist, y=y_usd_price, name="BTC (USD)",
        line=dict(color=COLORS["BTC"], width=2.5),
        visible=vis_usd,
        customdata=usd["Date"].dt.strftime("%b %Y"),
        hovertemplate="BTC | (%{customdata}, %{y:$,.0f})<extra></extra>",
        hoverlabel=hover_light
    ))

    # --- Gold traces (hidden initially unless denom=='Gold') ---
    for name in ["Top", "Frothy"]:
        fig.add_trace(go.Scatter(
            x=x_full, y=y_gld[name], name=f"{name} (Gold)",
            line=dict(color=COLORS[name], width=2, dash="dash"),
            visible=vis_gld,
            customdata=pd.to_datetime(full_dates).strftime("%b %Y"),
            hovertemplate=f"{name} | (%{{customdata}}, %{{y:,.0f}} oz)<extra></extra>",
            hoverlabel=hover_dark
        ))
    fig.add_trace(go.Scatter(
        x=x_full, y=y_gld["PL Best Fit"], name="PL Best Fit (Gold)",
        line=dict(color="white", width=2, dash="dash"),
        visible=vis_gld,
        customdata=pd.to_datetime(full_dates).strftime("%b %Y"),
        hovertemplate="PL Best Fit | (%{customdata}, %{y:,.0f} oz)<extra></extra>",
        hoverlabel=hover_dark
    ))
    for name in ["Bear", "Support"]:
        fig.add_trace(go.Scatter(
            x=x_full, y=y_gld[name], name=f"{name} (Gold)",
            line=dict(color=COLORS[name], width=2, dash="dash"),
            visible=vis_gld,
            customdata=pd.to_datetime(full_dates).strftime("%b %Y"),
            hovertemplate=f"{name} | (%{{customdata}}, %{{y:,.0f}} oz)<extra></extra>",
            hoverlabel=hover_dark
        ))
    fig.add_trace(go.Scatter(
        x=x_hist, y=y_gld_price, name="BTC (Gold)",
        line=dict(color=COLORS["BTC"], width=2.5),
        visible=vis_gld,
        customdata=usd["Date"].dt.strftime("%b %Y"),
        hovertemplate="BTC | (%{customdata}, %{y:,.0f} oz)<extra></extra>",
        hoverlabel=hover_light
    ))

    # x tick labels (display helper — we already have the gridlines)
    tickvals, ticktext = year_ticks(2012, 2020, 2040)
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)

    return fig, bands_usd, bands_gld, usd, gld

# ---------------------- page writer ----------------------
def page_html(fig, usd_df, bands_usd, gld_df, bands_gld):
    """Return full HTML with controls + JS to switch denomination, legend, and zone."""
    full_dates = pd.to_datetime(bands_usd["Date"])
    full_iso   = full_dates.dt.strftime("%Y-%m-%d").tolist()

    # USD history
    usd_hist_iso   = usd_df["Date"].dt.strftime("%Y-%m-%d").tolist()
    usd_hist_price = usd_df["BTC"].astype(float).round(6).tolist()

    # GOLD history (oz/BTC) aligned to USD dates
    gld_aligned      = gld_df.set_index("Date").reindex(usd_df["Date"]).fillna(method="ffill")
    gld_hist_series  = gld_aligned["GLD"].astype(float)
    gld_hist_price   = gld_hist_series.round(6).tolist()

    pack = dict(
        full_dates = full_iso,
        usd_hist_dates = usd_hist_iso,
        usd_hist_price = usd_hist_price,
        gld_hist_price = gld_hist_price,
        usd_bands = {k: bands_usd[k].astype(float).round(6).tolist() for k in LEVELS},
        gld_bands = {k: bands_gld[k].astype(float).round(6).tolist() for k in LEVELS},
    )

    # initial zone (today) in USD
    last_date = usd_df["Date"].iloc[-1]
    idx = int(np.searchsorted(full_dates.values, last_date.to_datetime64(), side="right") - 1)
    last_price = float(usd_df["BTC"].iloc[-1])
    row_bands = {k: float(bands_usd[k].iloc[idx]) for k in LEVELS}
    zone, dot_fg, chip_bg = zone_for(last_price, row_bands)

    PJSON   = json.dumps(pack, separators=(",", ":"))
    FIGJSON = fig.to_json()

    html = dedent("""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width,initial-scale=1"/>
      <title>BTC Purchase Indicator</title>
      <script src="https://cdn.plot.ly/plotly-2.27.1.min.js"></script>
      <style>
        :root { --bg:#0d0f13; --card:#141820; --muted:#a7b0c0; --chip:#222735; }
        body { margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto; background:var(--bg); color:#e8ecf3;}
        .wrap { max-width: 1080px; margin: 28px auto 80px; padding: 0 14px; }
        h1 { margin: 0 0 14px; font-size: 44px; line-height: 1.05; }
        .row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
        .chip { display:inline-flex; align-items:center; gap:10px; padding:10px 16px; border-radius:999px; background: var(--chip); }
        .dot { width:12px; height:12px; border-radius:50%; background:#fff; }
        .btn { background:var(--card); border:1px solid #2a3242; color:#dbe2ee; padding:10px 14px; border-radius:12px; cursor:pointer; }
        .btn:hover{filter:brightness(1.05)}
        select { background:var(--card); border:1px solid #2a3242; color:#dbe2ee; padding:8px 10px; border-radius:10px; font-size:14px; }
        .label { color:#a7b0c0; }
        .panel { background:var(--card); border:1px solid #263044; border-radius:16px; padding:10px; }
        #fig { width:100%; height: 72vh; min-height:520px; }
        .slider-row{margin-top:14px; gap:14px;}
        input[type=range]{ width: 100%; accent-color:#8ac6ff;}
        .legend-btn{ margin-left:auto;}
      </style>
    </head>
    <body>
      <div class="wrap">
        <h1>BTC Purchase Indicator</h1>

        <div class="row">
          <div id="zoneChip" class="chip" style="background:%(chip_bg)s">
            <div id="zoneDot" class="dot" style="background:%(dot_fg)s"></div>
            <div class="label">Price Zone:</div><b id="zoneTxt">%(zone)s</b>
          </div>

          <button class="btn" onclick="location.href='dmas.html'">Open DMA chart →</button>

          <div class="label" style="margin-left:6px">Denomination</div>
          <select id="denomSel" onchange="setDenom(this.value)">
            <option value="USD">USD</option>
            <option value="Gold">Gold</option>
          </select>

          <button id="legendBtn" class="btn legend-btn" onclick="toggleLegend()">Legend</button>
        </div>

        <div class="panel" style="margin-top:12px">
          <div class="label">View at date:</div>
          <div class="row slider-row">
            <input id="dateSlider" type="range" min="0" max="0" step="1" value="0" oninput="onSlide(this.value)"/>
            <div id="slideVal" class="label"></div>
          </div>
        </div>

        <div class="panel" style="margin-top:12px">
          <div id="fig"></div>
        </div>
      </div>

      <script>
        const PACK = %(PJSON)s;
        const FIG  = %(FIGJSON)s;

        // mount figure
        const el = document.getElementById('fig');
        const fig = JSON.parse(FIG);
        Plotly.newPlot(el, fig.data, fig.layout, {displaylogo:false, responsive:true});

        // state
        let denom = 'USD';
        let showLegend = true;

        // slider setup
        const slider = document.getElementById('dateSlider');
        const slideVal = document.getElementById('slideVal');
        slider.max = PACK.usd_hist_dates.length - 1;
        slider.value = slider.max;

        function fmtUSD(x){ return '$' + Number(x).toLocaleString('en-US', {maximumFractionDigits:0}); }
        function fmtOZ(x){  return Number(x).toLocaleString('en-US', {maximumFractionDigits:0}) + ' oz'; }

        function updateZone(idx){
          // pick date (from USD history timeline, also aligns gold series)
          const dt = PACK.usd_hist_dates[idx];
          // locate same date in full_dates to read bands
          const fidx = PACK.full_dates.indexOf(dt);
          if (fidx < 0) return;

          const price = (denom==='USD') ? PACK.usd_hist_price[idx] : PACK.gld_hist_price[idx];
          const bands = (denom==='USD') ? PACK.usd_bands : PACK.gld_bands;

          const row = {
            "Support": bands["Support"][fidx],
            "Bear":    bands["Bear"][fidx],
            "Frothy":  bands["Frothy"][fidx],
            "Top":     bands["Top"][fidx]
          };

          let z = "Frothy", dot="#111", chip="#ff4d4d";
          if (price < row["Support"])      { z = "SELL THE HOUSE!!"; dot="#fff"; chip="#7a0000"; }
          else if (price < row["Bear"])    { z = "Buy";             dot="#101010"; chip="#78ff8c"; }
          else if (price < row["Frothy"])  { z = "DCA";             dot="#101010"; chip="#ffffff"; }
          else if (price < row["Top"])     { z = "Relax";           dot="#101010"; chip="#ffb84d"; }
          else                              { z = "Frothy";         dot="#101010"; chip="#ff4d4d"; }

          document.getElementById('zoneTxt').innerText = z;
          document.getElementById('zoneDot').style.background = dot;
          document.getElementById('zoneChip').style.background = chip;

          const shown = (denom==='USD') ? fmtUSD(price) : fmtOZ(price);
          slideVal.innerText = dt + ' · ' + shown;
        }

        function setDenom(v){
          denom = v;
          // data ordering:
          // 0..5 = USD bands + BTC; 6..11 = Gold bands + BTC
          const usdVisible  = (denom==='USD');
          const gldVisible  = !usdVisible;
          const vis = [
            usdVisible, usdVisible, usdVisible, usdVisible, usdVisible, usdVisible,
            gldVisible, gldVisible, gldVisible, gldVisible, gldVisible, gldVisible
          ];
          Plotly.update(el, {}, {yaxis: {title: (denom==='USD'?'USD / BTC':'Gold oz / BTC'), tickformat:(denom==='USD'?'$,.0f':',.0f')}});
          Plotly.restyle(el, 'visible', vis);
          updateZone(slider.value);
        }

        function toggleLegend(){
          showLegend = !showLegend;
          Plotly.relayout(el, {showlegend: showLegend});
        }

        function onSlide(v){ updateZone(parseInt(v)); }
        // init view
        setDenom('USD');  // default
        document.getElementById('denomSel').value = 'USD';
        updateZone(slider.value);
      </script>
    </body>
    </html>
    """) % dict(
        zone=zone, chip_bg=chip_bg, dot_fg=dot_fg
    )
    return html

# ---------------------- main build ----------------------
def main():
    outdir = Path("dist")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_prices()

    fig, bands_usd, bands_gld, usd_df, gld_df = make_powerlaw_figure(df, denom="USD")
    html = page_html(fig, usd_df, bands_usd, gld_df, bands_gld)
    (outdir / "index.html").write_text(html, encoding="utf-8")

    # simple placeholder DMA page link (kept so the button works)
    (outdir / "dmas.html").write_text(
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>DMA – Coming back</title>"
        "<body style='background:#0d0f13;color:#e8ecf3;font-family:Inter,system-ui;'>"
        "<div style='max-width:960px;margin:24px auto;padding:12px'>"
        "<a href='index.html' style='color:#8ac6ff;text-decoration:none'>&larr; Back</a>"
        "<h1>DMA chart</h1><p>This placeholder remains so the navigation works while we finish the new DMA page build.</p>"
        "</div></body>", encoding="utf-8"
    )

    print("[build] Wrote dist/index.html and dist/dmas.html")

if __name__ == "__main__":
    main()