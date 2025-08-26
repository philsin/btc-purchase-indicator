# build_static.py  — builds dist/index.html with FIG & ARR JSON for template
import os, json, math, io, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

OUTDIR = "dist"
TEMPLATE = "template.html"

GENESIS = pd.Timestamp("2009-01-03")
PROJ_END = pd.Timestamp("2040-12-31")
FD_SUPPLY = 21_000_000  # not used here; price-only bands (denom handled later)

# -------------------- Data loaders (Stooq + fallbacks) --------------------
def _read_stooq_csv(url, date_col="Date", price_col="Close"):
    df = pd.read_csv(url)
    cols = [c.lower() for c in df.columns]
    dcol = [c for c in df.columns if "date" in c.lower()][0]
    pcol = [c for c in df.columns if ("close" in c.lower()) or ("price" in c.lower())][0]
    df = df.rename(columns={dcol: "Date", pcol: "Price"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(
        df["Price"].astype(str).str.replace(",", ""), errors="coerce"
    )
    df = df.dropna().sort_values("Date")
    return df[["Date", "Price"]]

def load_btc():
    # BTC from Stooq
    return _read_stooq_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")

def load_gold():
    # Gold (XAUUSD) from Stooq
    return _read_stooq_csv("https://stooq.com/q/d/l/?s=xauusd&i=d")

def load_spy():
    # SPY ETF from Stooq (USD)
    return _read_stooq_csv("https://stooq.com/q/d/l/?s=spy.us&i=d")

# -------------------- Power-law fit & bands (on BTC/USD) --------------------
def fit_power(df_price):
    # df_price: columns Date, Price (BTC/USD)
    X = np.log10(np.maximum(1.0, (df_price["Date"] - GENESIS).dt.days.values))
    y = np.log10(df_price["Price"].values)
    slope, intercept = np.polyfit(X, y, 1)
    mid_log = slope * X + intercept
    resid = y - mid_log
    sigma = np.std(resid)
    return slope, intercept, max(sigma, 0.25)

LEVELS = {
    "Support": -1.5,
    "Bear":    -0.75,  # updated per your request
    "Mid":      0.0,   # PL Best Fit
    "Frothy":  +1.0,
    "Top":     +2.0,   # updated to +2.0σ
}

# -------------------- Build time series to 2040 --------------------
def build_full_dates(last_hist_date):
    # end monthly to PROJ_END (first day of month)
    future = pd.date_range(
        last_hist_date + pd.offsets.MonthBegin(1),
        PROJ_END, freq="MS", inclusive="both"
    )
    return future

def halving_dates():
    # Known + projected (approx. every 210k blocks ~4 years)
    # Use actual dates for past, estimate for future mid-April windows
    dates = [
        "2012-11-28",
        "2016-07-09",
        "2020-05-11",
        "2024-04-20"  # 2024 actual
    ]
    # Next estimates (approx every 4 years, mid-April)
    last = pd.Timestamp("2024-04-20")
    for yr in (2028, 2032, 2036, 2040):
        dates.append(f"{yr}-04-15")
    return [pd.Timestamp(d) for d in dates]

# -------------------- Main build --------------------
def main():
    btc = load_btc()
    gold = load_gold()
    spy = load_spy()

    # align (inner join) btc/gold/spy on Date for ratios and shared calendar
    df = btc.merge(gold.rename(columns={"Price": "Gold"}), on="Date", how="inner")
    df = df.merge(spy.rename(columns={"Price": "SPY"}), on="Date", how="inner")
    # sanity
    df = df[(df["Price"] > 0) & (df["Gold"] > 0) & (df["SPY"] > 0)].copy()
    df = df.rename(columns={"Price": "BTC"})
    df = df.sort_values("Date").reset_index(drop=True)

    # Fit power law on BTC/USD
    slope, intercept, sigma = fit_power(df[["Date", "BTC"]])
    days_full = np.maximum(1, (df["Date"] - GENESIS).dt.days.values)
    mid_log_hist = slope * np.log10(days_full) + intercept
    mid_hist = 10 ** mid_log_hist

    # extend monthly to 2040 for band projections
    future = build_full_dates(df["Date"].iloc[-1])
    full = pd.concat([df[["Date"]], pd.DataFrame({"Date": future})], ignore_index=True)
    full["days"] = np.maximum(1, (full["Date"] - GENESIS).dt.days)
    mid_log = slope * np.log10(full["days"].values) + intercept
    mid = 10 ** mid_log

    # Construct USD bands on the full monthly grid
    bands_usd = {}
    for name, k in LEVELS.items():
        bands_usd[name] = (10 ** (mid_log + sigma * k)).tolist()

    # Construct Gold & SPY denominations by dividing USD bands by Gold/SPY price
    # We need Gold/SPY price on the same monthly grid -> forward-fill from last known daily
    daily = df.set_index("Date")[["BTC", "Gold", "SPY"]].copy()
    # fill to cover future range for ffill
    last_daily = daily.index[-1]
    ext_idx = pd.date_range(daily.index[0], PROJ_END, freq="D")
    daily_ext = daily.reindex(ext_idx).ffill()

    # Helper: map monthly date -> latest available daily price (same or before)
    def snap_monthly_prices(monthly_idx, col):
        # For each monthly date, use that exact day if present, else previous day.
        out = []
        for d in monthly_idx:
            if d in daily_ext.index:
                out.append(float(daily_ext.loc[d, col]))
            else:
                # pick last available before d
                loc = daily_ext.index.searchsorted(d) - 1
                if loc < 0:
                    out.append(float('nan'))
                else:
                    out.append(float(daily_ext.iloc[loc][col]))
        return out

    full_gold = snap_monthly_prices(full["Date"], "Gold")
    full_spy  = snap_monthly_prices(full["Date"], "SPY")

    def safe_div(num_list, den_list):
        out = []
        for n, d in zip(num_list, den_list):
            if d and d > 0 and np.isfinite(n) and np.isfinite(d):
                out.append(n / d)
            else:
                out.append(float('nan'))
        return out

    bands_gld = {name: safe_div(bands_usd[name], full_gold) for name in LEVELS.keys()}
    bands_spy = {name: safe_div(bands_usd[name], full_spy)  for name in LEVELS.keys()}

    # Historical series for hover/slider (BTC in USD, Gold-denom, SPY-denom)
    hist_usd = df["BTC"].tolist()
    hist_gld = (df["BTC"] / df["Gold"]).tolist()  # BTC / (USD per oz) = oz/BTC
    hist_spy = (df["BTC"] / df["SPY"]).tolist()   # BTC / (USD per SPY) = SPY/BTC

    # Build FIG (traces) on log-time x = log10(days since genesis)
    def logx_from_dates(dts):
        d = np.maximum(1, (np.array(dts) - GENESIS).astype("timedelta64[D]").astype(int))
        return np.log10(d)

    # full monthly x, hist daily x
    x_full = logx_from_dates(full["Date"].values)
    x_hist = logx_from_dates(df["Date"].values)

    # Colors
    C = {
        "top": "#22c55e",        # green
        "frothy": "#86efac",
        "mid": "#e5e7eb",        # white
        "bear": "#fca5a5",
        "support": "#ef4444",
        "btc": "#facc15",        # yellow
        "zone_soft": {
            "buy":"rgba(239,68,68,0.10)",
            "bear":"rgba(252,165,165,0.10)",
            "mid":"rgba(229,231,235,0.06)",
            "frothy":"rgba(134,239,172,0.10)"
        }
    }

    # Build traces for each denomination as legendgroups: USD, GLD, SPY
    def band_traces(group, ydict, visible):
        # zone fills (Support-Bear, Bear-Mid, Mid-Frothy, Frothy-Top)
        t = []
        # Support to Bear
        t += [dict(type="scatter", mode="lines", x=x_full.tolist(), y=ydict["Support"],
                   line=dict(color="rgba(0,0,0,0)"),
                   fill=None, name="", hoverinfo="skip",
                   legendgroup=group, showlegend=False, visible=visible)]
        t += [dict(type="scatter", mode="lines", x=x_full.tolist(), y=ydict["Bear"],
                   line=dict(color="rgba(0,0,0,0)"),
                   fill="tonexty", fillcolor=C["zone_soft"]["buy"], name="",
                   hoverinfo="skip", legendgroup=group, showlegend=False, visible=visible)]
        # Bear to Mid
        t += [dict(type="scatter", mode="lines", x=x_full.tolist(), y=ydict["Mid"],
                   line=dict(color="rgba(0,0,0,0)"),
                   fill="tonexty", fillcolor=C["zone_soft"]["bear"], name="",
                   hoverinfo="skip", legendgroup=group, showlegend=False, visible=visible)]
        # Mid to Frothy
        t += [dict(type="scatter", mode="lines", x=x_full.tolist(), y=ydict["Frothy"],
                   line=dict(color="rgba(0,0,0,0)"),
                   fill="tonexty", fillcolor=C["zone_soft"]["mid"], name="",
                   hoverinfo="skip", legendgroup=group, showlegend=False, visible=visible)]
        # Frothy to Top
        t += [dict(type="scatter", mode="lines", x=x_full.tolist(), y=ydict["Top"],
                   line=dict(color="rgba(0,0,0,0)"),
                   fill="tonexty", fillcolor=C["zone_soft"]["frothy"], name="",
                   hoverinfo="skip", legendgroup=group, showlegend=False, visible=visible)]

        # Band lines (dashed)
        t += [
            dict(type="scatter", mode="lines", x=x_full.tolist(), y=ydict["Top"],
                 line=dict(color=C["top"], dash="dash", width=1.5),
                 name="Top (+2σ)", legendgroup=group, visible=visible),
            dict(type="scatter", mode="lines", x=x_full.tolist(), y=ydict["Frothy"],
                 line=dict(color=C["frothy"], dash="dash", width=1.5),
                 name="Frothy (+1σ)", legendgroup=group, visible=visible),
            dict(type="scatter", mode="lines", x=x_full.tolist(), y=ydict["Mid"],
                 line=dict(color=C["mid"], dash="dash", width=1.5),
                 name="PL Best Fit (0σ)", legendgroup=group, visible=visible),
            dict(type="scatter", mode="lines", x=x_full.tolist(), y=ydict["Bear"],
                 line=dict(color=C["bear"], dash="dash", width=1.5),
                 name="Bear (-0.75σ)", legendgroup=group, visible=visible),
            dict(type="scatter", mode="lines", x=x_full.tolist(), y=ydict["Support"],
                 line=dict(color=C["support"], dash="dash", width=1.5),
                 name="Support (-1.5σ)", legendgroup=group, visible=visible),
        ]
        return t

    data = []
    # USD group (default visible)
    data += band_traces("USD", bands_usd, True)
    data += [dict(type="scatter", mode="lines", x=x_hist.tolist(), y=hist_usd,
                  line=dict(color=C["btc"], width=2), name="BTC",
                  legendgroup="USD", visible=True)]
    # Yellow marker (USD)
    data += [dict(type="scatter", mode="markers", x=[x_hist[-1]], y=[hist_usd[-1]],
                  marker=dict(color=C["btc"], size=8), name="",
                  hoverinfo="skip", legendgroup="USD", visible=True)]

    # GLD group (hidden initially)
    data += band_traces("GLD", bands_gld, False)
    data += [dict(type="scatter", mode="lines", x=x_hist.tolist(), y=hist_gld,
                  line=dict(color=C["btc"], width=2), name="BTC",
                  legendgroup="GLD", visible=False)]
    data += [dict(type="scatter", mode="markers", x=[x_hist[-1]], y=[hist_gld[-1]],
                  marker=dict(color=C["btc"], size=8), name="", hoverinfo="skip",
                  legendgroup="GLD", visible=False)]

    # SPY group (hidden initially)
    data += band_traces("SPY", bands_spy, False)
    data += [dict(type="scatter", mode="lines", x=x_hist.tolist(), y=hist_spy,
                  line=dict(color=C["btc"], width=2), name="BTC",
                  legendgroup="SPY", visible=False)]
    data += [dict(type="scatter", mode="markers", x=[x_hist[-1]], y=[hist_spy[-1]],
                  marker=dict(color=C["btc"], size=8), name="", hoverinfo="skip",
                  legendgroup="SPY", visible=False)]

    # Remember indices of the three marker traces (for slider moves)
    USD_MARK_IDX = len(data) - 6  # marker in USD group
    GLD_MARK_IDX = len(data) - 4
    SPY_MARK_IDX = len(data) - 2

    # Halving vertical shapes (by x in log-time)
    def x_from_date(dt):
        d = max(1, (pd.Timestamp(dt) - GENESIS).days)
        return math.log10(d)
    halvings = halving_dates()
    halving_shapes = []
    for d in halvings:
        x = x_from_date(d)
        halving_shapes.append(dict(
            type="line", xref="x", x0=x, x1=x, yref="paper", y0=0, y1=1,
            line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot")
        ))

    layout = dict(
        template="plotly_dark",
        showlegend=True,
        xaxis=dict(
            title="Year (log-time)",
            type="linear",  # log of time already used in x values
            showgrid=True, gridwidth=0.5,
        ),
        yaxis=dict(
            title="USD / BTC",
            type="log",
            showgrid=True, gridwidth=0.5,
            tickformat="$,d"
        ),
        paper_bgcolor="#0f1116",
        plot_bgcolor="#151821",
        margin=dict(l=60, r=30, t=40, b=50),
        shapes=halving_shapes,
        meta=dict(
            USD_MARK_IDX=USD_MARK_IDX,
            GLD_MARK_IDX=GLD_MARK_IDX,
            SPY_MARK_IDX=SPY_MARK_IDX
        )
    )

    FIG = dict(data=data, layout=layout)

    # -------------------- ARR payload for the client --------------------
    ARR = dict(
        dates=[d.strftime("%Y-%m-%d") for d in full["Date"]],
        usd=bands_usd,
        gld=bands_gld,
        spy=bands_spy,
        hist=dict(
            dates=[d.strftime("%Y-%m-%d") for d in df["Date"]],
            usd=hist_usd,
            gld=hist_gld,
            spy=hist_spy
        )
    )

    # -------------------- Emit template --------------------
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__TITLE__", "BTC Purchase Indicator")
    html = html.replace("__FIGJSON__", json.dumps(FIG, separators=(",",":")))
    html = html.replace("__ARRAYS__", json.dumps(ARR, separators=(",",":")))

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[build] rows: {len(df):,}  (BTC/Gold/SPY)")
    print("[build] wrote dist/index.html")

if __name__ == "__main__":
    main()
