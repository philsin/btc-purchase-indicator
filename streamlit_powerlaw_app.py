# streamlit_powerlaw_app.py — BTC Purchase Indicator
# --------------------------------------------------
#  ✓ Mobile‑friendly Streamlit front‑end
#  ✓ Power‑law mid‑line + σ support/resistance channel
#  ✓ Triple‑fallback price loader (CoinMetrics → CoinGecko → Stooq CSV)
#  ✓ 21 M FDV toggle, σ‑width slider, Value/Frothy badge

import io, requests, pandas as pd, numpy as np
import streamlit as st
import plotly.graph_objects as go

UA = {"User-Agent": "btc-pl-tool/1.0"}         # avoids CoinGecko 403

# ───────────────────────── data loader ──────────────────────────

def fetch_prices() -> pd.DataFrame:
    """Return BTC daily closes in USD from 2010‑01‑01 to today.
    1️⃣ CoinMetrics  2️⃣ CoinGecko  3️⃣ Stooq CSV."""

    # 1️⃣ CoinMetrics CSV (may 401 without API key)
    cm = ("https://api.coinmetrics.io/v4/timeseries/asset-metrics"
          "?assets=btc&metrics=PriceUSD&frequency=1d&start_time=2010-01-01")
    try:
        r = requests.get(cm, headers=UA, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        df = df.rename(columns={"time": "Date", "PriceUSD": "Price"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df[["Date", "Price"]]
    except requests.RequestException:
        pass  # fall through

    # 2️⃣ CoinGecko JSON (no key, needs UA)
    try:
        cg = ("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
              "?vs_currency=usd&days=max&interval=daily")
        r = requests.get(cg, headers=UA, timeout=15)
        r.raise_for_status()
        data = r.json()["prices"]                # list of [epoch_ms, price]
        df = pd.DataFrame(data, columns=["ts", "Price"])
        df["Date"] = pd.to_datetime(df["ts"], unit="ms")
        return df[["Date", "Price"]]
    except requests.RequestException:
        pass

    # 3️⃣ Stooq CSV (static mirror, always 200)
    try:
        stooq = "https://stooq.com/q/d/l/?s=btcusd&i=d"
        df = pd.read_csv(stooq)
        df.columns = [c.lower() for c in df.columns]
        date_col  = [c for c in df.columns if "date"  in c][0]
        price_col = [c for c in df.columns if "close" in c or "price" in c][0]
        df = df[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "Price"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna()
        return df
    except Exception as e:
        st.error(f"All data sources failed → {e}")
        st.stop()

# ───────────────────── power‑law utilities ──────────────────────
GENESIS = pd.Timestamp("2009-01-03")

def days_since_genesis(ts):
    return (ts - GENESIS).days

def fit_power_law(df):
    X = np.log10(df["days"].values.reshape(-1, 1))
    y = np.log10(df["Price"].values)
    A, B = np.polyfit(X.flatten(), y, 1)
    mid = 10 ** (A * X + B).flatten()
    sigma = np.std(np.log10(df["Price"]) - np.log10(mid))
    return mid, sigma

# ────────────────────── Streamlit interface ─────────────────────
st.set_page_config("BTC Purchase Indicator", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    df = fetch_prices()
    df["days"] = df["Date"].apply(days_since_genesis)
    mid, sigma = fit_power_law(df)
    df["mid"] = mid
    df["support"] = 10 ** (np.log10(mid) - sigma)
    df["resist"]  = 10 ** (np.log10(mid) + sigma)
    return df, sigma

df, sigma = load_data()

# Sidebar controls
sigma_mult = st.sidebar.slider("σ band width", 0.5, 2.5, 1.0, 0.25)
use_mcap   = st.sidebar.toggle("Show fully‑diluted Market‑Cap (21 M)")

# Toggle price ↔ market‑cap
if use_mcap:
    df[["Price", "mid", "support", "resist"]] *= 21_000_000
    y_title = "Market‑Cap (USD)"
else:
    y_title = "Price (USD)"

# Zone badge
latest = df.iloc[-1]
zone = (
    "🟢 Value" if latest["Price"] < df["support"].iloc[-1] * sigma_mult else
    "🔴 Frothy" if latest["Price"] > df["resist"].iloc[-1]  * sigma_mult else
    "⚪ Neutral"
)

st.markdown(f"## Current zone: **{zone}**  (σ×{sigma_mult})")

# Plotly chart
fig = go.Figure()
fig.update_layout(
    yaxis_type="log", yaxis_title=y_title, xaxis_title="Year",
    template="plotly_white", height=500)

fig.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="BTC", line=dict(color="gold")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["mid"], name="Mid‑line", line=dict(color="black", dash="dash")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["support"]*sigma_mult, name="Support", line=dict(color="green", dash="dash")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["resist"]*sigma_mult, name="Resistance", line=dict(color="red", dash="dash")))

st.plotly_chart(fig, use_container_width=True)
