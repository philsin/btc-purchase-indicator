import io, requests, pandas as pd, numpy as np, streamlit as st, plotly.graph_objects as go

UA = {"User-Agent": "btc-pl-tool/1.0"}  # helps pass CoinGecko rate‑limit

# ───────────────────────── data loaders ──────────────────────────
def fetch_prices() -> pd.DataFrame:
    """Return daily BTC/USD closes. Tries CoinMetrics → CoinGecko → GitHub CSV."""
    # 1️⃣ CoinMetrics CSV
    cm_url = ("https://api.coinmetrics.io/v4/timeseries/asset-metrics"
              "?assets=btc&metrics=PriceUSD&frequency=1d&start_time=2010-01-01")
    try:
        r = requests.get(cm_url, timeout=15, headers=UA); r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        df = df.rename(columns={"time": "Date", "PriceUSD": "Price"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df[["Date", "Price"]]
    except requests.RequestException:
        pass

    # 2️⃣ CoinGecko JSON
    try:
        cg = ("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
              "?vs_currency=usd&days=max&interval=daily")
        r = requests.get(cg, timeout=15, headers=UA); r.raise_for_status()
        data = r.json()["prices"]
        df = pd.DataFrame(data, columns=["ts", "Price"])
        df["Date"] = pd.to_datetime(df["ts"], unit="ms")
        return df[["Date", "Price"]]
    except requests.RequestException:
        pass

    # 3️⃣ Static mirror
    gh = ("https://raw.githubusercontent.com/datasets/bitcoin-price/"
          "master/data/bitcoin_price.csv")
    df = pd.read_csv(gh)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Closing Price (USD)"]].rename(columns={"Closing Price (USD)": "Price"})

# ────────────────────── power‑law utilities ───────────────────────
GENESIS = pd.Timestamp("2009-01-03")

def days_since_genesis(ts): return (ts - GENESIS).days

def fit_power_law(df):
    X = np.log10(df["days"].values.reshape(-1, 1))
    y = np.log10(df["Price"].values)
    A, B = np.polyfit(X.flatten(), y, 1)
    mid = 10 ** (A * X + B).flatten()
    sigma = np.std(np.log10(df["Price"]) - np.log10(mid))
    return mid, A, B, sigma

# ────────────────────────── Streamlit UI ──────────────────────────
st.set_page_config(page_title="BTC Purchase Indicator", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    df = fetch_prices()
    df["days"] = df["Date"].apply(days_since_genesis)
    mid, A, B, sigma = fit_power_law(df)
    df["mid"] = mid
    df["support"] = 10 ** (np.log10(mid) - sigma)
    df["resist"]  = 10 ** (np.log10(mid) + sigma)
    return df, sigma

df, sigma = load_data()

sigma_mult = st.sidebar.slider("σ band width", 0.5, 2.5, 1.0, 0.25)
use_mcap   = st.sidebar.toggle("Show fully‑diluted Market‑Cap (21 M)")

if use_mcap:
    df["Price"] = df["Price"] * 21_000_000
    df[["mid","support","resist"]] *= 21_000_000
    y_title = "Market‑Cap (USD)"
else:
    y_title = "Price (USD)"

latest = df.iloc[-1]
zone = ("Value 🟢" if latest["Price"] < df["support"].iloc[-1] * sigma_mult else
        "Frothy 🔴" if latest["Price"] > df["resist"].iloc[-1]  * sigma_mult else
        "Neutral ⚪")

st.markdown(f"### **Current zone:** {zone}")

fig = go.Figure()
fig.update_layout(yaxis_type="log", xaxis_title="Year", yaxis_title=y_title, template="plotly_white")
fig.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="BTC", line=dict(color="gold")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["mid"],    name="Mid‑line", line=dict(color="black", dash="dash")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["support"]*sigma_mult, name="-σ", line=dict(color="green", dash="dash")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["resist"] *sigma_mult, name="+σ", line=dict(color="red",   dash="dash")))

st.plotly_chart(fig, use_container_width=True)
