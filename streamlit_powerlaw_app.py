# streamlit_powerlaw_app.py  — BTC Purchase Indicator
import io, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

UA = {"User-Agent": "btc-pl-tool/1.0"}
GENESIS = pd.Timestamp("2009-01-03")
FD_SUPPLY = 21_000_000
GRID_D = "M24"

# ─── 1️⃣ CoinMetrics (needs key → will fail) ─────────────────────────
def _coinmetrics():
    raise RuntimeError("CoinMetrics now requires an API key")  # skip

# ─── 2️⃣ CoinGecko (≤365 days) ───────────────────────────────────────
def _coingecko():
    raise RuntimeError("CoinGecko free tier limited to 365 days")  # skip

# ─── 3️⃣ Stooq daily CSV ─────────────────────────────────────────────
def _stooq():
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    date_col  = [c for c in df.columns if "date"  in c][0]
    price_col = [c for c in df.columns if "close" in c or "price" in c][0]

    clean_date  = df[date_col].astype(str).str.replace(r"[^0-9\\-]", "", regex=True)
    clean_price = (df[price_col].astype(str)
                               .str.replace(",", "", regex=False)
                               .str.replace(r"[^0-9.]", "", regex=True))

    df = pd.DataFrame({"Date": clean_date, "Price": clean_price})
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna().query("Price>0").sort_values("Date")

# ─── 4️⃣ Static GitHub mirror (never blocks) ─────────────────────────
def _github():
    raw = ("https://raw.githubusercontent.com/datasets/bitcoin-price/"
           "master/data/bitcoin_price.csv")
    df = pd.read_csv(raw).rename(columns={"Closing Price (USD)": "Price"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Price"]]

# ─── diagnostic loader ───────────────────────────────────────────────
def get_price_history():
    for name, fn in [("Stooq CSV", _stooq),
                     ("GitHub mirror", _github)]:
        try:
            df = fn()
            if len(df):
                st.info(f"Loaded {len(df):,} rows from **{name}**")
                st.dataframe(df.head())
                return df
        except Exception as e:
            st.warning(f"{name} failed → {e}")
    st.error("No price data from any source."); st.stop()

# ─── power-law helpers ───────────────────────────────────────────────
def fit_power(df):
    X = np.log10((df["Date"] - GENESIS).dt.days.values)
    y = np.log10(df["Price"].values)
    slope, intercept = np.polyfit(X, y, 1)
    mid_log = slope * X + intercept
    sigma   = np.std(y - mid_log)
    return mid_log, sigma

# ─── Streamlit UI ────────────────────────────────────────────────────
st.set_page_config(page_title="BTC Purchase Indicator", layout="wide")

raw = get_price_history()
mid_log, σ = fit_power(raw)

k      = st.sidebar.slider("σ band width", 0.5, 2.5, 1.0, 0.25)
as_cap = st.sidebar.toggle("Market-Cap")

raw["mid"]     = 10 ** mid_log
raw["support"] = 10 ** (mid_log - σ * k)
raw["resist"]  = 10 ** (mid_log + σ * k)

df = raw.copy()
y_title = "Price (USD)"
if as_cap:
    df[["Price", "mid", "support", "resist"]] *= FD_SUPPLY
    y_title = "Market-Cap (USD)"

price, sup, res = df.iloc[-1][["Price", "support", "resist"]]
zone = "🟢 Value" if price < sup else "🔴 Frothy" if price > res else "⚪ Neutral"
st.markdown(f"### **Current zone:** {zone}")

fig = go.Figure(layout=dict(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),
    xaxis=dict(type="date", title="Year", dtick=GRID
