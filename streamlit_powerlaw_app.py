import io, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

UA = {"User-Agent": "btc-pl-tool/1.0"}
GENESIS   = pd.Timestamp("2009-01-03")
FD_SUPPLY = 21_000_000              # fully‑diluted BTC
GRID_D    = "M24"                   # vertical grid every 24 months

# ────────────────  three raw loaders  ────────────────
def _coinmetrics():
    url = ("https://api.coinmetrics.io/v4/timeseries/asset-metrics"
           "?assets=btc&metrics=PriceUSD&frequency=1d&start_time=2010-01-01")
    r = requests.get(url, headers=UA, timeout=15); r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df = df.rename(columns={"time": "Date", "PriceUSD": "Price"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Price"]]

def _coingecko():
    url = ("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
           "?vs_currency=usd&days=max&interval=daily")
    r = requests.get(url, headers=UA, timeout=15); r.raise_for_status()
    df = pd.DataFrame(r.json()["prices"], columns=["ts", "Price"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms")
    return df[["Date", "Price"]]

def _stooq():
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    date_col  = [c for c in df.columns if "date"  in c][0]
    price_col = [c for c in df.columns if "close" in c or "price" in c][0]
    clean = df[date_col].astype(str).str.replace(r"[^0-9\\-]", "", regex=True)
    df["Date"]  = pd.to_datetime(clean, format="%Y-%m-%d", errors="coerce")
    df["Price"] = pd.to_numeric(df[price_col], errors="coerce")
    return df.dropna(subset=["Date", "Price"]).sort_values("Date")[["Date", "Price"]]

# ─────────────── diagnostic wrapper ────────────────
def get_price_history() -> pd.DataFrame:
    sources = [
        ("CoinMetrics", _coinmetrics),
        ("CoinGecko",   _coingecko),
        ("Stooq CSV",   _stooq),
    ]
    for name, fn in sources:
        try:
            df = fn()
            if not df.empty:
                st.info(f"Loaded {len(df):,} rows from **{name}**")
                return df
        except Exception as e:
            st.warning(f"{name} failed → {e}")
    st.error("No price data from any source."); st.stop()

# ───────────── power‑law helpers ─────────────
def days_since_genesis(ts): return (ts - GENESIS).days

def fit_power(df):
    X = np.log10((df["Date"] - GENESIS).dt.days.values)
    y = np.log10(df["Price"].values)
    slope, intercept = np.polyfit(X, y, 1)
    mid_log = slope * X + intercept
    sigma   = np.std(y - mid_log)
    return mid_log, sigma

# ─────────────  Streamlit UI  ─────────────
st.set_page_config(page_title="BTC Purchase Indicator", layout="wide")

raw = get_price_history()                         # ← now defined
mid_log, σ = fit_power(raw)

# sidebar controls
k        = st.sidebar.slider("σ band width", 0.5, 2.5, 1.0, 0.25)
as_cap   = st.sidebar.toggle("Market‑Cap")        # your label

# compute bands
raw["mid"]     = 10 ** mid_log
raw["support"] = 10 ** (mid_log - σ * k)
raw["resist"]  = 10 ** (mid_log + σ * k)

df = raw.copy()
y_title = "Price (USD)"
if as_cap:
    cols = ["Price", "mid", "support", "resist"]
    df[cols] *= FD_SUPPLY
    y_title = "Market‑Cap (USD)"

# zone badge
p, s, r = df.iloc[-1][["Price", "support", "resist"]]
zone = "🟢 Value" if p < s else "🔴 Frothy" if p > r else "⚪ Neutral"
st.markdown(f"### **Current zone:** {zone}")

# build figure
fig = go.Figure()
fig.update_layout(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),
    xaxis=dict(type="date", title="Year", dtick=GRID_D,
               showgrid=True, gridwidth=0.5),
    yaxis=dict(type="log",  title=y_title,
               showgrid=True, gridwidth=0.5),
    plot_bgcolor="#111", paper_bgcolor="#111",
)
fig.add_trace(go.Scatter(x=df["Date"], y=df["Price"],
                         name="BTC", line=dict(color="gold", width=1.8)))
fig.add_trace(go.Scatter(x=df["Date"], y=df["mid"],
                         name="Mid‑line", line=dict(color="white", dash="dash")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["support"],
                         name="-σ", line=dict(color="green", dash="dash")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["resist"],
                         name="+σ", line=dict(color="red", dash="dash")))

# keep user zoom
if "xrange" in st.session_state:
    fig.update_xaxes(range=st.session_state["xrange"])

ev = plotly_events(fig, override_height=620, key="evt", click_event=False)
if ev and "xaxis.range[0]" in ev[0]:
    st.session_state["xrange"] = [ev[0]["xaxis.range[0]"], ev[0]["xaxis.range[1]"]]
