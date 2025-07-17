import io, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

UA       = {"User-Agent": "btc-pl-tool/1.0"}
GENESIS  = pd.Timestamp("2009-01-03")
SUPPLY   = 21_000_000
GRID_D   = "M24"                     # vertical grid every 24 months

# ─────────────── data loaders ───────────────
def _coinmetrics() -> pd.DataFrame:
    url = ("https://api.coinmetrics.io/v4/timeseries/asset-metrics"
           "?assets=btc&metrics=PriceUSD&frequency=1d&start_time=2010-01-01")
    r = requests.get(url, headers=UA, timeout=15); r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df = df.rename(columns={"time": "Date", "PriceUSD": "Price"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Price"]]

def _coingecko() -> pd.DataFrame:
    url = ("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
           "?vs_currency=usd&days=max&interval=daily")
    r = requests.get(url, headers=UA, timeout=15); r.raise_for_status()
    data = r.json()["prices"]
    df = pd.DataFrame(data, columns=["ts", "Price"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms")
    return df[["Date", "Price"]]

def _stooq() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]

    date_col  = [c for c in df.columns if "date"  in c][0]
    price_col = [c for c in df.columns if "close" in c or "price" in c][0]

    clean_dates = (
        df[date_col].astype(str)
        .str.replace(r"[^0-9\-]", "", regex=True)           # keep digits & ASCII hyphen
    )
    df["Date"]  = pd.to_datetime(clean_dates, format="%Y-%m-%d", errors="coerce")
    df["Price"] = pd.to_numeric(df[price_col], errors="coerce")
    return df.dropna(subset=["Date", "Price"]).sort_values("Date")[["Date", "Price"]]

@st.cache_data(ttl=3600, show_spinner="Downloading price history…")
def load_prices() -> pd.DataFrame:
    for fn in (_coinmetrics, _coingecko, _stooq):
        try:
            return fn()
        except Exception:
            continue
    st.error("All data sources failed — please retry later."); st.stop()

# ───────────── power‑law helpers ─────────────
def days_since_genesis(ts): return (ts - GENESIS).days

def fit_power_law(df):
    X = np.log10(df["days"].values); y = np.log10(df["Price"].values)
    B, A = np.polyfit(X, y, 1)        # y = A + B·log10(days)
    mid_log = A + B * X
    sigma   = np.std(y - mid_log)
    mid     = 10 ** mid_log
    return mid, mid_log, sigma

# ─────────────── Streamlit UI ───────────────
st.set_page_config(page_title="BTC Purchase Indicator", layout="wide")

df = load_prices()
df["days"] = df["Date"].apply(days_since_genesis)
mid, mid_log, sigma = fit_power_law(df)
df["mid"]     = mid
df["support"] = 10 ** (mid_log - sigma)
df["resist"]  = 10 ** (mid_log + sigma)

# sidebar
sigma_mult = st.sidebar.slider("σ band width", 0.5, 2.5, 1.0, 0.25)
use_mcap   = st.sidebar.toggle("Market‑Cap")      # your new label

# scale toggle
if use_mcap:
    df_plot = df.assign(
        Price   = df["Price"]   * SUPPLY,
        mid     = df["mid"]     * SUPPLY,
        support = df["support"] * SUPPLY,
        resist  = df["resist"]  * SUPPLY,
    )
    y_title = "Market‑Cap (USD)"
else:
    df_plot = df.copy(); y_title = "Price (USD)"

# zone badge
price_now   = df_plot["Price"].iloc[-1]
support_now = df_plot["support"].iloc[-1] * sigma_mult
resist_now  = df_plot["resist"].iloc[-1]  * sigma_mult
zone = ("🟢 Value" if price_now < support_now else
        "🔴 Frothy" if price_now > resist_now else
        "⚪ Neutral")
st.markdown(f"### **Current zone:** {zone}")

# figure
fig = go.Figure()
fig.update_layout(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),
    xaxis=dict(title="Year", type="date", showgrid=True, gridwidth=0.5,
               tick0="2010-01-01", dtick=GRID_D),
    yaxis=dict(title=y_title, type="log", showgrid=True, gridwidth=0.5),
    plot_bgcolor="#111", paper_bgcolor="#111",
)

fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Price"],
                         name="BTC", line=dict(color="gold", width=1.8)))
fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["mid"],
                         name="Mid‑line", line=dict(color="white", dash="dash")))
fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["support"]*sigma_mult,
                         name="-σ", line=dict(color="green", dash="dash")))
fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["resist"]*sigma_mult,
                         name="+σ", line=dict(color="red",   dash="dash")))

# keep zoom
if "xrange" not in st.session_state: st.session_state["xrange"] = None
sel = plotly_events(fig, override_height=600, key="pl_events")
if sel and "xaxis.range[0]" in sel[0]:
    st.session_state["xrange"] = [sel[0]["xaxis.range[0]"], sel[0]["xaxis.range[1]"]]
if st.session_state["xrange"]:
    fig.update_xaxes(range=st.session_state["xrange"])

st.plotly_chart(fig, use_container_width=True)
