# ─────────────────────────────────────────────────────────────
# streamlit_powerlaw_app.py  ·  BTC Purchase Indicator
# cleaned – commas on log axis
# ─────────────────────────────────────────────────────────────
import io, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

UA       = {"User-Agent": "btc-pl-tool/1.0"}
GENESIS  = pd.Timestamp("2009-01-03")
FD_SUPPLY = 21_000_000
GRID_D   = "M24"

# ─── data loaders (Stooq → GitHub fallback) ─────────────────
def _stooq():
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    date_col  = [c for c in df.columns if "date" in c][0]
    price_col = [c for c in df.columns if "close" in c or "price" in c][0]

    clean_date  = df[date_col].astype(str).str.replace(r"[^0-9\-]", "", regex=True)
    clean_price = (df[price_col].astype(str)
                               .str.replace(",", "", regex=False)
                               .str.replace(r"[^0-9.]", "", regex=True))

    df = pd.DataFrame({"Date": clean_date, "Price": clean_price})
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna().query("Price > 0").sort_values("Date")

def _github():
    raw = ("https://raw.githubusercontent.com/datasets/bitcoin-price/"
           "master/data/bitcoin_price.csv")
    df = pd.read_csv(raw).rename(columns={"Closing Price (USD)": "Price"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Price"]]

def get_price_history():
    try:
        df = _stooq()
        if len(df) > 1000:
            st.info(f"Loaded {len(df):,} rows from **Stooq CSV**")
            return df
        st.warning("Stooq returned <1000 rows, falling back to GitHub")
    except Exception as e:
        st.warning(f"Stooq failed → {e}")

    df = _github()
    st.info(f"Loaded {len(df):,} rows from **GitHub mirror**")
    return df

# ─── power‑law fit ───────────────────────────────────────────
def fit_power(df):
    X = np.log10((df["Date"] - GENESIS).dt.days)
    y = np.log10(df["Price"])
    slope, intercept = np.polyfit(X, y, 1)
    mid_log = slope * X + intercept
    sigma   = np.std(y - mid_log)
    return mid_log, sigma

# ─── Streamlit layout ────────────────────────────────────────
st.set_page_config(page_title="BTC Purchase Indicator", layout="wide")

raw = get_price_history()
mid_log, sigma = fit_power(raw)

# sidebar
k      = st.sidebar.slider("σ band width", 0.5, 2.5, 1.0, 0.25)
as_cap = st.sidebar.toggle("Market‑Cap")

# bands
sigma_vis = max(sigma, 0.25)
raw["mid"]     = 10 ** mid_log
raw["support"] = 10 ** (mid_log - sigma_vis * k)
raw["resist"]  = 10 ** (mid_log + sigma_vis * k)

df = raw.copy()
y_title = "Price (USD)"
if as_cap:
    df[["Price", "mid", "support", "resist"]] *= FD_SUPPLY
    y_title = "Market‑Cap (USD)"

# zone badge
p, s, r = df.iloc[-1][["Price", "support", "resist"]]
zone = "🟢 Value" if p < s else "🔴 Frothy" if p > r else "⚪ Neutral"
st.markdown(f"### **Current zone:** {zone}")

# ─── figure ──────────────────────────────────────────────────
fig = go.Figure(layout=dict(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),
    xaxis=dict(type="date", title="Year", dtick=GRID_D,
               showgrid=True, gridwidth=0.5),
    yaxis=dict(type="log",  title=y_title, tickformat="~,d",  # ← comma format
               showgrid=True, gridwidth=0.5),
    plot_bgcolor="#111", paper_bgcolor="#111",
))

# bands first
fig.add_trace(go.Scatter(x=df["Date"], y=df["mid"],
                         name="Mid‑line", line=dict(color="white", dash="dash")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["support"],
                         name="-σ", line=dict(color="green", dash="dash")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["resist"],
                         name="+σ", line=dict(color="red", dash="dash")))

# BTC last
