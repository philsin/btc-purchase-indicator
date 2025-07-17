# ─────────────────────────────────────────────────────────────
# streamlit_powerlaw_app.py  ·  BTC Purchase Indicator
# fixed ±σ bands & custom zones
# ─────────────────────────────────────────────────────────────
import io, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

UA        = {"User-Agent": "btc-pl-tool/1.0"}
GENESIS   = pd.Timestamp("2009-01-03")
FD_SUPPLY = 21_000_000
GRID_D    = "M24"          # vertical grid every 2 years

# ─── data loaders (Stooq → GitHub fallback) ─────────────────
def _stooq():
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    date_col  = [c for c in df.columns if "date"  in c][0]
    price_col = [c for c in df.columns if "close" in c or "price" in c][0]

    clean_date  = df[date_col].astype(str).str.replace(r"[^0-9\-]", "", regex=True)
    clean_price = (df[price_col].astype(str)
                               .str.replace(",", "", regex=False)
                               .str.replace(r"[^0-9.]", "", regex=True))

    df = pd.DataFrame({"Date": clean_date, "Price": clean_price})
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna().query("Price>0").sort_values("Date")

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

# ─── Streamlit page ──────────────────────────────────────────
st.set_page_config(page_title="BTC Purchase Indicator", layout="wide")

raw = get_price_history()
mid_log, σ = fit_power(raw)

#  fixed bands
σ_vis = max(σ, 0.25)
levels = {
    "Base":      -1.5,
    "Support":   -0.5,
    "PL Best Fit": 0.0,
    "Resistance": +1.0,
    "Top":       +2.0,
}
colors = {
    "Base":      "red",
    "Support":   "rgba(255,100,100,1)",   # light red
    "PL Best Fit": "white",
    "Resistance":"rgba(100,255,100,1)",   # light green
    "Top":       "green",
}
styles = {n: ("dash" if n != "PL Best Fit" else "dash") for n in levels}

for name, k in levels.items():
    raw[name] = 10 ** (mid_log + σ_vis * k)

# optional Market‑Cap toggle
as_cap = st.sidebar.toggle("Market‑Cap")
df = raw.copy()
y_title = "Price (USD)"
if as_cap:
    band_cols = list(levels.keys())
    df[["Price", *band_cols]] *= FD_SUPPLY
    y_title = "Market‑Cap (USD)"

# determine zone
p = df.iloc[-1]["Price"]
zones = [
    ("SELL THE HOUSE!!",          -np.inf,       -1.5),
    ("Buy",                       -1.5,          -0.5),
    ("DCA",                       -0.5,           1.0),
    ("Relax",                      1.0,           2.0),
    ("Frothy",                     2.0,         np.inf),
]
for label, lo, hi in zones:
    if lo*σ_vis + df.iloc[-1]["PL Best Fit"] < p <= hi*σ_vis + df.iloc[-1]["PL Best Fit"]:
        zone = label
        break
st.markdown(f"### **Current zone:** {zone}")

# ─── figure ──────────────────────────────────────────────────
fig = go.Figure(layout=dict(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),
    xaxis=dict(type="date", title="Year", dtick=GRID_D,
               showgrid=True, gridwidth=0.5),
    yaxis=dict(type="log",  title=y_title,
               tickformat="$,d",
               showgrid=True, gridwidth=0.5),
    plot_bgcolor="#111", paper_bgcolor="#111",
))

# plot bands first
for name, k in levels.items():
    if name == "PL Best Fit":
        continue
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df[name],
        name=name, line=dict(color=colors[name], dash=styles[name])
    ))

# mid-line (white) on top of other bands
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["PL Best Fit"],
    name="PL Best Fit", line=dict(color="white", dash="dash")
))

# BTC price LAST (foreground)
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Price"],
    name="BTC", line=dict(color="gold", width=3)
))

# keep zoom between reruns
if "xrange" in st.session_state:
    fig.update_xaxes(range=st.session_state["xrange"])

st.plotly_chart(fig, use_container_width=True)

ev = plotly_events(fig, select_event=False, click_event=False, key="zoom")
if ev and "xaxis.range[0]" in ev[0]:
    st.session_state["xrange"] = [ev[0]["xaxis.range[0]"],
                                  ev[0]["xaxis.range[1]"]]
# ─────────────────────────────────────────────────────────────
