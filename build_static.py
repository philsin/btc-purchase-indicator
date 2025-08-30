# ───────────────────────────────────────────────
# BTC Purchase Indicator — Percentile Rails Version
# ───────────────────────────────────────────────
import os, json, math, requests
import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg

BTC_FILE = "data/btc_usd.csv"
GENESIS_DATE = pd.Timestamp("2009-01-03")
EPS_LOG = 0.015  # small buffer so bands don't touch

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────
def days_since_genesis(dates, start=GENESIS_DATE):
    return (pd.to_datetime(dates) - start).dt.days + 1.0

def load_series_csv(path):
    return pd.read_csv(path, parse_dates=["date"])

def robust_qfit(x_years: np.ndarray, y_series: np.ndarray, q: float = 0.5):
    """Quantile regression in log-log space"""
    m = (x_years > 0) & (y_series > 0) & np.isfinite(x_years) & np.isfinite(y_series)
    x = np.log10(x_years[m]); y = np.log10(y_series[m])
    X = pd.DataFrame({"const": 1.0, "logx": x})
    res = QuantReg(y, X).fit(q=q)
    a0 = float(res.params["const"]); b = float(res.params["logx"])
    resid = y - (a0 + b * x)
    return a0, b, resid, m

def suggest_defaults_for_series(dates, x_years, y_series, eps=EPS_LOG, half_window_days=90):
    """Auto percentiles mode: midline q50, rails q2.5/20/80/97.5 (parallel lines)"""
    a0, b, resid, mask = robust_qfit(x_years.values, y_series.values, q=0.5)
    c025 = float(np.nanquantile(resid, 0.025)) - eps
    c200 = float(np.nanquantile(resid, 0.200))
    c800 = float(np.nanquantile(resid, 0.800))
    c975 = float(np.nanquantile(resid, 0.975)) + eps

    # keep placeholder ranges for Date Ranges UI
    idx_all = np.where(mask)[0]
    if len(idx_all) < 2:
        ranges = {"ceil1":["2013-01-01","2013-12-31"],
                  "ceil2":["2017-01-01","2017-12-31"],
                  "floor":["2015-01-01","2015-12-31"]}
    else:
        d0 = dates.iloc[idx_all[0]].date()
        d1 = dates.iloc[idx_all[-1]].date()
        ranges = {"ceil1":[str(d0),str(d1)],
                  "ceil2":[str(d0),str(d1)],
                  "floor":[str(d0),str(d1)]}

    return {"a0":a0,"b":b,"c025":c025,"c200":c200,"c800":c800,"c975":c975,"ranges":ranges}

# ───────────────────────────────────────────────
# Build payload per denominator
# ───────────────────────────────────────────────
def build_payload(base, denom_key=None):
    y_main = base["btc"] if denom_key is None else base["btc"]/base[denom_key]
    y_label = "BTC / USD" if denom_key is None else f"BTC / {denom_key.upper()}"
    defaults = suggest_defaults_for_series(base["date"], base["x_days"], y_main)

    return {
        "label": y_label,
        "x_main": base["x_days"].tolist(),
        "y_main": y_main.tolist(),
        "date_iso_main": base["date"].astype(str).tolist(),
        "x_grid": base["x_days"].tolist(),
        "defaults": defaults
    }

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
if __name__ == "__main__":
    # fetch btc data if missing
    if not os.path.exists(BTC_FILE):
        os.makedirs(os.path.dirname(BTC_FILE), exist_ok=True)
        url = "https://raw.githubusercontent.com/datasets/bitcoin/master/data/bitcoin.csv"
        df = pd.read_csv(url, parse_dates=["Date"])
        df = df.rename(columns={"Date":"date","Closing Price (USD)":"btc"})
        df[["date","btc"]].to_csv(BTC_FILE,index=False)

    base = load_series_csv(BTC_FILE)
    base = base.sort_values("date").reset_index(drop=True)
    base["x_days"] = days_since_genesis(base["date"])

    PRECOMP = {"USD": build_payload(base, None)}

    html = f"""
    <html>
    <head><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>
    <body>
    <div id="chart" style="width:100%;height:95vh;"></div>
    <script>
    const PRECOMP = {json.dumps(PRECOMP)};
    function railsFromQuantiles(P){{
      const a0=P.defaults.a0, b=P.defaults.b;
      const c025=P.defaults.c025, c200=P.defaults.c200, c800=P.defaults.c800, c975=P.defaults.c975;
      const lx=P.x_grid.map(v=>Math.log10(v));
      const logM=lx.map(v=>a0+b*v);
      const logF=logM.map(v=>v+c025);
      const log20=logM.map(v=>v+c200);
      const log80=logM.map(v=>v+c800);
      const logC=logM.map(v=>v+c975);
      const exp10=a=>a.map(v=>Math.pow(10,v));
      return {{
        FLOOR:exp10(logF),P20:exp10(log20),P50:exp10(logM),P80:exp10(log80),CEILING:exp10(logC)
      }};
    }}
    const P=PRECOMP.USD; const rails=railsFromQuantiles(P);
    const traces=[
      {{x:P.x_main,y:P.y_main,name:P.label,mode:"lines",line:{{color:"black"}}}},
      {{x:P.x_grid,y:rails.FLOOR,name:"Floor",mode:"lines",line:{{color:"red"}}}},
      {{x:P.x_grid,y:rails.P20,name:"20%",mode:"lines",line:{{color:"orange",dash:"dot"}}}},
      {{x:P.x_grid,y:rails.P50,name:"50%",mode:"lines",line:{{color:"gold"}}}},
      {{x:P.x_grid,y:rails.P80,name:"80%",mode:"lines",line:{{color:"green",dash:"dot"}}}},
      {{x:P.x_grid,y:rails.CEILING,name:"Ceiling",mode:"lines",line:{{color:"darkgreen"}}}}
    ];
    Plotly.newPlot("chart",traces,{{
      title:"BTC Purchase Indicator — Rails (p≈auto)",
      xaxis:{{title:"Years",type:"log"}},
      yaxis:{{title:P.label,type:"log",tickformat:",.0f"}}
    }});
    </script>
    </body></html>
    """
    with open("index.html","w",encoding="utf-8") as f: f.write(html)
    print("index.html written.")