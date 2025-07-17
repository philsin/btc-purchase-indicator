import io, requests, pandas as pd

UA = {"User-Agent": "btc-pl-tool/1.0"}

def fetch_prices() -> pd.DataFrame:
    """Return BTC daily closes (USD). Two‑stage fallback."""
    # 1️⃣ CoinMetrics CSV
    cm_url = (
        "https://api.coinmetrics.io/v4/timeseries/asset-metrics"
        "?assets=btc&metrics=PriceUSD&frequency=1d&start_time=2010-01-01"
    )
    try:
        r = requests.get(cm_url, timeout=15, headers=UA)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        df = df.rename(columns={"time": "Date", "PriceUSD": "Price"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df[["Date", "Price"]]
    except requests.RequestException:
        pass  # fall through

    # 2️⃣ CoinGecko JSON (add UA)
    try:
        cg_url = (
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            "?vs_currency=usd&days=max&interval=daily"
        )
        r = requests.get(cg_url, timeout=15, headers=UA)
        r.raise_for_status()
        data = r.json()["prices"]
        df = pd.DataFrame(data, columns=["ts", "Price"])
        df["Date"] = pd.to_datetime(df["ts"], unit="ms")
        return df[["Date", "Price"]]
    except requests.RequestException:
        pass  # fall through

    # 3️⃣ Static GitHub CSV mirror (never blocks)
    gh_raw = (
        "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    )
    df = pd.read_csv(gh_raw)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Closing Price (USD)"]].rename(columns={"Closing Price (USD)": "Price"})
