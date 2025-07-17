def fetch_prices() -> pd.DataFrame:
    """
    Returns daily BTC/USD closes from 2010‑01‑01 to today.
    Tries CoinMetrics first; if that 429/403s or returns != 200,
    we fall back to CoinGecko.
    """
    import io, requests

    cm_url = (
        "https://api.coinmetrics.io/v4/timeseries/asset-metrics"
        "?assets=btc&metrics=PriceUSD&frequency=1d"
        "&start_time=2010-01-01"
    )
    try:
        r = requests.get(cm_url, timeout=10)
        r.raise_for_status()
        csv_bytes = r.content
    except requests.exceptions.RequestException:
        # --- fallback to CoinGecko ---
        cg = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        r = requests.get(f"{cg}?vs_currency=usd&days=max&interval=daily", timeout=10)
        r.raise_for_status()
        data = r.json()["prices"]        # list of [ms_epoch, price]
        df = pd.DataFrame(data, columns=["ts", "price"])
        df["Date"] = pd.to_datetime(df["ts"], unit="ms").dt.tz_localize(None)
        df["Price"] = df["price"].astype(float)
        return df[["Date", "Price"]]

    # CoinMetrics returns plaintext CSV
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df["Date"] = pd.to_datetime(df["time"])
    df["Price"] = df["PriceUSD"].astype(float)
    return df[["Date", "Price"]]
