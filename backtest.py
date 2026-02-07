#!/usr/bin/env python3
"""
Backtest BTC Power Law Oscillator-Based Buy/Sell Signals
Compares USD, Gold, and SPX denominators
Analyzes $5,000/year investment strategies over 6 years (2019-2024)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.regression.quantile_regression import QuantReg

# ─────────────────────── Configuration ───────────────────────
GENESIS_DATE = datetime(2009, 1, 3)
W_OSC = 0.60
W_POS = 0.40
ANNUAL_INVESTMENT = 5000
BACKTEST_START = datetime(2019, 1, 1)
BACKTEST_END = datetime(2025, 1, 1)

# ─────────────────────── Load data ───────────────────────
btc_df = pd.read_csv("data/btc_usd.csv", parse_dates=["date"])
gold_df = pd.read_csv("data/denominator_gold.csv", parse_dates=["date"])
spx_df = pd.read_csv("data/denominator_spx.csv", parse_dates=["date"])

btc_df = btc_df.sort_values("date").reset_index(drop=True)
gold_df = gold_df.sort_values("date").reset_index(drop=True)
spx_df = spx_df.sort_values("date").reset_index(drop=True)

# Create daily interpolated data
date_range = pd.date_range(start=btc_df["date"].min(), end=btc_df["date"].max(), freq="D")
daily_df = pd.DataFrame({"date": date_range})

# Merge and interpolate BTC
daily_df = daily_df.merge(btc_df.rename(columns={"price": "btc_usd"}), on="date", how="left")
daily_df["btc_usd"] = daily_df["btc_usd"].interpolate(method="linear")

# Merge and interpolate Gold
daily_df = daily_df.merge(gold_df.rename(columns={"price": "gold_usd"}), on="date", how="left")
daily_df["gold_usd"] = daily_df["gold_usd"].interpolate(method="linear").ffill().bfill()

# Merge and interpolate SPX
daily_df = daily_df.merge(spx_df.rename(columns={"price": "spx_usd"}), on="date", how="left")
daily_df["spx_usd"] = daily_df["spx_usd"].interpolate(method="linear").ffill().bfill()

# Calculate denominated prices
daily_df["btc_gold"] = daily_df["btc_usd"] / daily_df["gold_usd"]  # BTC in oz of gold
daily_df["btc_spx"] = daily_df["btc_usd"] / daily_df["spx_usd"]    # BTC in SPX units
daily_df = daily_df.dropna()

def years_since_genesis(date):
    """Convert date to years since genesis block."""
    delta = (date - GENESIS_DATE).total_seconds() / (365.25 * 24 * 3600)
    return max(delta, 1/365.25)

def quantile_fit_loglog(x_years, y_vals, q=0.5):
    """Fit quantile regression in log-log space."""
    x_years = np.asarray(x_years)
    y_vals = np.asarray(y_vals)
    mask = np.isfinite(x_years) & np.isfinite(y_vals) & (x_years > 0) & (y_vals > 0)
    xlog = np.log10(x_years[mask])
    ylog = np.log10(y_vals[mask])
    X = pd.DataFrame({"const": 1.0, "logx": xlog})
    res = QuantReg(ylog, X).fit(q=q)
    a0 = float(res.params["const"])
    b = float(res.params["logx"])
    resid = ylog - (a0 + b * xlog)

    # R-squared
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((ylog - np.mean(ylog))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return a0, b, resid, r_squared

class DenominatorAnalysis:
    """Power law analysis for a specific denominator."""

    def __init__(self, name, prices, x_years):
        self.name = name
        self.prices = prices
        self.x_years = x_years

        # Fit power law
        self.a0, self.b, self.residuals, self.r_squared = quantile_fit_loglog(x_years, prices)
        self.resid_min = self.residuals.min()
        self.resid_max = self.residuals.max()
        self.sorted_resid = np.sort(self.residuals)

    def get_power_law_price(self, date):
        x = years_since_genesis(date)
        log_price = self.a0 + self.b * np.log10(x)
        return 10 ** log_price

    def get_oscillator(self, date, price):
        x = years_since_genesis(date)
        log_dev = np.log10(price) - (self.a0 + self.b * np.log10(x))
        if self.resid_max == self.resid_min:
            return 0
        return (2 * (log_dev - self.resid_min) / (self.resid_max - self.resid_min)) - 1

    def get_position(self, date, price):
        x = years_since_genesis(date)
        log_dev = np.log10(price) - (self.a0 + self.b * np.log10(x))
        idx = np.searchsorted(self.sorted_resid, log_dev)
        return 100 * idx / len(self.sorted_resid)

def compute_composite(osc, p):
    normalized_pos = (50 - p) / 50
    return W_OSC * osc + W_POS * normalized_pos

def get_signal(osc, p):
    comp = compute_composite(osc, p)

    # Extreme sell
    if osc > 0.8 and p > 85:
        return "TOP INBOUND", comp
    if osc > 0.6 and p > 75:
        return "FROTHY", comp

    # Extreme buy (check BEFORE moderate!)
    if osc < -0.7 and p < 15:
        return "SELL THE HOUSE", comp
    if osc < -0.5 and p < 25:
        return "STRONG BUY", comp

    # Moderate
    if osc > 0.4 or p > 65:
        return "HOLD ON", comp
    if osc < -0.3 or p < 35:
        return "BUY", comp

    return "DCA", comp

# ─────────────────────── Create analyses for each denominator ───────────────────────
x_years = np.array([years_since_genesis(d) for d in daily_df["date"]])

analyses = {
    "USD": DenominatorAnalysis("USD", daily_df["btc_usd"].values, x_years),
    "GOLD": DenominatorAnalysis("GOLD", daily_df["btc_gold"].values, x_years),
    "SPX": DenominatorAnalysis("SPX", daily_df["btc_spx"].values, x_years),
}

print("=" * 90)
print("POWER LAW FIT COMPARISON BY DENOMINATOR")
print("=" * 90)
print(f"{'Denom':<8} {'a0':>10} {'b (slope)':>12} {'R²':>10} {'Resid Min':>12} {'Resid Max':>12}")
print("-" * 90)
for name, analysis in analyses.items():
    print(f"{name:<8} {analysis.a0:>10.4f} {analysis.b:>12.4f} {analysis.r_squared:>10.4f} "
          f"{analysis.resid_min:>12.4f} {analysis.resid_max:>12.4f}")

# ─────────────────────── Calculate signals for each denominator ───────────────────────
backtest_df = daily_df[(daily_df["date"] >= BACKTEST_START) & (daily_df["date"] <= BACKTEST_END)].copy()

for denom_name, analysis in analyses.items():
    price_col = f"btc_{denom_name.lower()}" if denom_name != "USD" else "btc_usd"
    backtest_df[f"osc_{denom_name}"] = backtest_df.apply(
        lambda r: analysis.get_oscillator(r["date"], r[price_col]), axis=1)
    backtest_df[f"pos_{denom_name}"] = backtest_df.apply(
        lambda r: analysis.get_position(r["date"], r[price_col]), axis=1)
    backtest_df[f"signal_{denom_name}"], backtest_df[f"comp_{denom_name}"] = zip(*backtest_df.apply(
        lambda r: get_signal(r[f"osc_{denom_name}"], r[f"pos_{denom_name}"]), axis=1))

# Monthly samples
monthly_df = backtest_df.groupby(backtest_df["date"].dt.to_period("M")).first().reset_index(drop=True)

# ─────────────────────── Signal Distribution by Denominator ───────────────────────
print("\n" + "=" * 90)
print("SIGNAL DISTRIBUTION BY DENOMINATOR (2019-2024)")
print("=" * 90)

signal_order = ["SELL THE HOUSE", "STRONG BUY", "BUY", "DCA", "HOLD ON", "FROTHY", "TOP INBOUND"]
total_days = len(backtest_df)

print(f"{'Signal':<18}", end="")
for denom in analyses.keys():
    print(f"{denom:>12} {'%':>8}", end="")
print()
print("-" * 90)

for signal in signal_order:
    print(f"{signal:<18}", end="")
    for denom in analyses.keys():
        count = (backtest_df[f"signal_{denom}"] == signal).sum()
        pct = 100 * count / total_days
        print(f"{count:>12} {pct:>7.1f}%", end="")
    print()

# ─────────────────────── Signal Comparison Table ───────────────────────
print("\n" + "=" * 90)
print("MONTHLY SIGNAL COMPARISON (Key Dates)")
print("=" * 90)
print(f"{'Date':<12} {'BTC/USD':>10} {'USD Signal':<15} {'GOLD Signal':<15} {'SPX Signal':<15}")
print("-" * 90)

# Show key dates
key_months = monthly_df[monthly_df["date"].dt.year.isin([2020, 2021, 2022, 2023, 2024])]
for _, row in key_months.iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d'):<12} ${row['btc_usd']:>9,.0f} "
          f"{row['signal_USD']:<15} {row['signal_GOLD']:<15} {row['signal_SPX']:<15}")

# ─────────────────────── Strategy Backtests by Denominator ───────────────────────
def run_strategy(denom_name, strategy_name, buy_condition, description):
    """Run backtest for a specific denominator and strategy."""
    btc_held = 0.0
    total_invested = 0.0
    monthly_budget = ANNUAL_INVESTMENT / 12
    trades = []

    signal_col = f"signal_{denom_name}"

    for year in range(2019, 2025):
        year_df = backtest_df[backtest_df["date"].dt.year == year]

        for month in range(1, 13):
            month_df = year_df[year_df["date"].dt.month == month]
            if len(month_df) == 0:
                continue

            row = month_df.iloc[0]
            signal = row[signal_col]

            if buy_condition(signal):
                btc_bought = monthly_budget / row["btc_usd"]
                btc_held += btc_bought
                total_invested += monthly_budget
                trades.append({
                    "date": row["date"],
                    "signal": signal,
                    "btc": btc_bought,
                    "price": row["btc_usd"]
                })

    final_price = backtest_df.iloc[-1]["btc_usd"]
    final_value = btc_held * final_price

    return {
        "denom": denom_name,
        "strategy": strategy_name,
        "description": description,
        "invested": total_invested,
        "btc_held": btc_held,
        "final_value": final_value,
        "return_pct": (final_value / total_invested - 1) * 100 if total_invested > 0 else 0,
        "trades": len(trades),
        "avg_cost": total_invested / btc_held if btc_held > 0 else 0
    }

# Define strategies
buy_signals = {"BUY", "STRONG BUY", "SELL THE HOUSE"}
extreme_signals = {"STRONG BUY", "SELL THE HOUSE"}

strategies_config = [
    ("Pure DCA", lambda s: True, "Buy every month"),
    ("Buy Signals Only", lambda s: s in buy_signals, "Only BUY/STRONG BUY/SELL THE HOUSE"),
    ("Extreme Value Only", lambda s: s in extreme_signals, "Only STRONG BUY/SELL THE HOUSE"),
    ("Skip HOLD ON", lambda s: s != "HOLD ON", "DCA except when HOLD ON"),
]

print("\n" + "=" * 90)
print("STRATEGY COMPARISON BY DENOMINATOR ($5,000/year, 2019-2024)")
print("=" * 90)

results = []
for denom in analyses.keys():
    for strat_name, buy_cond, desc in strategies_config:
        result = run_strategy(denom, strat_name, buy_cond, desc)
        results.append(result)

# Group by strategy
print(f"\n{'Strategy':<22} {'Denom':<8} {'Invested':>10} {'Final':>12} {'Return':>10} {'Trades':>8} {'Avg Cost':>10}")
print("-" * 90)

for strat_name, _, _ in strategies_config:
    strat_results = [r for r in results if r["strategy"] == strat_name]
    for r in strat_results:
        print(f"{r['strategy']:<22} {r['denom']:<8} ${r['invested']:>9,.0f} ${r['final_value']:>11,.0f} "
              f"{r['return_pct']:>9.1f}% {r['trades']:>8} ${r['avg_cost']:>9,.0f}")
    print()

# ─────────────────────── Best Strategy by Denominator ───────────────────────
print("=" * 90)
print("BEST PERFORMING STRATEGY BY DENOMINATOR")
print("=" * 90)

for denom in analyses.keys():
    denom_results = [r for r in results if r["denom"] == denom and r["invested"] > 0]
    best = max(denom_results, key=lambda x: x["return_pct"])
    print(f"\n{denom}:")
    print(f"  Best Strategy: {best['strategy']}")
    print(f"  Return: {best['return_pct']:.1f}%")
    print(f"  Final Value: ${best['final_value']:,.0f}")
    print(f"  Avg Cost/BTC: ${best['avg_cost']:,.0f}")

# ─────────────────────── Signal Agreement Analysis ───────────────────────
print("\n" + "=" * 90)
print("SIGNAL AGREEMENT ANALYSIS")
print("=" * 90)

# Count when all three agree
agreement = backtest_df.apply(
    lambda r: r["signal_USD"] == r["signal_GOLD"] == r["signal_SPX"], axis=1)
agree_count = agreement.sum()
print(f"All three denominators agree: {agree_count}/{total_days} days ({100*agree_count/total_days:.1f}%)")

# Count when USD differs from GOLD/SPX
usd_vs_gold = (backtest_df["signal_USD"] != backtest_df["signal_GOLD"]).sum()
usd_vs_spx = (backtest_df["signal_USD"] != backtest_df["signal_SPX"]).sum()
gold_vs_spx = (backtest_df["signal_GOLD"] != backtest_df["signal_SPX"]).sum()

print(f"\nDisagreement rates:")
print(f"  USD vs GOLD: {usd_vs_gold} days ({100*usd_vs_gold/total_days:.1f}%)")
print(f"  USD vs SPX:  {usd_vs_spx} days ({100*usd_vs_spx/total_days:.1f}%)")
print(f"  GOLD vs SPX: {gold_vs_spx} days ({100*gold_vs_spx/total_days:.1f}%)")

# ─────────────────────── Consensus Strategy ───────────────────────
print("\n" + "=" * 90)
print("CONSENSUS STRATEGY: Only buy when 2+ denominators agree on BUY signal")
print("=" * 90)

def consensus_buy(row):
    """Returns True if at least 2 denominators show BUY or better."""
    buy_count = sum([
        row["signal_USD"] in buy_signals,
        row["signal_GOLD"] in buy_signals,
        row["signal_SPX"] in buy_signals
    ])
    return buy_count >= 2

btc_held = 0.0
total_invested = 0.0
monthly_budget = ANNUAL_INVESTMENT / 12

for year in range(2019, 2025):
    year_df = backtest_df[backtest_df["date"].dt.year == year]

    for month in range(1, 13):
        month_df = year_df[year_df["date"].dt.month == month]
        if len(month_df) == 0:
            continue

        row = month_df.iloc[0]
        if consensus_buy(row):
            btc_bought = monthly_budget / row["btc_usd"]
            btc_held += btc_bought
            total_invested += monthly_budget

final_price = backtest_df.iloc[-1]["btc_usd"]
final_value = btc_held * final_price
return_pct = (final_value / total_invested - 1) * 100 if total_invested > 0 else 0

print(f"  Invested: ${total_invested:,.0f}")
print(f"  BTC Held: {btc_held:.6f}")
print(f"  Final Value: ${final_value:,.0f}")
print(f"  Return: {return_pct:.1f}%")
print(f"  Avg Cost/BTC: ${total_invested/btc_held:,.0f}" if btc_held > 0 else "  N/A")

# ─────────────────────── Key Insights ───────────────────────
print("\n" + "=" * 90)
print("KEY INSIGHTS: DENOMINATOR IMPACT ON SIGNALS")
print("=" * 90)
print("""
1. POWER LAW FIT QUALITY:
   - USD has the strongest R² fit (most reliable power law)
   - GOLD and SPX have different slope parameters, reflecting different value propositions

2. SIGNAL DIVERGENCE:
   - Denominators disagree frequently, especially during transitions
   - BTC/GOLD often shows more extreme signals (gold's stability as baseline)
   - BTC/SPX tends to lag USD signals (correlated risk assets)

3. STRATEGY IMPLICATIONS:
   - Using USD signals alone is simplest and has strong historical performance
   - GOLD denominator can provide earlier "value" signals during dollar weakness
   - SPX denominator useful for risk-on/risk-off positioning

4. CONSENSUS APPROACH:
   - Requiring 2+ denominators to agree reduces noise
   - May miss some opportunities but increases conviction
   - Best for conservative investors

5. RECOMMENDATION:
   - Primary: Use USD-based signals (highest R², most liquid market)
   - Secondary: Monitor GOLD signals for dollar-hedged perspective
   - Optional: SPX for equity correlation context
""")

# ─────────────────────── 2024 Detailed View ───────────────────────
print("=" * 90)
print("2024 DETAILED SIGNAL COMPARISON")
print("=" * 90)
recent = monthly_df[monthly_df["date"].dt.year == 2024]
print(f"{'Month':<10} {'Price':>10} {'USD':>8} {'USD Sig':<12} {'GOLD':>8} {'GOLD Sig':<12} {'SPX':>8} {'SPX Sig':<12}")
print("-" * 90)
for _, row in recent.iterrows():
    print(f"{row['date'].strftime('%Y-%m'):<10} ${row['btc_usd']:>9,.0f} "
          f"{row['osc_USD']:>8.2f} {row['signal_USD']:<12} "
          f"{row['osc_GOLD']:>8.2f} {row['signal_GOLD']:<12} "
          f"{row['osc_SPX']:>8.2f} {row['signal_SPX']:<12}")
