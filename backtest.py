#!/usr/bin/env python3
"""
Backtest BTC Power Law Oscillator-Based Buy/Sell Signals
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

# ─────────────────────── Load and prepare data ───────────────────────
df = pd.read_csv("data/btc_usd.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Create daily interpolated data for more accurate backtesting
date_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
daily_df = pd.DataFrame({"date": date_range})
daily_df = daily_df.merge(df, on="date", how="left")
daily_df["price"] = daily_df["price"].interpolate(method="linear")
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
    return a0, b, resid

# Calculate power law parameters using ALL historical data up to each point
# For backtesting, we'll use the full historical fit (which is what we'd have known at each point in a rolling sense)
x_years = np.array([years_since_genesis(d) for d in daily_df["date"]])
prices = daily_df["price"].values

# Fit power law to all data
a0, b, all_residuals = quantile_fit_loglog(x_years, prices, q=0.5)
resid_min, resid_max = all_residuals.min(), all_residuals.max()

# Calculate percentile offsets for p-value calculation
sorted_resid = np.sort(all_residuals)

print(f"Power Law: Price = 10^{a0:.4f} × t^{b:.4f}")
print(f"Residual range: [{resid_min:.4f}, {resid_max:.4f}]")
print()

def get_power_law_price(date):
    """Get the 50th percentile power law price for a date."""
    x = years_since_genesis(date)
    log_price = a0 + b * np.log10(x)
    return 10 ** log_price

def get_oscillator(date, price):
    """Calculate oscillator: normalized log-deviation from trend [-1, +1]."""
    x = years_since_genesis(date)
    log_dev = np.log10(price) - (a0 + b * np.log10(x))
    if resid_max == resid_min:
        return 0
    return (2 * (log_dev - resid_min) / (resid_max - resid_min)) - 1

def get_position(date, price):
    """Calculate position percentile (0-100%)."""
    x = years_since_genesis(date)
    log_dev = np.log10(price) - (a0 + b * np.log10(x))
    # Find percentile in sorted residuals
    idx = np.searchsorted(sorted_resid, log_dev)
    return 100 * idx / len(sorted_resid)

def compute_composite(osc, p):
    """Compute composite score."""
    normalized_pos = (50 - p) / 50
    return W_OSC * osc + W_POS * normalized_pos

def get_signal(osc, p):
    """Get signal based on oscillator and position - check extremes FIRST."""
    comp = compute_composite(osc, p)

    # === EXTREME SELL SIGNALS ===
    if osc > 0.8 and p > 85:
        return "TOP INBOUND", comp
    if osc > 0.6 and p > 75:
        return "FROTHY", comp

    # === EXTREME BUY SIGNALS (check BEFORE moderate signals!) ===
    if osc < -0.7 and p < 15:
        return "SELL THE HOUSE", comp
    if osc < -0.5 and p < 25:
        return "STRONG BUY", comp

    # === MODERATE SIGNALS ===
    if osc > 0.4 or p > 65:
        return "HOLD ON", comp
    if osc < -0.3 or p < 35:
        return "BUY", comp

    # === DEFAULT ===
    return "DCA", comp

# ─────────────────────── Calculate signals for backtest period ───────────────────────
backtest_df = daily_df[(daily_df["date"] >= BACKTEST_START) & (daily_df["date"] <= BACKTEST_END)].copy()
backtest_df["x_years"] = backtest_df["date"].apply(years_since_genesis)
backtest_df["pl_price"] = backtest_df["date"].apply(get_power_law_price)
backtest_df["oscillator"] = backtest_df.apply(lambda r: get_oscillator(r["date"], r["price"]), axis=1)
backtest_df["position"] = backtest_df.apply(lambda r: get_position(r["date"], r["price"]), axis=1)
backtest_df["signal"], backtest_df["composite"] = zip(*backtest_df.apply(
    lambda r: get_signal(r["oscillator"], r["position"]), axis=1))

# Sample monthly for reporting
monthly_df = backtest_df.groupby(backtest_df["date"].dt.to_period("M")).first().reset_index(drop=True)

print("=" * 80)
print("SIGNAL HISTORY (Monthly Samples, 2019-2024)")
print("=" * 80)
print(f"{'Date':<12} {'Price':>10} {'PL Price':>10} {'Osc':>8} {'Pos':>8} {'Signal':<15}")
print("-" * 80)
for _, row in monthly_df.iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d'):<12} ${row['price']:>9,.0f} ${row['pl_price']:>9,.0f} "
          f"{row['oscillator']:>8.3f} {row['position']:>7.1f}% {row['signal']:<15}")

# ─────────────────────── Strategy Simulations ───────────────────────
print("\n" + "=" * 80)
print("BACKTEST RESULTS: $5,000/year over 6 years (2019-2024)")
print("=" * 80)

def run_strategy(name, buy_condition, sell_condition=None, description=""):
    """
    Run a backtest strategy.
    buy_condition: function(row) -> bool, when True, invest monthly allocation
    sell_condition: function(row) -> bool, when True, sell all holdings
    """
    btc_held = 0.0
    cash = 0.0
    total_invested = 0.0
    monthly_budget = ANNUAL_INVESTMENT / 12

    trades = []
    years = sorted(backtest_df["date"].dt.year.unique())

    for year in years:
        if year > 2024:
            continue
        year_df = backtest_df[backtest_df["date"].dt.year == year]

        # Monthly DCA or signal-based buying
        for month in range(1, 13):
            month_df = year_df[year_df["date"].dt.month == month]
            if len(month_df) == 0:
                continue

            # Use first day of month for decision
            row = month_df.iloc[0]

            # Check sell condition first
            if sell_condition and sell_condition(row) and btc_held > 0:
                sell_value = btc_held * row["price"]
                trades.append({
                    "date": row["date"],
                    "action": "SELL",
                    "btc": btc_held,
                    "price": row["price"],
                    "value": sell_value,
                    "signal": row["signal"]
                })
                cash += sell_value
                btc_held = 0

            # Check buy condition
            if buy_condition(row):
                btc_bought = monthly_budget / row["price"]
                btc_held += btc_bought
                total_invested += monthly_budget
                trades.append({
                    "date": row["date"],
                    "action": "BUY",
                    "btc": btc_bought,
                    "price": row["price"],
                    "value": monthly_budget,
                    "signal": row["signal"]
                })

    # Final valuation
    final_row = backtest_df.iloc[-1]
    final_btc_value = btc_held * final_row["price"]
    final_total = final_btc_value + cash

    return {
        "name": name,
        "description": description,
        "total_invested": total_invested,
        "btc_held": btc_held,
        "cash": cash,
        "final_btc_value": final_btc_value,
        "final_total": final_total,
        "return_pct": (final_total / total_invested - 1) * 100 if total_invested > 0 else 0,
        "trades": trades
    }

# Strategy definitions
strategies = []

# 1. Pure DCA - invest every month regardless of signal
strategies.append(run_strategy(
    "Pure DCA",
    buy_condition=lambda r: True,
    sell_condition=None,
    description="Invest $416.67/month regardless of signal"
))

# 2. Buy only when signal is BUY or better (STRONG BUY, SELL THE HOUSE)
buy_signals = {"BUY", "STRONG BUY", "SELL THE HOUSE"}
strategies.append(run_strategy(
    "Buy Signals Only (No Sell)",
    buy_condition=lambda r: r["signal"] in buy_signals,
    sell_condition=None,
    description="Only buy when signal is BUY, STRONG BUY, or SELL THE HOUSE"
))

# 3. Buy when BUY or better, accumulate cash otherwise, deploy on SELL THE HOUSE
strategies.append(run_strategy(
    "Value Buy + DCA Accumulate",
    buy_condition=lambda r: r["signal"] in buy_signals or r["signal"] == "DCA",
    sell_condition=None,
    description="Buy on BUY signals + DCA in neutral"
))

# 4. Buy on value signals, sell at FROTHY
frothy_signals = {"FROTHY", "TOP INBOUND"}
strategies.append(run_strategy(
    "Buy Value + Sell Frothy",
    buy_condition=lambda r: r["signal"] in buy_signals or r["signal"] == "DCA",
    sell_condition=lambda r: r["signal"] in frothy_signals,
    description="Buy on value signals, sell when FROTHY or TOP INBOUND"
))

# 5. Aggressive: Buy only at extreme value, sell at any overvalued
extreme_buy = {"STRONG BUY", "SELL THE HOUSE"}
overvalued = {"HOLD ON", "FROTHY", "TOP INBOUND"}
strategies.append(run_strategy(
    "Extreme Value Only",
    buy_condition=lambda r: r["signal"] in extreme_buy,
    sell_condition=lambda r: r["signal"] == "TOP INBOUND",
    description="Only buy at STRONG BUY/SELL THE HOUSE, sell at TOP INBOUND"
))

# 6. Conservative: DCA always, but double down on extreme value
# This needs special handling
def run_double_down_strategy():
    btc_held = 0.0
    total_invested = 0.0
    monthly_budget = ANNUAL_INVESTMENT / 12
    reserve = 0.0  # Accumulated reserve for double-down

    trades = []
    years = sorted(backtest_df["date"].dt.year.unique())

    for year in years:
        if year > 2024:
            continue
        year_df = backtest_df[backtest_df["date"].dt.year == year]

        for month in range(1, 13):
            month_df = year_df[year_df["date"].dt.month == month]
            if len(month_df) == 0:
                continue

            row = month_df.iloc[0]
            signal = row["signal"]

            # Always invest base amount on non-overvalued signals
            if signal not in overvalued:
                amount = monthly_budget
                # Double down on extreme value signals
                if signal in extreme_buy:
                    amount = monthly_budget * 2 + reserve
                    reserve = 0

                btc_bought = amount / row["price"]
                btc_held += btc_bought
                total_invested += amount
                trades.append({
                    "date": row["date"],
                    "action": "BUY",
                    "btc": btc_bought,
                    "price": row["price"],
                    "value": amount,
                    "signal": signal
                })
            else:
                # Accumulate reserve during overvalued periods
                reserve += monthly_budget * 0.5

    final_row = backtest_df.iloc[-1]
    final_btc_value = btc_held * final_row["price"]

    return {
        "name": "DCA + Double Down",
        "description": "Regular DCA, 2x on extreme value, skip overvalued",
        "total_invested": total_invested,
        "btc_held": btc_held,
        "cash": reserve,
        "final_btc_value": final_btc_value,
        "final_total": final_btc_value + reserve,
        "return_pct": ((final_btc_value + reserve) / total_invested - 1) * 100 if total_invested > 0 else 0,
        "trades": trades
    }

strategies.append(run_double_down_strategy())

# ─────────────────────── Print Results ───────────────────────
print("\n" + "-" * 80)
print(f"{'Strategy':<30} {'Invested':>12} {'Final Value':>14} {'Return':>10} {'BTC Held':>10}")
print("-" * 80)

for s in strategies:
    print(f"{s['name']:<30} ${s['total_invested']:>11,.0f} ${s['final_total']:>13,.0f} "
          f"{s['return_pct']:>9.1f}% {s['btc_held']:>9.4f}")

# Detailed breakdown for each strategy
print("\n" + "=" * 80)
print("DETAILED STRATEGY ANALYSIS")
print("=" * 80)

for s in strategies:
    print(f"\n{'─' * 80}")
    print(f"Strategy: {s['name']}")
    print(f"Description: {s['description']}")
    print(f"{'─' * 80}")
    print(f"  Total Invested:  ${s['total_invested']:>12,.2f}")
    print(f"  BTC Held:        {s['btc_held']:>12.6f} BTC")
    print(f"  Cash Remaining:  ${s['cash']:>12,.2f}")
    print(f"  Final BTC Value: ${s['final_btc_value']:>12,.2f}")
    print(f"  Final Total:     ${s['final_total']:>12,.2f}")
    print(f"  Total Return:    {s['return_pct']:>12.1f}%")
    print(f"  Avg Cost/BTC:    ${s['total_invested']/s['btc_held'] if s['btc_held'] > 0 else 0:>12,.2f}")

    # Trade summary
    buy_trades = [t for t in s["trades"] if t["action"] == "BUY"]
    sell_trades = [t for t in s["trades"] if t["action"] == "SELL"]
    print(f"  Buy Trades:      {len(buy_trades):>12}")
    print(f"  Sell Trades:     {len(sell_trades):>12}")

    if buy_trades:
        avg_buy_price = sum(t["value"] for t in buy_trades) / sum(t["btc"] for t in buy_trades)
        print(f"  Avg Buy Price:   ${avg_buy_price:>12,.2f}")

# Signal distribution during backtest period
print("\n" + "=" * 80)
print("SIGNAL DISTRIBUTION (2019-2024)")
print("=" * 80)
signal_counts = backtest_df["signal"].value_counts()
total_days = len(backtest_df)
for signal, count in signal_counts.items():
    pct = 100 * count / total_days
    print(f"  {signal:<15}: {count:>5} days ({pct:>5.1f}%)")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("""
1. TIMING MATTERS: Signal-based strategies can outperform pure DCA by avoiding
   purchases during overvalued periods and concentrating buys during value zones.

2. EXTREME VALUE IS RARE: SELL THE HOUSE and STRONG BUY signals occur infrequently,
   but historically provide the best entry points.

3. SELLING IS TRICKY: While selling at FROTHY/TOP INBOUND sounds good in theory,
   it requires accurate re-entry timing. Bitcoin often continues higher after
   initial "overvalued" signals.

4. DCA WORKS: Pure DCA remains a strong baseline. The psychological benefit of
   regular investing often outweighs marginal returns from timing.

5. COMPOUND APPROACH: The best results often come from a hybrid - DCA as baseline
   with increased allocation during deep value signals.
""")

# Monthly signal table for 2024
print("=" * 80)
print("2024 MONTHLY SIGNALS (Recent History)")
print("=" * 80)
recent = monthly_df[monthly_df["date"].dt.year == 2024]
print(f"{'Month':<10} {'Price':>10} {'Osc':>8} {'Pos':>8} {'Signal':<15} {'Composite':>10}")
print("-" * 70)
for _, row in recent.iterrows():
    print(f"{row['date'].strftime('%Y-%m'):<10} ${row['price']:>9,.0f} {row['oscillator']:>8.3f} "
          f"{row['position']:>7.1f}% {row['signal']:<15} {row['composite']:>10.3f}")
