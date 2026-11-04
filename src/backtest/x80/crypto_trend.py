"""python -m src.backtest.x80.crypto_trend OUT_CSV : time-series trend rules on BTC / ETH (Yahoo daily, 00:00 UTC closes).

Honest set-up:
* every rule is decided on the close of day t and earns day t+1 (no same-bar look-ahead);
* 0.15% per side (exchange fee 0.10% + slippage 0.05%) on every change of exposure, including the daily
  rebalancing a constant-leverage position needs;
* idle cash earns the T-bill rate, borrowed money costs T-bill + 5%/yr (crypto margin / perp funding is
  usually dearer than that), a day that loses more than the equity ends the account (ruin);
* parameters are ranked on the early period only (BTC 2014-09..2019, ETH 2017-11..2019) and the
  in-sample winner is then read on 2020-26, next to the median of all rules (what an average pick gets).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, load, stats

PPY = 365
COST = 0.0015
FIN_SPREAD = 0.05
IS_END = "2019-12-31"
PERIODS = {"IS": ("2000-01-01", IS_END), "OOS 2020-26": ("2020-01-01", "2030-01-01"), "2020-21": ("2020-01-01", "2021-12-31"),
           "2022-23": ("2022-01-01", "2023-12-31"), "2024-26": ("2024-01-01", "2030-01-01"),
           "2022-26": ("2022-01-01", "2030-01-01")}


def signals(close: pd.Series) -> dict[str, pd.Series]:
    """0/1 long-or-cash exposure decided at each close."""
    out: dict[str, pd.Series] = {}
    for n in (5, 10, 20, 30, 50, 75, 100, 150, 200):
        out[f"sma{n}"] = (close > close.rolling(n).mean()).astype(float)
    for f, s in ((5, 20), (10, 30), (10, 50), (20, 50), (20, 100), (50, 200)):
        out[f"ema{f}_{s}"] = (close.ewm(span=f, min_periods=s).mean() > close.ewm(span=s, min_periods=s).mean()).astype(float)
    for n, m in ((20, 10), (20, 20), (55, 20), (50, 25), (100, 50)):
        hi, lo = close.rolling(n).max().shift(1), close.rolling(m).min().shift(1)
        st = pd.Series(np.nan, index=close.index)
        st[close > hi] = 1.0
        st[close < lo] = 0.0
        out[f"don{n}_{m}"] = st.ffill().fillna(0.0)
    for k in (7, 14, 30, 60, 90, 120, 180, 365):
        out[f"mom{k}"] = (close.pct_change(k) > 0).astype(float)
    return out


def run_exposure(r: pd.Series, expo: pd.Series, cash: pd.Series, cost: float = COST, fin_spread: float = FIN_SPREAD,
                 ppy: int = PPY) -> pd.Series:
    """Daily account return of holding ``expo`` (decided at the previous close) x the asset."""
    e = expo.shift(1).fillna(0.0)
    c = cash.reindex(r.index).fillna(0.0)
    drift = (e * (1 + r) / (1 + e * r) - e).abs()                  # exposure drift a constant-leverage book must trade back
    turn = e.diff().abs().fillna(e.abs()) + drift.shift(1).fillna(0.0)
    borrow, idle = (e - 1).clip(lower=0), (1 - e).clip(lower=0)
    ret = e * r + idle * c - borrow * (c + fin_spread / ppy) - turn * cost
    ret = ret.clip(lower=-1.0)
    dead = (ret <= -1.0).cumsum().shift(1).fillna(0) > 0           # ruined accounts stay at zero
    return ret.mask(dead, 0.0)


def main() -> None:
    out_path = sys.argv[1]
    px = load()
    rows = []
    for asset in ("BTC-USD", "ETH-USD"):
        close = px[asset]["close"]
        r = close.pct_change().dropna()
        cash = cash_daily(px, r.index, PPY)
        sig = signals(close)
        variants = {"buy_hold": pd.Series(1.0, index=close.index)}
        variants.update(sig)
        rv = r.rolling(30).std() * np.sqrt(PPY)
        for name, s in sig.items():                                  # vol-targeted versions of each rule
            variants[f"{name}|vt80"] = s * (0.8 / rv).clip(upper=3.0).reindex(s.index).fillna(0.0)
        for name, s in variants.items():
            for lev in ((1.0, 1.5, 2.0, 3.0) if "|vt" not in name else (1.0,)):
                ret = run_exposure(r, s.reindex(r.index).fillna(0.0) * lev, cash)
                trades = (s.diff().abs() > 0).sum() / (len(s) / PPY)
                for lab, (a, b) in PERIODS.items():
                    rr = ret[(ret.index >= a) & (ret.index <= b)]
                    st = stats(rr, cash, PPY)
                    if st:
                        rows.append({"asset": asset, "rule": name, "lev": lev, "period": lab, "switches_per_yr": trades, **st})
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    show(t)


def show(t: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    for asset in t["asset"].unique():
        x = t[(t["asset"] == asset)]
        is_ = x[(x["period"] == "IS") & (x["lev"] == 1.0) & (x["rule"] != "buy_hold")]
        best = is_.sort_values("sharpe", ascending=False).head(3)["rule"].tolist()
        print(f"\n== {asset}: in-sample top-3 by Sharpe {best}")
        keep = ["buy_hold"] + best
        y = x[x["rule"].isin(keep)].pivot_table(index=["rule", "lev"], columns="period", values=["cagr", "max_dd"])
        print((y * 100).round(0).to_string())
        o = x[(x["period"] == "OOS 2020-26") & (x["lev"] == 1.0) & (x["rule"] != "buy_hold")]
        print("OOS 2020-26, all 1x rules: median CAGR %.0f%%, median maxDD %.0f%%, share with CAGR > buy&hold %.0f%%" % (
            o["cagr"].median() * 100, o["max_dd"].median() * 100,
            (o["cagr"] > x[(x["rule"] == "buy_hold") & (x["lev"] == 1.0) & (x["period"] == "OOS 2020-26")]["cagr"].iloc[0]).mean() * 100))
        o80 = x[(x["period"] == "OOS 2020-26") & (x["cagr"] >= 0.8)]
        print("OOS 2020-26 variants with CAGR >= 80%:", len(o80), "of", len(x[x["period"] == "OOS 2020-26"]))
        if len(o80):
            print(o80.sort_values("cagr", ascending=False).head(10)[["rule", "lev", "cagr", "max_dd", "sharpe"]].round(2).to_string())


if __name__ == "__main__":
    main()
