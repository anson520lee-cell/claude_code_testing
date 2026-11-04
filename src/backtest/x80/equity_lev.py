"""python -m src.backtest.x80.equity_lev OUT_CSV : leveraged index ideas with long history (1985-2026).

Index total-return series: QQQ (from 1999-03) / SPY (from 1993) adjusted closes; before that ^NDX / ^GSPC price
plus a flat dividend accrual (0.5% / 3.0% a year). Leverage is daily-rebalanced (like a futures book or a
leveraged ETF): L x r - (L - 1) x (T-bill + 0.5%/yr) - 2 bp per unit of turnover. Ruin (a day below -100%) ends it.

Ideas (none of the plain 200-day timing / buy-and-hold TQQQ already covered by the reality check):
1. overnight only: buy the close, sell the next open (QQQ / SPY, real open prices), 1x-3x, 1.5 bp per side;
2. turn of the month: hold only the last trading day + first three of each month, 1x-4x;
3. volatility targeting: exposure = min(cap, target / 20-day realised vol); target and cap ranked on 1985-2009,
   read on 2010-26;
4. leveraged risk parity: S&P + long Treasuries (TLT, from 2002-07), 55/45 (the "HFEA" mix) or inverse-vol
   weights, monthly rebalanced, total leverage 1x-4x.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, load, stats

SPREAD = 0.005
COST = 0.0002
PERIODS = {"1985-99": ("1985-01-01", "1999-12-31"), "2000-09": ("2000-01-01", "2009-12-31"), "2010-17": ("2010-01-01", "2017-12-31"),
           "2018-26": ("2018-01-01", "2030-01-01"), "2022-26": ("2022-01-01", "2030-01-01"), "IS 1985-09": ("1985-01-01", "2009-12-31"),
           "OOS 2010-26": ("2010-01-01", "2030-01-01")}


def total_return(px: dict, etf: str, index: str, div: float) -> pd.Series:
    a = px[etf]["adj_close"].pct_change().dropna()
    b = px[index]["close"].pct_change().dropna() + div / 252
    return pd.concat([b[b.index < a.index[0]], a]).sort_index()


def book(r: pd.Series, expo: pd.Series, cash: pd.Series, cost: float = COST, spread: float = SPREAD) -> pd.Series:
    """Account return of exposure ``expo`` (already aligned: expo[t] is held over day t)."""
    e = expo.reindex(r.index).fillna(0.0)
    c = cash.reindex(r.index).fillna(0.0)
    drift = (e * (1 + r) / (1 + e * r) - e).abs()
    turn = e.diff().abs().fillna(e.abs()) + drift.shift(1).fillna(0.0)
    ret = e * r + (1 - e) * c - (e - 1).clip(lower=0) * spread / 252 - turn * cost
    ret = ret.clip(lower=-1.0)
    dead = (ret <= -1.0).cumsum().shift(1).fillna(0) > 0
    return ret.mask(dead, 0.0)


def overnight(px: dict, etf: str, cash: pd.Series, lev: float, side_cost: float = 0.00015) -> pd.Series:
    x = px[etf]
    f = x["adj_close"] / x["close"]
    ao, ac = x["open"] * f, x["adj_close"]
    on = (ao / ac.shift(1) - 1).dropna()
    c = cash.reindex(on.index).fillna(0.0)
    gap_days = on.index.to_series().diff().dt.days.fillna(1).clip(lower=1)
    fin = (lev - 1) * (c * 252 / 365 + SPREAD / 365) * gap_days        # borrowed money is paid per calendar night
    ret = lev * on - fin - 2 * side_cost * lev + 0 * c                 # out of the market in the day: no cash earned intraday
    return ret.clip(lower=-1.0)


def tom_mask(index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(index, index=index)
    month = index.to_period("M")
    pos_in_month = s.groupby(month).cumcount()
    from_end = s.groupby(month).cumcount(ascending=False)
    return ((pos_in_month <= 2) | (from_end == 0)).astype(float)


def main() -> None:
    out_path = sys.argv[1]
    px = load()
    rows = []

    def add(family, name, lev, ret, cash, extra=None):
        for lab, (a, b) in PERIODS.items():
            rr = ret[(ret.index >= a) & (ret.index <= b)]
            st = stats(rr, cash)
            if st and st["years"] > 1.5:
                rows.append({"family": family, "rule": name, "lev": lev, "period": lab, **(extra or {}), **st})

    for etf, idx, div in (("QQQ", "^NDX", 0.005), ("SPY", "^GSPC", 0.03)):
        r = total_return(px, etf, idx, div)
        cash = cash_daily(px, r.index)
        for lev in (1.0, 2.0, 3.0):
            add("buy_hold", f"{etf} buy&hold", lev, book(r, pd.Series(lev, index=r.index), cash), cash)
        # 1. overnight only (real opens: QQQ from 1999, SPY from 1993)
        for lev in (1.0, 2.0, 3.0):
            add("overnight", f"{etf} overnight", lev, overnight(px, etf, cash, lev), cash)
        # 2. turn of month (decided on the calendar, known in advance)
        m = tom_mask(r.index)
        for lev in (1.0, 2.0, 3.0, 4.0):
            add("turn_of_month", f"{etf} TOM", lev, book(r, m * lev, cash), cash, {"time_in": float(m.mean())})
        # 3. volatility targeting (exposure decided on the close before)
        rv = r.rolling(20).std() * np.sqrt(252)
        sma = (1 + r).cumprod()
        trend = (sma > sma.rolling(210).mean()).astype(float)
        for tv in (0.15, 0.20, 0.30, 0.40, 0.50):
            for cap in (2.0, 3.0, 4.0):
                e = (tv / rv).clip(upper=cap).shift(1)
                add("vol_target", f"{etf} vt{int(tv * 100)} cap{cap:g}", cap, book(r, e, cash), cash)
                add("vol_target+trend", f"{etf} vt{int(tv * 100)} cap{cap:g} +10m trend", cap, book(r, (e * trend.shift(1)), cash), cash)
    # 4. leveraged risk parity, 2002-07 onwards
    eq = px["SPY"]["adj_close"].pct_change()
    bd = px["TLT"]["adj_close"].pct_change()
    d = pd.concat({"eq": eq, "bd": bd}, axis=1).dropna()
    cash = cash_daily(px, d.index)
    vol = d.rolling(60).std()
    inv = (1 / vol).div((1 / vol).sum(axis=1), axis=0)
    month_start = d.index.to_series().dt.to_period("M").diff().ne(pd.offsets.MonthEnd(0) * 0)
    for wname, w in (("55/45", pd.DataFrame({"eq": 0.55, "bd": 0.45}, index=d.index)), ("inv-vol", inv)):
        w = w.where(month_start).ffill().shift(1).dropna()
        for lev in (1.0, 2.0, 3.0, 4.0):
            port = (w * d.reindex(w.index)).sum(axis=1)
            ret = book(port, pd.Series(lev, index=port.index), cash)
            add("risk_parity", f"SPY+TLT {wname}", lev, ret, cash)
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    show(t)


def show(t: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    cols = ["1985-99", "2000-09", "2010-17", "2018-26", "2022-26"]
    main_rules = t[t["family"].isin(["buy_hold", "overnight", "turn_of_month", "risk_parity"])]
    y = main_rules.pivot_table(index=["family", "rule", "lev"], columns="period", values="cagr")[cols]
    z = main_rules.pivot_table(index=["family", "rule", "lev"], columns="period", values="max_dd")[cols]
    print("CAGR %\n", (y * 100).round(0).to_string())
    print("max DD %\n", (z * 100).round(0).to_string())
    for fam in ("vol_target", "vol_target+trend"):
        v = t[t["family"] == fam]
        is_ = v[v["period"] == "IS 1985-09"].sort_values("cagr", ascending=False)
        for etf in ("QQQ", "SPY"):
            best = is_[is_["rule"].str.startswith(etf)].head(1)
            if not len(best):
                continue
            rule = best["rule"].iloc[0]
            o = v[(v["rule"] == rule)].set_index("period")[["cagr", "max_dd", "sharpe"]]
            print(f"\n{fam} {etf}: in-sample best by CAGR = {rule}\n", (o.loc[[p for p in ['IS 1985-09', 'OOS 2010-26', '2018-26', '2022-26'] if p in o.index]] * [100, 100, 1]).round(2).to_string())
    o80 = t[(t["period"] == "OOS 2010-26") & (t["cagr"] >= 0.8)]
    print("\nanything >= 80% CAGR in 2010-26:", len(o80))


if __name__ == "__main__":
    main()
