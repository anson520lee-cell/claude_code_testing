"""python -m src.backtest.x80.misc CBOE_DIR EXTRA_PARQUET OUT_CSV : the remaining families.

1. option selling: Cboe PutWrite (PUT, WPUT weekly, PUTY 2% OTM), BuyWrite (BXM, BXMW) and Condor (CNDR) indices -
   fully collateralised option-writing benchmarks (real option prices, no model), levered 1x-3x daily at
   T-bill + 0.5%;
2. leveraged-ETF decay: short TQQQ and short SQQQ in equal dollars, weekly rebalanced; borrow fees are not in any
   public data set, so two assumptions (TQQQ 1% / SQQQ 5% a year, and 1% / 15%);
3. multi-asset trend following (time-series momentum) on 18 ETFs: monthly, sign of the 12-month excess return,
   each asset scaled to 40% / N vol, long and short, then the book scaled to 10% vol; levered 1x-4x;
4. "dual momentum" (an online favourite): each month hold the better of SPY / EFA by 12-month return if it beat
   T-bills, else IEF; 1x-3x.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, load, stats
from src.backtest.x80.equity_lev import book

PERIODS = {"1986-99": ("1986-01-01", "1999-12-31"), "2000-09": ("2000-01-01", "2009-12-31"), "2010-17": ("2010-01-01", "2017-12-31"),
           "2018-26": ("2018-01-01", "2030-01-01"), "2022-26": ("2022-01-01", "2030-01-01"), "all": ("1900-01-01", "2030-01-01")}
TREND = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "TIP", "GLD", "SLV", "DBC", "USO", "UUP", "FXE", "FXY", "VNQ"]


def cboe(path: str) -> pd.Series:
    d = pd.read_csv(path)
    d["DATE"] = pd.to_datetime(d["DATE"], format="%m/%d/%Y")
    s = d.set_index("DATE").iloc[:, 0].sort_index()
    s = s[~s.index.duplicated()]
    gaps = s.index.to_series().diff().dt.days
    start = gaps[gaps > 7].index.max() if (gaps > 7).any() else s.index[0]   # the early files have sparse months
    return s[s.index >= start].pct_change().dropna()


def main() -> None:
    cboe_dir, extra_path, out_path = sys.argv[1:4]
    px = load()
    ex = pd.read_parquet(extra_path)
    ex["date"] = pd.to_datetime(ex["date"])
    for s, g in ex.groupby("symbol"):
        px[s] = g.set_index("date").sort_index()[["open", "high", "low", "close", "adj_close", "volume"]]
    rows = []

    def add(family, name, lev, ret):
        cash = cash_daily(px, ret.index)
        for lab, (a, b) in PERIODS.items():
            rr = ret[(ret.index >= a) & (ret.index <= b)]
            st = stats(rr, cash)
            if st and st["years"] > 1.5:
                rows.append({"family": family, "rule": name, "lev": lev, "period": lab, **st})

    # 1. option-writing indices
    for name in ("PUT", "WPUT", "PUTY", "BXM", "BXMW", "CNDR"):
        r = cboe(os.path.join(cboe_dir, f"{name}.csv"))
        cash = cash_daily(px, r.index)
        for lev in (1.0, 2.0, 3.0):
            add("option_writing", name, lev, book(r, pd.Series(lev, index=r.index), cash))
    # 2. short TQQQ + short SQQQ
    rt, rs = px["TQQQ"]["adj_close"].pct_change(), px["SQQQ"]["adj_close"].pct_change()
    d = pd.concat({"t": rt, "s": rs}, axis=1).dropna()
    cash = cash_daily(px, d.index)
    for ft, fs in ((0.01, 0.05), (0.01, 0.15)):
        for gross in (1.0, 2.0):
            wt = ws = gross / 2
            ret, et, es = [], wt, ws
            for i, (dt, row) in enumerate(d.iterrows()):
                if i % 5 == 0:                                            # weekly back to equal dollars
                    turn = abs(et - wt) + abs(es - ws)
                    et, es = wt, ws
                else:
                    turn = 0.0
                pnl = -et * row["t"] - es * row["s"] + cash.loc[dt] * (1 + et + es) - (et * ft + es * fs) / 252 - turn * 0.0003
                et, es = et * (1 + row["t"]), es * (1 + row["s"])
                ret.append(pnl / 1.0)
                tot = 1 + pnl
                et, es = et / tot, es / tot
            add("letf_decay", f"short TQQQ+SQQQ borrow {ft:.0%}/{fs:.0%}", gross, pd.Series(ret, index=d.index).clip(lower=-1))
    # 3. multi-asset time-series momentum
    adj = pd.DataFrame({s: px[s]["adj_close"] for s in TREND if s in px}).sort_index()
    adj = adj[adj.index >= "2003-01-01"]
    r = adj.pct_change(fill_method=None)
    cash = cash_daily(px, r.index)
    me = adj.groupby(adj.index.to_period("M")).tail(1).index
    vol = r.rolling(60, min_periods=40).std() * np.sqrt(252)
    w = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    for dt in me:
        past = adj.loc[:dt]
        if len(past) < 260:
            continue
        ret12 = past.iloc[-1] / past.iloc[-253] - 1
        cash12 = (1 + cash.loc[:dt].iloc[-252:]).prod() - 1
        live = ret12.dropna().index
        n = len(live)
        if n < 5:
            continue
        sig = np.sign(ret12[live] - cash12)
        w.loc[w.index > dt, :] = 0.0
        w.loc[w.index > dt, live] = (sig * 0.40 / n / vol.loc[dt, live]).values
    port = (w.shift(0) * r.fillna(0)).sum(axis=1)
    port = port[port.index >= "2004-01-01"]
    scale = 0.10 / (port.rolling(252, min_periods=120).std() * np.sqrt(252)).shift(1).clip(lower=0.02)
    tsm = (port * scale.clip(upper=3)).dropna()
    turn_cost = (w.diff().abs().sum(axis=1) * 0.0003).reindex(tsm.index).fillna(0)
    tsm = tsm - turn_cost + cash.reindex(tsm.index).fillna(0)            # collateral earns T-bills
    for lev in (1.0, 2.0, 3.0, 4.0):
        add("trend_multi_asset", "TSMOM 18 ETFs (10% vol)", lev, book(tsm - cash.reindex(tsm.index).fillna(0) + 0, pd.Series(lev, index=tsm.index), cash.reindex(tsm.index)) if False else
            lev * (tsm - cash.reindex(tsm.index).fillna(0)) + cash.reindex(tsm.index).fillna(0) - (lev - 1) * 0.005 / 252)
    # 4. dual momentum
    a = pd.DataFrame({s: px[s]["adj_close"] for s in ("SPY", "EFA", "IEF")}).dropna()
    rr = a.pct_change()
    cash = cash_daily(px, rr.index)
    hold = pd.Series(index=rr.index, dtype=object)
    for dt in a.groupby(a.index.to_period("M")).tail(1).index:
        past = a.loc[:dt]
        if len(past) < 253:
            continue
        m12 = past.iloc[-1] / past.iloc[-253] - 1
        c12 = (1 + cash.loc[:dt].iloc[-252:]).prod() - 1
        best = "SPY" if m12["SPY"] >= m12["EFA"] else "EFA"
        hold[hold.index > dt] = best if m12[best] > c12 else "IEF"
    hold = hold.dropna()
    base = pd.Series([rr.loc[d, h] for d, h in hold.items()], index=hold.index)
    sw = (hold != hold.shift(1)).astype(float) * 0.0003
    for lev in (1.0, 2.0, 3.0):
        add("dual_momentum", "SPY/EFA/IEF dual momentum", lev, book(base - sw, pd.Series(lev, index=base.index), cash))
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    cols = [c for c in PERIODS if c in set(t["period"])]
    print("CAGR %\n", (t.pivot_table(index=["family", "rule", "lev"], columns="period", values="cagr")[cols] * 100).round(0).to_string())
    print("max DD %\n", (t.pivot_table(index=["family", "rule", "lev"], columns="period", values="max_dd")[cols] * 100).round(0).to_string())
    print("Sharpe\n", t[t["lev"] == 1.0].pivot_table(index=["family", "rule"], columns="period", values="sharpe")[cols].round(2).to_string())
    pd.DataFrame({"tsmom": tsm}).to_parquet(out_path.replace(".csv", "_tsmom.parquet"))


if __name__ == "__main__":
    main()
