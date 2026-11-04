"""python -m src.backtest.x80.crypto_carry BINANCE_DIR OUT_CSV : perpetual-futures funding carry on Binance USDT-M.

Long spot + short perpetual of the same coin: no price exposure, the short side collects funding when it is
positive (longs pay shorts) and pays it when negative. Daily P&L per unit of notional =
funding received that UTC day - change of the perp premium (perp vs spot) - trading costs
(0.15% per side per leg: 0.10% spot fee / 0.05% perp fee + slippage). At 1x the capital is the spot coin (no T-bill
income); leverage above 1x borrows USDT at T-bill + 4%/yr. All coins Binance ever listed are in the data, delisted ones included.
Rules: always-on BTC and ETH; a rotation that each day holds the 5 coins (spot turnover >= $20M a day, perp listed
>= 30 days) with the highest funding over the past 3 days, kept while they stay in the top 10; and a weekly version of
it (kept while in the top 15) because the daily one pays more in costs than it collects.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, load, stats

LEG_COST = 0.0015
PPY = 365
BORROW_SPREAD = 0.04
PERIODS = {"2020-21": ("2020-01-01", "2021-12-31"), "2022-23": ("2022-01-01", "2023-12-31"), "2024-26": ("2024-01-01", "2030-01-01"),
           "all": ("2020-01-01", "2030-01-01")}


def daily_inputs(bdir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f = pd.read_parquet(f"{bdir}/funding.parquet")
    f["date"] = pd.to_datetime(f["time"]).dt.floor("D")
    fund = f.groupby(["date", "symbol"])["rate"].sum().unstack().sort_index()
    p = pd.read_parquet(f"{bdir}/premium.parquet")
    prem = p.pivot_table(index="date", columns="symbol", values="close").sort_index()
    s = pd.read_parquet(f"{bdir}/spot.parquet", columns=["date", "symbol", "quote_volume"])
    qv = s.pivot_table(index="date", columns="symbol", values="quote_volume").sort_index()
    idx = pd.date_range("2020-01-01", max(fund.index.max(), prem.index.max()), freq="D")
    return fund.reindex(idx), prem.reindex(idx), qv.reindex(idx)


def carry_returns(fund: pd.DataFrame, prem: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    """Daily return per unit of capital at 1x notional for weights ``w`` (held over the day)."""
    dprem = prem.diff().clip(-0.05, 0.05)                              # perp vs spot: short perp gains when premium falls
    pnl = (fund.fillna(0.0) - dprem.fillna(0.0))
    turn = w.diff().abs().sum(axis=1)
    turn.iloc[0] = w.iloc[0].abs().sum()                                # the first day opens every position
    return (w * pnl.reindex(columns=w.columns)).sum(axis=1) - turn * 2 * LEG_COST


def main() -> None:
    bdir, out_path = sys.argv[1:3]
    fund, prem, qv = daily_inputs(bdir)
    px = load()
    cash = cash_daily(px, fund.index, PPY)
    series = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        w = pd.DataFrame(0.0, index=fund.index, columns=[sym])
        w[sym] = fund[sym].notna().astype(float)
        series[f"{sym[:-4]} always on"] = carry_returns(fund[[sym]], prem[[sym]], w)
    liq = qv.rolling(30, min_periods=20).median().shift(1)
    age = fund.notna().cumsum()
    score = fund.rolling(3, min_periods=3).sum().shift(1)              # known before the day starts
    cols = [c for c in fund.columns if c in liq.columns and c in prem.columns]   # premium was fetched for the liquid ones
    elig = (liq[cols] >= 2e7) & (age[cols] >= 30 * 3) & prem[cols].notna()
    sc = score[cols].where(elig)
    w = pd.DataFrame(0.0, index=fund.index, columns=cols)
    held: list[str] = []
    for d in fund.index:
        row = sc.loc[d].dropna()
        row = row[row > 0]
        top10 = list(row.nlargest(10).index)
        held = [h for h in held if h in top10]
        for c in row.nlargest(5).index:
            if len(held) >= 5:
                break
            if c not in held:
                held.append(c)
        if held:
            w.loc[d, held] = 1.0 / len(held)
    series["top-5 funding rotation"] = carry_returns(fund[cols], prem[cols], w)
    # lower-turnover design (the daily one pays more in costs than it collects): decide once a week, keep while top 15
    w2 = pd.DataFrame(0.0, index=fund.index, columns=cols)
    held, cur = [], pd.Series(dtype=float)
    for i, d in enumerate(fund.index):
        if i % 7 == 0:
            row = sc.loc[d].dropna()
            row = row[row > 0]
            top15 = list(row.nlargest(15).index)
            held = [h for h in held if h in top15]
            for c in row.nlargest(5).index:
                if len(held) >= 5:
                    break
                if c not in held:
                    held.append(c)
        if held:
            w2.loc[d, held] = 1.0 / len(held)
    series["top-5 funding, weekly"] = carry_returns(fund[cols], prem[cols], w2)
    rows = []
    for name, base in series.items():
        c = cash.reindex(base.index).fillna(0.0)
        for lev in (1.0, 2.0, 3.0, 5.0):
            ret = lev * base - (lev - 1) * (c + BORROW_SPREAD / PPY)     # capital sits in the spot coin: no T-bill income
            ret = ret.clip(lower=-1.0)
            for lab, (a, b) in PERIODS.items():
                rr = ret[(ret.index >= a) & (ret.index <= b)]
                st = stats(rr, cash, PPY)
                if st:
                    rows.append({"rule": name, "lev": lev, "period": lab, **st})
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    pd.DataFrame(series).to_parquet(out_path.replace(".csv", "_series.parquet"))
    pd.set_option("display.width", 250)
    print("CAGR %\n", (t.pivot_table(index=["rule", "lev"], columns="period", values="cagr")[list(PERIODS)] * 100).round(1).to_string())
    print("max DD %\n", (t.pivot_table(index=["rule", "lev"], columns="period", values="max_dd")[list(PERIODS)] * 100).round(1).to_string())
    yearly = pd.DataFrame({k: v.groupby(v.index.year).sum() * 100 for k, v in series.items()})
    print("funding carry by year, % of notional (before borrowing):\n", yearly.round(1).to_string())


if __name__ == "__main__":
    main()
