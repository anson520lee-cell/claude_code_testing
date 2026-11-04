"""python -m src.backtest.x80.crypto_ls BINANCE_DIR OUT_CSV : long Bitcoin / short altcoins, and short new listings.

Only coins that could really be shorted that day are used (a Binance USDT-M perpetual existed, so the coin has a
funding print); the short side pays or receives that coin's funding. Universe each week: perp listed, spot
history >= 60 days, 30-day median spot turnover >= $5M, top N by turnover, BTC excluded; delisted coins included
(a short is closed at the last close). Weekly rebalance, 0.10% per side per leg. The capital is the BTC position
(no T-bill income); 2x borrows at T-bill + 4%.
1. BTC vs alts: long 1x BTC spot, short 1x an equal-weight basket of the top 10 / 20 / 50 alts (dollar neutral);
   also a version with the short sized to the basket's trailing 90-day beta to BTC (beta neutral).
2. Short new listings: short every coin in its 30th-180th day after its spot listing (perp available), equal weight,
   against long BTC.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, load, stats
from src.backtest.x80.crypto_xs import panel

COST = 0.001
PPY = 365
PERIODS = {"2020-21": ("2020-01-01", "2021-12-31"), "2022-23": ("2022-01-01", "2023-12-31"), "2024-26": ("2024-01-01", "2030-01-01"),
           "2022-26": ("2022-01-01", "2030-01-01"), "all": ("2020-01-01", "2030-01-01")}


def run(close, r, fund, pick_fn, beta_neutral: bool, cash, gate: pd.Series | None = None) -> pd.Series:
    dates = close.index[close.index >= "2020-01-01"]
    reb = dates[::7]
    btc = r["BTCUSDT"]
    w_prev = pd.Series(dtype=float)
    out = []
    for i, d in enumerate(reb):
        prev = d - pd.Timedelta(days=1)
        alts = pick_fn(prev)
        if len(alts) < 5 or (gate is not None and not bool(gate.get(prev, False))):
            w = pd.Series({"BTCUSDT": 0.0})
        else:
            hs = 1.0
            if beta_neutral:
                past = r.loc[prev - pd.Timedelta(days=90):prev]
                basket = past[alts].mean(axis=1)
                b = past["BTCUSDT"]
                var = b.var()
                hs = float(np.clip(basket.cov(b) / var, 0.5, 2.0)) if var > 0 else 1.0
                hs = 1.0 / hs
            w = pd.concat([pd.Series({"BTCUSDT": 1.0}), pd.Series(-hs / len(alts), index=alts)])
        turn = w.sub(w_prev, fill_value=0).abs().sum()
        w_prev = w
        end = reb[i + 1] if i + 1 < len(reb) else dates[-1] + pd.Timedelta(days=1)
        span = r.loc[(r.index >= d) & (r.index < end)]
        if span.empty:
            continue
        rr = span.reindex(columns=w.index).fillna(0.0)
        fr = fund.reindex(index=span.index, columns=w.index).fillna(0.0)
        day = rr.mul(w, axis=1).sum(axis=1) - fr.mul(w, axis=1).sum(axis=1)   # a short (w<0) receives positive funding
        day.iloc[0] -= turn * COST
        out.append(day)
    return pd.concat(out).clip(lower=-1.0)


def main() -> None:
    bdir, out_path = sys.argv[1:3]
    close, qv = panel(f"{bdir}/spot.parquet")
    close = close[close.index >= "2019-06-01"]
    qv = qv.reindex(close.index)
    f = pd.read_parquet(f"{bdir}/funding.parquet")
    f["date"] = pd.to_datetime(f["time"]).dt.floor("D")
    fund = f.groupby(["date", "symbol"])["rate"].sum().unstack().reindex(close.index)
    r = close.pct_change(fill_method=None).where(close.shift(1).notna() & close.notna())
    age = close.notna().cumsum()
    first = close.apply(lambda s: s.first_valid_index())
    liq = qv.rolling(30, min_periods=20).median()
    has_perp = fund.reindex(columns=close.columns).notna().rolling(3, min_periods=1).max().astype(bool)
    px = load()
    cash = cash_daily(px, close.index, PPY)

    def top_alts(n):
        def fn(d):
            ok = (age.loc[d] >= 60) & (liq.loc[d] >= 5e6) & has_perp.loc[d] & close.loc[d].notna()
            ok["BTCUSDT"] = False
            return list(liq.loc[d][ok].nlargest(n).index)
        return fn

    def new_listings(d):
        days = (d - first).dt.days
        ok = (days >= 30) & (days <= 180) & has_perp.loc[d] & close.loc[d].notna() & (liq.loc[d] >= 2e6)
        ok["BTCUSDT"] = False
        return list(ok[ok].index)

    series = {}
    for n in (10, 20, 50):
        for bn in (False, True):
            series[f"long BTC / short top{n} alts{' (beta neutral)' if bn else ''}"] = run(close, r, fund, top_alts(n), bn, cash)
    series["long BTC / short new listings (day 30-180)"] = run(close, r, fund, new_listings, False, cash)
    # trend filter on the BTC-vs-alts spread itself: hold the pair only while the spread is above its N-day average
    uni = pd.DataFrame(False, index=close.index, columns=close.columns)
    for d in close.index[close.index >= "2019-12-01"]:
        uni.loc[d, top_alts(50)(d)] = True
    spread = (r["BTCUSDT"] - r.where(uni.shift(1).fillna(False).astype(bool)).mean(axis=1)).fillna(0.0)
    idx = (1 + spread).cumprod()
    for n in (30, 60, 90):
        gate = idx > idx.rolling(n).mean()
        series[f"long BTC / short top50 alts + spread > {n}d avg"] = run(close, r, fund, top_alts(50), False, cash, gate)
    b = r["BTCUSDT"].fillna(0.0)
    series["BTC buy&hold"] = b[b.index >= "2020-01-01"]
    rows = []
    for name, ret in series.items():
        for lev in (1.0, 2.0):
            c = cash.reindex(ret.index).fillna(0.0)
            rr = (lev * ret - (lev - 1) * (c + 0.04 / PPY)).clip(lower=-1.0) if lev != 1.0 else ret   # capital is in BTC: no T-bill
            dead = (rr <= -1.0).cumsum().shift(1).fillna(0) > 0
            rr = rr.mask(dead, 0.0)
            for lab, (a, z) in PERIODS.items():
                x = rr[(rr.index >= a) & (rr.index <= z)]
                st = stats(x, cash, PPY)
                if st:
                    rows.append({"rule": name, "lev": lev, "period": lab, **st})
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    pd.DataFrame(series).to_parquet(out_path.replace(".csv", "_series.parquet"))
    pd.set_option("display.width", 250)
    print("CAGR %\n", (t.pivot_table(index=["rule", "lev"], columns="period", values="cagr")[list(PERIODS)] * 100).round(0).to_string())
    print("max DD %\n", (t.pivot_table(index=["rule", "lev"], columns="period", values="max_dd")[list(PERIODS)] * 100).round(0).to_string())
    print("Sharpe 1x\n", t[t["lev"] == 1.0].pivot_table(index="rule", columns="period", values="sharpe")[list(PERIODS)].round(2).to_string())


if __name__ == "__main__":
    main()
