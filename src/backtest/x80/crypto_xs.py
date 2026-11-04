"""python -m src.backtest.x80.crypto_xs SPOT_PARQUET OUT_CSV : cross-sectional crypto momentum on Binance USDT pairs.

Universe is point-in-time and includes every coin Binance later delisted (LUNA, FTT, ...): on each date the coins
with >= 60 days of history whose 30-day median turnover is >= $5M, top 50 by turnover; the first 14 days after a
listing are skipped. Weekly rebalance on the 00:00 UTC close using returns up to the previous close.
Rules: long the top 5 / top 10 by past 7 / 14 / 28 / 56-day return, equal weight; optional BTC trend filter
(hold cash when BTC <= its 50-day average); a coin that stops trading is sold at its last close.
Costs 0.2% per side (fee 0.1% + slippage on alts). Lookbacks ranked on 2018-19, read on 2020-26.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, load, stats

COST = 0.002
PPY = 365
PERIODS = {"IS 2018-19": ("2018-01-01", "2019-12-31"), "OOS 2020-26": ("2020-01-01", "2030-01-01"), "2020-21": ("2020-01-01", "2021-12-31"),
           "2022-23": ("2022-01-01", "2023-12-31"), "2024-26": ("2024-01-01", "2030-01-01")}


def panel(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(path)
    d = d[d["close"] > 0]
    close = d.pivot_table(index="date", columns="symbol", values="close").sort_index()
    qv = d.pivot_table(index="date", columns="symbol", values="quote_volume").sort_index()
    close = close.asfreq("D")
    qv = qv.reindex(close.index)
    return close, qv


def universe(close: pd.DataFrame, qv: pd.DataFrame, top: int = 50, min_qv: float = 5e6) -> pd.DataFrame:
    age = close.notna().cumsum()
    liq = qv.rolling(30, min_periods=20).median()
    ok = (age >= 60) & (liq >= min_qv) & close.notna()
    rank = liq.where(ok).rank(axis=1, ascending=False)
    return ok & (rank <= top)


def backtest(close: pd.DataFrame, uni: pd.DataFrame, look: int, k: int, btc_filter: bool, cash: pd.Series) -> pd.Series:
    r = close.pct_change(fill_method=None)
    r = r.where(close.shift(1).notna() & close.notna())               # no return across gaps / after delisting
    mom = close / close.shift(look) - 1
    btc = close["BTCUSDT"]
    trend = btc > btc.rolling(50).mean()
    dates = close.index[close.index >= "2018-01-01"]
    reb = dates[::7]
    w_prev = pd.Series(dtype=float)
    out = []
    for i, d in enumerate(reb):
        prev = d - pd.Timedelta(days=1)
        if prev not in close.index:
            continue
        cand = mom.loc[prev][uni.loc[prev]].dropna()
        hold = not btc_filter or bool(trend.loc[prev])
        if hold and len(cand) >= k:
            pick = cand.nlargest(k).index
            w = pd.Series(1.0 / k, index=pick)
        else:
            w = pd.Series(dtype=float)
        turn = w.sub(w_prev, fill_value=0).abs().sum()
        w_prev = w
        end = reb[i + 1] if i + 1 < len(reb) else dates[-1] + pd.Timedelta(days=1)
        span = r.loc[(r.index >= d) & (r.index < end)]
        if span.empty:
            continue
        c = cash.reindex(span.index).fillna(0.0)
        if len(w):
            rr = span[w.index].fillna(0.0)                               # delisted: flat after the last close
            day = rr.mul(w, axis=1).sum(axis=1) + (1 - w.sum()) * c
        else:
            day = c.copy()
        day.iloc[0] -= turn * COST
        out.append(day)
    return pd.concat(out)


def main() -> None:
    spot, out_path = sys.argv[1:3]
    close, qv = panel(spot)
    uni = universe(close, qv)
    px = load()
    cash = cash_daily(px, close.index, PPY)
    print("universe size by year:", uni.sum(axis=1).groupby(uni.index.year).mean().round(0).to_dict())
    n_del, n_ever = int((uni.any() & close.iloc[-30:].isna().all()).sum()), int(uni.any().sum())
    print("delisted coins that were ever in the universe:", n_del, "of", n_ever)
    import json
    json.dump({"delisted": n_del, "ever": n_ever, "symbols": int(close.shape[1])}, open(out_path.replace(".csv", "_universe.json"), "w"))
    rows, series = [], {}
    btc = close["BTCUSDT"].pct_change(fill_method=None)
    ew = close.pct_change(fill_method=None).where(uni.shift(1).fillna(False).astype(bool)).mean(axis=1)
    series["BTC buy&hold"] = btc[btc.index >= "2018-01-01"].fillna(0.0)
    series["top-50 equal weight (daily)"] = ew[ew.index >= "2018-01-01"].fillna(0.0)
    for look in (7, 14, 28, 56):
        for k in (5, 10):
            for filt in (False, True):
                series[f"mom{look} top{k}{' +BTC>sma50' if filt else ''}"] = backtest(close, uni, look, k, filt, cash)
    for name, ret in series.items():
        for lab, (a, b) in PERIODS.items():
            rr = ret[(ret.index >= a) & (ret.index <= b)]
            st = stats(rr, cash, PPY)
            if st:
                rows.append({"rule": name, "period": lab, **st})
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    pd.DataFrame(series).to_parquet(out_path.replace(".csv", "_series.parquet"))
    pd.set_option("display.width", 250)
    y = t.pivot_table(index="rule", columns="period", values="cagr")[list(PERIODS)]
    z = t.pivot_table(index="rule", columns="period", values="max_dd")[list(PERIODS)]
    print("CAGR %\n", (y * 100).round(0).to_string())
    print("max DD %\n", (z * 100).round(0).to_string())
    is_ = t[(t["period"] == "IS 2018-19") & t["rule"].str.startswith("mom")].sort_values("sharpe", ascending=False)
    print("in-sample best:", is_.head(3)[["rule", "sharpe", "cagr"]].round(2).to_string())


if __name__ == "__main__":
    main()
