"""python -m src.backtest.x80.improve BINANCE_DIR GRID_DIR SP_PRICES OUT_CSV : second pass on the best candidates.

1. BTC / ETH trend with the rule re-chosen every 1 January from past data only (56 rules: the 28 crypto_trend signals,
   raw and vol-targeted to 80%): best trailing Sharpe on all past data or on the last 3 years, top 1 or the average
   exposure of the top 5; plus the average of all 28 signals (no choice at all), raw and vol-targeted.
   The selection method itself is picked on BTC 2016-19; 2020-26 is out of sample for both coins.
2. Multi-coin trend on Binance spot (point in time, delisted coins included): each week hold the top 10 / 20 / 50 coins
   by turnover with inverse-vol weights, each scaled by its trend strength (share of 7 trend rules that are long);
   long only, never more than 100% invested, 0.2% per side. Chosen on 2018-19.
3. BTC vs alts, two-way: while the BTC-minus-alts spread is above its N-day average, long BTC / short the top-50 alts
   (perps); otherwise the reverse (long alts / short BTC). Funding paid and received, 0.1% per side. N picked on 2020-21.
4. Concentrated S&P 500 momentum (point-in-time members, top 10 / 20 by 6-1 month return, monthly) with a SPY 200-day
   filter (cash when SPY is below it) and/or a 20% / 30% vol target (trailing 63 days, cap 2x, T-bill + 0.5%). Chosen on 2010-17.
5. BTC / ETH trend long AND short with perpetuals (Binance closes and funding, from 2020): exposure = 2 x (share of the
   28 rules that are long) - 1, raw or vol-targeted to 80% (cap 3x); the long side pays funding, the short side receives
   it; 0.05% per side; the margin earns nothing. Long-only on the same data next to it. Picked on 2020-21.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.backtest.x80 import crypto_trend as ct
from src.backtest.x80.common import cash_daily, funding_daily, load, stats
from src.backtest.x80.crypto_factors import simulate
from src.backtest.x80.crypto_xs import panel, universe
from src.backtest.x80.equity_lev import book
from src.backtest.x80.pairs_mom import momentum

PPY = 365


def trend_rules(close: pd.Series, r: pd.Series) -> dict[str, pd.Series]:
    sig = ct.signals(close)
    rv = r.rolling(30).std() * np.sqrt(PPY)
    out = {}
    for k, s in sig.items():
        out[k] = s.reindex(r.index).fillna(0.0)
        out[f"{k}|vt80"] = (s * (0.8 / rv).clip(upper=3.0).reindex(s.index)).reindex(r.index).fillna(0.0)
    return out


def walk_forward_trend(close: pd.Series, cash: pd.Series, first_year: int, window: int | None, top: int) -> tuple[pd.Series, dict]:
    """Each 1 January rank the rules by Sharpe on past data only, then hold the top ``top`` (average exposure) all year."""
    r = close.pct_change().dropna()
    rules = trend_rules(close, r)
    rets = pd.DataFrame({k: ct.run_exposure(r, e, cash) for k, e in rules.items()})
    expo = pd.DataFrame(rules)
    ex = rets.sub(cash.reindex(rets.index).fillna(0.0), axis=0)
    out = pd.Series(0.0, index=r.index)
    picks = {}
    for y in range(first_year, r.index[-1].year + 1):
        a, b = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
        h = ex[ex.index < a]
        if window:
            h = h[h.index >= a - pd.DateOffset(years=window)]
        sh = h.mean() / h.std()
        pick = list(sh.nlargest(top).index)
        picks[y] = pick
        m = (r.index >= a) & (r.index < b)
        out[m] = expo.loc[m, pick].mean(axis=1)
    out[r.index < pd.Timestamp(f"{first_year}-01-01")] = 0.0
    return ct.run_exposure(r, out, cash), picks


def ensemble_trend(close: pd.Series, cash: pd.Series, vt: bool) -> pd.Series:
    r = close.pct_change().dropna()
    sig = pd.DataFrame(ct.signals(close)).reindex(r.index).fillna(0.0)
    e = sig.mean(axis=1)
    if vt:
        rv = r.rolling(30).std() * np.sqrt(PPY)
        e = e * (0.8 / rv).clip(upper=3.0).fillna(0.0)
    return ct.run_exposure(r, e, cash)


def coin_trend_strength(close: pd.DataFrame) -> pd.DataFrame:
    """Share of 7 trend rules that are long (NaN where fewer than 3 rules have enough history)."""
    votes, cnt = 0, 0
    for n in (20, 50, 100, 200):
        ma = close.rolling(n, min_periods=n).mean()
        v = (close > ma).astype(float).where(ma.notna())
        votes = votes + v.fillna(0.0)
        cnt = cnt + v.notna()
    for k in (30, 90, 180):
        past = close.shift(k)
        v = (close > past).astype(float).where(past.notna() & close.notna())
        votes = votes + v.fillna(0.0)
        cnt = cnt + v.notna()
    return (votes / cnt).where(cnt >= 3)


def multi_coin_trend(close: pd.DataFrame, qv: pd.DataFrame, top: int, start: str = "2018-01-01") -> pd.Series:
    r = close.pct_change(fill_method=None).where(close.shift(1).notna() & close.notna())
    uni = universe(close, qv, top=top)
    ts = coin_trend_strength(close)
    rv = r.rolling(30, min_periods=20).std()
    idx = close.index
    reb = np.where(idx >= start)[0][::7]
    W = np.zeros((len(reb), close.shape[1]))
    for i, s in enumerate(reb):
        p = s - 1
        ok = uni.values[p] & np.isfinite(ts.values[p]) & np.isfinite(rv.values[p]) & (rv.values[p] > 0)
        if ok.sum() < 3:
            continue
        inv = np.where(ok, 1.0 / np.where(ok, rv.values[p], 1.0), 0.0)
        W[i] = inv / inv.sum() * np.where(ok, ts.values[p], 0.0)
    d = simulate(W, reb, r.fillna(0.0).values, np.zeros(close.shape), cost=0.002)
    return pd.Series(d, index=idx).iloc[reb[0]:]


def btc_alts_two_way(close: pd.DataFrame, qv: pd.DataFrame, fund: pd.DataFrame, n_ma: int, cost: float = 0.001) -> pd.Series:
    r = close.pct_change(fill_method=None).where(close.shift(1).notna() & close.notna())
    age = close.notna().cumsum()
    liq = qv.rolling(30, min_periods=20).median()
    has_perp = fund.reindex(columns=close.columns).notna().rolling(3, min_periods=1).max().astype(bool)
    ok = (age >= 60) & (liq >= 5e6) & has_perp & close.notna()
    ok["BTCUSDT"] = False
    alts = ok & (liq.where(ok).rank(axis=1, ascending=False) <= 50)
    basket = r.where(alts.shift(1).fillna(False).astype(bool)).mean(axis=1)
    spread = (r["BTCUSDT"] - basket).fillna(0.0)
    sidx = (1 + spread).cumprod()
    gate = sidx > sidx.rolling(n_ma).mean()
    idx = close.index
    reb = np.where(idx >= "2020-01-01")[0][::7]
    cols = list(close.columns)
    jb = cols.index("BTCUSDT")
    W = np.zeros((len(reb), len(cols)))
    for i, s in enumerate(reb):
        p = s - 1
        a = alts.values[p]
        if a.sum() < 5:
            continue
        sign = 1.0 if gate.iloc[p] else -1.0
        W[i, a] = -sign / a.sum()
        W[i, jb] = sign
    d = simulate(W, reb, r.fillna(0.0).values, fund.reindex(index=idx, columns=cols).fillna(0.0).values, cost=cost)
    return pd.Series(d, index=idx).iloc[reb[0]:]


def run_perp(r: pd.Series, expo: pd.Series, fund: pd.Series, cost: float = 0.0005) -> pd.Series:
    """Daily return of holding ``expo`` (decided at the previous close, may be negative) in a perpetual."""
    e = expo.shift(1).fillna(0.0)
    drift = (e * (1 + r) / (1 + e * r) - e).abs()
    turn = e.diff().abs().fillna(e.abs()) + drift.shift(1).fillna(0.0)
    ret = (e * r - e * fund.reindex(r.index).fillna(0.0) - turn * cost).clip(lower=-1.0)
    dead = (ret <= -1.0).cumsum().shift(1).fillna(0) > 0
    return ret.mask(dead, 0.0)


def periods_stats(name: str, ret: pd.Series, cash: pd.Series, ppy: int, periods: dict, extra: dict | None = None) -> list[dict]:
    rows = []
    for lab, (a, b) in periods.items():
        st = stats(ret[(ret.index >= a) & (ret.index <= b)], cash, ppy)
        if st:
            rows.append({"rule": name, "period": lab, **(extra or {}), **st})
    return rows


def main() -> None:
    bdir, grid_dir, sp_prices, out_path = sys.argv[1:5]
    px = load()
    rows, series, chosen = [], {}, {}
    cP = {"2016-19": ("2016-01-01", "2019-12-31"), "OOS 2020-26": ("2020-01-01", "2030-01-01"), "2020-21": ("2020-01-01", "2021-12-31"),
          "2022-26": ("2022-01-01", "2030-01-01")}
    # 1. walk-forward trend
    methods = {f"{'expanding' if w is None else str(w) + 'y'} top{k}": (w, k) for w in (None, 3) for k in (1, 5)}
    pick_log = {}
    for asset, fy in (("BTC-USD", 2016), ("ETH-USD", 2019)):
        close = px[asset]["close"]
        cash = cash_daily(px, close.index, PPY)
        for m, (w, k) in methods.items():
            ret, picks = walk_forward_trend(close, cash, fy, w, k)
            series[f"{asset[:3]} trend, re-picked yearly ({m})"] = ret
            pick_log[f"{asset[:3]} {m}"] = {str(y): p for y, p in picks.items()}
            rows += periods_stats(f"{asset[:3]} trend, re-picked yearly ({m})", ret, cash, PPY, cP, {"part": "trend"})
        for vt in (False, True):
            ret = ensemble_trend(close, cash, vt)
            name = f"{asset[:3]} trend, average of 28 rules{' vt80' if vt else ''}"
            series[name] = ret
            rows += periods_stats(name, ret, cash, PPY, cP, {"part": "trend"})
    t = pd.DataFrame(rows)
    btc_early = t[(t["period"] == "2016-19") & t["rule"].str.startswith("BTC")].set_index("rule")["sharpe"]
    best_m = btc_early.idxmax().replace("BTC trend, ", "")
    chosen["trend method (BTC 2016-19)"] = best_m
    print("trend methods, Sharpe BTC 2016-19:", btc_early.round(2).to_dict(), "-> chosen:", best_m)
    for a in ("BTC", "ETH"):
        series[f"{a} trend, walk-forward pick"] = series[f"{a} trend, {best_m}"]
    # 2. multi-coin trend
    close, qv = panel(f"{bdir}/spot.parquet")
    c365 = cash_daily(px, close.index, PPY)
    mP = {"IS 2018-19": ("2018-01-01", "2019-12-31"), "OOS 2020-26": ("2020-01-01", "2030-01-01"), "2020-21": ("2020-01-01", "2021-12-31"),
          "2022-26": ("2022-01-01", "2030-01-01")}
    for top in (10, 20, 50):
        ret = multi_coin_trend(close, qv, top)
        series[f"multi-coin trend top{top}"] = ret
        rows += periods_stats(f"multi-coin trend top{top}", ret, c365, PPY, mP, {"part": "multi-coin"})
    t = pd.DataFrame(rows)
    mc = t[(t["period"] == "IS 2018-19") & t["rule"].str.startswith("multi-coin")].set_index("rule")["sharpe"]
    chosen["multi-coin (2018-19)"] = mc.idxmax()
    series["multi-coin trend, chosen"] = series[mc.idxmax()]
    print("multi-coin trend Sharpe 2018-19:", mc.round(2).to_dict())
    # 3. BTC vs alts two-way
    c2 = close[close.index >= "2019-06-01"]
    q2 = qv.reindex(c2.index)
    fund = funding_daily(pd.read_parquet(f"{bdir}/funding.parquet")).reindex(index=c2.index, columns=c2.columns)
    bP = {"IS 2020-21": ("2020-01-01", "2021-12-31"), "2022-23": ("2022-01-01", "2023-12-31"), "2024-26": ("2024-01-01", "2030-01-01"),
          "OOS 2022-26": ("2022-01-01", "2030-01-01")}
    for n in (30, 60, 90):
        ret = btc_alts_two_way(c2, q2, fund, n)
        series[f"BTC vs alts two-way, spread vs {n}d avg"] = ret
        rows += periods_stats(f"BTC vs alts two-way, spread vs {n}d avg", ret, c365, PPY, bP, {"part": "btc-alts"})
    t = pd.DataFrame(rows)
    ba = t[(t["period"] == "IS 2020-21") & t["rule"].str.startswith("BTC vs alts")].set_index("rule")["sharpe"]
    chosen["BTC vs alts two-way (2020-21)"] = ba.idxmax()
    series["BTC vs alts two-way, chosen"] = series[ba.idxmax()]
    print("BTC vs alts two-way Sharpe 2020-21:", ba.round(2).to_dict())
    # 4. concentrated momentum + filter + vol target
    g = pd.read_parquet(f"{grid_dir}/grid.parquet", columns=["symbol", "date"])
    g["m"] = True
    member = g.pivot_table(index="date", columns="symbol", values="m", aggfunc="any").fillna(False).astype(bool).sort_index()
    p = pd.read_parquet(sp_prices, columns=["symbol", "date", "adj_close"])
    spx = p.pivot_table(index="date", columns="symbol", values="adj_close").sort_index()
    spx = spx[spx.index >= "2008-06-01"]
    member = member.reindex(spx.index).ffill().fillna(False).astype(bool)
    spy = px["SPY"]["adj_close"]
    filt = (spy > spy.rolling(200).mean()).astype(float)
    sP = {"IS 2010-17": ("2010-01-01", "2017-12-31"), "OOS 2018-26": ("2018-01-01", "2030-01-01"), "2022-26": ("2022-01-01", "2030-01-01")}
    for n in (10, 20):
        base = momentum(spx, member, 126, 21, n)
        cash = cash_daily(px, base.index)
        rv = base.rolling(63, min_periods=40).std() * np.sqrt(252)
        for use_f in (False, True):
            for tv in (None, 0.2, 0.3):
                e = pd.Series(1.0, index=base.index)
                if tv:
                    e = (tv / rv).clip(upper=2.0).shift(1).fillna(0.0)
                if use_f:
                    e = e * filt.reindex(base.index).ffill().shift(1).fillna(0.0)
                ret = book(base, e, cash)
                name = f"momentum 6-1 top{n}{' + SPY>200d' if use_f else ''}{f' + vol {int(tv * 100)}%' if tv else ''}"
                series[name] = ret
                rows += periods_stats(name, ret, cash, 252, sP, {"part": "momentum"})
    t = pd.DataFrame(rows)
    mo = t[(t["period"] == "IS 2010-17") & (t["part"] == "momentum")].set_index("rule")["sharpe"]
    chosen["momentum (2010-17)"] = mo.idxmax()
    series["momentum, chosen"] = series[mo.idxmax()]
    print("momentum Sharpe 2010-17:", mo.round(2).to_dict())
    # 5. BTC / ETH trend long-short with perps
    pP = {"IS 2020-21": ("2020-01-01", "2021-12-31"), "2022-23": ("2022-01-01", "2023-12-31"), "2024-26": ("2024-01-01", "2030-01-01"),
          "OOS 2022-26": ("2022-01-01", "2030-01-01")}
    for coin in ("BTC", "ETH"):
        cl = close[f"{coin}USDT"].dropna()
        cl = cl[cl.index >= "2018-06-01"]
        rr = cl.pct_change().dropna()
        sig = pd.DataFrame(ct.signals(cl)).reindex(rr.index).fillna(0.0).mean(axis=1)
        rv = rr.rolling(30).std() * np.sqrt(PPY)
        vt = (0.8 / rv).clip(upper=3.0).fillna(0.0)
        fb = fund[f"{coin}USDT"] if f"{coin}USDT" in fund else pd.Series(0.0, index=rr.index)
        for lab, e in ((f"{coin} trend long/short perp", 2 * sig - 1), (f"{coin} trend long/short perp vt80", (2 * sig - 1) * vt),
                       (f"{coin} trend long-only perp", sig), (f"{coin} trend long-only perp vt80", sig * vt)):
            ret = run_perp(rr, e, fb)
            ret = ret[ret.index >= "2020-01-01"]
            series[lab] = ret
            rows += periods_stats(lab, ret, c365, PPY, pP, {"part": "perp trend"})
    t = pd.DataFrame(rows)
    pt = t[(t["period"] == "IS 2020-21") & (t["part"] == "perp trend") & t["rule"].str.startswith("BTC")].set_index("rule")["sharpe"]
    pick = pt.idxmax()
    chosen["perp trend (BTC 2020-21)"] = pick
    series["BTC trend perp, chosen"] = series[pick]
    series["ETH trend perp, chosen"] = series[pick.replace("BTC", "ETH")]
    print("perp trend Sharpe 2020-21:", pt.round(2).to_dict())
    t.to_csv(out_path, index=False)
    pd.DataFrame(series).to_parquet(out_path.replace(".csv", "_series.parquet"))
    with open(out_path.replace(".csv", "_chosen.json"), "w") as fh:
        json.dump({"chosen": chosen, "trend_picks": pick_log}, fh, indent=1)
    pd.set_option("display.width", 250)
    for part in ("trend", "multi-coin", "btc-alts", "momentum", "perp trend"):
        x = t[t["part"] == part]
        print(f"--- {part}: Sharpe / CAGR % / max DD %")
        print(x.pivot_table(index="rule", columns="period", values="sharpe").round(2).to_string())
        print((x.pivot_table(index="rule", columns="period", values="cagr") * 100).round(0).to_string())
        print((x.pivot_table(index="rule", columns="period", values="max_dd") * 100).round(0).to_string())
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
