"""python -m src.backtest.x80.pairs_mom GRID_DIR PRICES OUT_CSV : two stock-level ideas on the point-in-time S&P 500.

1. Concentrated momentum (a common "huge backtest" recipe): each month buy the top 5 / 10 / 20 members by 12-1, 6-1
   or 3-month return, equal weight, 5 bp per side; 1x and 2x (financed at T-bill + 0.5%).
2. Pairs trading, distance method (Gatev, Goetzmann & Rouwenhorst): every 6 months take 12 months of normalised
   prices of the members with full data, pick the 20 pairs with the smallest squared distance, then trade for
   6 months: open when the spread is 2 formation standard deviations apart (long the laggard, short the leader,
   $0.5 each per pair slot), close when it crosses zero or at the end. 5 bp per side per leg, 0.5%/yr borrow.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, load, stats
from src.backtest.x80.equity_lev import book

PERIODS = {"2010-17": ("2010-01-01", "2017-12-31"), "2018-26": ("2018-01-01", "2030-01-01"), "2022-26": ("2022-01-01", "2030-01-01"),
           "all": ("2010-01-01", "2030-01-01")}
COST = 0.0005


def momentum(px: pd.DataFrame, member: pd.DataFrame, look: int, skip: int, n: int) -> pd.Series:
    r = px.pct_change(fill_method=None)
    me = px.groupby(px.index.to_period("M")).tail(1).index
    out, w_prev = [], pd.Series(dtype=float)
    for i, d in enumerate(me[:-1]):
        pos = px.index.get_loc(d)
        if pos < look + 1:
            continue
        score = px.iloc[pos - skip] / px.iloc[pos - look] - 1
        score = score[member.loc[d].reindex(score.index).fillna(False).astype(bool)].dropna()
        if len(score) < 100:
            continue
        pick = score.nlargest(n).index
        w = pd.Series(1.0 / n, index=pick)
        turn = w.sub(w_prev, fill_value=0).abs().sum()
        w_prev = w
        span = r.loc[(r.index > d) & (r.index <= me[i + 1]), pick].fillna(0.0)
        day = span.mean(axis=1)
        day.iloc[0] -= turn * COST
        out.append(day)
    return pd.concat(out)


def pairs(px: pd.DataFrame, member: pd.DataFrame) -> pd.Series:
    r = px.pct_change(fill_method=None)
    dates = px.index
    starts = [d for d in px.groupby(px.index.to_period("6M")).head(1).index if d >= dates[0] + pd.Timedelta(days=380)]
    out = []
    for s in starts:
        i0 = dates.get_loc(s)
        form = px.iloc[i0 - 252:i0]
        ok = form.notna().all() & member.loc[dates[i0 - 1]].reindex(form.columns).fillna(False).astype(bool)
        f = form.loc[:, ok]
        f = f / f.iloc[0]
        if f.shape[1] < 100:
            continue
        a = f.values
        sq = (a ** 2).sum(axis=0)
        ssd = sq[:, None] + sq[None, :] - 2 * a.T @ a
        np.fill_diagonal(ssd, np.inf)
        iu = np.triu_indices(len(sq), 1)
        flat = ssd[iu]
        best = np.argsort(flat)[:20]
        prs = [(f.columns[iu[0][k]], f.columns[iu[1][k]]) for k in best]
        trade = px.iloc[i0:i0 + 126]
        rr = r.iloc[i0:i0 + 126]
        daily = pd.Series(0.0, index=trade.index)
        for x, y in prs:
            sd = (f[x] - f[y]).std()
            px_x, px_y = trade[x] / form[x].iloc[0], trade[y] / form[y].iloc[0]
            spread = (px_x - px_y)
            pos = 0                                                # +1: long x short y
            for t in range(1, len(trade)):
                d = trade.index[t]
                if pos != 0:
                    rx, ry = rr[x].iloc[t], rr[y].iloc[t]
                    if np.isnan(rx) or np.isnan(ry):
                        pos = 0
                        continue
                    daily.loc[d] += pos * 0.5 * (rx - ry) / len(prs) - 0.5 * 0.005 / 252 / len(prs)
                sp = spread.iloc[t - 1]
                if np.isnan(sp):
                    continue
                if pos == 0 and abs(sp) > 2 * sd:
                    pos = -1 if sp > 0 else 1
                    daily.loc[d] -= 2 * COST * 0.5 * 2 / len(prs) * 0 + 2 * 0.5 * COST / len(prs)
                elif pos != 0 and np.sign(sp) == pos:              # spread crossed zero
                    pos = 0
                    daily.loc[d] -= 2 * 0.5 * COST / len(prs)
        out.append(daily)
    s = pd.concat(out)
    return s[~s.index.duplicated()]


def main() -> None:
    grid_dir, prices_path, out_path = sys.argv[1:4]
    g = pd.read_parquet(f"{grid_dir}/grid.parquet", columns=["symbol", "date"])
    g["m"] = True
    member = g.pivot_table(index="date", columns="symbol", values="m", aggfunc="any").fillna(False).astype(bool).sort_index()
    p = pd.read_parquet(prices_path, columns=["symbol", "date", "adj_close"])
    px = p.pivot_table(index="date", columns="symbol", values="adj_close").sort_index()
    px = px[px.index >= "2008-06-01"]
    member = member.reindex(px.index).ffill().fillna(False).astype(bool)
    mx = load()
    rows, series = [], {}
    for look, skip, lab in ((252, 21, "12-1"), (126, 21, "6-1"), (63, 0, "3m")):
        for n in (5, 10, 20):
            series[f"momentum {lab} top{n}"] = momentum(px, member, look, skip, n)
    series["pairs distance top20"] = pairs(px, member)
    spy = mx["SPY"]["adj_close"].pct_change().dropna()
    series["SPY"] = spy[spy.index >= "2009-06-01"]
    for name, ret in series.items():
        cash = cash_daily(mx, ret.index)
        if name.startswith("pairs"):
            ret = ret + cash                                       # capital idles in T-bills
        for lev in ((1.0, 2.0) if not name.startswith("pairs") else (1.0, 3.0, 5.0)):
            rr = book(ret, pd.Series(lev, index=ret.index), cash) if lev != 1.0 else ret
            for plab, (a, b) in PERIODS.items():
                x = rr[(rr.index >= a) & (rr.index <= b)]
                st = stats(x, cash)
                if st:
                    rows.append({"rule": name, "lev": lev, "period": plab, **st})
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    pd.set_option("display.width", 250)
    print("CAGR %\n", (t.pivot_table(index=["rule", "lev"], columns="period", values="cagr")[list(PERIODS)] * 100).round(0).to_string())
    print("max DD %\n", (t.pivot_table(index=["rule", "lev"], columns="period", values="max_dd")[list(PERIODS)] * 100).round(0).to_string())
    print("Sharpe 1x\n", t[t["lev"] == 1.0].pivot_table(index="rule", columns="period", values="sharpe")[list(PERIODS)].round(2).to_string())


if __name__ == "__main__":
    main()
