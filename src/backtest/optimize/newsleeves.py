"""python -m src.backtest.optimize.newsleeves RAW_SP RAW_X80 RAW_FX OUT_PARQUET

Three sleeves that no earlier round had, each a DAILY excess return over T-bills (calendar of its own market):

  RSI(2) mean reversion  Connors RSI(2) < 5 with the close above its 200-day average, point-in-time S&P 500 members
                         (unadjusted close >= $5, >= 260 sessions); buy the next open, sell at the first close above
                         the 5-day average or after 10 sessions; 10 slots of 10% each, most oversold first;
                         15 bp round trip (the swing42 rule that held out of sample in both universes).
  G10 FX carry           each month end rank USD + 9 currencies by their 3-month interbank rate (FRED/OECD, used one
                         month late because it is published late), long the top 3, short the bottom 3, equal weight;
                         daily return = spot moves + the rate differential accrued daily; 3 bp per unit traded.
                         A pure long/short return (no cash), so it is an unfunded sleeve.
  turn of the month      SPY from the close two sessions before month end to the close of the 3rd session of the
                         next month (the -1..+3 window); 2 bp per side.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from src.backtest.edge_hunt.panel import load_prices, membership
from src.backtest.swing42.core import rsi
from src.backtest.x80.common import cash_daily, load

RSI_X, SLOTS, MAX_DAYS, RT_COST = 5.0, 10, 10, 0.0015
FX_TOP, FX_COST, FX_LAG_MONTHS = 3, 0.0003, 1
TOM_COST = 0.0002


def rsi2_sleeve(raw_sp: str, cash: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """(daily excess return, trade list) of the RSI(2) book."""
    px = load_prices(os.path.join(raw_sp, "prices.parquet"))
    cal = px["SPY"].index
    cal = cal[cal >= "2006-01-01"]
    mem = membership(os.path.join(raw_sp, "sp500_hist.csv"), cal)
    syms = [s for s in mem if s in px and not s.startswith("^")]
    adj = pd.DataFrame({s: px[s]["adj_close"].reindex(cal) for s in syms})
    raw = pd.DataFrame({s: px[s]["close"].reindex(cal) for s in syms})
    opn = pd.DataFrame({s: (px[s]["open"] * px[s]["adj_close"] / px[s]["close"]).reindex(cal) for s in syms})
    A, R, O = adj.to_numpy(float), raw.to_numpy(float), opn.to_numpy(float)
    M = np.column_stack([mem[s] for s in syms])
    first = np.argmax(np.isfinite(A), axis=0)
    seasoned = np.arange(len(cal))[:, None] >= first[None, :] + 260
    rs = rsi(A, 2)
    sma200 = adj.rolling(200, min_periods=200).mean().to_numpy()
    sma5 = adj.rolling(5, min_periods=5).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        sig = (rs < RSI_X) & (A > sma200) & M & seasoned & (R >= 5) & np.isfinite(O)
    c = cash.reindex(cal).fillna(0.0).to_numpy()
    eq, pos, out, trades = 1.0, [], np.zeros(len(cal)), []          # pos: [col, value now, entry_i, value at entry]
    for i in range(1, len(cal)):
        e0 = eq
        pnl, invested = 0.0, 0.0
        keep = []
        for p in pos:                                                # held from yesterday's close
            j, v, ei, v0 = p
            invested += v
            r = A[i, j] / A[i - 1, j] - 1 if np.isfinite(A[i, j]) and np.isfinite(A[i - 1, j]) else 0.0
            pnl += v * r
            p[1] = v * (1 + r)
            done = (np.isfinite(sma5[i, j]) and A[i, j] > sma5[i, j]) or (i - ei + 1 >= MAX_DAYS) or not np.isfinite(A[i, j])
            if done:
                pnl -= p[1] * RT_COST / 2
                trades.append((cal[ei], cal[i], syms[j], p[1] * (1 - RT_COST / 2) / (v0 * (1 + RT_COST / 2)) - 1))
            else:
                keep.append(p)
        pos = keep
        cand = np.flatnonzero(sig[i - 1])                            # yesterday's signal, bought at today's open
        held = {p[0] for p in pos}
        cand = [j for j in cand[np.argsort(rs[i - 1, cand])] if j not in held]
        for j in cand[:max(SLOTS - len(pos), 0)]:
            v = e0 / SLOTS
            r = A[i, j] / O[i, j] - 1 if np.isfinite(A[i, j]) else 0.0
            pnl += v * r - v * RT_COST / 2
            invested += v
            pos.append([j, v * (1 + r), i, v])
        out[i] = (pnl - invested * c[i]) / e0                        # excess of T-bills: the money in stocks would have earned c
        eq = e0 + pnl + (e0 - invested) * c[i]
    ex = pd.Series(out, index=cal)
    ex = ex[ex.index >= cal[260]]
    t = pd.DataFrame(trades, columns=["entry", "exit", "symbol", "ret"])
    return ex, t


def fx_carry_sleeve(raw_fx: str) -> tuple[pd.Series, pd.DataFrame]:
    """(daily long/short carry return, monthly weights)."""
    spot = pd.read_parquet(os.path.join(raw_fx, "fx_spot.parquet")).sort_index()
    rates = pd.read_parquet(os.path.join(raw_fx, "fx_rates.parquet")).sort_index() / 100
    spot = spot[spot.index >= "2004-01-01"]
    cal = pd.bdate_range(spot.index.min(), spot.index.max())
    s = spot.reindex(cal).ffill()
    lr = np.log(s).diff()
    lr = lr.mask(lr.abs() > 0.08, 0.0)                               # Yahoo FX has a few bad prints; no G10 pair moves 8% in a day here
    sr = np.expm1(lr).fillna(0.0)
    sr["USD"] = 0.0
    mrate = rates.resample("MS").last().ffill()
    known = mrate.shift(FX_LAG_MONTHS + 1)                           # rate for month m is published after m ends
    month_ends = pd.Series(cal, index=cal).groupby(cal.to_period("M")).max()
    W = pd.DataFrame(0.0, index=cal, columns=sr.columns)
    wrows = {}
    for d in month_ends.values[:-1]:
        d = pd.Timestamp(d)
        m = pd.Timestamp(d.year, d.month, 1)
        if m not in known.index:
            continue
        r = known.loc[m].reindex(sr.columns)
        live = r.notna() & (s.loc[d].reindex(sr.columns).notna() | (r.index == "USD"))
        r = r[live]
        if len(r) < 2 * FX_TOP + 1:
            continue
        order = r.sort_values()
        w = pd.Series(0.0, index=sr.columns)
        w[order.index[-FX_TOP:]] = 1 / FX_TOP
        w[order.index[:FX_TOP]] = -1 / FX_TOP
        wrows[d] = w
    wdf = pd.DataFrame(wrows).T.sort_index()
    W = wdf.reindex(cal).ffill().shift(1).fillna(0.0)                # decided at the month-end close, held from the next day
    rr = rates.resample("D").last().ffill().reindex(cal).ffill()      # the carry actually earned (forward points) is today's rate
    rr = rr.reindex(columns=sr.columns).fillna(0.0)
    carry = (W * rr).sum(axis=1) / 260
    spot_part = (W * sr).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(0.0)
    ret = spot_part + carry - FX_COST * turn
    first = W.abs().sum(axis=1).gt(0).idxmax()
    return ret[ret.index >= first], wdf


def tom_sleeve(px: dict, cash: pd.Series) -> pd.Series:
    """Daily excess return of holding SPY over the turn-of-the-month window (-1..+3)."""
    a = px["SPY"]["adj_close"]
    a = a[a.index >= "2004-01-01"]
    r = a.pct_change().fillna(0.0)
    cal = a.index
    per = cal.to_period("M")
    pos_in_month = pd.Series(np.arange(len(cal)), index=cal).groupby(per).transform(lambda x: x - x.min())
    from_end = pd.Series(np.arange(len(cal)), index=cal).groupby(per).transform(lambda x: x.max() - x)
    held = (from_end == 0) | (pos_in_month <= 2)                     # last session, then sessions 1-3 of the next month
    enter = held & ~held.shift(1, fill_value=False)
    leave = held & ~held.shift(-1, fill_value=False)
    c = cash.reindex(cal).fillna(0.0)
    ex = (r - c).where(held, 0.0) - TOM_COST * (enter.astype(float) + leave.astype(float))
    return ex


def main() -> None:
    raw_sp, raw_x80, raw_fx, out = sys.argv[1:5]
    px = load(os.path.join(raw_x80, "prices.parquet"))
    cal = pd.date_range("2004-01-01", "2026-11-06")
    c252 = cash_daily(px, cal, 252)
    rsi2, trades = rsi2_sleeve(raw_sp, c252)
    fx, fxw = fx_carry_sleeve(raw_fx)
    tom = tom_sleeve(px, c252)
    df = pd.DataFrame({"RSI(2) mean reversion": rsi2, "G10 FX carry": fx, "turn of the month": tom})
    df.to_parquet(out)
    trades.to_parquet(out.replace(".parquet", "_rsi2_trades.parquet"))
    fxw.to_parquet(out.replace(".parquet", "_fx_weights.parquet"))
    for k, v in df.items():
        v = v.dropna()
        v = v[v.index < "2025-01-01"]
        print(f"{k:24s} {v.index.min().date()}..{v.index.max().date()}  ann.mean {v.mean() * 252:.3%}  vol {v.std() * np.sqrt(252):.2%}"
              f"  Sharpe {v.mean() / v.std() * np.sqrt(252):.2f}  (before the holdout)")
    print("RSI(2) trades", len(trades), "mean", round(trades["ret"].mean(), 4), "win", round((trades["ret"] > 0).mean(), 3))


if __name__ == "__main__":
    main()
