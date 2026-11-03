"""Honest backtest engine for the "can anyone really get 90%/yr" study (research only, no live rules touched).

Conventions
 * daily adjusted closes (splits + dividends), weights decided at the close of day t are earned from the
   close of day t+2 on (trade at the t+1 close, i.e. one extra day of delay: conservative)
 * cost = bps * traded notional (sum of |delta weight|) per rebalance, on top of that nothing is hidden
 * names whose data end while held (acquired / delisted) are assumed to be converted to cash at the last price
"""
from __future__ import annotations
import numpy as np, pandas as pd

TRADING_DAYS = 252
PERIODS = {"全期 2005-26": ("2005-01-01", "2026-12-31"),
           "GFC 2005-09": ("2005-01-01", "2009-12-31"),
           "樣本外 2010-19": ("2010-01-01", "2019-12-31"),
           "近期 2020-26": ("2020-01-01", "2026-12-31"),
           "AI行情 2023-26": ("2023-01-01", "2026-12-31")}


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 30:
        return {"cagr": np.nan, "mdd": np.nan, "sharpe": np.nan, "vol": np.nan, "n": len(r)}
    eq = (1 + r).cumprod()
    yrs = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    mdd = (eq / eq.cummax() - 1).min()
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return {"cagr": cagr, "mdd": mdd, "sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan,
            "vol": vol, "n": len(r)}


def by_period(r: pd.Series) -> dict:
    return {k: stats(r.loc[a:b]) for k, (a, b) in PERIODS.items()}


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).last().values)


def run_weights(ret: pd.DataFrame, W: pd.DataFrame, bps: float, delay: int = 2) -> tuple[pd.Series, pd.Series]:
    """W: target weights at rebalance dates (rows) over ret columns. Returns (net daily return, turnover per day).
    Weights drift between rebalances; unfilled NaN returns count as 0 (cash)."""
    dates = ret.index
    rb = [d for d in W.index if d in dates]
    pos = {d: dates.get_loc(d) for d in rb}
    out = np.zeros(len(dates)); turn = np.zeros(len(dates))
    cols = ret.columns
    R = ret.to_numpy(dtype=float); R = np.nan_to_num(R, nan=0.0)
    w = np.zeros(len(cols))
    nxt = 0
    sched = sorted(((pos[d] + delay, d) for d in rb))
    si = 0
    for i in range(len(dates)):
        while si < len(sched) and sched[si][0] == i:
            tgt = W.loc[sched[si][1]].reindex(cols).fillna(0.0).to_numpy()
            tv = np.abs(tgt - w).sum()
            turn[i] = tv; out[i] -= tv * bps / 1e4
            w = tgt.copy(); si += 1
        if i > 0:
            gross = R[i] @ w
            out[i] += gross
            # drift
            w = w * (1 + R[i]); s = (1 + gross)
            w = w / s if s > 0 else w
    return pd.Series(out, index=dates), pd.Series(turn, index=dates)


def leveraged(ret: pd.Series, lev: float, expense: float = 0.0095, borrow_spread: float = 0.005,
              cash: pd.Series | None = None) -> pd.Series:
    """Daily-reset leveraged ETF return from an underlying daily return series."""
    c = cash.reindex(ret.index).ffill().fillna(0.02) / TRADING_DAYS if cash is not None else 0.02 / TRADING_DAYS
    fin = (lev - 1) * (c + borrow_spread / TRADING_DAYS)
    return lev * ret - fin - expense / TRADING_DAYS
