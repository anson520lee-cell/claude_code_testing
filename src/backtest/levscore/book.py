"""The x80 round-2 book (equal risk contribution, 104 weeks) and a levering function that takes ANY weekly leverage path.

Financing and trading costs are exactly meta.lever's: funded sleeves borrow beyond 100% at T-bill + 1% (crypto
T-bill + 4%), the market-neutral perpetual books use margin only (their margin earns T-bills), re-weighting and
leverage changes cost 0.10% of the notional traded, a week that loses everything ends the account.
On top, `intraweek_trough` uses the sleeves' DAILY returns to find the worst point inside each week: at 8x a book that
is down 12.5% on a Wednesday has wiped the account out even if it is back up by Friday, which weekly data cannot see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.x80 import meta

METHOD, LOOK, VOL_WIN, CAP = "equal risk", 104, 0, 10.0       # the rule x80 round 2 picked on 2010-17


def base_book(w_ex: pd.DataFrame, cash_w: pd.Series, method: str = METHOD, look: int = LOOK):
    """(unlevered weekly excess return, financing table, weights) - meta.run unchanged."""
    return meta.run(w_ex, cash_w, method, look)


def vol_leverage(base: pd.Series, target: float, vol_win: int = VOL_WIN, cap: float = CAP) -> pd.Series:
    """x80's leverage: target vol / the book's vol known at the previous close (whole history so far when vol_win=0)."""
    roll = base.expanding(min_periods=52) if vol_win == 0 else base.rolling(vol_win, min_periods=max(8, vol_win // 2))
    bvol = roll.std().shift(1) * np.sqrt(52)
    return (target / bvol).clip(upper=cap)


def lever_path(base: pd.Series, fin: pd.DataFrame, cash_w: pd.Series, lev: pd.Series, cost: float | None = None,
               trough: pd.Series | None = None) -> pd.Series:
    """Weekly account return for a given leverage path (lev[t] is decided at the close of week t-1).
    With ``trough`` (the unlevered book's worst cumulative return inside each week), a week whose intraweek low
    reaches -100% of equity is a wipe-out (-100%) even if the week closes higher."""
    lev = lev.reindex(base.index)
    c = cash_w.reindex(base.index).fillna(0.0)
    ret = lev * base + c + lev * fin["unfunded"] * c - (lev * fin["funded"] - 1).clip(lower=0) * fin["spread"] / 52
    ret = ret - meta.trading_cost(lev.fillna(0.0), base.fillna(0.0), fin, ret.fillna(0.0), cost)
    if trough is not None:
        wiped = (1 + lev * trough.reindex(base.index).fillna(0.0)) <= 0
        ret = ret.mask(wiped, -1.0)
    ret = ret.dropna().clip(lower=-1.0)
    dead = (ret <= -1.0).cumsum().shift(1).fillna(0) > 0
    return ret.mask(dead, 0.0)


def intraweek_trough(daily: pd.DataFrame, W: pd.DataFrame, w_ex: pd.DataFrame) -> pd.DataFrame:
    """For each week: the book's cumulative return at its worst day inside the week (<= 0) and at the week's end,
    holding that week's weights without rebalancing (buy and hold inside the week, like the weekly book)."""
    out = {}
    weeks = W.index
    for t in weeks:
        row = w_ex.loc[t]
        live = row.notna()
        w = W.loc[t].where(live, 0.0)
        if w.sum() <= 0:
            out[t] = (0.0, 0.0)
            continue
        w = w / w.sum()
        d = daily.loc[(daily.index > t - pd.Timedelta(days=7)) & (daily.index <= t), w.index[w > 0]].fillna(0.0)
        if d.empty:
            out[t] = (0.0, 0.0)
            continue
        g = (1 + d).cumprod() - 1
        path = g.to_numpy() @ w[w > 0].to_numpy()
        out[t] = (float(min(0.0, path.min())), float(path[-1]))
    return pd.DataFrame(out, index=["trough", "week_end"]).T


def perf(r: pd.Series, cash_w: pd.Series | None = None) -> dict:
    """CAGR, max drawdown, vol, Sharpe (excess of T-bills), worst week, share of positive weeks and years."""
    r = r.dropna()
    if len(r) < 26:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 52
    out = {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0,
           "max_dd": float((eq / eq.cummax() - 1).min()), "vol": float(r.std() * np.sqrt(52)),
           "worst_week": float(r.min()), "hit": float((r > 0).mean())}
    if cash_w is not None:
        ex = r - cash_w.reindex(r.index).fillna(0.0)
        out["sharpe"] = float(ex.mean() / ex.std() * np.sqrt(52)) if ex.std() > 0 else np.nan
    yr = (1 + r).groupby(r.index.year).prod() - 1
    out["pos_years"] = float((yr > 0).mean())
    out["worst_year"] = float(yr.min())
    out["calmar"] = out["cagr"] / abs(out["max_dd"]) if out["max_dd"] < 0 else np.nan
    return out
