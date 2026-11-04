"""Shared helpers: price loading, cash rate, performance statistics, leverage arithmetic."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

RAW = "/home/user/research_data/x80/raw/prices.parquet"


def load(path: str = RAW) -> dict[str, pd.DataFrame]:
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"]).dt.tz_localize(None) if getattr(p["date"].dt, "tz", None) else pd.to_datetime(p["date"])
    out = {}
    for s, g in p.groupby("symbol"):
        x = g.set_index("date").sort_index()[["open", "high", "low", "close", "adj_close", "volume"]]
        out[s] = x[~x.index.duplicated()].dropna(subset=["close"])
    return out


def cash_daily(px: dict[str, pd.DataFrame], index: pd.DatetimeIndex, periods_per_year: int = 252) -> pd.Series:
    """Daily T-bill return from ^IRX (13-week discount rate, %), forward-filled onto ``index``."""
    irx = px["^IRX"]["close"].reindex(index, method="ffill").fillna(0) / 100
    return irx / periods_per_year


def stats(r: pd.Series, cash: pd.Series | None = None, ppy: int = 252) -> dict:
    """CAGR, vol, Sharpe (excess of cash), max drawdown, worst year, of a daily return series."""
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / ppy
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    c = cash.reindex(r.index).fillna(0) if cash is not None else 0.0
    ex = r - c
    vol = r.std() * math.sqrt(ppy)
    sharpe = ex.mean() / ex.std() * math.sqrt(ppy) if ex.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    yr = (1 + r).groupby(r.index.year).prod() - 1
    return {"cagr": float(cagr), "vol": float(vol), "sharpe": float(sharpe), "max_dd": float(dd),
            "worst_year": float(yr.min()), "years": float(yrs), "start": str(r.index[0].date()), "end": str(r.index[-1].date())}


def kelly(r: pd.Series, cash: pd.Series, ppy: int = 252) -> dict:
    """Growth-optimal (Kelly) leverage on a daily return stream financed at the cash rate, using realised moments
    (an upper bound: it assumes the future equals the past and leverage is free of extra costs)."""
    r = r.dropna()
    c = cash.reindex(r.index).fillna(0)
    ex = r - c
    mu, var = ex.mean() * ppy, ex.var() * ppy
    rf = c.mean() * ppy
    lev = mu / var if var > 0 else np.nan
    g = rf + mu ** 2 / (2 * var) if var > 0 else np.nan          # continuous-time growth at full Kelly
    return {"kelly_leverage": float(lev), "kelly_growth": float(math.exp(g) - 1), "half_kelly_growth": float(math.exp(rf + 0.375 * mu ** 2 / var) - 1)}


def levered(r: pd.Series, cash: pd.Series, lev: float, fee: float = 0.0, ppy: int = 252) -> pd.Series:
    """Daily-rebalanced leverage: lev x r - (lev - 1) x cash - fee/ppy (like a leveraged ETF or a margin account)."""
    c = cash.reindex(r.index).fillna(0)
    return lev * r - (lev - 1) * c - fee / ppy


def required_sharpe(target: float, rf: float = 0.02, fraction: float = 1.0) -> float:
    """Sharpe needed for growth ``target`` at a Kelly fraction (1 = full, 0.5 = half): g = rf + (f - f^2/2) S^2."""
    k = fraction - fraction ** 2 / 2
    return math.sqrt(max(math.log(1 + target) - rf, 0) / k)
