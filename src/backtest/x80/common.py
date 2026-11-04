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


def deflated_sharpe(sr: float, sr_var: float | None, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014): the probability that the best of ``n_trials`` rules has a
    true Sharpe above what the best of that many zero-skill rules would show by luck. ``sr`` and ``sr_var`` (the
    variance of the Sharpe ratios across the trials) are per period, not annualised; ``kurt`` is raw (normal = 3).
    ``sr_var=None`` uses the sampling variance of a zero Sharpe, 1 / (n_obs - 1): the lenient version, for when the
    trials include real effects of both signs that would inflate the cross-trial variance."""
    from statistics import NormalDist

    nd = NormalDist()
    g = 0.5772156649                                                       # Euler-Mascheroni
    sr0 = 0.0
    if sr_var is None:
        sr_var = 1.0 / (n_obs - 1)
    if n_trials > 1:
        sr0 = math.sqrt(max(sr_var, 0.0)) * ((1 - g) * nd.inv_cdf(1 - 1 / n_trials) + g * nd.inv_cdf(1 - 1 / (n_trials * math.e)))
    den = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    return float(nd.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(max(den, 1e-12))))


def funding_daily(f: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    """Funding per UTC day d = the prints in (00:00 d, 00:00 d+1]: what a position held from one 00:00 close to the next
    pays or receives. The print stamped 00:00 of day d is settled on the position held through day d-1."""
    day = (pd.to_datetime(f["time"]) - pd.Timedelta(seconds=1)).dt.floor("D")
    return f.assign(date=day).groupby(["date", symbol_col])["rate"].sum().unstack().sort_index()
