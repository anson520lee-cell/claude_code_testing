"""Turning a score into a leverage multiplier without looking ahead.

Every rule returns m[t] for week t from information up to the close of week t-1 (the features are shifted by one).
Tiers use EXPANDING quantiles of the feature's own past (at least two years of history; before that m = 1), so a
threshold never uses a value from the future. Multipliers are relative to the x80 vol-target leverage:
L[t] = clip(m[t] x target / vol, 0, cap).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HIST = 104


def expanding_rank(x: pd.Series, min_hist: int = MIN_HIST) -> pd.Series:
    """Percentile of x[t] among x[..t] (itself included), NaN until min_hist observations."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    hist: list[float] = []
    for i, a in enumerate(v):
        if np.isfinite(a):
            hist.append(a)
            if len(hist) >= min_hist:
                h = np.asarray(hist)
                out[i] = (np.sum(h < a) + 0.5 * np.sum(h == a)) / len(h)
    return pd.Series(out, index=x.index)


def tiers(pct: pd.Series, mults=(0.5, 1.0, 1.5), cuts=(1 / 3, 2 / 3)) -> pd.Series:
    """Map a percentile (0-1) to multipliers: bottom third, middle, top third (NaN -> 1)."""
    m = pd.Series(mults[1], index=pct.index)
    m[pct <= cuts[0]] = mults[0]
    m[pct > cuts[1]] = mults[2]
    return m.where(pct.notna(), 1.0)


def single(feature: pd.Series, direction: int, mults=(0.5, 1.0, 1.5)) -> pd.Series:
    """direction +1: more leverage when the feature is high; -1: when it is low. Uses the value at t-1."""
    pct = expanding_rank(feature)
    if direction < 0:
        pct = 1 - pct
    return tiers(pct, mults).shift(1).fillna(1.0)


def expanding_z(x: pd.Series, min_hist: int = MIN_HIST) -> pd.Series:
    mu = x.expanding(min_periods=min_hist).mean()
    sd = x.expanding(min_periods=min_hist).std()
    return (x - mu) / sd


def composite(F: pd.DataFrame, signs: dict[str, int]) -> pd.Series:
    """Average of sign-adjusted expanding z-scores (each clipped at +-3) of the chosen features, at week t."""
    z = pd.DataFrame({k: (expanding_z(F[k]) * s).clip(-3, 3) for k, s in signs.items()})
    return z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1) >= max(1, len(signs) // 2))


def from_score(score: pd.Series, mults=(0.5, 1.0, 1.5)) -> pd.Series:
    return tiers(expanding_rank(score), mults).shift(1).fillna(1.0)


def continuous(score: pd.Series, k: float = 0.5, lo: float = 0.25, hi: float = 1.75) -> pd.Series:
    """m = clip(1 + k x z(score), lo, hi), z from the score's own expanding history."""
    z = expanding_z(score)
    return (1 + k * z).clip(lo, hi).shift(1).fillna(1.0)


def walk_forward_model(F: pd.DataFrame, y: pd.Series, horizon: int, first_year: int, kind: str = "ridge",
                       alpha: float = 10.0, start: str = "2006-01-01") -> pd.Series:
    """Refit every January on all weeks whose target is fully known by then (target = book return over the next
    ``horizon`` weeks); predict each week of that year. Features: expanding z-scores (past only), NaN -> 0.
    kind 'ridge' predicts the mean, 'logit' the probability of a winning week (target > 0). Returns the prediction
    for week t made with the features at t (shift it by one before use)."""
    from sklearn.linear_model import LogisticRegression, Ridge

    Z = pd.DataFrame({c: expanding_z(F[c]).clip(-3, 3) for c in F.columns}).fillna(0.0)
    Z = Z[Z.index >= start]
    yy = y.reindex(Z.index)
    pred = pd.Series(np.nan, index=Z.index)
    for year in range(first_year, Z.index.max().year + 1):
        t0 = pd.Timestamp(f"{year}-01-01")
        # a row's target spans weeks t+1..t+horizon, so it is known only when t + horizon weeks < t0
        known = Z.index < t0 - pd.Timedelta(weeks=horizon)
        Xtr, ytr = Z[known], yy[known]
        ok = ytr.notna()
        Xtr, ytr = Xtr[ok], ytr[ok]
        rows = (Z.index >= t0) & (Z.index < pd.Timestamp(f"{year + 1}-01-01"))
        if len(ytr) < 156 or not rows.any():
            continue
        if kind == "ridge":
            mdl = Ridge(alpha=alpha).fit(Xtr.to_numpy(), ytr.to_numpy())
            pred[rows] = mdl.predict(Z[rows].to_numpy())
        else:
            mdl = LogisticRegression(C=1.0 / alpha, max_iter=1000).fit(Xtr.to_numpy(), (ytr > 0).astype(int).to_numpy())
            pred[rows] = mdl.predict_proba(Z[rows].to_numpy())[:, 1]
    return pred
