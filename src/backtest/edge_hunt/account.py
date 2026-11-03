"""Account simulation of the live rules on the point-in-time signals: 10 slots, 1% risk per trade, position cap 25%,
priority crash rebound > squeeze > breakout (the live setup bonus order), ties broken at random (several seeds).
P&L is booked at the exit session (no intra-trade marking), so drawdowns are slightly understated."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIORITY = {"crash_rebound": 0, "squeeze": 1, "breakout": 2}


def simulate(trades: pd.DataFrame, cal: pd.DatetimeIndex, risk: float = 0.01, slots: int = 10, cap: float = 0.25,
             seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = trades.copy()
    t["risk_pct"] = (t["ret"] / t["r"]).where(t["r"].abs() > 1e-6)
    t = t.dropna(subset=["risk_pct"])
    t = t[t["risk_pct"] > 0]
    pos = np.searchsorted(cal.values, t["date"].values.astype("datetime64[ns]"))
    t["entry_i"] = pos + 1
    t["exit_i"] = t["entry_i"] + t["days"].astype(int) - 1
    # optional prio_adj (0 / 0.5): a score bonus that puts favoured candidates first inside their setup's priority
    t["prio"] = t["setup"].map(PRIORITY) + rng.random(len(t)) * 0.5 + (t["prio_adj"] if "prio_adj" in t else 0.0)
    t = t.sort_values(["entry_i", "prio"])
    eq = 1.0
    open_pos: list[tuple[int, float]] = []          # (exit_i, pnl)
    curve = np.full(len(cal), np.nan)
    by_day = {i: g for i, g in t.groupby("entry_i")}
    for i in range(len(cal)):
        keep = []
        for ex, pnl in open_pos:
            if ex <= i:
                eq += pnl
            else:
                keep.append((ex, pnl))
        open_pos = keep
        g = by_day.get(i)
        if g is not None:
            for _, row in g.iterrows():
                if len(open_pos) >= slots:
                    break
                size = min(risk / row["risk_pct"], cap) * eq
                open_pos.append((int(row["exit_i"]), size * row["ret"]))
        curve[i] = eq
    return pd.Series(curve, index=cal)


def stats(curve: pd.Series, start: str, end: str) -> dict:
    c = curve[(curve.index >= start) & (curve.index <= end)].dropna()
    if len(c) < 30:
        return {}
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    cagr = (c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1
    dd = (c / c.cummax() - 1).min()
    m = c.resample("ME").last().pct_change().dropna()
    sharpe = m.mean() / m.std() * np.sqrt(12) if m.std() > 0 else np.nan
    return {"cagr": float(cagr), "max_dd": float(dd), "sharpe": float(sharpe)}
