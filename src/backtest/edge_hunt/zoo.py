"""Every online swing strategy, long AND short, on the point-in-time S&P 500 panel (Neo 2026-11-03: "test all the
swing short strategies too, think about why each one fails and whether our data can fix it").

Long: the 29 long swing42 strategies (Neo's list of online styles) + the year-end tax-loss bounce, with the
parameters swing42 chose on 2000-2014. Short (pre-registered here; the live short setups breakdown / rally_fail /
ema21_loss come from the live code in panel.py):
    s28  parabolic short (swing42 #28, chosen n=10, gain 40%)
    x01  RSI(2) >= 95 below the 200-day average (Connors, short side)
    x02  20-day z-score >= +2 below the 200-day average
    x03  TTM squeeze firing down (swing42 #25, side -1)
    x04  liquidity sweep of equal highs (swing42 #17, side -1)
    x05  gap-up fade: open >= 5% above the prior close and the close below the open
    x06  failed breakout: yesterday closed above the prior 20-day high, today closes back below it
    x07  dead-cat bounce: a crash 3-10 sessions ago, bounced >= 1.5 ATR, today a red close below EMA21 < EMA50
    x08  new 52-week low on >= 1.5x the 50-day volume
    x09  box breakdown: 40-day range <= 12% deep, close below it on >= 1.3x volume (mirror of swing42 #4)
Every trade uses the live exit in R (long: 1.5 ATR stop + 3-ATR chandelier trail, 20 sessions; short: the mirror),
entry at the next open, costs as swing42 (long 15 bps round trip, short 25 bps + 2%/yr borrow).
"""
from __future__ import annotations

import numpy as np

from src.backtest.edge_hunt.confluence import COOLDOWN, FN
from src.backtest.swing42 import core, strategies as st
from src.backtest.swing42.core import shift


def long_signals(p: core.Panel, chosen: dict[int, dict]) -> dict[str, np.ndarray]:
    out = {}
    fns = dict(FN)
    fns[42] = st.s42_tax_loss
    for sid, fn in fns.items():
        prm = (chosen.get(sid) or {}).get("params", {})
        sig = np.asarray(fn(p, None, **prm), dtype=bool) & p.elig
        out[f"L{sid:02d}"] = core.fresh(sig, COOLDOWN.get(sid, 10)) if sid != 42 else sig
    return out


def short_signals(p: core.Panel) -> dict[str, np.ndarray]:
    I = st.I
    C, O, H, L, V = p.C, p.O, p.H, p.L, p.V
    a = I(p, "atr", 14)
    sma200 = I(p, "sma", 200)
    e21, e50 = I(p, "ema", 21), I(p, "ema", 50)
    out = {}
    out["s28"] = st.s28_parabolic_short(p, None, n=10, gain=0.4, first_red=False)
    out["x01"] = (I(p, "rsi", 2) >= 95) & (C < sma200)
    m20, sd20 = I(p, "sma", 20), I(p, "std", 20)
    out["x02"] = ((C - m20) / sd20 >= 2) & (C < sma200)
    out["x03"] = st.s25_squeeze(p, None, n=3, kc=2.0, side=-1)
    out["x04"] = st.s17_sweep(p, None, side=-1, k=3, tol_atr=0.6)
    gap = O / shift(C, 1) - 1
    out["x05"] = (gap >= 0.05) & (C < O)
    hh20_prior = shift(I(p, "hh", 20), 2)                      # 20-day high ending two sessions ago
    out["x06"] = (shift(C, 1) > hh20_prior) & (C < hh20_prior)
    ret5, ret3 = I(p, "ret", 5), I(p, "ret", 3)
    crash = (ret5 <= -0.20) | (ret3 <= -0.15) | ((C - e21) <= -3 * a)
    recent_crash = core.roll(shift(crash.astype(float), 3), 8, "max", minp=1) > 0     # t-10 .. t-3
    low10 = core.roll(L, 10, "min")
    out["x07"] = recent_crash & (C - low10 >= 1.5 * a) & (C < O) & (C < e21) & (e21 < e50)
    ll252 = shift(core.roll(L, 252, "min"), 1)
    vavg50 = shift(I(p, "vsma", 50), 1)
    out["x08"] = (C < ll252) & (V >= 1.5 * vavg50)
    hh40, ll40 = shift(I(p, "hh", 40), 1), shift(core.roll(L, 40, "min"), 1)
    out["x09"] = ((hh40 / ll40 - 1) <= 0.12) & (C < ll40) & (V >= 1.3 * vavg50)
    return {k: core.fresh(np.nan_to_num(v.astype(float)) > 0, 10) & p.elig for k, v in out.items()}


def outcomes(p: core.Panel) -> dict[str, np.ndarray]:
    """R of the live exit for every stock-day, long and short (unit = 1.5 ATR / entry)."""
    a = core.atr(p)
    entry = shift(p.O, -1)
    unit = 1.5 * a / entry
    unit = np.where(unit > 1e-4, unit, np.nan)
    rl, hl = core.exit_path(p, "live", side=1, maxh=20, stop_k=1.5, trail_k=3.0)
    rs, hs = core.exit_path(p, "live", side=-1, maxh=20, stop_k=1.5, trail_k=3.0)
    cff = core.cached(p, "cff", lambda: core._paths(p))
    fwd = {k: shift(cff, -k) / entry - 1 for k in (1, 3, 5, 10, 20)}
    return {"RL": rl / unit, "RS": rs / unit, "retL": rl, "retS": rs, **{f"fwd{k}": v for k, v in fwd.items()}}
