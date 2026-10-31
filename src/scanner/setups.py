"""Swing setup detection - explicit rules, evaluated on every day (so they can be back-tested).

A setup is a COMBINATION of evidence (trend context + location + volume + momentum),
never a single indicator crossover. Each setup returns a boolean mask and a 0-100
quality score built from documented sub-conditions.

LONG setups
-----------
breakout   Close > highest High of the previous 20 sessions, Volume >= 1.3 x MAVOL20,
           CLV >= 0.6 (closed in the upper part of the day's range), Close > EMA50.
           Quality: volume (1.3x -> 0, 3x -> 35), a volatility squeeze in the prior 10 sessions
           (BOLL width percentile <= 25: +20), also a 55-session high (+15), CLV (+0..15),
           not gapped > 2 ATR (+15, exhaustion risk otherwise).
pullback   Uptrend context: EMA21 > EMA50, EMA50 rising (5d), Close > EMA50.
           Location: the Low of the last 3 sessions came within 0.5 ATR of EMA21 or EMA9, Close >= EMA21 - 0.25 ATR,
           and price is >= 1 ATR below its 10-session high (a real pullback, not a pause at highs).
           Momentum reset: 40 <= RSI14 <= 62. Light pullback volume: 3-day avg volume < 1.1 x MAVOL20.
           Quality: lighter volume (+0..25), RSI6 turning up today (+20), HH/HL structure (+20),
           EMA21 > EMA50 > EMA200 (+15), depth 1-3 ATR from the 10-day high (+20).
squeeze    BOLL width in the lowest 15% of the previous 126 sessions, Close > EMA21 > EMA50,
           Close within 1.5 ATR of the 20-session high. A coiled range BEFORE a breakout - the plan
           only enters on a break of the 20-session high (conditional entry).
           Quality: tighter (+0..30), RS vs SPY 20d > 0 (+20), HH/HL (+20), OBV slope20 > 0 (+15), ADX < 20 (+15).
reclaim    Close crossed above EMA21 today or yesterday after >= 3 sessions below it,
           Volume >= 1.0 x MAVOL20, and (Close > EMA50 or EMA50 rising).
           Quality: volume (+0..30), CLV (+0..20), MACD histogram rising (+20), Close > EMA50 (+15), RSI14 > 50 (+15).
momentum   Close > EMA9 > EMA21 > EMA50, 20d return > 0 and outperforming SPY over 20 and 60 sessions,
           Close within 1 ATR of the 20-session high, extension (Close - EMA21) <= 2.5 ATR.
           Quality: RS 60d (+0..30), HH/HL (+20), up/down volume ratio > 1.2 (+20), RSI14 55-72 (+15), ADX >= 25 (+15).

SHORT setups (mirror images)
----------------------------
breakdown  Close < lowest Low of the previous 20 sessions, Volume >= 1.3 x MAVOL20, CLV <= 0.4, Close < EMA50.
rally_fail Downtrend: EMA21 < EMA50, EMA50 falling, Close < EMA50; the High of the last 3 sessions came within
           0.5 ATR of EMA21 (or EMA9) from below, Close <= EMA21 + 0.25 ATR, >= 1 ATR above the 10-session low,
           38 <= RSI14 <= 60, 3-day volume < 1.1 x MAVOL20.
ema21_loss Close crossed below EMA21 today or yesterday after >= 3 sessions above, Volume >= 1.0 x MAVOL20,
           and (Close < EMA50 or EMA50 falling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LONG_SETUPS = ("breakout", "pullback", "squeeze", "reclaim", "momentum")
SHORT_SETUPS = ("breakdown", "rally_fail", "ema21_loss")
SETUP_DIRECTION = {**{s: "long" for s in LONG_SETUPS}, **{s: "short" for s in SHORT_SETUPS}}
SETUP_NAMES = {
    "breakout": "Breakout (20D high, volume)", "pullback": "Pullback to EMA9/21 in uptrend",
    "squeeze": "Volatility squeeze near highs", "reclaim": "EMA21 reclaim", "momentum": "Momentum leader near highs",
    "breakdown": "Breakdown (20D low, volume)", "rally_fail": "Failed rally into falling EMA21",
    "ema21_loss": "EMA21 loss",
}
# When several setups are active the primary one is the first in this order.
PRIORITY = ("breakout", "breakdown", "pullback", "rally_fail", "reclaim", "ema21_loss", "squeeze", "momentum")


def _scale(x: pd.Series, lo: float, hi: float, points: float) -> pd.Series:
    """0 at ``lo``, ``points`` at ``hi`` (linear, clipped; works for lo > hi too)."""
    return (((x - lo) / (hi - lo)).clip(0, 1) * points).fillna(0)


def _b(mask: pd.Series) -> pd.Series:
    return mask.fillna(False).astype(bool)


def detect_setups(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean column ``setup_<name>`` and 0-100 ``quality_<name>`` for every setup and day."""
    c, a = df["close"], df["atr"]
    e9, e21, e50, e200 = df["ema9"], df["ema21"], df["ema50"], df["ema200"]
    vr, clv = df["volume_ratio"], df["clv"]
    rsi14, rsi6 = df["rsi14"], df["rsi6"]
    hhhl = df["swing_structure"].eq("HH/HL")
    lhll = df["swing_structure"].eq("LH/LL")
    out = pd.DataFrame(index=df.index)

    # ---------------------------------------------------------------- long
    breakout = _b((c > df["prior_high20"]) & (vr >= 1.3) & (clv >= 0.6) & (c > e50))
    q = (_scale(vr, 1.3, 3.0, 35) + 20 * _b(df["bbw_pct_min10"] <= 25) + 15 * _b(c > df["prior_high55"])
         + _scale(clv, 0.6, 1.0, 15) + 15 * _b(df["gap_atr"] <= 2))
    out["setup_breakout"], out["quality_breakout"] = breakout, q.where(breakout, 0)

    touched = (df["low3"] <= e21 + 0.5 * a) | (df["low3"] <= e9 + 0.5 * a)
    depth = (df["high10"] - c) / a
    pullback = _b((e21 > e50) & (df["ema50_slope"] > 0) & (c > e50) & touched & (c >= e21 - 0.25 * a)
                  & (depth >= 1) & rsi14.between(40, 62) & (df["vol3_vs_mavol"] < 1.1))
    q = (_scale(df["vol3_vs_mavol"], 1.1, 0.6, 25) + 20 * _b(rsi6 > df["rsi6_prev"]) + 20 * hhhl
         + 15 * _b((e21 > e50) & (e50 > e200)) + 20 * _b(depth.between(1, 3)))
    out["setup_pullback"], out["quality_pullback"] = pullback, q.where(pullback, 0)

    squeeze = _b((df["bb_width_percentile"] <= 15) & (c > e21) & (e21 > e50) & (df["high20"] - c <= 1.5 * a))
    q = (_scale(df["bb_width_percentile"], 15, 0, 30) + 20 * _b(df["rel_20d"] > 0) + 20 * hhhl
         + 15 * _b(df["obv_slope20"] > 0) + 15 * _b(df["adx"] < 20))
    out["setup_squeeze"], out["quality_squeeze"] = squeeze, q.where(squeeze, 0)

    # crossed up today (1st session above) or yesterday (2nd), after a run of >= 3 sessions below
    below_run_before = df["sessions_below_ema21"].shift(1).where(df["sessions_above_ema21"] == 1)
    below_run_before = below_run_before.fillna(df["sessions_below_ema21"].shift(2).where(df["sessions_above_ema21"] == 2))
    crossed_up = (c > e21) & (below_run_before >= 3)
    reclaim = _b(crossed_up & (vr >= 1.0) & ((c > e50) | (df["ema50_slope"] > 0)))
    q = (_scale(vr, 1.0, 2.5, 30) + _scale(clv, 0.5, 1.0, 20) + 20 * _b(df["macd_hist_3d"] > 0)
         + 15 * _b(c > e50) + 15 * _b(rsi14 > 50))
    out["setup_reclaim"], out["quality_reclaim"] = reclaim, q.where(reclaim, 0)

    momentum = _b((c > e9) & (e9 > e21) & (e21 > e50) & (df["return_20d"] > 0) & (df["rel_20d"] > 0)
                  & (df["rel_60d"] > 0) & (df["high20"] - c <= a) & (df["extension_atr"] <= 2.5))
    q = (_scale(df["rel_60d"], 0, 0.25, 30) + 20 * hhhl + 20 * _b(df["updown_vol_ratio"] > 1.2)
         + 15 * _b(rsi14.between(55, 72)) + 15 * _b(df["adx"] >= 25))
    out["setup_momentum"], out["quality_momentum"] = momentum, q.where(momentum, 0)

    # ---------------------------------------------------------------- short
    breakdown = _b((c < df["prior_low20"]) & (vr >= 1.3) & (clv <= 0.4) & (c < e50))
    q = (_scale(vr, 1.3, 3.0, 35) + 20 * _b(df["bbw_pct_min10"] <= 25) + 15 * _b(c < df["prior_low55"])
         + _scale(clv, 0.4, 0.0, 15) + 15 * _b(df["gap_atr"] >= -2))
    out["setup_breakdown"], out["quality_breakdown"] = breakdown, q.where(breakdown, 0)

    tested = (df["high3"] >= e21 - 0.5 * a) | (df["high3"] >= e9 - 0.5 * a)
    bounce = (c - df["low10"]) / a
    rally_fail = _b((e21 < e50) & (df["ema50_slope"] < 0) & (c < e50) & tested & (c <= e21 + 0.25 * a)
                    & (bounce >= 1) & rsi14.between(38, 60) & (df["vol3_vs_mavol"] < 1.1))
    q = (_scale(df["vol3_vs_mavol"], 1.1, 0.6, 25) + 20 * _b(rsi6 < df["rsi6_prev"]) + 20 * lhll
         + 15 * _b((e21 < e50) & (e50 < e200)) + 20 * _b(bounce.between(1, 3)))
    out["setup_rally_fail"], out["quality_rally_fail"] = rally_fail, q.where(rally_fail, 0)

    above_run_before = df["sessions_above_ema21"].shift(1).where(df["sessions_below_ema21"] == 1)
    above_run_before = above_run_before.fillna(df["sessions_above_ema21"].shift(2).where(df["sessions_below_ema21"] == 2))
    loss = _b((c < e21) & (above_run_before >= 3) & (vr >= 1.0) & ((c < e50) | (df["ema50_slope"] < 0)))
    q = (_scale(vr, 1.0, 2.5, 30) + _scale(clv, 0.5, 0.0, 20) + 20 * _b(df["macd_hist_3d"] < 0)
         + 15 * _b(c < e50) + 15 * _b(rsi14 < 50))
    out["setup_ema21_loss"], out["quality_ema21_loss"] = loss, q.where(loss, 0)

    # ---------------------------------------------------------------- primary
    masks = [out[f"setup_{name}"].to_numpy() for name in PRIORITY]
    out["primary_setup"] = np.select(masks, list(PRIORITY), default="")
    out["direction"] = out["primary_setup"].map(SETUP_DIRECTION).fillna("")
    return out


def active_setups(row: pd.Series) -> list[str]:
    return [name for name in PRIORITY if bool(row.get(f"setup_{name}", False))]
