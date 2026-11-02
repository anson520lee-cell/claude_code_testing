"""Candlestick patterns on any timeframe (daily, weekly, hourly) - one detector for the backtests AND the live terminal.

Neo (2026-11-02) asked what candles really predict for a held position ("does a hammer mean a rebound, or does the
price fall faster afterwards?"). This module only DETECTS patterns with explicit rules; what they predict is measured
in ``src/backtest/holding_study.py`` and shown next to every holding (``src/portfolio/holdings.py``). The ``folklore``
field records the textbook claim so the results can say where the data agree or disagree.

Notation per bar: O H L C V, body = |C - O|, range = H - L, upper wick = H - max(O, C), lower wick = min(O, C) - L,
CLV (close location) = (C - L) / range, ATR = Wilder ATR(14) of the same timeframe, previous bar = suffix 1
(C1 = previous close). "After a decline" = the previous close is below the close 5 bars earlier (C1 < C6); "after a
rise" = C1 > C6. Volume ratio = V / average V of the previous 20 bars (hourly: of the same hour of day over the
previous 20 sessions, so the open and close hours are compared with themselves).

Every flag at bar t uses bars up to and including t only (no look-ahead). Patterns need range >= 0.3 ATR unless the
rule says otherwise, so a flat, illiquid bar never counts as a doji or hammer.

Volume variants: each base pattern P is also tested as P@hv (volume ratio >= 1.5) and P@lv (<= 0.7).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

HV, LV = 1.5, 0.7

# id, name, bars, family, textbook claim ("bull" = expected to rise next, "bear" = to fall, "cont" = continuation)
PATTERNS: list[dict[str, Any]] = [
    # ---- one bar
    {"id": "hammer", "name": "Hammer (after a decline)", "bars": 1, "folklore": "bull"},
    {"id": "hanging_man", "name": "Hanging man (hammer shape after a rise)", "bars": 1, "folklore": "bear"},
    {"id": "inverted_hammer", "name": "Inverted hammer (after a decline)", "bars": 1, "folklore": "bull"},
    {"id": "shooting_star", "name": "Shooting star (after a rise)", "bars": 1, "folklore": "bear"},
    {"id": "doji", "name": "Doji (body <= 10% of range)", "bars": 1, "folklore": "reversal"},
    {"id": "long_legged_doji", "name": "Long-legged doji", "bars": 1, "folklore": "reversal"},
    {"id": "dragonfly_doji", "name": "Dragonfly doji", "bars": 1, "folklore": "bull"},
    {"id": "gravestone_doji", "name": "Gravestone doji", "bars": 1, "folklore": "bear"},
    {"id": "spinning_top", "name": "Spinning top", "bars": 1, "folklore": "indecision"},
    {"id": "bull_marubozu", "name": "Bullish marubozu (full-body up bar, >= 1 ATR)", "bars": 1, "folklore": "bull"},
    {"id": "bear_marubozu", "name": "Bearish marubozu (full-body down bar, >= 1 ATR)", "bars": 1, "folklore": "bear"},
    {"id": "long_lower_wick", "name": "Long lower wick / pin bar (lower wick >= 60% of a >= 1 ATR range)", "bars": 1, "folklore": "bull"},
    {"id": "long_upper_wick", "name": "Long upper wick (upper wick >= 60% of a >= 1 ATR range)", "bars": 1, "folklore": "bear"},
    {"id": "close_top10", "name": "Close in the top 10% of the range", "bars": 1, "folklore": "bull"},
    {"id": "close_bottom10", "name": "Close in the bottom 10% of the range", "bars": 1, "folklore": "bear"},
    {"id": "big_up", "name": "Big up bar (C - O >= 1.2 ATR)", "bars": 1, "folklore": "cont"},
    {"id": "big_down", "name": "Big down bar (O - C >= 1.2 ATR)", "bars": 1, "folklore": "cont"},
    {"id": "wide_range", "name": "Wide-range bar (range >= 2 ATR)", "bars": 1, "folklore": "climax"},
    {"id": "nr4", "name": "NR4 (narrowest range of 4 bars)", "bars": 4, "folklore": "breakout soon"},
    {"id": "nr7", "name": "NR7 (narrowest range of 7 bars)", "bars": 7, "folklore": "breakout soon"},
    # ---- two bars
    {"id": "bull_engulfing", "name": "Bullish engulfing", "bars": 2, "folklore": "bull"},
    {"id": "bear_engulfing", "name": "Bearish engulfing", "bars": 2, "folklore": "bear"},
    {"id": "bull_harami", "name": "Bullish harami", "bars": 2, "folklore": "bull"},
    {"id": "bear_harami", "name": "Bearish harami", "bars": 2, "folklore": "bear"},
    {"id": "piercing", "name": "Piercing line", "bars": 2, "folklore": "bull"},
    {"id": "dark_cloud", "name": "Dark cloud cover", "bars": 2, "folklore": "bear"},
    {"id": "tweezer_bottom", "name": "Tweezer bottom", "bars": 2, "folklore": "bull"},
    {"id": "tweezer_top", "name": "Tweezer top", "bars": 2, "folklore": "bear"},
    {"id": "inside_bar", "name": "Inside bar", "bars": 2, "folklore": "pause"},
    {"id": "outside_up", "name": "Outside bar closing up", "bars": 2, "folklore": "bull"},
    {"id": "outside_down", "name": "Outside bar closing down", "bars": 2, "folklore": "bear"},
    {"id": "gap_up", "name": "Gap up (low above the previous high)", "bars": 2, "folklore": "cont"},
    {"id": "gap_down", "name": "Gap down (high below the previous low)", "bars": 2, "folklore": "cont"},
    {"id": "gap_up_faded", "name": "Opened >= 0.5 ATR up, closed below the previous close", "bars": 2, "folklore": "bear"},
    {"id": "gap_down_recovered", "name": "Opened >= 0.5 ATR down, closed above the previous close", "bars": 2, "folklore": "bull"},
    {"id": "key_reversal_up", "name": "Key reversal up (lower low, close above the previous high)", "bars": 2, "folklore": "bull"},
    {"id": "key_reversal_down", "name": "Key reversal down (higher high, close below the previous low)", "bars": 2, "folklore": "bear"},
    # ---- three bars
    {"id": "morning_star", "name": "Morning star", "bars": 3, "folklore": "bull"},
    {"id": "evening_star", "name": "Evening star", "bars": 3, "folklore": "bear"},
    {"id": "three_white_soldiers", "name": "Three white soldiers", "bars": 3, "folklore": "bull"},
    {"id": "three_black_crows", "name": "Three black crows", "bars": 3, "folklore": "bear"},
    {"id": "three_inside_up", "name": "Three inside up", "bars": 3, "folklore": "bull"},
    {"id": "three_inside_down", "name": "Three inside down", "bars": 3, "folklore": "bear"},
    {"id": "up3", "name": "Three higher closes", "bars": 3, "folklore": "cont"},
    {"id": "down3", "name": "Three lower closes", "bars": 3, "folklore": "cont"},
]
PATTERN_IDS = [p["id"] for p in PATTERNS]
PATTERN_INFO = {p["id"]: p for p in PATTERNS}

# Hourly-only patterns (need sessions: previous session high / low, session VWAP, first / last hour)
HOURLY_EXTRA: list[dict[str, Any]] = [
    {"id": "spring", "name": "Pin through the previous day's low and back above it (打針收回)", "folklore": "bull"},
    {"id": "upthrust", "name": "Spike above the previous day's high and back below it", "folklore": "bear"},
    {"id": "session_low_reclaim", "name": "New low of the day, closed in the upper half of the hour", "folklore": "bull"},
    {"id": "session_high_reject", "name": "New high of the day, closed in the lower half of the hour", "folklore": "bear"},
    {"id": "vwap_cross_up", "name": "Close crossed above the session VWAP", "folklore": "bull"},
    {"id": "vwap_cross_down", "name": "Close crossed below the session VWAP", "folklore": "bear"},
    {"id": "first_hour_reversal_up", "name": "First hour: opened below yesterday's close, closed the hour up", "folklore": "bull"},
    {"id": "first_hour_reversal_down", "name": "First hour: opened above yesterday's close, closed the hour down", "folklore": "bear"},
    {"id": "last_hour_strong", "name": "Last hour up, closing in its top 20%", "folklore": "bull"},
    {"id": "last_hour_weak", "name": "Last hour down, closing in its bottom 20%", "folklore": "bear"},
    {"id": "vol_spike_up", "name": "Hour volume >= 2x its usual, hour closed up", "folklore": "bull"},
    {"id": "vol_spike_down", "name": "Hour volume >= 2x its usual, hour closed down", "folklore": "bear"},
]
HOURLY_BASE = ["hammer", "shooting_star", "doji", "long_lower_wick", "long_upper_wick", "bull_engulfing", "bear_engulfing",
               "big_up", "big_down", "wide_range", "inside_bar", "up3", "down3"]
HOURLY_IDS = HOURLY_BASE + [p["id"] for p in HOURLY_EXTRA]
HOURLY_INFO = {**{k: PATTERN_INFO[k] for k in HOURLY_BASE}, **{p["id"]: p for p in HOURLY_EXTRA}}


def wilder_atr(h: np.ndarray, lo: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder ATR(period) (same formula as ``src.technical.indicators.atr``), numpy, NaN-tolerant at the start."""
    n = len(c)
    tr = np.full(n, np.nan)
    if n == 0:
        return tr
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - lo, np.abs(h - pc), np.abs(lo - pc)]), axis=0)
    tr[0] = h[0] - lo[0]
    out = np.full(n, np.nan)
    ok = np.isfinite(tr)
    idx = np.flatnonzero(ok)
    if idx.size < period:
        return out
    i0 = idx[period - 1]
    out[i0] = np.nanmean(tr[idx[0]:i0 + 1])
    for j in range(i0 + 1, n):
        out[j] = out[j - 1] if not np.isfinite(tr[j]) else (out[j - 1] * (period - 1) + tr[j]) / period
    return out


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k] if k else x
    return out


def bar_features(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, v: np.ndarray, atr: np.ndarray,
                 vol_ratio: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Continuous shape of each bar (scale-free): body / wicks / range in ATR and as a share of the range, CLV,
    gap in ATR, volume ratio."""
    rng = h - lo
    safe = np.where(rng > 0, rng, np.nan)
    a = np.where(atr > 0, atr, np.nan)
    body = np.abs(c - o)
    up = h - np.maximum(o, c)
    dn = np.minimum(o, c) - lo
    c1 = _shift(c, 1)
    if vol_ratio is None:
        vs = pd.Series(v, dtype=float)
        vol_ratio = (vs / vs.shift(1).rolling(20, min_periods=10).mean()).to_numpy()
    return {"body_atr": (c - o) / a, "range_atr": rng / a, "upper_frac": up / safe, "lower_frac": dn / safe,
            "body_frac": body / safe, "clv": (c - lo) / safe, "gap_atr": (o - c1) / a, "ret_atr": (c - c1) / a,
            "vol_ratio": vol_ratio}


def detect(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, atr: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean flag per base pattern (``PATTERN_IDS``) for every bar."""
    o, h, lo, c, atr = (np.asarray(x, dtype=float) for x in (o, h, lo, c, atr))
    n = len(c)
    rng = h - lo
    a = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
    body = np.abs(c - o)
    up = h - np.maximum(o, c)
    dn = np.minimum(o, c) - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        clv = np.where(rng > 0, (c - lo) / np.where(rng > 0, rng, 1), np.nan)
        real = rng >= 0.3 * a
        o1, h1, l1, c1 = (_shift(x, 1) for x in (o, h, lo, c))
        o2, h2, l2, c2 = (_shift(x, 2) for x in (o, h, lo, c))
        c6 = _shift(c, 6)
        body1, body2 = np.abs(c1 - o1), np.abs(c2 - o2)
        rng1 = h1 - l1
        down_before = c1 < c6
        up_before = c1 > c6
        bull, bear = c > o, c < o
        bull1, bear1 = c1 > o1, c1 < o1
        bull2, bear2 = c2 > o2, c2 < o2
        hammer_shape = real & (dn >= 2 * body) & (up <= 0.2 * rng) & (rng >= 0.5 * a)
        inv_shape = real & (up >= 2 * body) & (dn <= 0.2 * rng) & (rng >= 0.5 * a)
        doji = real & (body <= 0.1 * rng)
        F: dict[str, np.ndarray] = {
            "hammer": hammer_shape & down_before,
            "hanging_man": hammer_shape & up_before,
            "inverted_hammer": inv_shape & down_before,
            "shooting_star": inv_shape & up_before,
            "doji": doji,
            "long_legged_doji": doji & (up >= 0.3 * rng) & (dn >= 0.3 * rng) & (rng >= a),
            "dragonfly_doji": doji & (up <= 0.1 * rng) & (dn >= 0.6 * rng),
            "gravestone_doji": doji & (dn <= 0.1 * rng) & (up >= 0.6 * rng),
            "spinning_top": real & (body > 0.1 * rng) & (body <= 0.3 * rng) & (up >= 0.25 * rng) & (dn >= 0.25 * rng),
            "bull_marubozu": bull & (body >= 0.9 * rng) & (rng >= a),
            "bear_marubozu": bear & (body >= 0.9 * rng) & (rng >= a),
            "long_lower_wick": (dn >= 0.6 * rng) & (rng >= a),
            "long_upper_wick": (up >= 0.6 * rng) & (rng >= a),
            "close_top10": real & (clv >= 0.9) & (rng >= 0.5 * a),
            "close_bottom10": real & (clv <= 0.1) & (rng >= 0.5 * a),
            "big_up": (c - o) >= 1.2 * a,
            "big_down": (o - c) >= 1.2 * a,
            "wide_range": rng >= 2 * a,
            "bull_engulfing": bear1 & bull & (o <= c1) & (c >= o1) & (body > body1) & (body >= 0.3 * a),
            "bear_engulfing": bull1 & bear & (o >= c1) & (c <= o1) & (body > body1) & (body >= 0.3 * a),
            "bull_harami": bear1 & (body1 >= 0.6 * a) & bull & (o >= c1) & (c <= o1) & (body < body1),
            "bear_harami": bull1 & (body1 >= 0.6 * a) & bear & (o <= c1) & (c >= o1) & (body < body1),
            "piercing": bear1 & (body1 >= 0.6 * a) & (o < c1) & bull & (c > (o1 + c1) / 2) & (c < o1),
            "dark_cloud": bull1 & (body1 >= 0.6 * a) & (o > c1) & bear & (c < (o1 + c1) / 2) & (c > o1),
            "tweezer_bottom": down_before & (np.abs(lo - l1) <= 0.1 * a) & bear1 & bull,
            "tweezer_top": up_before & (np.abs(h - h1) <= 0.1 * a) & bull1 & bear,
            "inside_bar": (h <= h1) & (lo >= l1) & (rng1 > 0),
            "outside_up": (h > h1) & (lo < l1) & (c > c1) & (clv >= 0.5),
            "outside_down": (h > h1) & (lo < l1) & (c < c1) & (clv <= 0.5),
            "gap_up": lo > h1,
            "gap_down": h < l1,
            "gap_up_faded": (o >= c1 + 0.5 * a) & (c < c1),
            "gap_down_recovered": (o <= c1 - 0.5 * a) & (c > c1),
            "key_reversal_up": (lo < l1) & (c > h1),
            "key_reversal_down": (h > h1) & (c < l1),
            "morning_star": bear2 & (body2 >= 0.6 * a) & (body1 <= 0.3 * a) & (np.maximum(o1, c1) <= c2 + 0.1 * a)
                            & bull & (c > (o2 + c2) / 2),
            "evening_star": bull2 & (body2 >= 0.6 * a) & (body1 <= 0.3 * a) & (np.minimum(o1, c1) >= c2 - 0.1 * a)
                            & bear & (c < (o2 + c2) / 2),
            "three_white_soldiers": bull & bull1 & bull2 & (c > c1) & (c1 > c2) & (o >= o1) & (o <= c1) & (o1 >= o2)
                                    & (o1 <= c2) & (clv >= 0.7) & (body >= 0.5 * a) & (body1 >= 0.5 * a) & (body2 >= 0.5 * a),
            "three_black_crows": bear & bear1 & bear2 & (c < c1) & (c1 < c2) & (o <= o1) & (o >= c1) & (o1 <= o2)
                                 & (o1 >= c2) & (clv <= 0.3) & (body >= 0.5 * a) & (body1 >= 0.5 * a) & (body2 >= 0.5 * a),
            "up3": (c > c1) & (c1 > c2) & (c2 > _shift(c, 3)),
            "down3": (c < c1) & (c1 < c2) & (c2 < _shift(c, 3)),
        }
        rs = pd.Series(rng)
        F["nr4"] = real & (rng < rs.shift(1).rolling(3).min().to_numpy())
        F["nr7"] = real & (rng < rs.shift(1).rolling(6).min().to_numpy())
        harami_bull_prev = _shift(F["bull_harami"].astype(float), 1) == 1
        harami_bear_prev = _shift(F["bear_harami"].astype(float), 1) == 1
        F["three_inside_up"] = harami_bull_prev & bull & (c > o2)
        F["three_inside_down"] = harami_bear_prev & bear & (c < o2)
    out = {}
    for k in PATTERN_IDS:
        x = np.asarray(F[k], dtype=bool) & np.isfinite(a)
        out[k] = x if len(x) == n else np.zeros(n, dtype=bool)
    return out


def with_volume(flags: dict[str, np.ndarray], vol_ratio: np.ndarray, ids: list[str] | None = None) -> dict[str, np.ndarray]:
    """P, P@hv (volume ratio >= 1.5) and P@lv (<= 0.7) for every base pattern."""
    vr = np.asarray(vol_ratio, dtype=float)
    hv, lv = vr >= HV, vr <= LV
    out: dict[str, np.ndarray] = {}
    for k in ids or list(flags):
        out[k] = flags[k]
        out[f"{k}@hv"] = flags[k] & hv
        out[f"{k}@lv"] = flags[k] & lv
    return out


def frame_flags(f: pd.DataFrame) -> pd.DataFrame:
    """Base daily patterns for a feature / price frame with open/high/low/close (+ atr if present)."""
    o, h, lo, c = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = f["atr"].to_numpy(dtype=float) if "atr" in f else wilder_atr(h, lo, c)
    return pd.DataFrame(detect(o, h, lo, c, atr), index=f.index)


# ====================================================================== weekly context (from daily bars, causal)
def weekly_context(f: pd.DataFrame) -> pd.DataFrame:
    """Per DAY: the state of the last COMPLETED week (known on every day of the following week) and the current week
    so far. Columns: w_up_trend (last weekly close > SMA10 of weekly closes), w_rsi14 (Wilder RSI of weekly closes),
    w_ret (last week's return), w_close_vs_sma10 (in %), wtd_ret (week-to-date return vs last week's close),
    w_prev_pattern_* for a few weekly candles of the last completed week."""
    idx = f.index
    wk = pd.Series(idx.to_period("W-FRI"), index=idx)
    g = f.groupby(wk.to_numpy())
    W = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    wc = W["close"]
    sma10 = wc.rolling(10, min_periods=10).mean()
    delta = wc.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gain / loss.where(loss > 0))
    watr = wilder_atr(W["high"].to_numpy(float), W["low"].to_numpy(float), wc.to_numpy(float), 10)
    wf = detect(W["open"].to_numpy(float), W["high"].to_numpy(float), W["low"].to_numpy(float), wc.to_numpy(float), watr)
    weekly = pd.DataFrame({"w_up_trend": (wc > sma10).astype(float).where(sma10.notna()), "w_rsi14": rsi,
                           "w_ret": wc.pct_change(), "w_close_vs_sma10": wc / sma10 - 1,
                           **{f"w_{k}": wf[k].astype(float) for k in ("hammer", "shooting_star", "bull_engulfing", "bear_engulfing",
                                                                       "long_lower_wick", "long_upper_wick", "doji")}}, index=W.index)
    prev = weekly.shift(1)                       # the last COMPLETED week, as seen during the current one
    out = prev.reindex(wk.to_numpy())
    out.index = idx
    last_close = wc.shift(1).reindex(wk.to_numpy()).to_numpy()
    out["wtd_ret"] = f["close"].to_numpy(float) / last_close - 1
    return out


# ====================================================================== hourly bars
def hourly_flags(hb: pd.DataFrame) -> pd.DataFrame:
    """Hourly patterns (``HOURLY_IDS``) + volume ratio + session columns for a frame of regular-hours 1-hour bars
    indexed in New York time (Yahoo's 60-minute bars start at 9:30, the last one is the 15:30-16:00 half hour)."""
    if hb is None or hb.empty:
        return pd.DataFrame()
    hb = hb.sort_index()
    o, h, lo, c, v = (hb[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    atr = wilder_atr(h, lo, c, 14)
    day = np.asarray(hb.index.date)
    hour = np.asarray([t.hour * 60 + t.minute for t in hb.index])
    df = pd.DataFrame({"day": day, "slot": hour, "v": v})
    # usual volume of the same slot over the previous 20 sessions
    piv = df.pivot_table(index="day", columns="slot", values="v", aggfunc="sum")
    usual = piv.shift(1).rolling(20, min_periods=8).mean()
    usual_v = usual.stack().rename("usual").reset_index()
    df = df.merge(usual_v, on=["day", "slot"], how="left")
    vr = (df["v"] / df["usual"]).to_numpy()
    flags = detect(o, h, lo, c, atr)
    F = {k: flags[k] for k in HOURLY_BASE}
    new_day = np.r_[True, day[1:] != day[:-1]]
    last_of_day = np.r_[day[1:] != day[:-1], True]
    # session aggregates up to this bar
    sid = np.cumsum(new_day) - 1
    s = pd.DataFrame({"sid": sid, "h": h, "lo": lo, "tpv": (h + lo + c) / 3 * np.nan_to_num(v), "v": np.nan_to_num(v)})
    run_hi = s.groupby("sid")["h"].cummax().to_numpy()
    run_lo = s.groupby("sid")["lo"].cummin().to_numpy()
    prior_hi = np.where(new_day, np.nan, np.r_[np.nan, run_hi[:-1]])
    prior_lo = np.where(new_day, np.nan, np.r_[np.nan, run_lo[:-1]])
    vwap = (s.groupby("sid")["tpv"].cumsum() / s.groupby("sid")["v"].cumsum().replace(0, np.nan)).to_numpy()
    sess_hi = pd.Series(run_hi).groupby(sid).transform("last").to_numpy()
    sess_lo = pd.Series(run_lo).groupby(sid).transform("last").to_numpy()
    sess_close = pd.Series(c).groupby(sid).transform("last").to_numpy()
    first_idx = np.flatnonzero(new_day)
    prev_day_hi = np.full(n, np.nan)
    prev_day_lo = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    sh = sess_hi[first_idx]
    sl_ = sess_lo[first_idx]
    sc = sess_close[first_idx]
    prev_hi_s = np.r_[np.nan, sh[:-1]]
    prev_lo_s = np.r_[np.nan, sl_[:-1]]
    prev_c_s = np.r_[np.nan, sc[:-1]]
    prev_day_hi[:] = prev_hi_s[sid]
    prev_day_lo[:] = prev_lo_s[sid]
    prev_day_close[:] = prev_c_s[sid]
    with np.errstate(invalid="ignore"):
        rng = h - lo
        clv = np.where(rng > 0, (c - lo) / np.where(rng > 0, rng, 1), np.nan)
        c1 = _shift(c, 1)
        vw1 = np.where(new_day, np.nan, _shift(vwap, 1))
        F["spring"] = (lo < prev_day_lo) & (c > prev_day_lo) & (clv >= 0.5)
        F["upthrust"] = (h > prev_day_hi) & (c < prev_day_hi) & (clv <= 0.5)
        F["session_low_reclaim"] = ~new_day & (lo < prior_lo) & (clv >= 0.5) & (rng >= 0.5 * atr)
        F["session_high_reject"] = ~new_day & (h > prior_hi) & (clv <= 0.5) & (rng >= 0.5 * atr)
        F["vwap_cross_up"] = ~new_day & (c > vwap) & (c1 <= vw1)
        F["vwap_cross_down"] = ~new_day & (c < vwap) & (c1 >= vw1)
        F["first_hour_reversal_up"] = new_day & (o < prev_day_close) & (c > o)
        F["first_hour_reversal_down"] = new_day & (o > prev_day_close) & (c < o)
        F["last_hour_strong"] = last_of_day & (c > o) & (clv >= 0.8)
        F["last_hour_weak"] = last_of_day & (c < o) & (clv <= 0.2)
        F["vol_spike_up"] = (vr >= 2.0) & (c > o)
        F["vol_spike_down"] = (vr >= 2.0) & (c < o)
    out = pd.DataFrame({k: np.asarray(F[k], dtype=bool) for k in HOURLY_IDS}, index=hb.index)
    out["vol_ratio"] = vr
    out["atr"] = atr
    out["vwap"] = vwap
    out["new_day"] = new_day
    out["last_of_day"] = last_of_day
    return out


def describe_today(flags_row: dict[str, Any], vol_ratio: float | None, info: dict[str, dict] | None = None) -> list[dict[str, Any]]:
    """Patterns present on one bar, with their volume variant id (used by the live holdings page)."""
    info = info or PATTERN_INFO
    out = []
    for k, on in flags_row.items():
        if k not in info or not bool(on):
            continue
        tag = "@hv" if vol_ratio is not None and np.isfinite(vol_ratio) and vol_ratio >= HV else \
              "@lv" if vol_ratio is not None and np.isfinite(vol_ratio) and vol_ratio <= LV else ""
        out.append({"id": k, "variant": k + tag, "name": info[k]["name"], "folklore": info[k].get("folklore")})
    return out
