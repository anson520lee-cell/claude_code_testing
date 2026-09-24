"""Deterministic classification rules for the 2-7 trading-day technical state.

Every rule is a plain comparison written below (and listed in README.md,
section 8). Rules are applied to every trading day, so the same labels can be
studied historically. A day whose inputs are missing gets "n/a" - a label is
never guessed.

Indicator priority for a 2-7 day swing, which is also the weight each one gets
in the Swing Technical Condition: price structure and EMA9 / EMA21 first, then
volume vs MAVOL20, RSI6 / RSI14, MACD 12/26/9, BOLL(20, 1.8), relative strength
and EMA50. EMA200 / EMA250 are context only; ATR and support / resistance
describe risk and location and add no points.

These labels DESCRIBE the current technical state. They are not predictions
and not trading recommendations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.technical.structure import STRUCTURE_POINTS

NA = "n/a"
CROSS_LOOKBACK = 3  # a crossover "counts" for this many sessions (today and the 2 before)


def _labels(conditions: list[pd.Series], choices: list[str], default: str, missing: pd.Series) -> pd.Series:
    """First matching condition wins; rows with missing inputs become "n/a"."""
    index = missing.index
    values = np.select([c.fillna(False).to_numpy(dtype=bool) for c in conditions], choices, default=default)
    return pd.Series(np.where(missing.to_numpy(dtype=bool), NA, values), index=index, dtype=object)


def recent_cross(fast: pd.Series, slow: pd.Series, lookback: int = CROSS_LOOKBACK) -> pd.Series:
    """+1 if ``fast`` is above ``slow`` today and was at or below it on one of the previous
    ``lookback`` sessions (a cross up within that window), -1 for the mirror case, else 0."""
    diff = fast - slow
    was_below = pd.concat([diff.shift(k) <= 0 for k in range(1, lookback + 1)], axis=1).any(axis=1)
    was_above = pd.concat([diff.shift(k) >= 0 for k in range(1, lookback + 1)], axis=1).any(axis=1)
    values = np.select([(diff > 0) & was_below, (diff < 0) & was_above], [1, -1], 0)
    return pd.Series(values, index=diff.index)


def last_cross_age(fast: pd.Series, slow: pd.Series, lookback: int = CROSS_LOOKBACK) -> int | None:
    """Sessions since the most recent cross on the LAST day (0 = today), or None if none within ``lookback``."""
    signs = np.sign((fast - slow).tail(lookback + 1).to_numpy(dtype=float))
    if len(signs) < 2 or not np.isfinite(signs).all() or signs[-1] == 0:
        return None
    for age in range(0, len(signs) - 1):
        if signs[-1 - age] != signs[-2 - age]:
            return age
    return None


# ----------------------------------------------------------------- swing trend (EMA9 / EMA21)
STRONG_UPTREND = "Strong short-term uptrend"
UPTREND = "Short-term uptrend"
IMPROVING = "Improving"
MIXED = "Mixed"
DETERIORATING = "Deteriorating"
DOWNTREND = "Short-term downtrend"
STRONG_DOWNTREND = "Strong short-term downtrend"
UP_TRENDS = (STRONG_UPTREND, UPTREND)
DOWN_TRENDS = (STRONG_DOWNTREND, DOWNTREND)


def classify_trend(df: pd.DataFrame) -> pd.Series:
    """Swing trend from price, EMA9 and EMA21 and their 3-session slopes (first matching rule wins).

    1. Strong short-term uptrend:   C > EMA9 > EMA21, EMA9 slope > 0 and EMA21 slope > 0
    2. Strong short-term downtrend: C < EMA9 < EMA21, EMA9 slope < 0 and EMA21 slope < 0
    3. Short-term uptrend:          EMA9 > EMA21, C > EMA21 and EMA21 slope > 0
    4. Short-term downtrend:        EMA9 < EMA21, C < EMA21 and EMA21 slope < 0
    5. Improving:                   C > EMA9 and EMA9 slope > 0 (the fast EMA turns first)
    6. Deteriorating:               C < EMA9 and EMA9 slope < 0
    7. Mixed:                       anything else
    EMA50 / EMA200 / EMA250 are deliberately NOT used here (secondary / context only).
    """
    c, e9, e21 = df["close"], df["ema9"], df["ema21"]
    s9, s21 = df["ema9_slope"], df["ema21_slope"]
    conditions = [
        (c > e9) & (e9 > e21) & (s9 > 0) & (s21 > 0),
        (c < e9) & (e9 < e21) & (s9 < 0) & (s21 < 0),
        (e9 > e21) & (c > e21) & (s21 > 0),
        (e9 < e21) & (c < e21) & (s21 < 0),
        (c > e9) & (s9 > 0),
        (c < e9) & (s9 < 0),
    ]
    choices = [STRONG_UPTREND, STRONG_DOWNTREND, UPTREND, DOWNTREND, IMPROVING, DETERIORATING]
    missing = df[["close", "ema9", "ema21", "ema9_slope", "ema21_slope"]].isna().any(axis=1)
    return _labels(conditions, choices, MIXED, missing)


EMA50_UP = "Price above rising EMA50"
EMA50_DOWN = "Price below falling EMA50"
EMA50_MIXED = "Mixed"


def classify_ema50(df: pd.DataFrame) -> pd.Series:
    """EMA50 as secondary confirmation: C > EMA50 with EMA50 rising, C < EMA50 with EMA50 falling, else Mixed."""
    c, e50, s50 = df["close"], df["ema50"], df["ema50_slope"]
    missing = df[["close", "ema50", "ema50_slope"]].isna().any(axis=1)
    return _labels([(c > e50) & (s50 > 0), (c < e50) & (s50 < 0)], [EMA50_UP, EMA50_DOWN], EMA50_MIXED, missing)


# ----------------------------------------------------------------- RSI6 (primary) and RSI14
RSI_ZONES = [  # (lower bound, label) checked from the top
    (80, "Extremely strong / extended"),
    (70, "Strong momentum"),
    (60, "Positive momentum"),
    (50, "Mild positive momentum"),
    (40, "Mild negative momentum"),
    (30, "Weak momentum"),
]
RSI_OVERSOLD = "Oversold"  # RSI6 < 30
RSI_EXTENDED_LEVEL, RSI_OVERSOLD_LEVEL, RSI_TURN_POINTS = 80, 30, 10


def rsi_zone(rsi: pd.Series) -> pd.Series:
    """RSI6 zone: >=80 extremely strong / extended, 70-80 strong, 60-70 positive, 50-60 mild positive,
    40-50 mild negative, 30-40 weak, <30 oversold. A high RSI is NOT read as bearish (nor a low one as
    bullish) by itself - it is always described together with trend, structure and volume."""
    conditions = [rsi >= bound for bound, _ in RSI_ZONES]
    return _labels(conditions, [label for _, label in RSI_ZONES], RSI_OVERSOLD, rsi.isna())


def rsi_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """RSI6 vs RSI14 and RSI6 turning points.

    rsi_cross:               +1 RSI6 crossed above RSI14 within the last 3 sessions, -1 crossed below, 0 none
    rsi_recovering_oversold: RSI6 was < 30 in the previous 5 sessions and is now >= that low + 10 points
    rsi_losing_extended:     RSI6 was >= 80 in the previous 5 sessions and is now <= that high - 10 points
    """
    r6, r14 = df["rsi6"], df["rsi14"]
    low = r6.shift(1).rolling(5, min_periods=1).min()
    high = r6.shift(1).rolling(5, min_periods=1).max()
    return pd.DataFrame({
        "rsi_cross": recent_cross(r6, r14),
        "rsi_recovering_oversold": (low < RSI_OVERSOLD_LEVEL) & (r6 >= low + RSI_TURN_POINTS),
        "rsi_losing_extended": (high >= RSI_EXTENDED_LEVEL) & (r6 <= high - RSI_TURN_POINTS),
    }, index=df.index)


# ----------------------------------------------------------------- MACD 12/26/9
MACD_IMPROVING = "Improving"
MACD_WEAKENING = "Weakening"
MACD_POSITIVE = "Positive"
MACD_NEGATIVE = "Negative"


def classify_macd(df: pd.DataFrame) -> pd.Series:
    """MACD 12/26/9 momentum from the histogram H = DIF - DEA and its change over 1 and 3 sessions.

    Improving: H rose over the last session and over the last 3 sessions (d1 > 0 and d3 > 0)
    Weakening: H fell over the last session and over the last 3 sessions (d1 < 0 and d3 < 0)
    Positive:  otherwise, DIF at or above DEA (H >= 0)
    Negative:  otherwise, DIF below DEA (H < 0)
    The CHANGE is checked first: for a 2-7 day swing it says more than the sign.
    """
    h = df["macd_hist"]
    d1, d3 = h.diff(1), h.diff(3)
    conditions = [(d1 > 0) & (d3 > 0), (d1 < 0) & (d3 < 0), h >= 0]
    return _labels(conditions, [MACD_IMPROVING, MACD_WEAKENING, MACD_POSITIVE], MACD_NEGATIVE,
                   h.isna() | d3.isna())


def macd_details(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive MACD details (no trade signals).

    macd_cross:        +1 bullish crossover (DIF crossed above DEA) within the last 3 sessions, -1 bearish, 0 none
    macd_hist_trend:   "expanding" if |H| grew vs 3 sessions ago with the same sign, "contracting" if it shrank
    macd_zero_context: "below zero but improving" (DIF < 0 and H higher than 3 sessions ago) or
                       "above zero but deteriorating" (DIF > 0 and H lower than 3 sessions ago)
    """
    h, dif = df["macd_hist"], df["macd"]
    before = h.shift(3)
    same_sign = np.sign(h) == np.sign(before)
    trend = np.select([same_sign & (h.abs() > before.abs()), same_sign & (h.abs() < before.abs())],
                      ["expanding", "contracting"], "")
    change = h - before
    zero = np.select([(dif < 0) & (change > 0), (dif > 0) & (change < 0)],
                     ["below zero but improving", "above zero but deteriorating"], "")
    return pd.DataFrame({"macd_cross": recent_cross(dif, df["macd_signal"]), "macd_hist_trend": trend,
                         "macd_zero_context": zero}, index=df.index)


# ----------------------------------------------------------------- short-term momentum (returns)
STRONG_POSITIVE = "Strong positive"
MOMENTUM_POSITIVE = "Positive"
MOMENTUM_MIXED = "Mixed"
MOMENTUM_NEGATIVE = "Negative"
STRONG_NEGATIVE = "Strong negative"


def classify_momentum(df: pd.DataFrame) -> pd.Series:
    """Short-term momentum from the 2D/3D/5D/7D returns and the 5-day move in ATRs (M5).

    P = number of positive returns among 2D, 3D, 5D, 7D;  M5 = (C_t - C_(t-5)) / ATR
    Strong positive: P = 4 and M5 >= 1.5    Strong negative: P = 0 and M5 <= -1.5
    Positive:        P >= 3 and M5 > 0      Negative:        P <= 1 and M5 < 0
    Mixed:           anything else
    """
    returns = df[["return_2d", "return_3d", "return_5d", "return_7d"]]
    positives = (returns > 0).sum(axis=1)
    m5 = df["move_5d_atr"]
    conditions = [
        (positives == 4) & (m5 >= 1.5),
        (positives == 0) & (m5 <= -1.5),
        (positives >= 3) & (m5 > 0),
        (positives <= 1) & (m5 < 0),
    ]
    choices = [STRONG_POSITIVE, STRONG_NEGATIVE, MOMENTUM_POSITIVE, MOMENTUM_NEGATIVE]
    return _labels(conditions, choices, MOMENTUM_MIXED, returns.isna().any(axis=1) | m5.isna())


# ----------------------------------------------------------------- volume vs MAVOL20
VERY_HIGH_PARTICIPATION = "Very high participation"
STRONG_PARTICIPATION = "Strong participation"
VOLUME_NORMAL = "Normal"
WEAK_PARTICIPATION = "Weak participation"
HEAVY_VOLUME = (VERY_HIGH_PARTICIPATION, STRONG_PARTICIPATION)


def classify_volume(df: pd.DataFrame) -> pd.Series:
    """Participation from Volume / MAVOL20: >= 2.0 very high, 1.5-2.0 strong, 0.8-1.5 normal, < 0.8 weak.

    Always read together with the day's direction (``day_direction``): heavy volume on a
    down day is selling pressure, not confirmation of strength.
    """
    r = df["volume_ratio"]
    return _labels([r >= 2.0, r >= 1.5, r >= 0.8], [VERY_HIGH_PARTICIPATION, STRONG_PARTICIPATION, VOLUME_NORMAL],
                   WEAK_PARTICIPATION, r.isna())


def day_direction(df: pd.DataFrame) -> pd.Series:
    """up / down / flat: today's close vs the previous close."""
    r1 = df["return_1d"]
    return _labels([r1 > 0, r1 < 0], ["up", "down"], "flat", r1.isna())


# ----------------------------------------------------------------- VWAP (secondary) / BOLL(20, 1.8)
def classify_vwap(df: pd.DataFrame) -> pd.Series:
    """Price vs 20D rolling VWAP: Near if within 0.5 ATR, otherwise Above / Below."""
    gap = (df["close"] - df["vwap20"]) / df["atr"]
    return _labels([gap > 0.5, gap < -0.5], ["Above", "Below"], "Near", gap.isna())


ABOVE_UPPER = "Above upper band"
NEAR_UPPER = "Near upper band"
UPPER_HALF = "Between middle and upper band"
NEAR_MIDDLE = "Near middle band"
LOWER_HALF = "Between middle and lower band"
NEAR_LOWER = "Near lower band"
BELOW_LOWER = "Below lower band"


def classify_bollinger_position(df: pd.DataFrame) -> pd.Series:
    """BOLL(20, 1.8) position from %B (0 = lower band, 0.5 = middle band, 1 = upper band).

    %B > 1 above upper band · >= 0.9 near upper band · > 0.6 between middle and upper ·
    >= 0.4 near middle band · > 0.1 between middle and lower · >= 0 near lower band · < 0 below lower band.
    Touching the upper band is NOT read as bearish by itself (see classify_band_context).
    """
    b = df["bb_pct_b"]
    conditions = [b > 1, b >= 0.9, b > 0.6, b >= 0.4, b > 0.1, b >= 0]
    choices = [ABOVE_UPPER, NEAR_UPPER, UPPER_HALF, NEAR_MIDDLE, LOWER_HALF, NEAR_LOWER]
    return _labels(conditions, choices, BELOW_LOWER, b.isna())


EXPANSION_AFTER_SQUEEZE = "Expansion after squeeze"
BAND_EXPANSION = "Band expansion"
VOLATILITY_SQUEEZE = "Volatility squeeze"
BAND_CONTRACTION = "Band contraction"
BAND_STABLE = "Stable"


def classify_band_state(df: pd.DataFrame) -> pd.Series:
    """BOLL band width W = (upper - lower) / middle; change = W_t / W_(t-5) - 1;
    W% = percentile of W vs the previous 126 sessions.

    Expansion after squeeze: change >= +20% and W% was <= 20 at some point in the last 10 sessions
    Band expansion:          change >= +20%
    Volatility squeeze:      W% <= 20
    Band contraction:        change <= -15%
    Stable:                  anything else
    """
    change, percentile = df["bb_width_change"], df["bb_width_percentile"]
    recently_squeezed = percentile.rolling(10, min_periods=1).min() <= 20
    conditions = [(change >= 0.2) & recently_squeezed, change >= 0.2, percentile <= 20, change <= -0.15]
    choices = [EXPANSION_AFTER_SQUEEZE, BAND_EXPANSION, VOLATILITY_SQUEEZE, BAND_CONTRACTION]
    return _labels(conditions, choices, BAND_STABLE, change.isna() | percentile.isna())


UPPER_WALK = "Upper-band walk (strong momentum)"
UPPER_PUSH = "Close above the upper band with trend and volume"
UPPER_EXTENSION = "Extension above the upper band without trend / volume confirmation"
LOWER_WALK = "Lower-band walk (strong downside momentum)"
LOWER_PUSH = "Close below the lower band with trend and volume"
LOWER_EXTENSION = "Drop below the lower band without trend / volume confirmation"
NO_BAND_CONTEXT = "None"


def classify_band_context(df: pd.DataFrame) -> pd.Series:
    """Separates a band walk in a strong trend from an unconfirmed one-off extension.

    Upper-band walk:  %B >= 0.9 (at / near the upper band) today and on >= 3 of the last 5 sessions,
                      C > EMA9 > EMA21 and EMA21 slope > 0
    Lower-band walk:  %B <= 0.1 today and on >= 3 of the last 5 sessions, C < EMA9 < EMA21, EMA21 slope < 0
    Close above the upper band with trend and volume: %B > 1, EMA9 > EMA21, EMA21 slope > 0, Volume >= MAVOL20
    Extension above the upper band without confirmation: %B > 1 and not one of the above
    (lower band: the mirror rules). Otherwise "None".
    """
    b, c, e9, e21, s21 = df["bb_pct_b"], df["close"], df["ema9"], df["ema21"], df["ema21_slope"]
    volume_ok = df["volume_ratio"] >= 1.0
    high_days = (b >= 0.9).astype(int).rolling(5, min_periods=5).sum()
    low_days = (b <= 0.1).astype(int).rolling(5, min_periods=5).sum()
    conditions = [
        (b >= 0.9) & (high_days >= 3) & (c > e9) & (e9 > e21) & (s21 > 0),
        (b <= 0.1) & (low_days >= 3) & (c < e9) & (e9 < e21) & (s21 < 0),
        (b > 1) & (e9 > e21) & (s21 > 0) & volume_ok,
        b > 1,
        (b < 0) & (e9 < e21) & (s21 < 0) & volume_ok,
        b < 0,
    ]
    choices = [UPPER_WALK, LOWER_WALK, UPPER_PUSH, UPPER_EXTENSION, LOWER_PUSH, LOWER_EXTENSION]
    return _labels(conditions, choices, NO_BAND_CONTEXT, b.isna())


# ----------------------------------------------------------------- volatility regime
def classify_volatility(percentile: pd.Series) -> pd.Series:
    """ATR% percentile vs the previous 252 sessions: <20 Low, 20-60 Normal, 60-80 Elevated,
    80-95 High, >=95 Extreme. n/a when fewer than 126 previous sessions exist."""
    conditions = [percentile >= 95, percentile >= 80, percentile >= 60, percentile >= 20, percentile < 20]
    choices = ["Extreme", "High", "Elevated", "Normal", "Low"]
    return _labels(conditions, choices, NA, percentile.isna())


# ----------------------------------------------------------------- relative strength
def relative_strength_score(df: pd.DataFrame) -> pd.Series:
    """Average of the 5D and 7D relative returns, each divided by its typical size.

        z_n = rel_n / (sigma_daily * sqrt(n)),  score = (z_5 + z_7) / 2
    sigma_daily = std of daily (stock - benchmark) returns over the previous 60 sessions.
    """
    sigma = df["rel_daily_sigma"].where(df["rel_daily_sigma"] > 0)
    return (df["rel_5d"] / (sigma * np.sqrt(5)) + df["rel_7d"] / (sigma * np.sqrt(7))) / 2


def classify_relative_strength(df: pd.DataFrame) -> pd.Series:
    """score >= 1.25 strongly outperforming, >= 0.5 outperforming, > -0.5 neutral,
    > -1.25 underperforming, otherwise strongly underperforming (n/a without a benchmark)."""
    score = relative_strength_score(df)
    conditions = [score >= 1.25, score >= 0.5, score > -0.5, score > -1.25]
    choices = ["Strongly outperforming", "Outperforming", "Neutral", "Underperforming"]
    return _labels(conditions, choices, "Strongly underperforming", score.isna())


# ----------------------------------------------------------------- ADX context
def classify_adx(adx: pd.Series) -> pd.Series:
    """ADX(14) >= 25 trending, < 20 weak / range-bound, 20-25 borderline. Secondary context only."""
    conditions = [adx >= 25, adx < 20]
    return _labels(conditions, ["Trending environment", "Weak / range-bound environment"],
                   "Borderline (ADX 20-25)", adx.isna())


# ----------------------------------------------------------------- overall swing condition
TREND_POINTS = {STRONG_UPTREND: 3, UPTREND: 2, IMPROVING: 1, MIXED: 0, DETERIORATING: -1,
                DOWNTREND: -2, STRONG_DOWNTREND: -3}
MACD_POINTS = {MACD_IMPROVING: 1, MACD_POSITIVE: 0, MACD_NEGATIVE: 0, MACD_WEAKENING: -1}
RS_POINTS = {"Strongly outperforming": 1, "Outperforming": 1, "Neutral": 0, "Underperforming": -1,
             "Strongly underperforming": -1}
BAND_POINTS = {UPPER_WALK: 1, LOWER_WALK: -1}
EMA50_POINTS = {EMA50_UP: 1, EMA50_DOWN: -1}
CONDITIONS = ("Strong", "Improving", "Mixed", "Weakening", "Weak")
MAX_SCORE = 12
COMPONENTS = ("structure", "trend", "volume", "rsi", "macd", "boll", "relative_strength", "ema50")


def swing_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Swing Technical Condition from a transparent, priority-weighted points score (-12 ... +12).

    1. Price structure (primary state): breakout with volume +2, breakout on normal volume +1,
       reclaiming EMA21 / EMA9 crossed above EMA21 +1, losing EMA21 / EMA9 crossed below EMA21 /
       momentum deterioration -1, breakdown -1, breakdown with heavy volume -2, others 0
    2. Swing trend (EMA9 / EMA21): strong uptrend +3, uptrend +2, improving +1, mixed 0,
       deteriorating -1, downtrend -2, strong downtrend -3
    3. Volume vs MAVOL20 with the day's direction: >= 2.0x +2 on an up day / -2 on a down day,
       1.5-2.0x +1 / -1, otherwise 0
    4. RSI6 / RSI14: RSI6 >= 60 and RSI6 >= RSI14 +1; RSI6 < 40 and RSI6 <= RSI14 -1
    5. MACD 12/26/9: improving +1, weakening -1, positive / negative 0
    6. BOLL(20, 1.8): upper-band walk +1, lower-band walk -1
    7. Relative strength: (strongly) outperforming +1, (strongly) underperforming -1
    8. EMA50 (secondary): price above rising EMA50 +1, below falling EMA50 -1
    EMA200 / EMA250 (context only), ATR and support / resistance add no points.

    Strong:    score >= 6 and the swing trend is an uptrend    Weak: score <= -6 and a downtrend
    Improving: score >= 2 (otherwise)                          Weakening: score <= -2 (otherwise)
    Mixed:     anything else. n/a if the trend, MACD or RSI6 is n/a.
    """
    trend = df["trend"]
    structure = df["structure"] if "structure" in df else pd.Series("", index=df.index)
    direction = np.sign(df["return_1d"]).fillna(0)
    ratio = df["volume_ratio"]
    points = pd.DataFrame({
        "structure": structure.map(STRUCTURE_POINTS).fillna(0),
        "trend": trend.map(TREND_POINTS),
        "volume": pd.Series(np.select([ratio >= 2.0, ratio >= 1.5], [2, 1], 0), index=df.index) * direction,
        "rsi": pd.Series(np.select([(df["rsi6"] >= 60) & (df["rsi6"] >= df["rsi14"]),
                                    (df["rsi6"] < 40) & (df["rsi6"] <= df["rsi14"])], [1, -1], 0), index=df.index),
        "macd": df["macd_state"].map(MACD_POINTS),
        "boll": df["band_context"].map(BAND_POINTS).fillna(0),
        "relative_strength": df["relative_strength"].map(RS_POINTS).fillna(0),
        "ema50": df["ema50_state"].map(EMA50_POINTS).fillna(0),
    }, index=df.index)
    score = points.sum(axis=1, min_count=len(COMPONENTS))
    up, down = trend.isin(UP_TRENDS), trend.isin(DOWN_TRENDS)
    conditions = [(score >= 6) & up, (score <= -6) & down, score >= 2, score <= -2]
    missing = trend.eq(NA) | df["macd_state"].eq(NA) | df["rsi6"].isna()
    label = _labels(conditions, ["Strong", "Weak", "Improving", "Weakening"], "Mixed", missing)
    out = points.add_prefix("points_").where(~missing, np.nan)
    out["swing_score"] = score.where(~missing)
    out["swing_condition"] = label
    return out


def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    """Add every indicator label to an indicator frame (the Swing Technical Condition is added
    separately by ``swing_condition`` once the structure states are known)."""
    out = df.copy()
    out["trend"] = classify_trend(out)
    out["ema50_state"] = classify_ema50(out)
    out["rsi_zone"] = rsi_zone(out["rsi6"])
    out = out.join(rsi_relationships(out))
    out["macd_state"] = classify_macd(out)
    out = out.join(macd_details(out))
    out["momentum"] = classify_momentum(out)
    out["volume_level"] = classify_volume(out)
    out["day_direction"] = day_direction(out)
    out["vwap_position"] = classify_vwap(out)
    out["bb_position"] = classify_bollinger_position(out)
    out["band_state"] = classify_band_state(out)
    out["band_context"] = classify_band_context(out)
    out["volatility"] = classify_volatility(out["atr_pct_percentile"])
    out["relative_strength_score"] = relative_strength_score(out)
    out["relative_strength"] = classify_relative_strength(out)
    out["adx_state"] = classify_adx(out["adx"])
    return out
