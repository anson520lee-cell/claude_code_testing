"""Recent price structure for swing trading: swing points, support / resistance,
breakout / pullback states and proximity to key references.

Only recent structure is used (by default the last 60 sessions); old levels
are deliberately ignored for a 2-7 day holding period. Distances are shown in
percent and in ATRs, because "1.5 ATR away" means the same thing for a calm
and a volatile stock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MERGE_ATR = 0.5   # levels closer than this (in ATRs) are treated as one level
MIN_GAP_ATR = 0.1  # a level closer than this to the price is "at" the price, not above/below it
NEAR_ATR = 0.5    # "near" EMA9 / EMA21 / EMA50 / VWAP20 means within 0.5 ATR
NEAR_EXTREME_ATR = 1.0  # "near" the 20D high / low means within 1 ATR


# ------------------------------------------------------------------ swing points
def swing_points(high: pd.Series, low: pd.Series, bars: int = 3) -> tuple[pd.Series, pd.Series]:
    """Swing high: the High is the highest of the ``bars`` sessions on each side (and itself).
    Swing low: the Low is the lowest of that window. The last ``bars`` sessions cannot be
    confirmed yet (their right-hand side is unknown), so they are never swing points."""
    window = 2 * bars + 1
    is_high = high.eq(high.rolling(window, center=True, min_periods=window).max())
    is_low = low.eq(low.rolling(window, center=True, min_periods=window).min())
    return is_high, is_low


# ------------------------------------------------------------------ support / resistance
@dataclass
class Level:
    price: float
    origin: str
    distance_pct: float = float("nan")  # level / close - 1
    distance_atr: float = float("nan")  # (level - close) / ATR
    also: list[str] = field(default_factory=list)          # origins of candidates merged into this level
    also_prices: list[float] = field(default_factory=list)  # their prices (within 0.5 ATR of this level)

    def label(self) -> str:
        merged = [origin if round(price, 2) == round(self.price, 2) else f"{origin} {price:,.2f}"
                  for origin, price in zip(self.also[:2], self.also_prices[:2])]
        return " / ".join([self.origin, *merged])


def support_resistance(df: pd.DataFrame, lookback: int = 60, bars: int = 3, count: int = 2) -> dict:
    """Nearest ``count`` resistance and support levels from the last ``lookback`` sessions.

    Candidates: every swing high and swing low in the window (a former high can be
    support once price is above it, and vice versa), plus the 20D and 60D high and low.
    Resistance = candidates above the close by at least 0.1 ATR, nearest first;
    support    = candidates below the close by at least 0.1 ATR, nearest first.
    A candidate within 0.5 ATR of a level already chosen is merged into it
    (so near-duplicate levels are not listed twice).
    """
    last = df.iloc[-1]
    close, atr_value = float(last["close"]), float(last["atr"])
    unit = atr_value if np.isfinite(atr_value) and atr_value > 0 else close * 0.01  # 1% if ATR unknown
    window = df.tail(lookback)
    is_high, is_low = swing_points(df["high"], df["low"], bars)
    candidates: list[tuple[float, str]] = []
    for date in window.index:
        if is_high.get(date, False):
            candidates.append((float(df.at[date, "high"]), f"swing high {date:%Y-%m-%d}"))
        if is_low.get(date, False):
            candidates.append((float(df.at[date, "low"]), f"swing low {date:%Y-%m-%d}"))
    for column, name in (("high20", "20D high"), ("low20", "20D low"), ("high60", "60D high"), ("low60", "60D low")):
        if pd.notna(last.get(column)):
            candidates.append((float(last[column]), name))

    def pick(side: str) -> list[Level]:
        if side == "resistance":
            pool = sorted((c for c in candidates if c[0] > close + MIN_GAP_ATR * unit), key=lambda c: c[0])
        else:
            pool = sorted((c for c in candidates if c[0] < close - MIN_GAP_ATR * unit), key=lambda c: -c[0])
        chosen: list[Level] = []
        for price, origin in pool:
            twin = next((lvl for lvl in chosen if abs(lvl.price - price) <= MERGE_ATR * unit), None)
            if twin is not None:
                if origin not in twin.also and origin != twin.origin:
                    twin.also.append(origin)
                    twin.also_prices.append(price)
                continue
            if len(chosen) == count:
                break
            chosen.append(Level(price, origin))
        for level in chosen:
            level.distance_pct = level.price / close - 1
            level.distance_atr = (level.price - close) / atr_value if atr_value > 0 else float("nan")
        return chosen

    return {"resistance": pick("resistance"), "support": pick("support"), "lookback": lookback}


# ------------------------------------------------------------------ structure states
BREAKOUT_VOLUME = "Breakout with volume confirmation"
BREAKOUT_NORMAL = "Breakout on normal volume"
BREAKOUT_WEAK = "Breakout without volume confirmation"
BREAKDOWN_VOLUME = "Short-term breakdown with heavy volume"
BREAKDOWN = "Short-term breakdown"
LOSING_EMA21 = "Price losing EMA21"
RECLAIMING_EMA21 = "Price reclaiming EMA21"
EMA9_CROSS_DOWN = "EMA9 crossed below EMA21"
EMA9_CROSS_UP = "EMA9 crossed above EMA21"
PULLBACK_EMA9 = "Pullback to EMA9"
PULLBACK_EMA21 = "Pullback to EMA21"
PULLBACK = "Pullback within swing uptrend"
LOSING_EMA9 = "Price losing EMA9"
RECLAIMING_EMA9 = "Price reclaiming EMA9"
NEAR_BREAKOUT = "Near breakout"
MOMENTUM_DETERIORATION = "Momentum deterioration"
EMA_COMPRESSION = "EMA9 / EMA21 compression"
VOLATILITY_SQUEEZE = "Volatility squeeze"
RANGE = "Range / consolidation"
BULL_STACK = "Price > EMA9 > EMA21"
BEAR_STACK = "Price < EMA9 < EMA21"

STATE_ORDER = [  # priority order: the first active state is the "primary" structure
    BREAKOUT_VOLUME, BREAKOUT_NORMAL, BREAKOUT_WEAK, BREAKDOWN_VOLUME, BREAKDOWN,
    LOSING_EMA21, RECLAIMING_EMA21, EMA9_CROSS_DOWN, EMA9_CROSS_UP,
    PULLBACK_EMA9, PULLBACK_EMA21, PULLBACK, LOSING_EMA9, RECLAIMING_EMA9,
    NEAR_BREAKOUT, MOMENTUM_DETERIORATION, EMA_COMPRESSION, VOLATILITY_SQUEEZE, RANGE,
    BULL_STACK, BEAR_STACK,
]
NO_STRUCTURE = "No distinct structure"
# Points of the PRIMARY structure in the Swing Technical Condition (all other states: 0).
STRUCTURE_POINTS = {
    BREAKOUT_VOLUME: 2, BREAKOUT_NORMAL: 1, RECLAIMING_EMA21: 1, EMA9_CROSS_UP: 1,
    BREAKDOWN_VOLUME: -2, BREAKDOWN: -1, LOSING_EMA21: -1, EMA9_CROSS_DOWN: -1, MOMENTUM_DETERIORATION: -1,
}
HEAVY_VOLUME_RATIO = 1.5    # Volume / MAVOL20 for "volume confirmation" / "heavy volume"
WEAK_VOLUME_RATIO = 0.8     # below this: weak participation
COMPRESSION_ATR = 0.3       # EMA9 and EMA21 closer than this (in ATRs) = compressed
RSI_DROP_POINTS = 20        # momentum deterioration: RSI6 fell at least this much in 3 sessions


def _crossed(fast: pd.Series, slow: pd.Series, lookback: int = 3) -> tuple[pd.Series, pd.Series]:
    """(crossed up, crossed down) within the last ``lookback`` sessions, still on the new side today."""
    diff = fast - slow
    was_below = pd.concat([diff.shift(k) <= 0 for k in range(1, lookback + 1)], axis=1).any(axis=1)
    was_above = pd.concat([diff.shift(k) >= 0 for k in range(1, lookback + 1)], axis=1).any(axis=1)
    return (diff > 0) & was_below, (diff < 0) & was_above


def structure_states(df: pd.DataFrame) -> pd.DataFrame:
    """One True/False column per descriptive structure state (see README section 8 for the rules).

    Breakout: close > highest High of the previous 20 sessions, split by Volume / MAVOL20:
      >= 1.5 with volume confirmation · 0.8-1.5 on normal volume · < 0.8 without volume confirmation
    Short-term breakdown: close < lowest Low of the previous 20 sessions (>= 1.5x MAVOL20: with heavy volume)
    Price reclaiming / losing EMA9 or EMA21: the close crossed that EMA today (yesterday on the other side)
    EMA9 crossed above / below EMA21: within the last 3 sessions
    Pullback within swing uptrend: EMA9 > EMA21, EMA21 slope > 0, 3D return < 0, close > EMA21
      ... to EMA9 / to EMA21: close within 0.5 ATR of that EMA (the nearer one if both);
      otherwise it counts only when the close is below EMA9
    Near breakout: not a breakout, close within 1 ATR below the previous 20-session high
    Momentum deterioration: EMA9 > EMA21, RSI6 fell >= 20 points in 3 sessions, MACD histogram below 3 sessions ago
    EMA9 / EMA21 compression: the two EMAs are within 0.3 ATR of each other
    Volatility squeeze: BOLL band width in the lowest 20% of the previous 126 sessions
    Range / consolidation: 20-day high-low range <= 4 ATR and ADX(14) < 20
    Price > EMA9 > EMA21 / Price < EMA9 < EMA21: the stacked order (lowest priority)
    These are descriptions of the chart, not buy or sell signals.
    """
    c, atr_value, ratio = df["close"], df["atr"], df["volume_ratio"]
    e9, e21 = df["ema9"], df["ema21"]
    c_prev, e9_prev, e21_prev = c.shift(1), e9.shift(1), e21.shift(1)
    breakout = c > df["prior_high20"]
    breakdown = c < df["prior_low20"]
    cross_up, cross_down = _crossed(e9, e21)
    pullback = (e9 > e21) & (df["ema21_slope"] > 0) & (df["return_3d"] < 0) & (c > e21)
    gap9 = (c - e9).abs() / atr_value
    gap21 = (c - e21).abs() / atr_value
    near9 = pullback & (gap9 <= NEAR_ATR) & ~((gap21 <= NEAR_ATR) & (gap21 < gap9))
    near21 = pullback & (gap21 <= NEAR_ATR) & ~near9
    below_high = df["prior_high20"] - c
    states = {
        BREAKOUT_VOLUME: breakout & (ratio >= HEAVY_VOLUME_RATIO),
        BREAKOUT_NORMAL: breakout & (ratio >= WEAK_VOLUME_RATIO) & (ratio < HEAVY_VOLUME_RATIO),
        BREAKOUT_WEAK: breakout & (ratio < WEAK_VOLUME_RATIO),
        BREAKDOWN_VOLUME: breakdown & (ratio >= HEAVY_VOLUME_RATIO),
        BREAKDOWN: breakdown & ~(ratio >= HEAVY_VOLUME_RATIO),
        LOSING_EMA21: (c < e21) & (c_prev >= e21_prev),
        RECLAIMING_EMA21: (c > e21) & (c_prev <= e21_prev),
        EMA9_CROSS_DOWN: cross_down,
        EMA9_CROSS_UP: cross_up,
        PULLBACK_EMA9: near9,
        PULLBACK_EMA21: near21,
        PULLBACK: pullback & ~near9 & ~near21 & (c < e9),
        LOSING_EMA9: (c < e9) & (c_prev >= e9_prev),
        RECLAIMING_EMA9: (c > e9) & (c_prev <= e9_prev),
        NEAR_BREAKOUT: ~breakout & (below_high >= 0) & (below_high <= NEAR_EXTREME_ATR * atr_value),
        MOMENTUM_DETERIORATION: ((e9 > e21) & (df["rsi6"].diff(3) <= -RSI_DROP_POINTS)
                                 & (df["macd_hist"].diff(3) < 0)),
        EMA_COMPRESSION: (e9 - e21).abs() <= COMPRESSION_ATR * atr_value,
        VOLATILITY_SQUEEZE: df["bb_width_percentile"] <= 20,
        RANGE: ((df["high20"] - df["low20"]) <= 4 * atr_value) & (df["adx"] < 20),
        BULL_STACK: (c > e9) & (e9 > e21),
        BEAR_STACK: (c < e9) & (e9 < e21),
    }
    frame = pd.DataFrame({name: mask.fillna(False).astype(bool) for name, mask in states.items()}, index=df.index)
    primary = np.select([frame[name].to_numpy() for name in STATE_ORDER], STATE_ORDER, default=NO_STRUCTURE)
    frame["structure"] = pd.Series(primary, index=df.index, dtype=object)
    return frame


def active_states(row: pd.Series) -> list[str]:
    """All structure states that are true for one day, in priority order."""
    return [name for name in STATE_ORDER if bool(row.get(name, False))]


# ------------------------------------------------------------------ proximity notes
def proximity_notes(row: pd.Series) -> list[str]:
    """Plain-language notes on where the close sits relative to key short-term references."""
    notes: list[str] = []
    c, a = row["close"], row["atr"]
    if not (np.isfinite(a) and a > 0):
        return notes
    if pd.notna(row.get("prior_high20")) and c > row["prior_high20"]:
        notes.append(f"Closed above the previous 20-session high ({row['prior_high20']:,.2f}).")
    elif pd.notna(row.get("high20")) and row["high20"] - c <= NEAR_EXTREME_ATR * a:
        notes.append(f"Within 1 ATR of the 20D high ({row['high20']:,.2f}, {(row['high20'] - c) / a:+.2f} ATR).")
    if pd.notna(row.get("prior_low20")) and c < row["prior_low20"]:
        notes.append(f"Closed below the previous 20-session low ({row['prior_low20']:,.2f}).")
    elif pd.notna(row.get("low20")) and c - row["low20"] <= NEAR_EXTREME_ATR * a:
        notes.append(f"Within 1 ATR of the 20D low ({row['low20']:,.2f}, {(row['low20'] - c) / a:+.2f} ATR).")
    for column, name in (("ema9", "EMA9"), ("ema21", "EMA21"), ("ema50", "EMA50"), ("vwap20", "VWAP20")):
        value = row.get(column)
        if pd.notna(value) and abs(c - value) <= NEAR_ATR * a:
            notes.append(f"Near {name} ({value:,.2f}, {(c - value) / a:+.2f} ATR from it).")
    return notes
