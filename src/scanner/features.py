"""Per-stock feature frame for the scanner (every row is causal: uses data up to that day only).

Reuses the swing indicator stack (``src/technical``): EMA 9/21/50/200/250, MACD 12/26/9,
RSI 6/14, BOLL(20, 1.8), ATR(14), MAVOL20, OBV, ROC, ADX, structure states.
Adds what a universe scanner needs:

* hv10 / hv20 / hv60 / hv120: annualised close-to-close historical volatility
* ret_60d / ret_120d, relative returns vs the stock's sector / industry ETF
* gap: open / previous close - 1, also in ATRs
* 52-week high / low (252 sessions) and distance from them
* CLV (close location value) = (C - L) / (H - L): where the close sits in the day's range
* up/down volume ratio (20 sessions): volume on up days / volume on down days
* swing structure, CAUSAL: a swing high/low (3 bars each side) is only known 3 sessions
  later, so it is used from then on. HH/HL = last confirmed swing high > previous one and
  last swing low > previous one; LH/LL mirrored.
* extension: (C - EMA21) / ATR
* dollar volume (20-day average of close x volume)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.technical.classification import classify_all, swing_condition
from src.technical.indicators import adjusted_ohlcv, compute_indicators
from src.technical.structure import structure_states, swing_points

PIVOT_BARS = 3


def _confirmed_pivots(values: pd.Series, is_pivot: pd.Series, bars: int) -> tuple[pd.Series, pd.Series]:
    """(last, previous) confirmed pivot value at each date."""
    known = values.shift(bars).where(is_pivot.shift(bars, fill_value=False).astype(bool))
    last = known.ffill()
    previous = known.dropna().shift(1).reindex(values.index).ffill()
    return last, previous


def build_features(prices: pd.DataFrame, spy_close: pd.Series | None, sector_close: pd.Series | None) -> pd.DataFrame:
    """Full feature frame for one stock. ``prices`` is a standard price frame (DB / provider)."""
    price_column = "adj_close" if prices["adj_close"].notna().all() and (prices["adj_close"] > 0).all() else "close"
    ohlcv = adjusted_ohlcv(prices, price_column)
    bench = spy_close.reindex(ohlcv.index).ffill() if spy_close is not None else None
    df = classify_all(compute_indicators(ohlcv, bench))
    df = df.join(structure_states(df))
    df = df.join(swing_condition(df))

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    daily = df["return_1d"]
    for n in (10, 20, 60, 120):
        df[f"hv{n}"] = daily.rolling(n, min_periods=n).std(ddof=1) * math.sqrt(252)
    for n in (60, 120):
        df[f"return_{n}d"] = close / close.shift(n) - 1
    if bench is not None:
        df["rel_60d"] = df["return_60d"] - (bench / bench.shift(60) - 1)
    else:
        df["rel_60d"] = np.nan
    if sector_close is not None:
        sector = sector_close.reindex(df.index).ffill()
        for n in (5, 20, 60):
            df[f"rel_sector_{n}d"] = df[f"return_{n}d"] - (sector / sector.shift(n) - 1)
    else:
        for n in (5, 20, 60):
            df[f"rel_sector_{n}d"] = np.nan

    prev_close = close.shift(1)
    df["gap"] = df["open"] / prev_close - 1
    df["gap_atr"] = (df["open"] - prev_close) / df["atr"].shift(1)
    df["high252"] = high.rolling(252, min_periods=200).max()
    df["low252"] = low.rolling(252, min_periods=200).min()
    df["from_52w_high"] = close / df["high252"] - 1
    df["from_52w_low"] = close / df["low252"] - 1
    df["prior_high55"] = high.shift(1).rolling(55, min_periods=55).max()
    df["prior_low55"] = low.shift(1).rolling(55, min_periods=55).min()
    rng = (high - low).where(high > low)
    df["clv"] = ((close - low) / rng).fillna(0.5)
    up_vol = volume.where(daily > 0, 0.0).rolling(20, min_periods=20).sum()
    down_vol = volume.where(daily < 0, 0.0).rolling(20, min_periods=20).sum()
    df["updown_vol_ratio"] = up_vol / down_vol.where(down_vol > 0)
    df["vol3_vs_mavol"] = volume.rolling(3).mean() / df["mavol20"]
    df["extension_atr"] = (close - df["ema21"]) / df["atr"]
    df["ema50_slope"] = df["ema50"] / df["ema50"].shift(5) - 1
    df["ema200_slope"] = df["ema200"] / df["ema200"].shift(10) - 1
    df["dollar_volume20"] = (close * volume).rolling(20, min_periods=20).mean()
    df["low3"], df["low5"], df["low10"] = (low.rolling(n).min() for n in (3, 5, 10))
    df["high3"], df["high5"], df["high10"] = (high.rolling(n).max() for n in (3, 5, 10))
    df["bbw_pct_min10"] = df["bb_width_percentile"].rolling(10, min_periods=1).min()
    df["sessions_below_ema21"] = _run_length(close < df["ema21"])
    df["sessions_above_ema21"] = _run_length(close > df["ema21"])
    df["rsi6_prev"] = df["rsi6"].shift(1)
    df["macd_hist_3d"] = df["macd_hist"] - df["macd_hist"].shift(3)
    df["roc10"] = df["return_10d"] * 100
    df["roc20"] = df["return_20d"] * 100

    is_high, is_low = swing_points(high, low, PIVOT_BARS)
    df["swing_high_last"], df["swing_high_prev"] = _confirmed_pivots(high, is_high, PIVOT_BARS)
    df["swing_low_last"], df["swing_low_prev"] = _confirmed_pivots(low, is_low, PIVOT_BARS)
    hh = df["swing_high_last"] > df["swing_high_prev"]
    hl = df["swing_low_last"] > df["swing_low_prev"]
    lh = df["swing_high_last"] < df["swing_high_prev"]
    ll = df["swing_low_last"] < df["swing_low_prev"]
    known = df[["swing_high_last", "swing_high_prev", "swing_low_last", "swing_low_prev"]].notna().all(axis=1)
    df["swing_structure"] = np.where(~known, "n/a", np.where(hh & hl, "HH/HL", np.where(lh & ll, "LH/LL", "Mixed")))
    return df


def _run_length(mask: pd.Series) -> pd.Series:
    """Consecutive sessions (ending today) for which ``mask`` has been True."""
    m = mask.fillna(False).astype(int)
    groups = (m != m.shift()).cumsum()
    return m.groupby(groups).cumsum()
