"""Technical indicators for a 2-7 trading-day swing horizon.

The PRIMARY indicators are the trader's own chart settings (constants below):
    EMA 9 / 21 / 50 / 200 / 250 · RSI 6 / 14 · MACD 12 / 26 / 9 · MAVOL20 · BOLL 20 / 1.8
Supporting measures: ATR(14), realised volatility, short-term and relative
returns, 20D / 60D highs and lows, OBV, a 20D rolling VWAP and ADX(14).

Every formula is written out with pandas / numpy (no TA-Lib) so it can be read
and checked. Notation: C = close, H = high, L = low, V = volume, t = today.

All indicators are causal: the value on day t only uses data up to day t.
A value that needs more history than is available is NaN ("n/a"), never guessed.

Prices are dividend-adjusted OHLC (see ``adjusted_ohlcv``): on the latest day
they equal the actual traded prices, and earlier prices are scaled for later
dividends so moves across ex-dividend dates are not mistaken for price drops.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# The trader's chart settings (primary indicators).
EMA_SPANS = (9, 21, 50, 200, 250)  # 9 momentum, 21 swing trend, 50 structure, 200 / 250 long-term context
EMA_SLOPE_SPANS = (9, 21, 50)
RSI_PERIODS = (6, 14)              # RSI6 primary, RSI14 secondary
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MAVOL_WINDOW = 20
BOLL_WINDOW, BOLL_STD = 20, 1.8
ATR_PERIOD = 14
SLOPE_DAYS = 3                     # EMA slope = % change over the last 3 sessions

RETURN_HORIZONS = (2, 3, 5, 7, 10, 20)
RELATIVE_HORIZONS = (2, 5, 7, 20)
TRADING_DAYS_PER_YEAR = 252


# ------------------------------------------------------------------ building blocks
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average.

        EMA_t = a * x_t + (1 - a) * EMA_(t-1),   a = 2 / (span + 1)

    It starts at the first available value; the first ``span - 1`` values are
    reported as NaN (warm-up) because they rest on too few observations.
    """
    values = series.astype(float)
    result = values.ewm(span=span, adjust=False).mean()
    return result.where(values.notna().cumsum() >= span)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average of the last ``window`` values (NaN until the window is full)."""
    return series.astype(float).rolling(window, min_periods=window).mean()


def wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by RSI, ATR and ADX).

    The first value is the simple average of the first ``period`` values;
    afterwards  avg_t = (avg_(t-1) * (period - 1) + x_t) / period.
    A missing x_t keeps the previous average.
    """
    x = values.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    full = pd.Series(~np.isnan(x)).rolling(period).sum().to_numpy() == period
    start = np.flatnonzero(full)
    if start.size:
        i = int(start[0])
        out[i] = x[i - period + 1: i + 1].mean()
        for j in range(i + 1, len(x)):
            out[j] = out[j - 1] if np.isnan(x[j]) else (out[j - 1] * (period - 1) + x[j]) / period
    return pd.Series(out, index=values.index)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope per day of the last ``window`` values (NaN until the window is full)."""
    x = np.arange(window) - (window - 1) / 2.0
    denominator = float((x ** 2).sum())
    return series.astype(float).rolling(window, min_periods=window).apply(
        lambda y: float(np.dot(x, y) / denominator), raw=True
    )


def percentile_of_previous(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Percentile (0-100) of today's value among the previous ``window`` values.

        percentile_t = 100 * share of values in (t-window ... t-1) that are <= value_t

    NaN when fewer than ``min_periods`` previous values exist.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        previous = values[max(0, i - window): i]
        previous = previous[~np.isnan(previous)]
        if previous.size >= min_periods:
            out[i] = 100.0 * float((previous <= value).mean())
    return pd.Series(out, index=series.index)


# ------------------------------------------------------------------ indicators
def rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index (Wilder).

        gain_t = max(C_t - C_(t-1), 0),  loss_t = max(C_(t-1) - C_t, 0)
        RS = WilderAverage(gain) / WilderAverage(loss);  RSI = 100 - 100 / (1 + RS)

    With no losses in the window RSI is 100 (50 if the price did not move at all).
    """
    change = close.astype(float).diff()
    avg_gain = wilder_average(change.clip(lower=0), period)
    avg_loss = wilder_average((-change).clip(lower=0), period)
    value = 100 - 100 / (1 + avg_gain / avg_loss.where(avg_loss > 0))
    no_loss = np.where(avg_gain > 0, 100.0, 50.0)
    return value.where(avg_loss > 0, no_loss).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """MACD (default 12, 26, 9).

        DIF = EMA12 - EMA26;  DEA (signal) = EMA9 of DIF;  Histogram = DIF - DEA
    Columns: macd (DIF), macd_signal (DEA), macd_hist.
    """
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line})


def bollinger(close: pd.Series, window: int = BOLL_WINDOW, num_std: float = BOLL_STD) -> pd.DataFrame:
    """Bollinger Bands, default BOLL(20, 1.8).

        Middle = SMA20;  Upper / Lower = Middle +/- 1.8 x std(C over the last 20 days)
        %B = (C - Lower) / (Upper - Lower);  Band width = (Upper - Lower) / Middle
    std is the population standard deviation (John Bollinger's definition, also used by
    TradingView); platforms that use the sample standard deviation draw bands ~2.6% wider.
    """
    middle = sma(close, window)
    std = close.astype(float).rolling(window, min_periods=window).std(ddof=0)
    upper, lower = middle + num_std * std, middle - num_std * std
    width = upper - lower
    return pd.DataFrame({
        "bb_mid": middle, "bb_upper": upper, "bb_lower": lower,
        "bb_pct_b": (close - lower) / width.where(width > 0),
        "bb_width": width / middle.where(middle > 0),
    })


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """TR_t = max(H_t - L_t, |H_t - C_(t-1)|, |L_t - C_(t-1)|)  (first day: H - L)."""
    previous = close.shift(1)
    parts = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1)
    return parts.max(axis=1, skipna=True).where((high - low).notna())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range: Wilder average of the true range over ``period`` days."""
    return wilder_average(true_range(high, low, close), period)


def rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                 window: int = 20) -> pd.Series:
    """Rolling VWAP from DAILY bars (an approximation, not an intraday session VWAP).

        Typical price TP = (H + L + C) / 3
        VWAP20_t = sum(TP * V over the last 20 days) / sum(V over the last 20 days)
    """
    typical = (high + low + close) / 3
    weighted = (typical * volume).rolling(window, min_periods=window).sum()
    total_volume = volume.rolling(window, min_periods=window).sum()
    return weighted / total_volume.where(total_volume > 0)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: OBV_t = OBV_(t-1) + sign(C_t - C_(t-1)) * V_t, starting at 0."""
    direction = np.sign(close.astype(float).diff()).fillna(0.0)
    return (direction * volume.astype(float).fillna(0.0)).cumsum()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (Wilder).

        +DM = H_t - H_(t-1) if it is > (L_(t-1) - L_t) and > 0, else 0   (-DM mirrored)
        +DI = 100 * Wilder(+DM) / Wilder(TR);  -DI likewise
        DX  = 100 * |+DI - -DI| / (+DI + -DI);  ADX = Wilder average of DX
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    tr = true_range(high, low, close).where(close.shift(1).notna())
    smoothed_tr = wilder_average(tr, period)
    plus_di = 100 * wilder_average(plus_dm, period) / smoothed_tr.where(smoothed_tr > 0)
    minus_di = 100 * wilder_average(minus_dm, period) / smoothed_tr.where(smoothed_tr > 0)
    total = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / total.where(total > 0)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": wilder_average(dx, period)})


# ------------------------------------------------------------------ data preparation
def adjusted_ohlcv(prices: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """Open/High/Low/Close on the same basis as the price series used for returns.

    When returns use the adjusted close, every OHLC value of a day is multiplied
    by adj_close / close for that day (the usual dividend adjustment). On the
    latest day this factor is 1, so the current price equals the actual close.
    A missing High/Low is replaced by the close of that day (counted in
    ``attrs["filled_high_low_days"]`` so the report can mention it).
    """
    factor = prices["adj_close"] / prices["close"] if price_column == "adj_close" else 1.0
    out = pd.DataFrame(index=prices.index)
    for column in ("open", "high", "low", "close"):
        out[column] = prices[column].astype(float) * factor
    missing = out["high"].isna() | out["low"].isna()
    out["high"] = out["high"].fillna(out["close"])
    out["low"] = out["low"].fillna(out["close"])
    out["volume"] = prices["volume"].astype(float)
    out.attrs["filled_high_low_days"] = int(missing.sum())
    return out


def compute_indicators(ohlcv: pd.DataFrame, benchmark_close: pd.Series | None = None) -> pd.DataFrame:
    """Calculate every swing indicator for every day. Returns one row per trading day."""
    df = ohlcv.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # Trend: EMA9 (momentum) and EMA21 (swing trend) first; EMA50 secondary; EMA200 / EMA250 context.
    for span in EMA_SPANS:
        df[f"ema{span}"] = ema(close, span)
    for span in EMA_SLOPE_SPANS:
        # Slope = % change of the EMA over the last 3 sessions: EMA_t / EMA_(t-3) - 1
        df[f"ema{span}_slope"] = df[f"ema{span}"] / df[f"ema{span}"].shift(SLOPE_DAYS) - 1

    # Momentum: RSI6 (primary), RSI14, MACD 12/26/9, short-term returns
    for period in RSI_PERIODS:
        df[f"rsi{period}"] = rsi(close, period)
    df = df.join(macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL))
    df["return_1d"] = close / close.shift(1) - 1
    for n in RETURN_HORIZONS:
        df[f"return_{n}d"] = close / close.shift(n) - 1  # ROC(n) = this value x 100

    # Volatility: ATR(14), realised volatility, BOLL(20, 1.8)
    df["atr"] = atr(high, low, close, ATR_PERIOD)
    df["atr_pct"] = df["atr"] / close * 100
    df["atr_pct_percentile"] = percentile_of_previous(df["atr_pct"], window=252, min_periods=126)
    daily = df["return_1d"]
    df["rv20"] = daily.rolling(20, min_periods=20).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    df["rv60"] = daily.rolling(60, min_periods=60).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    df = df.join(bollinger(close, BOLL_WINDOW, BOLL_STD))
    df["bb_width_change"] = df["bb_width"] / df["bb_width"].shift(5) - 1  # change over 5 sessions
    df["bb_width_percentile"] = percentile_of_previous(df["bb_width"], window=126, min_periods=60)
    # 5-day move measured in ATRs: (C_t - C_(t-5)) / ATR_t
    df["move_5d_atr"] = (close - close.shift(5)) / df["atr"]

    # Volume: MAVOL20 = average volume of the last 20 sessions INCLUDING today (as on charting
    # platforms); OBV and the 5-day average are secondary.
    df["mavol20"] = volume.rolling(MAVOL_WINDOW, min_periods=MAVOL_WINDOW).mean()
    df["volume_ratio"] = volume / df["mavol20"].where(df["mavol20"] > 0)
    df["avg_volume_5d"] = volume.rolling(5, min_periods=5).mean()
    df["volume_5d_vs_mavol20"] = df["avg_volume_5d"] / df["mavol20"].where(df["mavol20"] > 0)
    df["obv"] = obv(close, volume)
    for window in (10, 20):
        # OBV slope in "average daily volumes per day" so it is comparable across stocks
        mean_volume = volume.rolling(window, min_periods=window).mean()
        df[f"obv_slope{window}"] = rolling_slope(df["obv"], window) / mean_volume.where(mean_volume > 0)
    df["vwap20"] = rolling_vwap(high, low, close, volume, 20)  # secondary reference (daily-bar approximation)

    # Directional context
    df = df.join(adx(high, low, close, 14))

    # Recent highs / lows (reference levels include today; "prior" versions exclude it)
    for window in (20, 60):
        df[f"high{window}"] = high.rolling(window, min_periods=window).max()
        df[f"low{window}"] = low.rolling(window, min_periods=window).min()
    df["prior_high20"] = high.shift(1).rolling(20, min_periods=20).max()
    df["prior_low20"] = low.shift(1).rolling(20, min_periods=20).min()

    # Relative strength vs the benchmark: stock n-day return - benchmark n-day return
    if benchmark_close is not None and benchmark_close.notna().any():
        bench = benchmark_close.astype(float).reindex(df.index)
        bench_daily = bench / bench.shift(1) - 1
        for n in RELATIVE_HORIZONS:
            df[f"rel_{n}d"] = df[f"return_{n}d"] - (bench / bench.shift(n) - 1)
        # Typical size of daily stock-minus-benchmark moves over the previous 60 sessions
        df["rel_daily_sigma"] = (daily - bench_daily).shift(1).rolling(60, min_periods=40).std(ddof=1)
    else:
        for n in RELATIVE_HORIZONS:
            df[f"rel_{n}d"] = np.nan
        df["rel_daily_sigma"] = np.nan
    return df
