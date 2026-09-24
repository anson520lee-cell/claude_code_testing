"""Quantitative statistics calculated from daily prices.

Notation used in the comments (the same formulas are listed in README.md):

    P_t  = price on trading day t (adjusted close when available, else close)
    r_t  = daily return on day t
    V_t  = volume on day t
    W    = window length in trading days

Important design choice: the "baseline" for Z-scores and the volume ratio is
the window of W days *before* day t (days t-W ... t-1). Day t itself is left
out, so an unusually large move cannot inflate its own baseline and hide
itself. Every rolling value needs a full window; days without enough history
get NaN (shown as "n/a") rather than a number based on too little data.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

FORWARD_HORIZONS = (1, 5, 20)

# A standard deviation this small is floating-point rounding noise, not real
# variation (real stocks have daily-return std of ~0.0001 or more). Dividing by
# it would create huge meaningless Z-scores, so such days get NaN instead.
MIN_RETURN_STD = 1e-12
MIN_RELATIVE_VOLUME_STD = 1e-9
TRAILING_RETURN_HORIZONS = (5, 20)
MOVING_AVERAGE_WINDOWS = (20, 50)

# Look-back windows (in trading days) used in the "Recent Performance" table.
PERFORMANCE_WINDOWS = {
    "1 day": 1,
    "5 days": 5,
    "1 month (21 trading days)": 21,
    "3 months (63 trading days)": 63,
    "6 months (126 trading days)": 126,
    "1 year (252 trading days)": 252,
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series, returning NaN wherever the denominator is zero, negative or missing."""
    return numerator / denominator.where(denominator > 0)


def add_statistics(
    prices: pd.DataFrame,
    price_column: str = "adj_close",
    zscore_window: int = 60,
    volume_window: int = 20,
    volatility_window: int = 20,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Add return, volatility, volume, moving-average, drawdown and forward-return columns.

    Args:
        prices: standard price frame (see ``src.data.prices``).
        price_column: column used for returns ("adj_close" or "close").
        zscore_window: W for the return Z-score baseline.
        volume_window: W for average volume, volume ratio and volume Z-score.
        volatility_window: W for rolling volatility.
        trading_days_per_year: used to annualise volatility.

    Returns:
        A copy of ``prices`` with the new columns added (one row per trading day).
    """
    df = prices.copy()
    price = df[price_column].astype(float)
    df["price"] = price  # the series all returns are based on

    # Daily return:  r_t = P_t / P_(t-1) - 1
    df["daily_return"] = price / price.shift(1) - 1
    # Log return:  ln(P_t / P_(t-1))
    df["log_return"] = np.log(price / price.shift(1))
    # Trailing n-day return:  P_t / P_(t-n) - 1
    for n in TRAILING_RETURN_HORIZONS:
        df[f"return_{n}d"] = price / price.shift(n) - 1

    # Baseline of the previous W daily returns (days t-W ... t-1):
    #   mean_t = average(r_(t-W) ... r_(t-1))
    #   std_t  = sample standard deviation(r_(t-W) ... r_(t-1))
    previous_returns = df["daily_return"].shift(1).rolling(zscore_window, min_periods=zscore_window)
    df["rolling_mean_return"] = previous_returns.mean()
    df["rolling_std_return"] = previous_returns.std(ddof=1)
    # Return Z-score:  Z_t = (r_t - mean_t) / std_t
    return_std = df["rolling_std_return"]
    df["return_zscore"] = (df["daily_return"] - df["rolling_mean_return"]) / return_std.where(
        return_std > MIN_RETURN_STD
    )

    # Rolling volatility (annualised), includes day t:
    #   vol_t = std(r_(t-W+1) ... r_t) * sqrt(252)
    df["rolling_volatility"] = (
        df["daily_return"].rolling(volatility_window, min_periods=volatility_window).std(ddof=1)
        * math.sqrt(trading_days_per_year)
    )

    # Volume baseline over the previous W days (days t-W ... t-1):
    previous_volume = df["volume"].shift(1).rolling(volume_window, min_periods=volume_window)
    df["avg_volume"] = previous_volume.mean()
    df["volume_std"] = previous_volume.std(ddof=1)
    # Volume ratio:    V_t / average volume of previous W days
    df["volume_ratio"] = safe_divide(df["volume"], df["avg_volume"])
    # Volume Z-score:  (V_t - average volume) / std of volume
    volume_std = df["volume_std"]
    df["volume_zscore"] = (df["volume"] - df["avg_volume"]) / volume_std.where(
        volume_std > MIN_RELATIVE_VOLUME_STD * df["avg_volume"]
    )

    # Simple moving averages of the (unadjusted) close, as shown on price charts:
    #   MA_n = average(Close_(t-n+1) ... Close_t)
    for n in MOVING_AVERAGE_WINDOWS:
        df[f"ma_{n}"] = df["close"].rolling(n, min_periods=n).mean()

    # Drawdown: fall from the highest price seen so far.
    #   drawdown_t = P_t / max(P_0 ... P_t) - 1
    df["drawdown"] = price / price.cummax() - 1

    # Forward returns (what happened AFTER day t):  P_(t+h) / P_t - 1
    # These are NaN for the last h days of data (the future is not known yet).
    for h in FORWARD_HORIZONS:
        df[f"fwd_return_{h}d"] = price.shift(-h) / price - 1

    return df


def trailing_return(price: pd.Series, days: int) -> float | None:
    """Return over the last ``days`` trading days, or None if there is not enough history."""
    price = price.dropna()
    if len(price) <= days:
        return None
    return float(price.iloc[-1] / price.iloc[-1 - days] - 1)


def year_to_date_return(price: pd.Series) -> float | None:
    """Return from the last close of the previous calendar year to the latest close."""
    price = price.dropna()
    if price.empty:
        return None
    last_date = price.index[-1]
    previous_year = price[price.index.year < last_date.year]
    if previous_year.empty:
        return None
    return float(price.iloc[-1] / previous_year.iloc[-1] - 1)


def max_drawdown_details(price: pd.Series) -> dict[str, Any]:
    """Largest peak-to-trough fall in the series, with the dates involved.

    Returns a dict with keys: max_drawdown, peak_date, trough_date,
    recovery_date (None if the price never regained the peak), current_drawdown.
    """
    price = price.dropna()
    drawdown = price / price.cummax() - 1
    trough_date = drawdown.idxmin()
    peak_date = price.loc[:trough_date].idxmax()
    after_trough = price.loc[trough_date:]
    recovered = after_trough[after_trough >= price.loc[peak_date]]
    return {
        "max_drawdown": float(drawdown.min()),
        "peak_date": peak_date,
        "peak_price": float(price.loc[peak_date]),
        "trough_date": trough_date,
        "trough_price": float(price.loc[trough_date]),
        "recovery_date": recovered.index[0] if not recovered.empty else None,
        "current_drawdown": float(drawdown.iloc[-1]),
    }


def normal_two_sided_tail_probability(k: float) -> float:
    """P(|Z| >= k) for a standard normal variable:  erfc(k / sqrt(2))."""
    return math.erfc(k / math.sqrt(2))


def summarize_statistics(
    df: pd.DataFrame,
    zscore_threshold: float | None,
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    """Collect the headline numbers used in the report from a statistics frame.

    Args:
        df: output of ``add_statistics``.
        zscore_threshold: the |Z| threshold used for events (for the tail-frequency check).
        trading_days_per_year: used to annualise.
    """
    price = df["price"]
    returns = df["daily_return"].dropna()
    last = df.iloc[-1]
    n_days = len(df)

    performance = {label: trailing_return(price, days) for label, days in PERFORMANCE_WINDOWS.items()}
    performance["Year to date"] = year_to_date_return(price)
    performance["Full period"] = float(price.iloc[-1] / price.iloc[0] - 1)

    years = (n_days - 1) / trading_days_per_year
    annualised_return = (
        float((price.iloc[-1] / price.iloc[0]) ** (1 / years) - 1) if years >= 1 else None
    )

    last_year = df.tail(trading_days_per_year)
    close = df["close"]

    moving_averages = {}
    for n in MOVING_AVERAGE_WINDOWS:
        value = last.get(f"ma_{n}")
        if pd.notna(value):
            moving_averages[f"ma_{n}"] = {
                "value": float(value),
                "close_vs_ma": float(last["close"] / value - 1),
            }
        else:
            moving_averages[f"ma_{n}"] = None

    distribution = {
        "observations": int(returns.size),
        "mean": float(returns.mean()),
        "median": float(returns.median()),
        "std": float(returns.std(ddof=1)),
        "skewness": float(returns.skew()),
        "excess_kurtosis": float(returns.kurt()),
        "min": float(returns.min()),
        "min_date": returns.idxmin(),
        "max": float(returns.max()),
        "max_date": returns.idxmax(),
        "share_positive_days": float((returns > 0).mean()),
    }

    tail = None
    zscores = df["return_zscore"].dropna()
    if zscore_threshold and not zscores.empty:
        observed = float((zscores.abs() >= zscore_threshold).mean())
        expected = normal_two_sided_tail_probability(zscore_threshold)
        tail = {
            "threshold": float(zscore_threshold),
            "days_scored": int(zscores.size),
            "days_beyond_threshold": int((zscores.abs() >= zscore_threshold).sum()),
            "observed_share": observed,
            "normal_expected_share": expected,
            "ratio_to_normal": observed / expected if expected > 0 else None,
        }

    return {
        "start_date": df.index[0],
        "end_date": df.index[-1],
        "trading_days": n_days,
        "last_close": float(last["close"]),
        "last_price_for_returns": float(last["price"]),
        "high_52w_close": float(close.tail(trading_days_per_year).max()),
        "high_52w_date": close.tail(trading_days_per_year).idxmax(),
        "low_52w_close": float(close.tail(trading_days_per_year).min()),
        "low_52w_date": close.tail(trading_days_per_year).idxmin(),
        "close_vs_52w_high": float(last["close"] / close.tail(trading_days_per_year).max() - 1),
        "performance": performance,
        "annualised_return": annualised_return,
        "volatility": {
            "current_rolling_annualised": _float_or_none(last.get("rolling_volatility")),
            "full_period_annualised": float(returns.std(ddof=1) * math.sqrt(trading_days_per_year)),
            "last_year_annualised": float(
                last_year["daily_return"].dropna().std(ddof=1) * math.sqrt(trading_days_per_year)
            ) if last_year["daily_return"].notna().sum() > 2 else None,
            "daily_std": float(returns.std(ddof=1)),
        },
        "drawdown": max_drawdown_details(price),
        "moving_averages": moving_averages,
        "distribution": distribution,
        "tail_frequency": tail,
        "volume": {
            "last_volume": _float_or_none(last.get("volume")),
            "avg_volume_previous_window": _float_or_none(last.get("avg_volume")),
            "last_volume_ratio": _float_or_none(last.get("volume_ratio")),
        },
    }


def _float_or_none(value: Any) -> float | None:
    """Convert to float, mapping NaN/None to None."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value
