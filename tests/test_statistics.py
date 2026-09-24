"""Tests for return, Z-score, volume, volatility, drawdown and forward-return calculations.

Expected values are worked out by hand (or with plain numpy) so the formulas
in src/analysis/statistics.py are checked independently.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.statistics import (
    add_statistics,
    max_drawdown_details,
    normal_two_sided_tail_probability,
    summarize_statistics,
    trailing_return,
    year_to_date_return,
)


def price_frame(closes, volumes=None, start="2024-01-02"):
    dates = pd.bdate_range(start, periods=len(closes))
    volumes = volumes if volumes is not None else [1000.0] * len(closes)
    df = pd.DataFrame({
        "open": closes, "high": closes, "low": closes, "close": closes, "adj_close": closes, "volume": volumes,
    }, index=dates, dtype=float)
    df.index.name = "date"
    return df


def test_daily_and_log_returns():
    df = add_statistics(price_frame([100, 110, 99, 99]), zscore_window=5, volume_window=5, volatility_window=5)
    assert math.isnan(df["daily_return"].iloc[0])
    assert df["daily_return"].iloc[1] == pytest.approx(0.10)
    assert df["daily_return"].iloc[2] == pytest.approx(99 / 110 - 1)
    assert df["daily_return"].iloc[3] == pytest.approx(0.0)
    assert df["log_return"].iloc[1] == pytest.approx(math.log(1.10))


def test_trailing_multi_day_returns():
    closes = [100, 101, 102, 103, 104, 110] + [120] * 20
    df = add_statistics(price_frame(closes), zscore_window=5, volume_window=5, volatility_window=5)
    assert df["return_5d"].iloc[5] == pytest.approx(110 / 100 - 1)
    assert math.isnan(df["return_5d"].iloc[4])
    assert df["return_20d"].iloc[20] == pytest.approx(120 / 100 - 1)


def test_return_zscore_uses_previous_window_only():
    rng = np.random.default_rng(0)
    returns = list(rng.normal(0, 0.01, 30)) + [0.12]  # big jump on the last day
    closes = 100 * np.cumprod(1 + np.array([0.0] + returns))
    window = 20
    df = add_statistics(price_frame(list(closes)), zscore_window=window, volume_window=5, volatility_window=5)

    daily = df["daily_return"]
    last = len(df) - 1
    baseline = daily.iloc[last - window:last]  # the 20 returns BEFORE the last day
    expected = (daily.iloc[last] - baseline.mean()) / baseline.std(ddof=1)
    assert df["return_zscore"].iloc[last] == pytest.approx(expected)
    assert df["rolling_mean_return"].iloc[last] == pytest.approx(baseline.mean())
    assert df["rolling_std_return"].iloc[last] == pytest.approx(baseline.std(ddof=1))
    # The jump must NOT be part of its own baseline, so its Z-score is large.
    assert df["return_zscore"].iloc[last] > 5


def test_zscore_needs_full_window():
    closes = list(100 * np.cumprod(1 + np.full(30, 0.01) + np.linspace(0, 0.001, 30)))
    df = add_statistics(price_frame(closes), zscore_window=20, volume_window=5, volatility_window=5)
    # Row 0 has no return; rows 1..20 have fewer than 20 previous returns.
    assert df["return_zscore"].iloc[:21].isna().all()
    assert df["return_zscore"].iloc[21:].notna().all()


def test_zscore_is_nan_when_baseline_has_no_variation():
    closes = [100 * 1.01 ** i for i in range(30)]  # identical 1% returns -> std = 0
    df = add_statistics(price_frame(closes), zscore_window=10, volume_window=5, volatility_window=5)
    assert df["return_zscore"].iloc[15:].isna().all()


def test_volume_ratio_and_zscore():
    volumes = [100.0, 200.0] * 10 + [450.0]  # previous 20 days: mean 150, sample std ~51.3
    df = add_statistics(price_frame([10.0] * 21, volumes), zscore_window=5, volume_window=20, volatility_window=5)
    previous = pd.Series(volumes[:20])
    assert df["avg_volume"].iloc[20] == pytest.approx(150.0)
    assert df["volume_ratio"].iloc[20] == pytest.approx(450 / 150)
    assert df["volume_zscore"].iloc[20] == pytest.approx((450 - 150) / previous.std(ddof=1))
    assert df["volume_ratio"].iloc[:20].isna().all()


def test_volume_ratio_with_zero_average_is_nan():
    df = add_statistics(price_frame([10.0] * 8, [0.0] * 7 + [500.0]), zscore_window=5, volume_window=5,
                        volatility_window=5)
    assert math.isnan(df["volume_ratio"].iloc[7])


def test_rolling_volatility_is_annualised():
    rng = np.random.default_rng(3)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 40)))
    df = add_statistics(price_frame(closes), zscore_window=5, volume_window=5, volatility_window=20,
                        trading_days_per_year=252)
    expected = df["daily_return"].iloc[-20:].std(ddof=1) * math.sqrt(252)
    assert df["rolling_volatility"].iloc[-1] == pytest.approx(expected)


def test_moving_averages():
    closes = list(range(1, 61))
    df = add_statistics(price_frame(closes), zscore_window=5, volume_window=5, volatility_window=5)
    assert df["ma_20"].iloc[-1] == pytest.approx(np.mean(closes[-20:]))
    assert df["ma_50"].iloc[-1] == pytest.approx(np.mean(closes[-50:]))
    assert math.isnan(df["ma_50"].iloc[48])


def test_forward_returns():
    closes = [100, 102, 101, 105, 110, 120, 90]
    df = add_statistics(price_frame(closes), zscore_window=5, volume_window=5, volatility_window=5)
    assert df["fwd_return_1d"].iloc[0] == pytest.approx(0.02)
    assert df["fwd_return_5d"].iloc[0] == pytest.approx(120 / 100 - 1)
    assert df["fwd_return_5d"].iloc[1] == pytest.approx(90 / 102 - 1)
    # Not enough future data -> unknown, never zero
    assert math.isnan(df["fwd_return_1d"].iloc[-1])
    assert df["fwd_return_5d"].iloc[2:].isna().all()
    assert df["fwd_return_20d"].isna().all()


def test_drawdown_and_max_drawdown_details():
    closes = [100, 120, 90, 100, 130, 125]
    df = add_statistics(price_frame(closes), zscore_window=5, volume_window=5, volatility_window=5)
    assert df["drawdown"].iloc[2] == pytest.approx(90 / 120 - 1)
    info = max_drawdown_details(df["price"])
    assert info["max_drawdown"] == pytest.approx(-0.25)
    assert info["peak_date"] == df.index[1]
    assert info["trough_date"] == df.index[2]
    assert info["recovery_date"] == df.index[4]
    assert info["current_drawdown"] == pytest.approx(125 / 130 - 1)


def test_max_drawdown_not_recovered():
    df = add_statistics(price_frame([100, 80, 90]), zscore_window=5, volume_window=5, volatility_window=5)
    assert max_drawdown_details(df["price"])["recovery_date"] is None


def test_trailing_and_ytd_returns():
    dates = pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"])
    price = pd.Series([90.0, 100.0, 105.0, 110.0], index=dates)
    assert trailing_return(price, 1) == pytest.approx(110 / 105 - 1)
    assert trailing_return(price, 10) is None
    assert year_to_date_return(price) == pytest.approx(0.10)


def test_normal_tail_probability():
    # P(|Z| >= 2.5) for a standard normal distribution is about 1.242%
    assert normal_two_sided_tail_probability(2.5) == pytest.approx(0.012419, abs=1e-6)
    assert normal_two_sided_tail_probability(1.96) == pytest.approx(0.05, abs=1e-4)


def test_summary_statistics_values():
    rng = np.random.default_rng(7)
    closes = list(100 * np.cumprod(1 + rng.normal(0.0005, 0.02, 300)))
    df = add_statistics(price_frame(closes), zscore_window=60, volume_window=20, volatility_window=20)
    summary = summarize_statistics(df, 2.5, 252)
    returns = df["daily_return"].dropna()
    assert summary["trading_days"] == 300
    assert summary["performance"]["Full period"] == pytest.approx(closes[-1] / closes[0] - 1)
    assert summary["performance"]["1 day"] == pytest.approx(closes[-1] / closes[-2] - 1)
    assert summary["volatility"]["full_period_annualised"] == pytest.approx(returns.std() * math.sqrt(252))
    assert summary["distribution"]["mean"] == pytest.approx(returns.mean())
    assert summary["distribution"]["median"] == pytest.approx(returns.median())
    assert summary["high_52w_close"] == pytest.approx(max(closes[-252:]))
    tail = summary["tail_frequency"]
    zs = df["return_zscore"].dropna()
    assert tail["days_scored"] == len(zs)
    assert tail["days_beyond_threshold"] == int((zs.abs() >= 2.5).sum())
