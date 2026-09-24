"""Tests for benchmark alignment, abnormal returns, relative performance and beta."""

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.benchmark import add_benchmark_columns, summarize_benchmark
from src.analysis.statistics import add_statistics


def stats_from(closes, dates):
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "adj_close": closes,
                       "volume": 1000.0}, index=dates, dtype=float)
    return add_statistics(df, "adj_close", zscore_window=5, volume_window=5, volatility_window=5)


def test_abnormal_return_is_stock_minus_benchmark():
    dates = pd.bdate_range("2024-01-02", periods=4)
    stats = stats_from([100, 110, 99, 99], dates)
    bench = pd.Series([50.0, 51.0, 51.0, 49.98], index=dates)
    out = add_benchmark_columns(stats, bench)
    assert out["benchmark_return"].iloc[1] == pytest.approx(0.02)
    assert out["abnormal_return"].iloc[1] == pytest.approx(0.10 - 0.02)
    assert out["abnormal_return"].iloc[2] == pytest.approx((99 / 110 - 1) - 0.0)
    assert out["fwd_abnormal_1d"].iloc[0] == pytest.approx(0.10 - 0.02)


def test_missing_benchmark_day_is_not_spanned():
    dates = pd.bdate_range("2024-01-02", periods=5)
    stats = stats_from([100, 101, 102, 103, 104], dates)
    bench = pd.Series([50.0, 51.0, 52.0, 53.0], index=dates.delete(2))  # day 2 missing
    out = add_benchmark_columns(stats, bench)
    assert math.isnan(out["benchmark_return"].iloc[2])
    assert math.isnan(out["benchmark_return"].iloc[3])  # would silently span 2 days otherwise
    assert out["benchmark_return"].iloc[4] == pytest.approx(53 / 52 - 1)


def test_cumulative_relative_performance():
    dates = pd.bdate_range("2024-01-02", periods=3)
    stats = stats_from([100, 120, 150], dates)
    bench = pd.Series([200.0, 220.0, 250.0], index=dates)
    out = add_benchmark_columns(stats, bench)
    assert out["cum_return"].iloc[-1] == pytest.approx(0.5)
    assert out["benchmark_cum_return"].iloc[-1] == pytest.approx(0.25)
    assert out["relative_performance"].iloc[-1] == pytest.approx(1.5 / 1.25 - 1)


def test_beta_and_correlation():
    rng = np.random.default_rng(5)
    bench_returns = rng.normal(0, 0.01, 200)
    dates = pd.bdate_range("2023-01-02", periods=201)
    bench = pd.Series(100 * np.cumprod(np.concatenate([[1.0], 1 + bench_returns])), index=dates)
    stock = 50 * np.cumprod(np.concatenate([[1.0], 1 + 2 * bench_returns]))  # exactly 2x the daily return
    out = add_benchmark_columns(stats_from(list(stock), dates), bench)
    summary = summarize_benchmark(out, "BENCH")
    assert summary["beta"] == pytest.approx(2.0)
    assert summary["correlation"] == pytest.approx(1.0)
    full = [c for c in summary["comparison"] if c["window"] == "Full period"][0]
    assert full["stock_return"] == pytest.approx(stock[-1] / stock[0] - 1)
    assert full["difference"] == pytest.approx(full["stock_return"] - full["benchmark_return"])


def test_no_overlap_is_reported_not_crashed():
    dates = pd.bdate_range("2024-01-02", periods=5)
    stats = stats_from([100, 101, 102, 103, 104], dates)
    bench = pd.Series([1.0, 2.0], index=pd.bdate_range("2020-01-01", periods=2))
    out = add_benchmark_columns(stats, bench)
    summary = summarize_benchmark(out, "BENCH")
    assert summary["available"] is False
