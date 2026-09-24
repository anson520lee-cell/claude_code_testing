"""Compare a stock with a benchmark (default: SPY, the S&P 500 ETF).

Formulas (same notation as statistics.py, B_t = benchmark price):

    Benchmark return      rb_t = B_t / B_(t-1) - 1
    Abnormal return       ar_t = r_t - rb_t
    Cumulative return     R_t  = P_t / P_0 - 1         (same for the benchmark)
    Relative performance  RP_t = (P_t / P_0) / (B_t / B_0) - 1
    Beta                  cov(r, rb) / var(rb)

The benchmark is first aligned to the stock's trading days. Returns are
computed only after alignment, so the stock and benchmark returns on any row
always cover exactly the same dates. A missing benchmark day becomes NaN
instead of silently spanning two days.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.analysis.statistics import FORWARD_HORIZONS, PERFORMANCE_WINDOWS


def add_benchmark_columns(df: pd.DataFrame, benchmark_price: pd.Series) -> pd.DataFrame:
    """Add benchmark and abnormal-return columns to a statistics frame.

    Args:
        df: output of ``add_statistics`` (must contain "price", "daily_return"
            and the forward-return columns).
        benchmark_price: benchmark price series indexed by date.

    Returns:
        A copy of ``df`` with the columns benchmark_price, benchmark_return,
        abnormal_return, fwd_benchmark_{h}d, fwd_abnormal_{h}d, cum_return,
        benchmark_cum_return and relative_performance.
    """
    out = df.copy()
    bench = benchmark_price.astype(float).reindex(out.index)
    out["benchmark_price"] = bench
    out["benchmark_return"] = bench / bench.shift(1) - 1
    out["abnormal_return"] = out["daily_return"] - out["benchmark_return"]

    for h in FORWARD_HORIZONS:
        out[f"fwd_benchmark_{h}d"] = bench.shift(-h) / bench - 1
        out[f"fwd_abnormal_{h}d"] = out[f"fwd_return_{h}d"] - out[f"fwd_benchmark_{h}d"]

    # Cumulative performance starts on the first day where BOTH prices exist.
    both = out["price"].notna() & bench.notna()
    if both.any():
        start = both[both].index[0]
        stock_growth = out["price"] / out.loc[start, "price"]
        bench_growth = bench / bench.loc[start]
        out["cum_return"] = stock_growth - 1
        out["benchmark_cum_return"] = bench_growth - 1
        out["relative_performance"] = stock_growth / bench_growth - 1
        out.loc[out.index < start, ["cum_return", "benchmark_cum_return", "relative_performance"]] = float("nan")
    else:
        out["cum_return"] = float("nan")
        out["benchmark_cum_return"] = float("nan")
        out["relative_performance"] = float("nan")
    return out


def _window_return(series: pd.Series, days: int | None) -> float | None:
    """Return over the last ``days`` rows (None = whole series). None if not enough data."""
    series = series.dropna()
    if series.empty:
        return None
    if days is None:
        return float(series.iloc[-1] / series.iloc[0] - 1)
    if len(series) <= days:
        return None
    return float(series.iloc[-1] / series.iloc[-1 - days] - 1)


def summarize_benchmark(df: pd.DataFrame, benchmark: str, trading_days_per_year: int = 252) -> dict[str, Any]:
    """Headline benchmark comparison numbers for the report.

    Only days where both the stock and the benchmark have prices are used.
    """
    both = df[["price", "benchmark_price"]].dropna()
    if len(both) < 2:
        return {"benchmark": benchmark, "available": False, "reason": "No overlapping price history."}

    windows: dict[str, int | None] = dict(PERFORMANCE_WINDOWS)
    windows["Full period"] = None
    comparison = []
    for label, days in windows.items():
        stock_ret = _window_return(both["price"], days)
        bench_ret = _window_return(both["benchmark_price"], days)
        difference = stock_ret - bench_ret if stock_ret is not None and bench_ret is not None else None
        comparison.append({
            "window": label,
            "stock_return": stock_ret,
            "benchmark_return": bench_ret,
            "difference": difference,
        })

    paired = df[["daily_return", "benchmark_return"]].dropna()
    beta = correlation = None
    if len(paired) > 20 and paired["benchmark_return"].var() > 0:
        beta = float(paired["daily_return"].cov(paired["benchmark_return"]) / paired["benchmark_return"].var())
        correlation = float(paired["daily_return"].corr(paired["benchmark_return"]))

    abnormal = df["abnormal_return"].dropna()
    return {
        "benchmark": benchmark,
        "available": True,
        "overlap_start": both.index[0],
        "overlap_end": both.index[-1],
        "overlap_days": int(len(both)),
        "comparison": comparison,
        "beta": beta,
        "correlation": correlation,
        "abnormal_return_daily_std": float(abnormal.std(ddof=1)) if len(abnormal) > 2 else None,
        "abnormal_return_annualised_std": (
            float(abnormal.std(ddof=1) * math.sqrt(trading_days_per_year)) if len(abnormal) > 2 else None
        ),
        "final_relative_performance": (
            float(df["relative_performance"].dropna().iloc[-1])
            if df["relative_performance"].notna().any() else None
        ),
    }
