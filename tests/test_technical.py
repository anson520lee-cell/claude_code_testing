"""Tests for the swing technical layer (src/technical).

The primary indicators are the trader's chart settings: EMA 9/21/50/200/250,
RSI 6/14, MACD 12/26/9, MAVOL20 and BOLL(20, 1.8). Indicator values are checked
against independent, loop-based implementations of the documented formulas;
classification rules are checked on hand-made rows at and around their
thresholds; the reports are checked against the length and wording rules of
the report design.
"""

from __future__ import annotations

import copy
import math
import re

import numpy as np
import pandas as pd
import pytest

from src.analysis.benchmark import add_benchmark_columns
from src.analysis.statistics import add_statistics
from src.config import DEFAULT_SETTINGS, ConfigError, load_settings, validate_settings
from src.data.prices import standardize_price_frame
from src.technical import chart as technical_chart_module
from src.technical import classification as cls
from src.technical import indicators as ind
from src.technical import structure as st
from src.technical.analysis import next_earnings, run_technical_analysis
from src.technical.indicators import adjusted_ohlcv, atr, compute_indicators, macd, obv, rsi
from src.technical.reporting import (
    EVENT_RISK_BANNER,
    Money,
    build_summary_snapshot,
    build_technical_markdown,
    swing_summary,
    technical_json,
)
from src.technical.structure import NO_STRUCTURE, STATE_ORDER, structure_states, support_resistance
from src.technical.swing_stats import condition_outcomes, count_episodes
from tests.conftest import make_yf_prices

CONDITIONS = {"Strong", "Improving", "Mixed", "Weakening", "Weak"}


# ---------------------------------------------------------------- reference formulas
def ref_ema(values: np.ndarray, span: int) -> np.ndarray:
    """EMA_t = a * x_t + (1 - a) * EMA_(t-1), a = 2 / (span + 1); first span-1 values hidden."""
    a = 2 / (span + 1)
    out = np.empty(len(values))
    out[0] = values[0]
    for t in range(1, len(values)):
        out[t] = a * values[t] + (1 - a) * out[t - 1]
    out[: span - 1] = np.nan
    return out


def ref_wilder(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    out[period - 1] = np.mean(values[:period])
    for t in range(period, len(values)):
        out[t] = (out[t - 1] * (period - 1) + values[t]) / period
    return out


def ref_rsi(close: np.ndarray, period: int) -> np.ndarray:
    change = np.diff(close)
    avg_gain = ref_wilder(np.maximum(change, 0), period)
    avg_loss = ref_wilder(np.maximum(-change, 0), period)
    return np.concatenate([[np.nan], 100 - 100 / (1 + avg_gain / avg_loss)])


# Planted days: heavy-volume up and down days, so volume-dependent rules are exercised.
SHOCKS = {300: (0.07, 3.0), 420: (-0.06, 2.6), 505: (0.012, 1.8), 560: (-0.02, 2.2), 610: (0.05, 1.6)}


@pytest.fixture(scope="module")
def market():
    stock = standardize_price_frame(make_yf_prices(n=700, seed=21, shocks=SHOCKS))
    spy = standardize_price_frame(make_yf_prices(n=700, seed=99, base_price=400, daily_vol=0.008))
    return stock, spy


@pytest.fixture(scope="module")
def indicators(market):
    stock, spy = market
    return compute_indicators(adjusted_ohlcv(stock, "adj_close"), spy["adj_close"])


def analyse(stock, spy, end=None, earnings=None, benchmark=True, settings=None):
    prices = stock.iloc[:end]
    stats = add_statistics(prices, "adj_close")
    if benchmark:
        stats = add_benchmark_columns(stats, spy["adj_close"])
    return run_technical_analysis(prices, "adj_close", stats, earnings, settings, "SPY" if benchmark else None,
                                  "TEST")


# ---------------------------------------------------------------- the trader's parameters
def test_primary_indicator_parameters_are_the_traders_chart_settings(indicators):
    assert ind.EMA_SPANS == (9, 21, 50, 200, 250)
    assert ind.RSI_PERIODS == (6, 14)
    assert (ind.MACD_FAST, ind.MACD_SLOW, ind.MACD_SIGNAL) == (12, 26, 9)
    assert ind.MAVOL_WINDOW == 20
    assert (ind.BOLL_WINDOW, ind.BOLL_STD) == (20, 1.8)
    for column in ("ema9", "ema21", "ema50", "ema200", "ema250", "rsi6", "rsi14", "macd", "macd_signal",
                   "macd_hist", "mavol20", "volume_ratio", "bb_upper", "bb_mid", "bb_lower"):
        assert column in indicators, column
    for obsolete in ("ema5", "ema10", "ema20", "rsi7", "sma50", "sma200", "avg_volume_20d"):
        assert obsolete not in indicators, obsolete


# ---------------------------------------------------------------- indicators
def test_ema_9_21_50_200_250_match_the_recursive_definition(indicators):
    close = indicators["close"].to_numpy()
    for span in (9, 21, 50, 200, 250):
        np.testing.assert_allclose(indicators[f"ema{span}"], ref_ema(close, span), rtol=1e-12)
        assert indicators[f"ema{span}"].iloc[: span - 1].isna().all()  # warm-up is not reported
        assert indicators[f"ema{span}"].iloc[span - 1:].notna().all()
    t = 400
    for span in (9, 21, 50):  # slope = % change over the last 3 sessions
        assert indicators[f"ema{span}_slope"].iloc[t] == pytest.approx(
            indicators[f"ema{span}"].iloc[t] / indicators[f"ema{span}"].iloc[t - 3] - 1)


def test_rsi_6_and_14_match_wilders_definition(indicators):
    close = indicators["close"].to_numpy()
    for period in (6, 14):
        np.testing.assert_allclose(indicators[f"rsi{period}"], ref_rsi(close, period), rtol=1e-10)
        assert indicators[f"rsi{period}"].iloc[:period].isna().all()
        assert indicators[f"rsi{period}"].iloc[period:].between(0, 100).all()


def test_rsi_worked_example_and_edge_cases():
    # changes +1, -0.5, +1, +0.5, -1 -> first average gain 2/3, loss 1/6 (RS 4, RSI 80), then Wilder smoothing
    value = rsi(pd.Series([10, 11, 10.5, 11.5, 12, 11]), 3)
    assert value.iloc[:3].isna().all()
    assert value.iloc[3:].tolist() == pytest.approx([80.0, 100 - 100 / 6.5, 50.0])
    rising = pd.Series(np.arange(1.0, 31.0))
    assert rsi(rising, 6).iloc[-1] == 100.0                               # no losses
    assert rsi(pd.Series(np.full(30, 5.0)), 6).iloc[-1] == 50.0            # no movement at all
    assert rsi(pd.Series(rising.to_numpy()[::-1]), 6).iloc[-1] == pytest.approx(0.0)  # only losses


def test_macd_12_26_9_dif_dea_histogram(indicators):
    close = indicators["close"].to_numpy()
    dif = ref_ema(close, 12) - ref_ema(close, 26)
    first = 25  # first day on which EMA26 exists
    dea = np.full(len(close), np.nan)
    dea[first:] = ref_ema(dif[first:], 9)
    np.testing.assert_allclose(indicators["macd"], dif, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(indicators["macd_signal"], dea, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(indicators["macd_hist"], dif - dea, rtol=1e-9, atol=1e-12)
    assert indicators["macd_signal"].iloc[:33].isna().all() and indicators["macd_signal"].iloc[33:].notna().all()
    pd.testing.assert_frame_equal(macd(indicators["close"]), macd(indicators["close"], 12, 26, 9))


def test_boll_20_1_8_uses_population_std(indicators):
    close = indicators["close"].to_numpy()
    for t in (19, 150, 699):
        window = close[t - 19: t + 1]
        mid, sd = window.mean(), window.std(ddof=0)
        row = indicators.iloc[t]
        assert row["bb_mid"] == pytest.approx(mid)
        assert row["bb_upper"] == pytest.approx(mid + 1.8 * sd)
        assert row["bb_lower"] == pytest.approx(mid - 1.8 * sd)
        assert row["bb_pct_b"] == pytest.approx((close[t] - (mid - 1.8 * sd)) / (3.6 * sd))
        assert row["bb_width"] == pytest.approx(3.6 * sd / mid)
    t = 300
    assert indicators["bb_width_change"].iloc[t] == pytest.approx(
        indicators["bb_width"].iloc[t] / indicators["bb_width"].iloc[t - 5] - 1)
    assert indicators["bb_mid"].iloc[:19].isna().all()


def test_mavol20_includes_today_and_volume_ratio(indicators):
    volume = indicators["volume"].to_numpy()
    t = 420  # a planted heavy-volume down day
    mavol = volume[t - 19: t + 1].mean()  # the last 20 sessions INCLUDING today, as on a chart
    assert indicators["mavol20"].iloc[t] == pytest.approx(mavol)
    assert indicators["volume_ratio"].iloc[t] == pytest.approx(volume[t] / mavol)
    assert indicators["volume_ratio"].iloc[t] > 2.0
    assert indicators["avg_volume_5d"].iloc[t] == pytest.approx(volume[t - 4: t + 1].mean())
    assert indicators["volume_5d_vs_mavol20"].iloc[t] == pytest.approx(volume[t - 4: t + 1].mean() / mavol)
    assert indicators["mavol20"].iloc[:19].isna().all()


def test_atr_14_is_the_wilder_average_of_true_range(indicators):
    high, low, close = (indicators[k].to_numpy() for k in ("high", "low", "close"))
    tr = np.empty(len(close))
    tr[0] = high[0] - low[0]
    for t in range(1, len(close)):
        tr[t] = max(high[t] - low[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1]))
    np.testing.assert_allclose(indicators["atr"], ref_wilder(tr, 14), rtol=1e-10)
    assert indicators["atr_pct"].iloc[-1] == pytest.approx(indicators["atr"].iloc[-1] / close[-1] * 100)


def test_true_range_includes_overnight_gaps():
    value = atr(pd.Series([10.0, 16.0]), pd.Series([9.0, 15.0]), pd.Series([10.0, 15.5]), period=1)
    assert value.iloc[-1] == pytest.approx(6.0)  # range is only 1, but the gap from 10 to the high 16 is 6


def test_rolling_vwap_20_uses_typical_price_and_volume(indicators):
    high, low, close, volume = (indicators[k].to_numpy() for k in ("high", "low", "close", "volume"))
    typical = (high + low + close) / 3
    for t in (19, 300, 699):
        window = slice(t - 19, t + 1)
        expected = (typical[window] * volume[window]).sum() / volume[window].sum()
        assert indicators["vwap20"].iloc[t] == pytest.approx(expected)


def test_obv_adds_volume_on_up_days_and_subtracts_it_on_down_days():
    close = pd.Series([10.0, 11.0, 11.0, 10.0, 12.0])
    volume = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
    assert obv(close, volume).tolist() == [0.0, 200.0, 200.0, -200.0, 300.0]


def test_obv_slope_is_measured_in_average_daily_volumes():
    n = 40
    close = 100.0 + np.arange(n)  # every day is an up day with the same volume
    frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                          "volume": np.full(n, 5000.0)}, index=pd.bdate_range("2024-01-01", periods=n))
    out = compute_indicators(frame)
    assert out["obv_slope10"].iloc[-1] == pytest.approx(1.0)  # OBV rises by one average volume per day
    assert out["obv_slope20"].iloc[-1] == pytest.approx(1.0)


def test_20d_and_60d_highs_and_lows(indicators):
    high, low = indicators["high"].to_numpy(), indicators["low"].to_numpy()
    t = 500
    assert indicators["high20"].iloc[t] == high[t - 19: t + 1].max()
    assert indicators["low20"].iloc[t] == low[t - 19: t + 1].min()
    assert indicators["high60"].iloc[t] == high[t - 59: t + 1].max()
    assert indicators["low60"].iloc[t] == low[t - 59: t + 1].min()
    assert indicators["prior_high20"].iloc[t] == high[t - 20: t].max()  # breakout reference excludes today
    assert indicators["prior_low20"].iloc[t] == low[t - 20: t].min()


def test_relative_strength_vs_benchmark(market, indicators):
    stock, spy = market
    c, b = stock["adj_close"].to_numpy(), spy["adj_close"].to_numpy()
    t = 450
    for n in (2, 5, 7, 20):
        assert indicators[f"rel_{n}d"].iloc[t] == pytest.approx((c[t] / c[t - n] - 1) - (b[t] / b[t - n] - 1))
    daily = (c[1:] / c[:-1] - 1) - (b[1:] / b[:-1] - 1)  # daily[k] = relative return of day k + 1
    sigma = np.std(daily[t - 61: t - 1], ddof=1)          # days t-60 ... t-1 (not today)
    assert indicators["rel_daily_sigma"].iloc[t] == pytest.approx(sigma)
    expected = (indicators["rel_5d"].iloc[t] / (sigma * math.sqrt(5))
                + indicators["rel_7d"].iloc[t] / (sigma * math.sqrt(7))) / 2
    assert cls.relative_strength_score(indicators).iloc[t] == pytest.approx(expected)


def test_without_benchmark_relative_strength_is_not_available(market):
    stock, _ = market
    out = cls.classify_all(compute_indicators(adjusted_ohlcv(stock, "adj_close")))
    assert out["rel_5d"].isna().all()
    assert (out["relative_strength"] == "n/a").all()


def test_adjusted_ohlcv_uses_the_dividend_factor_and_fills_missing_high_low():
    prices = pd.DataFrame({"open": [10.0, 20.0], "high": [11.0, np.nan], "low": [9.0, 19.0],
                           "close": [10.0, 20.0], "adj_close": [9.0, 20.0], "volume": [1.0, 2.0]},
                          index=pd.bdate_range("2024-01-01", periods=2))
    out = adjusted_ohlcv(prices, "adj_close")
    assert out.iloc[0][["open", "high", "low", "close"]].tolist() == pytest.approx([9.0, 9.9, 8.1, 9.0])
    assert out.iloc[1]["close"] == 20.0 and out.iloc[1]["high"] == 20.0  # latest day = traded price
    assert out.attrs["filled_high_low_days"] == 1
    assert adjusted_ohlcv(prices, "close").iloc[0]["close"] == 10.0


def test_indicators_are_causal(market):
    stock, spy = market
    full = compute_indicators(adjusted_ohlcv(stock, "adj_close"), spy["adj_close"])
    cut = compute_indicators(adjusted_ohlcv(stock.iloc[:400], "adj_close"), spy["adj_close"])
    columns = ["ema9", "ema21", "ema250", "rsi6", "macd_hist", "mavol20", "atr", "vwap20", "obv_slope10", "bb_pct_b",
               "bb_width_change", "rel_5d", "adx", "atr_pct_percentile"]
    pd.testing.assert_frame_equal(full[columns].iloc[:400], cut[columns])


# ---------------------------------------------------------------- classifications
def rows(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_swing_trend_classes_use_ema9_and_ema21():
    df = rows(
        close=[105, 95, 103, 97, 101, 99, 100, 101],
        ema9=[104, 96, 104, 96, 100, 100, 100.5, 100],
        ema21=[102, 98, 102, 98, 102, 98, 99.5, np.nan],
        ema9_slope=[0.01, -0.01, -0.001, 0.001, 0.002, -0.002, 0.001, 0.01],
        ema21_slope=[0.01, -0.01, 0.001, -0.001, 0.001, -0.001, -0.001, 0.01],
    )
    assert cls.classify_trend(df).tolist() == [
        cls.STRONG_UPTREND, cls.STRONG_DOWNTREND, cls.UPTREND, cls.DOWNTREND, cls.IMPROVING, cls.DETERIORATING,
        cls.MIXED, "n/a"]


def test_ema50_is_secondary_confirmation():
    df = rows(close=[105, 95, 105, np.nan], ema50=[100, 100, 100, 100], ema50_slope=[0.01, -0.01, -0.01, 0.01])
    assert cls.classify_ema50(df).tolist() == [cls.EMA50_UP, cls.EMA50_DOWN, cls.EMA50_MIXED, "n/a"]


def test_rsi6_zone_boundaries():
    values = [85, 80, 79.9, 70, 65, 60, 55, 50, 49.9, 40, 35, 30, 29.9, 10, np.nan]
    assert cls.rsi_zone(pd.Series(values)).tolist() == [
        "Extremely strong / extended", "Extremely strong / extended", "Strong momentum", "Strong momentum",
        "Positive momentum", "Positive momentum", "Mild positive momentum", "Mild positive momentum",
        "Mild negative momentum", "Mild negative momentum", "Weak momentum", "Weak momentum", "Oversold", "Oversold",
        "n/a"]


def test_rsi6_vs_rsi14_relationships():
    cross = cls.rsi_relationships(rows(rsi6=[40, 45, 55, 60], rsi14=[50, 50, 50, 50])).iloc[-1]
    assert cross["rsi_cross"] == 1
    below = cls.rsi_relationships(rows(rsi6=[60, 55, 45, 40], rsi14=[50, 50, 50, 50])).iloc[-1]
    assert below["rsi_cross"] == -1
    steady = cls.rsi_relationships(rows(rsi6=[60, 62, 64, 66], rsi14=[50, 50, 50, 50])).iloc[-1]
    assert steady["rsi_cross"] == 0
    recovering = cls.rsi_relationships(rows(rsi6=[35, 25, 20, 28, 33, 36], rsi14=[40] * 6)).iloc[-1]
    assert recovering["rsi_recovering_oversold"] and not recovering["rsi_losing_extended"]
    fading = cls.rsi_relationships(rows(rsi6=[70, 85, 82, 76, 72, 71], rsi14=[60] * 6)).iloc[-1]
    assert fading["rsi_losing_extended"] and not fading["rsi_recovering_oversold"]
    assert not cls.rsi_relationships(rows(rsi6=[70, 85, 84, 83, 82, 81], rsi14=[60] * 6)).iloc[-1]["rsi_losing_extended"]


@pytest.mark.parametrize("histogram,expected", [
    ([0.1, 0.2, 0.3, 0.4], cls.MACD_IMPROVING),
    ([-0.4, -0.3, -0.2, -0.1], cls.MACD_IMPROVING),   # still negative, but improving: the change comes first
    ([0.4, 0.3, 0.2, 0.1], cls.MACD_WEAKENING),
    ([0.2, 0.1, 0.3, 0.25], cls.MACD_POSITIVE),
    ([-0.2, -0.1, -0.3, -0.25], cls.MACD_NEGATIVE),
    ([0.1, 0.2, 0.3, np.nan], "n/a"),
])
def test_macd_12_26_9_states(histogram, expected):
    assert cls.classify_macd(rows(macd_hist=histogram)).iloc[-1] == expected


def macd_rows(dif: list[float], dea: list[float]) -> pd.DataFrame:
    dif_s, dea_s = pd.Series(dif, dtype=float), pd.Series(dea, dtype=float)
    return pd.DataFrame({"macd": dif_s, "macd_signal": dea_s, "macd_hist": dif_s - dea_s})


def test_macd_crossovers_histogram_and_zero_line_context():
    bullish = macd_rows([-0.1, -0.05, 0.02, 0.05], [0.0] * 4)
    assert cls.macd_details(bullish).iloc[-1]["macd_cross"] == 1
    assert cls.last_cross_age(bullish["macd"], bullish["macd_signal"]) == 1  # crossed one session ago
    bearish = macd_rows([0.1, 0.05, 0.02, -0.03], [0.0] * 4)
    assert cls.macd_details(bearish).iloc[-1]["macd_cross"] == -1
    assert cls.last_cross_age(bearish["macd"], bearish["macd_signal"]) == 0  # today
    expanding = cls.macd_details(macd_rows([0.1, 0.12, 0.15, 0.2], [0.0] * 4)).iloc[-1]
    assert expanding["macd_hist_trend"] == "expanding" and expanding["macd_cross"] == 0
    improving = cls.macd_details(macd_rows([-0.5, -0.45, -0.4, -0.3], [-0.2] * 4)).iloc[-1]
    assert improving["macd_zero_context"] == "below zero but improving"
    assert improving["macd_hist_trend"] == "contracting"
    fading = cls.macd_details(macd_rows([0.5, 0.45, 0.4, 0.3], [0.2] * 4)).iloc[-1]
    assert fading["macd_zero_context"] == "above zero but deteriorating"


def test_momentum_classes():
    df = rows(
        return_2d=[0.01, -0.01, 0.01, 0.01, 0.01, 0.01, -0.02, np.nan],
        return_3d=[0.02, -0.02, 0.02, 0.02, -0.01, 0.02, -0.01, 0.01],
        return_5d=[0.04, -0.04, 0.03, 0.03, -0.02, -0.01, -0.03, 0.01],
        return_7d=[0.05, -0.05, -0.01, 0.04, -0.03, -0.02, 0.01, 0.01],
        move_5d_atr=[2.0, -2.0, 0.5, 1.0, -0.5, 1.0, -1.0, 1.0],
    )
    assert cls.classify_momentum(df).tolist() == [
        cls.STRONG_POSITIVE, cls.STRONG_NEGATIVE, cls.MOMENTUM_POSITIVE, cls.MOMENTUM_POSITIVE,
        cls.MOMENTUM_NEGATIVE, cls.MOMENTUM_MIXED, cls.MOMENTUM_NEGATIVE, "n/a"]


def test_volume_vs_mavol20_levels_and_day_direction():
    levels = cls.classify_volume(rows(volume_ratio=[2.5, 2.0, 1.99, 1.5, 1.49, 0.8, 0.79, np.nan]))
    assert levels.tolist() == [cls.VERY_HIGH_PARTICIPATION, cls.VERY_HIGH_PARTICIPATION, cls.STRONG_PARTICIPATION,
                               cls.STRONG_PARTICIPATION, cls.VOLUME_NORMAL, cls.VOLUME_NORMAL,
                               cls.WEAK_PARTICIPATION, "n/a"]
    assert cls.day_direction(rows(return_1d=[0.01, -0.01, 0.0, np.nan])).tolist() == ["up", "down", "flat", "n/a"]


def test_vwap_position_is_secondary_reference():
    vwap = rows(close=[102, 101, 99, 98.9, np.nan], vwap20=[100] * 5, atr=[2] * 5)
    assert cls.classify_vwap(vwap).tolist() == ["Above", "Near", "Near", "Below", "n/a"]


def test_boll_positions_from_percent_b():
    pct_b = [1.05, 0.95, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1, 0.05, -0.05, np.nan]
    assert cls.classify_bollinger_position(rows(bb_pct_b=pct_b)).tolist() == [
        cls.ABOVE_UPPER, cls.NEAR_UPPER, cls.NEAR_UPPER, cls.UPPER_HALF, cls.NEAR_MIDDLE, cls.NEAR_MIDDLE,
        cls.NEAR_MIDDLE, cls.LOWER_HALF, cls.NEAR_LOWER, cls.NEAR_LOWER, cls.BELOW_LOWER, "n/a"]


@pytest.mark.parametrize("change,percentiles,expected", [
    (0.25, [50, 15, 30, 40, 50, 60], cls.EXPANSION_AFTER_SQUEEZE),
    (0.25, [50] * 6, cls.BAND_EXPANSION),
    (0.0, [50, 40, 30, 20, 15, 10], cls.VOLATILITY_SQUEEZE),
    (-0.2, [60, 55, 50, 45, 40, 40], cls.BAND_CONTRACTION),
    (0.0, [50] * 6, cls.BAND_STABLE),
])
def test_boll_band_width_states(change, percentiles, expected):
    frame = rows(bb_width_change=[0.0] * 5 + [change], bb_width_percentile=percentiles)
    assert cls.classify_band_state(frame).iloc[-1] == expected


def band_rows(pct_b: list[float], ema9: float, ema21: float, slope: float, volume_ratio: float) -> pd.DataFrame:
    n = len(pct_b)
    return rows(bb_pct_b=pct_b, close=[100.0] * n, ema9=[ema9] * n, ema21=[ema21] * n, ema21_slope=[slope] * n,
                volume_ratio=[volume_ratio] * n)


@pytest.mark.parametrize("pct_b,ema9,ema21,slope,volume,expected", [
    ([0.5, 0.95, 0.92, 0.97, 1.02], 98, 96, 0.01, 1.0, cls.UPPER_WALK),       # walk in a strong uptrend
    ([0.5, 0.6, 0.7, 0.6, 1.1], 98, 96, 0.01, 1.3, cls.UPPER_PUSH),           # first close above, confirmed
    ([0.5, 0.6, 0.7, 0.6, 1.1], 98, 96, 0.01, 0.7, cls.UPPER_EXTENSION),      # ... but on light volume
    ([0.5, 0.6, 0.7, 0.6, 1.1], 96, 98, -0.01, 1.3, cls.UPPER_EXTENSION),     # ... or without a swing uptrend
    ([0.95, 0.97, 0.92, 0.9, 0.96], 96, 98, -0.01, 1.3, cls.NO_BAND_CONTEXT),  # near the band without a trend
    ([0.5, 0.05, 0.08, 0.03, -0.02], 102, 104, -0.01, 1.0, cls.LOWER_WALK),
    ([0.5, 0.4, 0.3, 0.4, -0.1], 104, 102, 0.01, 0.9, cls.LOWER_EXTENSION),
    ([0.5] * 5, 98, 96, 0.01, 1.0, cls.NO_BAND_CONTEXT),
])
def test_boll_upper_band_walk_versus_unconfirmed_extension(pct_b, ema9, ema21, slope, volume, expected):
    assert cls.classify_band_context(band_rows(pct_b, ema9, ema21, slope, volume)).iloc[-1] == expected


def test_volatility_regime_from_atr_percentile():
    values = pd.Series([96, 95, 94.9, 80, 79.9, 60, 59.9, 20, 19.9, np.nan])
    assert cls.classify_volatility(values).tolist() == [
        "Extreme", "Extreme", "High", "High", "Elevated", "Elevated", "Normal", "Normal", "Low", "n/a"]


def test_relative_strength_classes():
    sigma = 0.01
    scores = [1.5, 1.0, 0.0, -1.0, -1.5, 0.0]
    df = rows(rel_5d=[s * sigma * math.sqrt(5) for s in scores], rel_7d=[s * sigma * math.sqrt(7) for s in scores],
              rel_daily_sigma=[sigma] * 5 + [np.nan])
    assert cls.relative_strength_score(df).iloc[:5].tolist() == pytest.approx(scores[:5])
    assert cls.classify_relative_strength(df).tolist() == [
        "Strongly outperforming", "Outperforming", "Neutral", "Underperforming", "Strongly underperforming", "n/a"]


def test_adx_context_classes():
    assert cls.classify_adx(pd.Series([30, 25, 24.9, 20, 19.9, np.nan])).tolist() == [
        "Trending environment", "Trending environment", "Borderline (ADX 20-25)", "Borderline (ADX 20-25)",
        "Weak / range-bound environment", "n/a"]


def test_swing_condition_weights_follow_the_indicator_priority():
    df = rows(
        trend=[cls.STRONG_UPTREND, cls.IMPROVING, cls.STRONG_DOWNTREND, cls.MIXED, cls.DETERIORATING, cls.UPTREND,
               "n/a"],
        structure=[st.BREAKOUT_VOLUME, st.BREAKOUT_VOLUME, st.BREAKDOWN_VOLUME, NO_STRUCTURE, st.LOSING_EMA21,
                   st.PULLBACK_EMA9, NO_STRUCTURE],
        return_1d=[0.03, 0.03, -0.03, 0.001, -0.01, -0.02, 0.01],
        volume_ratio=[2.2, 2.2, 2.2, 1.0, 1.7, 2.5, 1.0],
        rsi6=[70, 70, 25, 50, 45, 55, 55],
        rsi14=[60, 60, 35, 50, 50, 50, 50],
        macd_state=[cls.MACD_IMPROVING, cls.MACD_IMPROVING, cls.MACD_WEAKENING, cls.MACD_POSITIVE,
                    cls.MACD_WEAKENING, cls.MACD_NEGATIVE, cls.MACD_POSITIVE],
        band_context=[cls.UPPER_WALK, cls.UPPER_WALK, cls.LOWER_WALK, cls.NO_BAND_CONTEXT, cls.NO_BAND_CONTEXT,
                      cls.NO_BAND_CONTEXT, cls.NO_BAND_CONTEXT],
        relative_strength=["Outperforming", "Outperforming", "Underperforming", "Neutral", "Neutral", "Neutral", "n/a"],
        ema50_state=[cls.EMA50_UP, cls.EMA50_UP, cls.EMA50_DOWN, cls.EMA50_MIXED, cls.EMA50_MIXED, cls.EMA50_UP,
                     cls.EMA50_UP],
    )
    out = cls.swing_condition(df)
    assert out["swing_score"].iloc[:6].tolist() == [12, 10, -12, 0, -4, 1]
    assert out["swing_condition"].tolist() == ["Strong", "Improving", "Weak", "Mixed", "Weakening", "Mixed", "n/a"]
    # heavy volume on a DOWN day counts against the stock even inside an uptrend
    assert out["points_volume"].iloc[5] == -2 and out["points_trend"].iloc[5] == 2
    assert pd.isna(out["swing_score"].iloc[6])
    assert cls.TREND_POINTS[cls.STRONG_UPTREND] > max(cls.MACD_POINTS.values())  # EMA9/21 outweigh MACD


def test_structure_points_refer_to_real_states():
    assert set(st.STRUCTURE_POINTS) <= set(STATE_ORDER)


def test_classifications_use_only_documented_labels(market):
    stock, spy = market
    out = analyse(stock, spy)["frame"]
    documented = {
        "trend": set(cls.TREND_POINTS), "macd_state": set(cls.MACD_POINTS),
        "volume_level": {cls.VERY_HIGH_PARTICIPATION, cls.STRONG_PARTICIPATION, cls.VOLUME_NORMAL,
                         cls.WEAK_PARTICIPATION},
        "bb_position": {cls.ABOVE_UPPER, cls.NEAR_UPPER, cls.UPPER_HALF, cls.NEAR_MIDDLE, cls.LOWER_HALF,
                        cls.NEAR_LOWER, cls.BELOW_LOWER},
        "band_context": {cls.UPPER_WALK, cls.UPPER_PUSH, cls.UPPER_EXTENSION, cls.LOWER_WALK, cls.LOWER_PUSH,
                         cls.LOWER_EXTENSION, cls.NO_BAND_CONTEXT},
        "relative_strength": set(cls.RS_POINTS), "swing_condition": CONDITIONS,
        "volatility": {"Low", "Normal", "Elevated", "High", "Extreme"},
        "structure": set(STATE_ORDER) | {NO_STRUCTURE},
    }
    for column, labels in documented.items():
        assert set(out[column].unique()) <= labels | {"n/a"}, column
    for column in ("trend", "macd_state", "swing_condition"):
        assert out[column].iloc[-1] != "n/a"  # enough history -> always classified
    assert {cls.VERY_HIGH_PARTICIPATION, cls.STRONG_PARTICIPATION} & set(out["volume_level"])


# ---------------------------------------------------------------- structure states
NEUTRAL = dict(close=100.0, atr=2.0, prior_high20=110.0, prior_low20=90.0, high20=110.0, low20=90.0,
               ema9=99.0, ema21=101.0, ema21_slope=0.0, return_3d=0.0, volume_ratio=1.0, rsi6=50.0,
               macd_hist=0.0, bb_width_percentile=50.0, adx=25.0)


def structure_case(last: dict, before: dict | None = None, first: dict | None = None) -> pd.Series:
    """Four sessions: three earlier ones (``before`` applies to all three, ``first`` to the oldest),
    then ``last``; returns the structure states of the last session."""
    earlier = {**NEUTRAL, **(before or {})}
    frame = pd.DataFrame([{**earlier, **(first or {})}, earlier, earlier, {**earlier, **last}])
    return structure_states(frame).iloc[-1]


UPTREND_EMAS = dict(ema9=104.0, ema21=100.0, close=105.0)


@pytest.mark.parametrize("last,before,first,expected", [
    (dict(close=111.0, volume_ratio=1.8), None, None, st.BREAKOUT_VOLUME),        # breakout + 1.8x MAVOL20
    (dict(close=111.0, volume_ratio=1.2), None, None, st.BREAKOUT_NORMAL),
    (dict(close=111.0, volume_ratio=0.6), None, None, st.BREAKOUT_WEAK),          # breakout + 0.6x MAVOL20
    (dict(close=89.0, volume_ratio=1.7), None, None, st.BREAKDOWN_VOLUME),
    (dict(close=89.0, volume_ratio=1.0), None, None, st.BREAKDOWN),
    (dict(close=100.0), dict(close=102.0), None, st.LOSING_EMA21),
    (dict(close=102.0), dict(close=100.0), None, st.RECLAIMING_EMA21),
    (dict(close=99.5, ema9=100.0), dict(close=100.0, ema9=102.0), None, st.EMA9_CROSS_DOWN),
    (dict(close=103.0, ema9=102.0), dict(close=102.0), None, st.EMA9_CROSS_UP),
    (dict(close=104.5, ema21_slope=0.01, return_3d=-0.01), UPTREND_EMAS, None, st.PULLBACK_EMA9),
    (dict(close=100.5, ema9=103.0, ema21=100.2, ema21_slope=0.01, return_3d=-0.02), UPTREND_EMAS, None,
     st.PULLBACK_EMA21),
    (dict(close=102.5, ema9=106.0, ema21_slope=0.01, return_3d=-0.02), dict(ema9=106.0, ema21=100.0, close=107.0),
     None, st.PULLBACK),
    (dict(close=98.9), None, None, st.LOSING_EMA9),
    (dict(close=100.0), dict(close=98.0), None, st.RECLAIMING_EMA9),
    (dict(close=109.0), dict(close=108.0), None, st.NEAR_BREAKOUT),
    (dict(close=102.5, rsi6=54.0, macd_hist=0.2), dict(ema9=102.0, ema21=100.0, close=103.0),
     dict(rsi6=75.0, macd_hist=0.5), st.MOMENTUM_DETERIORATION),
    (dict(), dict(ema9=100.2, ema21=100.0, close=100.5), None, st.EMA_COMPRESSION),
    (dict(bb_width_percentile=15.0), None, None, st.VOLATILITY_SQUEEZE),
    (dict(high20=104.0, low20=97.0, prior_high20=104.0, prior_low20=97.0, adx=18.0), None, None, st.RANGE),
    (dict(), dict(ema9=102.0, ema21=100.0, close=103.0), None, st.BULL_STACK),
    (dict(), dict(ema9=98.0, ema21=100.0, close=97.0), None, st.BEAR_STACK),
    (dict(), None, None, NO_STRUCTURE),
])
def test_structure_states(last, before, first, expected):
    states = structure_case(last, before, first)
    assert states["structure"] == expected
    if expected != NO_STRUCTURE:
        assert states[expected]


def test_breakout_has_priority_and_states_are_not_signals():
    states = structure_case(dict(close=111.0, volume_ratio=1.8, bb_width_percentile=10.0))
    assert states["structure"] == st.BREAKOUT_VOLUME
    assert states[st.VOLATILITY_SQUEEZE] and states[st.RECLAIMING_EMA21]  # still reported as secondary states
    assert not any(re.search(r"\b(buy|sell|hold)\b", name, re.IGNORECASE) for name in STATE_ORDER)


# ---------------------------------------------------------------- support / resistance
def level_frame() -> pd.DataFrame:
    """40 sessions drifting down (so the path itself has no swing points) with planted pivots."""
    n = 40
    close = 120.0 - 0.5 * np.arange(n)
    frame = pd.DataFrame({"close": close, "high": close + 0.5, "low": close - 0.5, "atr": 2.0},
                         index=pd.bdate_range("2024-01-01", periods=n))
    frame.iloc[10, frame.columns.get_loc("low")] = 109.5   # old swing low, now ABOVE the price
    frame.iloc[15, frame.columns.get_loc("low")] = 96.0    # swing low
    frame.iloc[25, frame.columns.get_loc("low")] = 99.0    # swing low
    frame.iloc[30, frame.columns.get_loc("high")] = 108.0  # swing high
    frame.iloc[34, frame.columns.get_loc("high")] = 108.6  # swing high within 0.5 ATR of the one above
    for window in (20, 60):
        frame[f"high{window}"] = frame["high"].rolling(window, min_periods=1).max()
        frame[f"low{window}"] = frame["low"].rolling(window, min_periods=1).min()
    return frame


def test_support_resistance_levels_are_nearest_distinct_and_polarity_aware():
    frame = level_frame()
    dates = frame.index
    levels = support_resistance(frame, lookback=60, bars=3)
    r1, r2 = levels["resistance"]
    s1, s2 = levels["support"]
    assert (r1.price, r2.price, s1.price, s2.price) == (108.0, 109.5, 99.0, 96.0)
    assert r1.origin == f"swing high {dates[30]:%Y-%m-%d}" and r1.also == [f"swing high {dates[34]:%Y-%m-%d}"]
    assert r2.origin == f"swing low {dates[10]:%Y-%m-%d}"  # a former low above the price acts as resistance
    assert r2.also == ["20D high"]                        # 110.5 is within 0.5 ATR -> merged, not listed twice
    assert r2.label() == f"swing low {dates[10]:%Y-%m-%d} / 20D high 110.50"  # merged price shown
    assert s1.origin == f"swing low {dates[25]:%Y-%m-%d}" and s1.also == ["20D low"]
    assert s2.origin == f"swing low {dates[15]:%Y-%m-%d}" and s2.also == ["60D low"]
    close = frame["close"].iloc[-1]
    assert r1.distance_pct == pytest.approx(108.0 / close - 1)
    assert r1.distance_atr == pytest.approx((108.0 - close) / 2.0)
    assert s1.distance_atr == pytest.approx((99.0 - close) / 2.0)
    chosen = [lv.price for lv in levels["resistance"] + levels["support"]]
    assert all(abs(a - b) > 0.5 * 2.0 for i, a in enumerate(chosen) for b in chosen[i + 1:])  # no duplicates


def test_support_resistance_mirror_image_turns_old_highs_into_support():
    frame = level_frame()
    mirror = pd.DataFrame({"close": 220 - frame["close"], "high": 220 - frame["low"], "low": 220 - frame["high"],
                           "atr": frame["atr"], "high20": 220 - frame["low20"], "low20": 220 - frame["high20"],
                           "high60": 220 - frame["low60"], "low60": 220 - frame["high60"]}, index=frame.index)
    levels = support_resistance(mirror, lookback=60, bars=3)
    assert [lv.price for lv in levels["support"]] == pytest.approx([220 - 108.0, 220 - 109.5])
    assert levels["support"][1].origin.startswith("swing high")  # a former high below the price acts as support
    assert [lv.price for lv in levels["resistance"]] == pytest.approx([220 - 99.0, 220 - 96.0])


def test_support_resistance_ignores_old_pivots_and_levels_at_the_price():
    frame = level_frame()
    levels = support_resistance(frame, lookback=20, bars=3)  # pivots before session 20 are ignored
    assert [lv.price for lv in levels["resistance"]] == [108.0, 110.5]
    assert levels["resistance"][1].origin == "20D high"
    assert levels["support"][1].origin == "60D low"  # the 60D low is still a candidate
    frame.loc[frame.index[-1], "high20"] = frame["close"].iloc[-1] + 0.1  # 0.05 ATR above the close
    assert all(lv.price != frame["high20"].iloc[-1] for lv in support_resistance(frame)["resistance"])


def test_last_sessions_cannot_be_swing_points():
    frame = level_frame()
    frame.iloc[-2, frame.columns.get_loc("high")] = 130.0  # unconfirmed: fewer than 3 sessions after it
    frame["high20"] = frame["high60"] = np.nan
    assert all(lv.price != 130.0 for lv in support_resistance(frame)["resistance"])


# ---------------------------------------------------------------- earnings event risk
LAST = pd.Timestamp("2026-09-23")  # a Wednesday


def calendar(*times: str) -> pd.DataFrame:
    return pd.DataFrame({"release_time": pd.to_datetime(list(times))})


@pytest.mark.parametrize("release,reaction,sessions,within", [
    ("2026-09-29 16:05", "2026-09-30", 5, True),   # after the close -> next session reacts
    ("2026-09-24 07:00", "2026-09-24", 1, True),   # pre-market -> same session
    ("2026-09-23 16:30", "2026-09-24", 1, True),   # after the close of the last session in the data
    ("2026-09-26 10:00", "2026-09-28", 3, True),   # Saturday -> Monday
    ("2026-10-02 09:00", "2026-10-02", 7, True),   # 7th session: still inside the window
    ("2026-10-02 16:00", "2026-10-05", 8, False),  # 8th session: outside
])
def test_next_earnings_reaction_session_and_window(release, reaction, sessions, within):
    risk = next_earnings(calendar(release, "2025-01-01 16:00"), LAST, 7)
    assert risk["status"] == "scheduled"
    assert risk["reaction_date"] == pd.Timestamp(reaction)
    assert risk["sessions_until"] == sessions
    assert risk["within_window"] is within


def test_next_earnings_never_guesses():
    assert next_earnings(None, LAST, 7)["status"] == "unknown"
    past_only = next_earnings(calendar("2026-09-23 08:00", "2026-06-01 16:00"), LAST, 7)
    assert past_only["status"] == "none" and past_only["within_window"] is False
    nearest = next_earnings(calendar("2026-12-02 16:00", "2026-10-28 16:00"), LAST, 7)
    assert nearest["release_time"] == pd.Timestamp("2026-10-28 16:00") and not nearest["within_window"]


# ---------------------------------------------------------------- historical 2-7 day context
def outcome_frame() -> pd.DataFrame:
    n = 12
    frame = pd.DataFrame(index=pd.bdate_range("2024-01-01", periods=n))
    for h in (1, 2, 3, 5, 7, 20):
        frame[f"fwd_return_{h}d"] = 0.01 * h * np.where(np.arange(n) % 2 == 0, 1, -1)
        frame[f"fwd_abnormal_{h}d"] = 0.001 * h
    for h in (3, 5, 7):
        frame[f"mfe_{h}d"] = 0.02 * h
        frame[f"mae_{h}d"] = -0.01 * h
    frame.iloc[-2:, frame.columns.get_loc("fwd_return_7d")] = np.nan  # no complete 7-day future yet
    return frame


def test_count_episodes():
    assert count_episodes(pd.Series([False, True, True, False, True, False, True, True, True])) == 3
    assert count_episodes(pd.Series([True, True])) == 1


def test_condition_outcomes_statistics_and_sample_flags():
    frame = outcome_frame()
    mask = pd.Series([True, True, True, False, True, False, True, False, False, False, True, True],
                     index=frame.index)
    result = condition_outcomes(frame, mask, "test", min_samples=20)
    assert result["n"] == 5                  # the last two days have no 7-day future and are excluded
    assert result["episodes"] == 3           # days 0-2, day 4 and day 6
    assert result["small_sample"] and not result["insufficient"]
    two, _, five, seven = result["rows"]
    # sample days 0, 1, 2, 4, 6: 2D returns +0.02, -0.02, +0.02, +0.02, +0.02
    assert two["avg"] == pytest.approx(0.012) and two["median"] == pytest.approx(0.02)
    assert two["pct_positive"] == pytest.approx(0.8)
    assert two["avg_vs_benchmark"] == pytest.approx(0.002)
    assert two["mfe_avg"] is None and two["mae_avg"] is None  # MFE / MAE only for 3, 5 and 7 days
    assert five["mfe_median"] == pytest.approx(0.10) and seven["mae_avg"] == pytest.approx(-0.07)
    tiny = condition_outcomes(frame, mask & (np.arange(12) < 2), "tiny")
    assert tiny["n"] == 2 and tiny["insufficient"]


def test_historical_context_groups_describe_today(market):
    stock, spy = market
    result = analyse(stock, spy)
    frame, groups = result["frame"], result["history"]
    today = frame.iloc[-1]
    labels = [g["label"] for g in groups]
    assert labels[-1] == "All trading days (baseline)" and groups[-1]["baseline"]
    assert f"Same swing trend and RSI6 zone: {today['trend']}, RSI6 {today['rsi_zone']}" in labels
    if today["structure"] != NO_STRUCTURE:
        assert labels[0] == f"Same structure: {today['structure']}"
    baseline = groups[-1]
    assert baseline["n"] == int(frame["fwd_return_7d"].notna().sum())
    assert baseline["rows"][2]["avg"] == pytest.approx(frame["fwd_return_5d"][frame["fwd_return_7d"].notna()].mean())
    group = next(g for g in groups if g["label"].startswith("Same swing trend"))
    same = (frame["trend"] == today["trend"]) & (frame["rsi_zone"] == today["rsi_zone"])
    assert group["n"] == int((same & frame["fwd_return_7d"].notna()).sum())


# ---------------------------------------------------------------- reports
HEADINGS = ["## Snapshot", "## 1. Short-Term Trend", "## 2. Momentum", "## 3. Volume & Participation",
            "## 4. Volatility & Swing Risk", "## 5. Relative Strength", "## 6. Price Structure",
            "## 7. 2–7 Day Historical Context", "## 8. Swing Summary"]
SENTENCE_LIMITS = {"## 1. Short-Term Trend": 4, "## 2. Momentum": 4, "## 3. Volume & Participation": 3,
                   "## 4. Volatility & Swing Risk": 4, "## 5. Relative Strength": 3}
SNAPSHOT_ROWS = ["Price", "EMA9", "EMA21", "EMA50", "EMA200", "EMA250", "RSI6", "RSI14", "MACD 12/26/9", "Volume",
                 "BOLL (20, 1.8)", "ATR(14)", "Relative Strength 5D", "Nearest Resistance", "Nearest Support",
                 "Event risk", "Swing Condition"]
RECOMMENDATION = re.compile(r"\b(buy|sell|hold)\b", re.IGNORECASE)
OBSOLETE = re.compile(r"EMA5\b|EMA10\b|EMA20\b|RSI\(7\)|RSI7|6, 13, 5|6,13,5|Swing MACD|SMA50|SMA200")


def split_sections(markdown: str) -> dict[str, str]:
    parts = re.split(r"^(## .+)$", markdown, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def sentence_count(body: str) -> int:
    paragraph = body.split("\n\n")[0]  # the prose paragraph (tables follow after a blank line)
    return len([s for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()])


@pytest.fixture(scope="module")
def report_cases(market):
    """Reports for many different days, chosen to include RSI6 extremes, heavy volume on up and down
    days, band walks / extensions, every condition and every structure state seen in the data."""
    stock, spy = market
    frame = analyse(stock, spy)["frame"]
    usable = np.arange(len(frame)) >= 260
    picks = set(range(260, len(frame), 45))
    conditions = [frame["rsi6"] >= 80, frame["rsi6"] < 30, frame["volume_ratio"] >= 2.0,
                  (frame["volume_ratio"] >= 1.5) & (frame["return_1d"] < 0)]
    conditions += [frame["swing_condition"] == c for c in CONDITIONS]
    conditions += [frame["band_context"] == c for c in (cls.UPPER_WALK, cls.UPPER_EXTENSION, cls.LOWER_WALK)]
    conditions += [frame[state] for state in STATE_ORDER]
    for condition in conditions:
        hits = np.flatnonzero(condition.to_numpy(dtype=bool) & usable)
        if hits.size:
            picks.add(int(hits[0]))
    cases = []
    for i, day in enumerate(sorted(picks)):
        last = stock.index[day]
        # every other case has earnings 3 sessions ahead (inside the 7-day window)
        release = (last + pd.offsets.BDay(3)).strftime("%Y-%m-%d 08:00") if i % 2 else "2020-01-02 16:00"
        result = analyse(stock, spy, end=day + 1, earnings=calendar(release))
        cases.append((result, build_technical_markdown(result, "USD"), build_summary_snapshot(result, "USD")))
    return cases


def test_report_cases_cover_extremes_volume_and_event_risk(report_cases):
    latest = [r["latest"] for r, _, _ in report_cases]
    assert max(row["rsi6"] for row in latest) >= 80 and min(row["rsi6"] for row in latest) < 30
    assert any(row["volume_ratio"] >= 2.0 for row in latest)
    assert any(row["volume_ratio"] >= 1.5 and row["return_1d"] < 0 for row in latest)
    assert any(r["event_risk"]["within_window"] for r, _, _ in report_cases)
    assert {r["condition"] for r, _, _ in report_cases} == CONDITIONS
    assert len({r["structure"] for r, _, _ in report_cases}) >= 12


def test_technical_report_structure_and_length_limits(report_cases):
    for result, markdown, _ in report_cases:
        assert markdown.startswith("# Swing Technical Analysis — TEST\n")
        assert "EMA 9 / 21 / 50 / 200 / 250 · RSI 6 / 14 · MACD 12 / 26 / 9 · MAVOL20 · BOLL (20, 1.8)" in markdown
        sections = split_sections(markdown)
        assert list(sections) == HEADINGS
        assert "\n\n---\n\n## 1. Short-Term Trend" in markdown
        snapshot = sections["## Snapshot"].splitlines()
        assert snapshot[0] == "| Metric | Reading | Interpretation |" and snapshot[1] == "|---|---:|---|"
        assert [line.split(" | ")[0].strip("| ") for line in snapshot[2:2 + len(SNAPSHOT_ROWS)]] == SNAPSHOT_ROWS
        for heading, limit in SENTENCE_LIMITS.items():
            assert sentence_count(sections[heading]) <= limit, (heading, sections[heading])
        summary = sections["## 8. Swing Summary"]
        assert len(summary.split()) <= 150
        for question in ("EMA9/EMA21:", "RSI6", "RSI14", "MACD 12/26/9", "MAVOL20", "BOLL(20, 1.8)", "ATR",
                         "support", "resistance"):
            assert question in summary, question
        assert summary.splitlines()[-1] == f"**Swing Technical Condition: {result['condition']}**"
        assert result["condition"] in CONDITIONS
        assert markdown.rstrip().endswith(f"**Swing Technical Condition: {result['condition']}**")
        structure = sections["## 6. Price Structure"]
        for name in ("Resistance 2", "Resistance 1", "Current", "Support 1", "Support 2", "20D high", "20D low",
                     "EMA9", "EMA21", "EMA50"):
            assert f"| {name}" in structure, name
        assert "N = " in sections["## 7. 2–7 Day Historical Context"]
        assert "MACD 12/26/9 is **" in sections["## 2. Momentum"] and "RSI6 is " in sections["## 2. Momentum"]
        assert "MAVOL20" in sections["## 3. Volume & Participation"]
        assert "BOLL (20, 1.8)" in sections["## 4. Volatility & Swing Risk"]


def test_reports_use_only_the_traders_indicators_and_never_recommend(report_cases):
    for result, markdown, snapshot in report_cases:
        for text in (markdown, snapshot):
            assert not RECOMMENDATION.search(text), RECOMMENDATION.search(text)
            assert not OBSOLETE.search(text), OBSOLETE.search(text)
            assert not re.search(r"\bnan\b", text, re.IGNORECASE)
            assert "target price" not in text.lower() and "probability" not in text.lower()
        assert "not a price target" in markdown


def test_heavy_volume_is_read_with_the_price_direction(report_cases):
    for result, markdown, _ in report_cases:
        row = result["latest"]
        volume = split_sections(markdown)["## 3. Volume & Participation"]
        if row["volume_ratio"] >= 1.5 and row["return_1d"] < 0:
            assert "selling pressure, not confirmation of strength" in volume
        if row["volume_ratio"] >= 1.5 and row["return_1d"] > 0:
            assert "buying pressure is confirmed by volume" in volume


def test_event_risk_banner_in_both_reports(report_cases):
    for result, markdown, snapshot in report_cases:
        within = result["event_risk"]["within_window"]
        assert (EVENT_RISK_BANNER in markdown) is within
        assert (EVENT_RISK_BANNER in snapshot) is within
        if not within:
            assert "No known earnings" in markdown or "No upcoming earnings date is listed" in markdown


def test_summary_snapshot_is_compact_and_shows_the_trader_setup(report_cases):
    result, _, snapshot = report_cases[0]
    lines = snapshot.splitlines()
    assert lines[0] == "## Swing Technical Snapshot"
    assert lines[-1] == "Full report: [technical_analysis.md](technical_analysis.md)"
    assert len(lines) <= 26
    for name in ("EMA9", "EMA21", "EMA50", "EMA200", "EMA250", "RSI6", "RSI14", "MACD 12/26/9", "Volume",
                 "BOLL (20, 1.8)", "ATR(14)", "Relative Strength 5D", "Nearest Resistance", "Nearest Support"):
        assert f"| {name} |" in snapshot, name
    assert "× MAVOL20" in snapshot
    assert f"| **Swing Condition** | **{result['condition']}** |" in snapshot


def test_swing_summary_stays_within_150_words_in_the_longest_case(market):
    stock, spy = market
    release = (stock.index[-1] + pd.offsets.BDay(2)).strftime("%Y-%m-%d 16:30")
    result = analyse(stock, spy, earnings=calendar(release))
    row = result["latest"].copy()
    for state in (st.RECLAIMING_EMA21, st.LOSING_EMA9, st.EMA_COMPRESSION):
        row[state] = True
    row["band_context"] = cls.UPPER_EXTENSION
    row["bb_position"] = cls.UPPER_HALF
    row["volume_level"], row["day_direction"] = cls.VERY_HIGH_PARTICIPATION, "down"
    row["volatility"], row["relative_strength"] = "Extreme", "Strongly underperforming"
    worst = {**result, "latest": row, "levels": {**result["levels"], "resistance": []}}
    text = swing_summary(worst, Money("USD"))
    assert EVENT_RISK_BANNER in text and "none above the price" in text
    assert len(text.split()) <= 150, len(text.split())


def test_small_samples_are_flagged_in_the_report(market):
    stock, spy = market
    result = analyse(stock, spy, settings={"min_history_samples": 10_000})
    markdown = build_technical_markdown(result, "USD")
    assert "small sample: treat as anecdotal" in markdown
    assert "no significance test" in markdown


def test_unknown_currency_gets_no_dollar_sign_and_json_is_plain(market):
    stock, spy = market
    result = analyse(stock, spy, benchmark=False)
    markdown = build_technical_markdown(result, None)
    assert "$" not in markdown
    assert "No benchmark data was available" in markdown
    assert "Earnings date unknown" in markdown  # no calendar -> never guessed
    data = technical_json(result)
    assert data["swing_condition"] in CONDITIONS
    assert data["parameters"] == {"ema": [9, 21, 50, 200, 250], "rsi": [6, 14], "macd": [12, 26, 9], "mavol": 20,
                                  "boll": {"period": 20, "std": 1.8}, "atr": 14}
    assert all(value is None or isinstance(value, float) for value in data["values"].values())


def test_short_history_is_reported_as_not_available(market):
    stock, spy = market
    result = analyse(stock, spy, end=40)
    markdown = build_technical_markdown(result, "USD")
    assert "Data notes" in markdown and "Only 40 trading days" in markdown and "EMA250 needs 250" in markdown
    assert "n/a" in markdown
    medium = analyse(stock, spy, end=300)
    assert any("EMA200 / EMA250 rest on 300 sessions" in note for note in medium["notes"])


# ---------------------------------------------------------------- chart
def test_technical_chart_panels_and_legends(market, monkeypatch, tmp_path):
    stock, spy = market
    result = analyse(stock, spy)
    figures = []

    def keep(fig, path):
        fig.canvas.draw()
        figures.append(fig)
        return path

    monkeypatch.setattr(technical_chart_module, "_save", keep)
    technical_chart_module.technical_chart(result, tmp_path / "technical_chart.png", days=126)
    fig = figures[0]
    price_ax, rsi_ax, macd_ax, volume_ax = fig.axes[:4]

    def legend(ax):
        return [t.get_text() for t in ax.get_legend().get_texts()]

    assert {"EMA9", "EMA21", "EMA50", "EMA200", "EMA250", "BOLL (20, 1.8)", "BOLL middle (MA20)"} <= set(legend(price_ax))
    assert legend(rsi_ax) == ["RSI6", "RSI14"]  # the primary RSI is listed first
    assert {"Histogram > 0", "Histogram < 0", "DIF (12, 26)", "DEA / signal (9)"} == set(legend(macd_ax))
    assert {"MAVOL20", "Volume (up day)", "Volume (down day)"} == set(legend(volume_ax))
    texts = [t.get_text() for t in price_ax.texts]
    assert any(t.startswith("R1 ") or t.startswith("S1 ") for t in texts)
    assert price_ax.get_xlim()[1] > 126  # room on the right for the level labels
    assert [line.get_linewidth() for line in price_ax.get_lines() if line.get_label() in ("EMA9", "EMA200")] == \
        [1.6, 1.1]  # EMA200 is drawn more subtly than EMA9
    technical_chart_module.plt.close(fig)


def test_far_away_ema200_and_ema250_do_not_squash_the_price_panel():
    n = 30
    view = pd.DataFrame({"low": np.full(n, 40.0), "high": np.full(n, 50.0), "close": np.full(n, 45.0),
                         "bb_upper": np.full(n, 49.0), "bb_lower": np.full(n, 41.0), "ema9": np.full(n, 45.0),
                         "ema21": np.full(n, 45.0), "ema50": np.full(n, 46.0), "ema200": np.full(n, 52.0),
                         "ema250": np.full(n, 90.0)})
    bottom, top = technical_chart_module._price_limits(view, [])
    assert top < 60 and top >= 52  # EMA200 (close by) is included, EMA250 (far away) is not


def test_technical_chart_is_written_even_for_short_or_close_only_data(market, tmp_path):
    stock, spy = market
    short = analyse(stock, spy, end=45)
    assert technical_chart_module.technical_chart(short, tmp_path / "short.png").stat().st_size > 10_000
    close_only = stock.copy()
    close_only["open"] = np.nan
    result = analyse(close_only, spy)
    assert technical_chart_module.technical_chart(result, tmp_path / "close.png").stat().st_size > 10_000


# ---------------------------------------------------------------- settings
@pytest.mark.parametrize("key,value", [("chart_days", 10), ("pivot_bars", 0), ("event_risk_days", True),
                                       ("min_history_samples", 2.5), ("structure_lookback_days", None)])
def test_invalid_technical_settings_are_rejected(key, value):
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    settings["technical"][key] = value
    with pytest.raises(ConfigError, match=f"technical.{key}"):
        validate_settings(settings)


def test_settings_file_contains_the_technical_defaults():
    assert load_settings()["technical"] == DEFAULT_SETTINGS["technical"]
