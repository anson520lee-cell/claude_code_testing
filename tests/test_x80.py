"""80%-a-year search (2026-11-04): leverage arithmetic, account mechanics and the funding-carry sign."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.x80.common import kelly, levered, required_sharpe, stats
from src.backtest.x80.crypto_carry import LEG_COST, carry_returns
from src.backtest.x80.crypto_trend import run_exposure, signals
from src.backtest.x80.equity_lev import tom_mask


def test_required_sharpe_matches_kelly_growth():
    s = required_sharpe(0.8, 0.02, 1.0)
    assert math.exp(0.02 + s * s / 2) - 1 == pytest.approx(0.8, rel=1e-9)
    assert required_sharpe(0.8, 0.02, 0.5) > s                      # half Kelly needs a better strategy
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-01", periods=5000)
    r = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    k = kelly(r, pd.Series(0.0, index=idx))
    assert k["kelly_leverage"] == pytest.approx(r.mean() / r.var(), rel=1e-6)


def test_levered_and_stats():
    idx = pd.bdate_range("2020-01-01", periods=504)
    r = pd.Series(0.001, index=idx)
    cash = pd.Series(0.0001, index=idx)
    lv = levered(r, cash, 3.0)
    assert lv.iloc[0] == pytest.approx(3 * 0.001 - 2 * 0.0001)
    st = stats(r, cash)
    assert st["cagr"] == pytest.approx(1.001 ** 252 - 1, rel=1e-6) and st["max_dd"] == 0


def test_run_exposure_lags_signal_and_ruins():
    idx = pd.date_range("2021-01-01", periods=6, freq="D")
    r = pd.Series([0.0, 0.10, -0.50, 0.20, 0.10, 0.10], index=idx)
    cash = pd.Series(0.0, index=idx)
    expo = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], index=idx)       # decided at each close, earns the next day
    ret = run_exposure(r, expo, cash, cost=0.0)
    assert ret.iloc[1] == pytest.approx(0.10) and ret.iloc[2] == pytest.approx(-0.50) and ret.iloc[3] == 0
    lev3 = run_exposure(r, pd.Series(3.0, index=idx), cash, cost=0.0, fin_spread=0.0)
    assert lev3.iloc[2] == -1.0 and (lev3.iloc[3:] == 0).all()          # a -150% day ends the account


def test_signals_have_no_lookahead():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    close = pd.Series(np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.03, 400))), index=idx)
    a = signals(close)
    b = signals(close.iloc[:300])
    for k in a:
        pd.testing.assert_series_equal(a[k].iloc[:300], b[k], check_names=False)   # the future never changes the past


def test_tom_mask_marks_last_and_first_three_days():
    idx = pd.bdate_range("2024-01-01", "2024-02-29")
    m = tom_mask(idx)
    assert m[pd.Timestamp("2024-01-31")] == 1 and m[pd.Timestamp("2024-02-01")] == 1 and m[pd.Timestamp("2024-02-05")] == 1
    assert m[pd.Timestamp("2024-02-06")] == 0 and m[pd.Timestamp("2024-01-15")] == 0


def test_carry_short_perp_collects_positive_funding():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    fund = pd.DataFrame({"X": [0.001, 0.001, -0.002]}, index=idx)
    prem = pd.DataFrame({"X": [0.0, 0.0, 0.0]}, index=idx)
    w = pd.DataFrame({"X": [1.0, 1.0, 1.0]}, index=idx)
    out = carry_returns(fund, prem, w)
    assert out.iloc[0] == pytest.approx(0.001 - 2 * LEG_COST)            # entry pays both legs
    assert out.iloc[1] == pytest.approx(0.001) and out.iloc[2] == pytest.approx(-0.002)
    prem2 = pd.DataFrame({"X": [0.0, 0.002, 0.002]}, index=idx)           # perp richening hurts the short perp
    assert carry_returns(fund, prem2, w).iloc[1] == pytest.approx(0.001 - 0.002)
