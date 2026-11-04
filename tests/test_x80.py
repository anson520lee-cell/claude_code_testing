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


# --- second round (2026-11-04): crypto factors, funding alignment, short vol, combinations ---------------------------

from src.backtest.x80.common import deflated_sharpe, funding_daily
from src.backtest.x80.crypto_factors import (
    band_weights,
    ls_weights,
    simulate,
    spot_name,
)
from src.backtest.x80.crypto_vrp import short_var_ladder
from src.backtest.x80.meta import (
    allocate,
    dd_brake,
    equal_risk_weights,
    lever,
)


def test_funding_print_at_midnight_belongs_to_the_previous_day():
    f = pd.DataFrame({"symbol": "X", "rate": [0.001, 0.002, 0.003, 0.004],
                      "time": pd.to_datetime(["2024-01-01 08:00:00.003", "2024-01-01 16:00:00.001", "2024-01-02 00:00:00.004",
                                              "2024-01-02 08:00:00.002"])})
    d = funding_daily(f)["X"]
    assert d[pd.Timestamp("2024-01-01")] == pytest.approx(0.006)        # 08:00 + 16:00 + the next midnight
    assert d[pd.Timestamp("2024-01-02")] == pytest.approx(0.004)


def test_perp_names_map_to_spot():
    assert spot_name("1000PEPEUSDT") == "PEPEUSDT" and spot_name("1000000MOGUSDT") == "MOGUSDT"
    assert spot_name("1MBABYDOGEUSDT") == "BABYDOGEUSDT" and spot_name("1INCHUSDT") == "1INCHUSDT" and spot_name("BTCUSDT") == "BTCUSDT"


def test_ls_weights_neutral():
    rng = np.random.default_rng(1)
    score = rng.normal(size=40)
    score[:5] = np.nan
    w = ls_weights(score, None)
    assert w.sum() == pytest.approx(0) and np.abs(w).sum() == pytest.approx(2) and (w[:5] == 0).all()
    beta = rng.uniform(0.5, 2.0, 40)
    wb = ls_weights(score, beta)
    hi, lo = wb > 0, wb < 0
    assert np.abs(wb).sum() == pytest.approx(2)
    assert (wb[hi].sum() * beta[hi].mean() + wb[lo].sum() * beta[lo].mean()) == pytest.approx(0, abs=1e-9)   # leg betas offset
    wbd, held_l, held_s = band_weights(score, np.zeros(40, bool), np.zeros(40, bool), 0.2, 0.4)
    assert wbd.sum() == pytest.approx(0) and held_l.sum() == held_s.sum()


def test_simulate_short_drifts_and_collects_funding():
    r = np.array([[0.10], [0.0], [0.0]])
    fund = np.array([[0.001], [0.001], [0.001]])
    out = simulate(np.array([[-1.0]]), np.array([0]), r, fund, cost=0.0)
    assert out[0] == pytest.approx(-0.10 + 0.001)                        # loses the rally, receives funding
    assert out[1] == pytest.approx(1.1 * 0.001 / 0.901)                  # the short grew to 1.1x of the start notional
    hold = simulate(np.array([[-1.0], [np.nan]]), np.array([0, 1]), r, np.zeros_like(r), cost=0.01)
    assert hold[1] == 0.0                                                 # a NaN row keeps the positions: no trading cost


def test_deflated_sharpe_penalises_many_trials():
    a = deflated_sharpe(0.1, None, 1, 1000)
    b = deflated_sharpe(0.1, None, 100, 1000)
    assert a > 0.99 and b < a and deflated_sharpe(0.0, None, 100, 1000) < 0.05


def test_short_variance_earns_when_nothing_moves():
    idx = pd.date_range("2022-01-01", periods=60, freq="D")
    d = pd.DataFrame({"iv": 0.6, "px": 100.0}, index=idx)
    ret = short_var_ladder(d, 0.3, 0.0, "always")
    assert (ret.iloc[2:] > 0).all()                                       # zero realised variance: every tranche gains
    spiky = d.copy()
    spiky.loc[idx[20], "px"] = 70.0                                       # a 30% crash day costs money
    assert short_var_ladder(spiky, 0.3, 0.0, "always").iloc[20] < 0


def test_inverse_vol_allocation():
    rng = np.random.default_rng(3)
    x = pd.DataFrame({"a": rng.normal(0, 0.01, 200), "b": rng.normal(0, 0.02, 200)})
    w = allocate(x, "inverse vol")
    assert w.sum() == pytest.approx(1) and w["a"] / w["b"] == pytest.approx(x["b"].std() / x["a"].std())
    neg = allocate(x - 0.05, "inverse vol, positive only")
    assert neg.sum() == 0                                                 # nothing with a positive mean: stay in T-bills


def test_equal_risk_survives_a_hedging_sleeve():
    S = np.array([[0.04, -0.012, 0.0], [-0.012, 0.01, 0.002], [0.0, 0.002, 0.09]])  # sleeve 2 hedges sleeve 1
    w = equal_risk_weights(S)
    rc = w * (S @ w)
    assert w.min() > 0.05 and w.sum() == pytest.approx(1)
    assert rc / rc.sum() == pytest.approx(np.ones(3) / 3, abs=1e-6)


def test_lever_whole_history_window_and_dd_brake():
    idx = pd.date_range("2010-01-01", periods=300, freq="W-FRI")
    rng = np.random.default_rng(5)
    base = pd.Series(rng.normal(0.002, 0.015, 300), index=idx)
    fin = pd.DataFrame({"funded": 1.0, "spread": 0.0, "unfunded": 0.0}, index=idx)
    cash = pd.Series(0.0, index=idx)
    r, lev = lever(base, fin, cash, "vol", 0.3, vol_win=0)
    want = 0.3 / (base.expanding(min_periods=52).std().shift(1) * np.sqrt(52))
    assert lev.iloc[100] == pytest.approx(want.iloc[100]) and lev.iloc[:52].isna().all()
    braked = dd_brake(base, fin, cash, lev, 0.3, "2011-01-01", "2030-01-01")
    eq = (1 + braked).cumprod()
    assert (eq / eq.cummax() - 1).min() > -0.3                             # weekly steps keep it inside the floor here
    plain = r[braked.index]
    assert braked.abs().sum() <= plain.abs().sum() + 1e-12               # the brake only ever cuts exposure
