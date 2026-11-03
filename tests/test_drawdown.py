"""Drawdown-study account layer (src/backtest/drawdown.py): risk caps, regime multipliers, causal context."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import drawdown as dd

CAL = pd.bdate_range("2020-01-01", periods=60)


def _trades(setups, finals, days=3, cal0=5):
    n = len(setups)
    tr = pd.DataFrame({"setup": setups, "sym": [f"S{i}" for i in range(n)], "sector": ["XLK"] * n, "date": CAL[cal0 - 1],
                       "cal0": cal0, "days": days, "notional_r": 5.0})
    pnl = np.zeros((n, 21))
    for i, f in enumerate(finals):
        pnl[i, :] = f                      # final result in R from day 0 (enough for accounting tests)
    cap = np.full((n, 21), 5.0)
    return tr, pnl, cap


def _run(tr, pnl, cap, v, ctx=None):
    return dd.run_account(tr, pnl, cap, CAL, v, ctx or {}, seed=0)


def test_single_winner_is_one_percent_risk():
    tr, pnl, cap = _trades(["breakout"], [2.0])
    r = _run(tr, pnl, cap, dd.Variant("b"))
    assert r["taken"] == 1
    assert abs(r["equity"].iloc[-1] - 1.02) < 1e-9


def test_heat_cap_limits_concurrent_open_risk():
    tr, pnl, cap = _trades(["breakout"] * 6, [0.0] * 6)
    assert _run(tr, pnl, cap, dd.Variant("b"))["taken"] == 6
    assert _run(tr, pnl, cap, dd.Variant("h", heat=0.03))["taken"] == 3
    assert _run(tr, pnl, cap, dd.Variant("s", slots=2))["taken"] == 2


def test_setup_slots_and_priority():
    tr, pnl, cap = _trades(["breakout", "breakout", "crash_rebound"], [0.0, 0.0, 0.0])
    r = _run(tr, pnl, cap, dd.Variant("b", slots=1, priority=True))
    assert r["taken"] == 1
    # with priority the single slot goes to the crash rebound: its trade is the one that opened
    r2 = _run(tr, pnl, cap, dd.Variant("c", setup_slots={"breakout": 0}))
    assert r2["taken"] == 1


def test_zero_regime_multiplier_pauses_new_entries():
    tr, pnl, cap = _trades(["breakout"] * 3, [1.0] * 3)
    ctx = {"spy_vs200": np.full(len(CAL), -0.1)}
    r = _run(tr, pnl, cap, dd.Variant("p", regime=("spy_ma", "spy_vs200", 0.0)), ctx)
    assert r["taken"] == 0 and r["equity"].iloc[-1] == 1.0
    half = _run(tr, pnl, cap, dd.Variant("h", regime=("spy_ma", "spy_vs200", 0.5)), ctx)
    assert abs(half["equity"].iloc[-1] - (1 + 3 * 0.005)) < 1e-9


def test_regime_filter_can_spare_crash_rebound():
    tr, pnl, cap = _trades(["breakout", "crash_rebound"], [1.0, 1.0])
    ctx = {"spy_vs200": np.full(len(CAL), -0.1)}
    v = dd.Variant("p", regime=("spy_ma", "spy_vs200", 0.0), regime_setups=("breakout",))
    assert _run(tr, pnl, cap, v, ctx)["taken"] == 1


def test_adaptive_heat_tightens_after_a_drawdown():
    # first trade loses 5R (-5% of the account), then 6 trades arrive later; the cap drops from 8% to 3%
    tr, pnl, cap = _trades(["breakout"] * 7, [0.0] * 7, days=3, cal0=5)
    tr.loc[0, "cal0"] = 5
    pnl[0, :] = -5.0
    tr.loc[1:, "cal0"] = 10
    v = dd.Variant("a", heat_adapt=(0.08, 0.03, 0.04, False))
    r = _run(tr, pnl, cap, v, {"spy_vs200": np.zeros(len(CAL))})
    assert r["taken"] == 1 + 3                       # 3% of the (lower) equity fits 3 more, not 6


def test_context_is_known_at_the_previous_close():
    px = pd.DataFrame({"adj_close": np.linspace(100, 160, len(CAL)), "close": np.linspace(100, 160, len(CAL))}, index=CAL)
    prices = {"SPY": px, "^VIX": px.assign(close=20.0), "AAA": px}
    ctx = dd.build_context(prices, CAL, ["AAA"])
    assert np.isnan(ctx["vix"][0]) and ctx["vix"][1] == 20.0
    assert np.isnan(ctx["spy_vol20"][:21]).all()


def test_perf_matches_manual_numbers():
    eq = pd.Series([1.0, 1.1, 0.88, 0.99] * 20, index=pd.bdate_range("2020-01-01", periods=80))
    p = dd.perf(eq)
    assert abs(p["max_dd"] - (0.88 / 1.1 - 1)) < 1e-12
