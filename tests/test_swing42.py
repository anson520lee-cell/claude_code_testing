import numpy as np
import pandas as pd

from src.backtest.swing42 import core


def _panel(close):
    c = np.asarray(close, float)[:, None]
    d = pd.bdate_range("2010-01-01", periods=len(c))
    return core.Panel(d, ["X"], c.copy(), c * 1.01, c * 0.99, c.copy(), np.ones_like(c), np.ones_like(c, bool))


def test_hold_exit_enters_next_open_and_charges_costs():
    p = _panel([10, 10, 11, 12, 13])
    r = core.exit_hold(p, 2, 1)
    # signal day 0 -> buy open of day 1 (10) -> sell close of day 2 (11)
    assert abs(r[0, 0] - (11 / 10 - 1 - core.COST_LONG)) < 1e-12


def test_fresh_rearms_after_cooldown():
    s = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1], bool)[:, None]
    f = core.fresh(s, 3)
    assert f[:, 0].tolist() == [True, False, False, False, False, False, False, False, True]


def test_evaluate_edge_is_vs_all_eligible_days():
    p = _panel(np.linspace(10, 20, 50))
    out = np.tile(np.arange(50, dtype=float)[:, None], (1, 1)) / 100
    sig = np.zeros((50, 1), bool)
    sig[40] = True
    p.dates = pd.bdate_range("2016-01-01", periods=50)
    r = core.evaluate(p, sig, out)
    assert abs(r["OOS"]["edge"] - (0.40 - out.mean())) < 1e-12
