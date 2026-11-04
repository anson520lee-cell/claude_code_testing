"""Execution timing and restart checks, using explicitly synthetic fixtures only."""
import json
import sys
import sqlite3
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from scripts.research import etf_expansion as ex


def test_trial_grid_is_complete_and_unique():
    cfgs = ex.configurations()
    assert len(cfgs) == len(set(cfgs)) == 1728
    assert sum(c.family == "momentum" for c in cfgs) == 1296
    assert sum(c.family == "reversal" for c in cfgs) == 432


def test_close_execution_earns_only_following_returns():
    ret = np.array([[.5], [.2], [.3]])
    target = np.ones_like(ret)
    got, _, dead = ex.simulate(ret, target, np.zeros(3), np.ones(3), 1, 1., 0.)
    assert np.allclose(got, [0., .2, .3]) and not dead


def test_entry_and_final_liquidation_pay_self_financing_costs():
    cost = .001
    r, _, _ = ex.simulate(np.zeros((2, 1)), np.ones((2, 1)), np.zeros(2), np.ones(2), 1, 1., cost)
    assert np.prod(1+r) == pytest.approx((1-cost)/(1+cost))


def test_weekend_borrowing_uses_calendar_days_and_ruin_is_permanent():
    r, peak, dead = ex.simulate(np.zeros((2, 1)), np.full((2, 1), 2.),
                                np.array([.05, .05]), np.array([1., 3.]), 1, 2., 0.)
    assert r[1] == pytest.approx(-(.05+.01)*3/365) and peak >= 2 and not dead
    r, _, dead = ex.simulate(np.array([[0.], [-.6], [100.]]), np.full((3, 1), 2.),
                             np.zeros(3), np.ones(3), 1, 2., 0.)
    assert dead and np.allclose(r, [0., -1., 0.])


def test_missing_allocated_returns_are_not_replaced_with_zero():
    with pytest.raises(ValueError, match="Missing return"):
        ex.simulate(np.array([[0.], [np.nan]]), np.ones((2, 1)), np.zeros(2), np.ones(2), 1, 1., 0.)


def test_drawdown_counts_initial_capital_loss():
    idx = pd.bdate_range("2010-01-01", periods=260)
    r = np.zeros(260)
    r[0] = -.5
    assert ex.statistics(r, idx)["max_dd"] == pytest.approx(-.5)


def test_signals_do_not_see_same_day_or_future_prices(tmp_path):
    rng = np.random.default_rng(82)
    idx = pd.bdate_range("2005-01-03", periods=1050)
    cols = sorted({s for names in ex.UNIVERSES.values() for s in names})
    values = 100*np.cumprod(1+rng.normal(.0005, .01, (len(idx), len(cols))), axis=0)
    frame = pd.DataFrame(values, index=idx, columns=cols).rename_axis("date").reset_index().melt(
        id_vars="date", var_name="symbol", value_name="adj_close")
    frame["close"] = frame.adj_close
    irx = pd.DataFrame({"date":idx, "symbol":"^IRX", "close":3., "adj_close":3.})
    frame = pd.concat([frame, irx], ignore_index=True)
    raw = tmp_path/"x80/raw"
    raw.mkdir(parents=True)
    frame.to_parquet(raw/"prices.parquet", index=False)
    frame.iloc[:0].to_parquet(raw/"prices_extra.parquet", index=False)
    before = ex.Data(tmp_path)
    cfg = ex.Config("global", "momentum", 21, 200, 3, .3, 2., 5)
    cutoff = pd.Timestamp("2008-06-02")
    a = before.signals(cfg).copy()
    m = (frame.symbol == "SPY") & (frame.date >= cutoff)
    frame.loc[m, "adj_close"] *= 2
    frame.to_parquet(raw/"prices.parquet", index=False)
    after = ex.Data(tmp_path)
    b = after.signals(cfg)
    assert np.array_equal(a[before.index <= cutoff], b[after.index <= cutoff])


def test_real_checkpoint_skips_completed_trials_and_freezes_before_later_evaluation(tmp_path, monkeypatch):
    cfgs = ex.configurations()[:2]
    called = []
    idx = pd.bdate_range("2008-01-02", "2026-11-03")
    cols = sorted({s for names in ex.UNIVERSES.values() for s in names})

    class FixtureData:
        def __init__(self, root):
            self.prices = pd.DataFrame(100., index=idx, columns=cols)
            self.rates = pd.Series(.02, index=idx)
            self.index = idx
            self.training = idx < "2025-01-01"
            self.cash = np.zeros(len(idx))
        def run(self, cfg, training=True):
            called.append((cfg, training))
            if not training:
                assert (tmp_path/"out/frozen_winner.json").exists()
            length = self.training.sum() if training else len(idx)
            return np.full(length, .0004), 1., False

    monkeypatch.setattr(ex, "Data", FixtureData)
    monkeypatch.setattr(ex, "configurations", lambda: cfgs)
    cmd = ["etf-expansion", str(tmp_path), str(tmp_path/"out"), "--max-new", "1"]
    monkeypatch.setattr(sys, "argv", cmd)
    ex.main()
    assert len(called) == 1
    assert not json.loads((tmp_path/"out/progress.json").read_text())["complete"]
    ex.main()
    assert [c for c, training in called if training] == cfgs
    assert len([1 for _, training in called if not training]) == 1
    assert json.loads((tmp_path/"out/progress.json").read_text())["complete"]
    with sqlite3.connect(tmp_path/"out/checkpoint.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM trials").fetchone()[0] == 2
