"""Leverage scoring study (2026-11-04): no look-ahead in the score rules, the levering arithmetic, intraweek
wipe-outs and the per-trade account with risk multipliers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.levscore import book as bk
from src.backtest.levscore import rules as rl
from src.backtest.levscore import trades as tr
from src.backtest.x80 import meta


def _weeks(n: int, start: str = "2010-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="W-FRI")


def test_expanding_rank_uses_only_the_past():
    idx = _weeks(300)
    x = pd.Series(np.random.default_rng(0).standard_normal(300), index=idx)
    p = rl.expanding_rank(x, min_hist=104)
    assert p.iloc[:103].isna().all() and p.iloc[103:].notna().all()
    y = x.copy()
    y.iloc[200:] += 100.0                                                # changing the future must not move earlier ranks
    assert rl.expanding_rank(y, 104).iloc[:200].equals(p.iloc[:200])
    assert p.iloc[150] == pytest.approx((np.sum(x.iloc[:151] < x.iloc[150]) + 0.5) / 151)


def test_tiers_and_single_rule_lag():
    idx = _weeks(5)
    pct = pd.Series([np.nan, 0.1, 0.5, 0.9, 1 / 3], index=idx)
    assert rl.tiers(pct).tolist() == [1.0, 0.5, 1.0, 1.5, 0.5]
    x = pd.Series(np.arange(300, dtype=float), index=_weeks(300))         # rising feature, direction +1
    m = rl.single(x, +1)
    raw = rl.tiers(rl.expanding_rank(x))
    assert m.iloc[1:].tolist() == raw.iloc[:-1].tolist()                 # the multiplier for week t uses week t-1
    assert m.iloc[0] == 1.0


def test_walk_forward_model_never_trains_on_unknown_targets():
    idx = _weeks(800, "2004-01-02")
    rng = np.random.default_rng(1)
    F = pd.DataFrame({"a": rng.standard_normal(800), "b": rng.standard_normal(800)}, index=idx)
    y = pd.Series(rng.standard_normal(800), index=idx)
    p1 = rl.walk_forward_model(F, y, 4, 2012, "ridge", start="2004-01-01")
    y2 = y.copy()
    y2[y2.index >= "2015-01-01"] = 50.0                                  # targets after 2015 cannot affect 2015's predictions
    p2 = rl.walk_forward_model(F, y2, 4, 2012, "ridge", start="2004-01-01")
    yr15 = (idx >= "2015-01-01") & (idx < "2016-01-01")
    assert np.allclose(p1[yr15], p2[yr15])
    yr16 = (idx >= "2016-01-01") & (idx < "2017-01-01")
    assert not np.allclose(p1[yr16], p2[yr16])


def test_lever_path_matches_meta_lever_and_wipes_out_intraweek():
    idx = _weeks(200)
    rng = np.random.default_rng(2)
    base = pd.Series(rng.normal(0.002, 0.01, 200), index=idx)
    fin = pd.DataFrame({"funded": 0.8, "spread": 0.02, "unfunded": 0.2, "turn": 0.05}, index=idx)
    cash = pd.Series(0.0005, index=idx)
    ref, _ = meta.lever(base, fin, cash, "vol", 0.3, vol_win=0)
    mine = bk.lever_path(base, fin, cash, bk.vol_leverage(base, 0.3))
    assert np.allclose(ref.reindex(mine.index), mine)
    trough = pd.Series(0.0, index=idx)
    trough.iloc[120] = -0.2                                              # 5x leverage, -20% inside the week -> dead
    lv = pd.Series(5.0, index=idx)
    r = bk.lever_path(base, fin, cash, lv, trough=trough)
    assert r.iloc[120] == -1.0 and (r.iloc[121:] == 0).all()
    r4 = bk.lever_path(base, fin, cash, pd.Series(4.0, index=idx), trough=trough)
    assert r4.iloc[120] > -1.0                                           # 4x survives a -20% trough


def test_intraweek_trough_buy_and_hold():
    days = pd.date_range("2020-01-04", "2020-01-17", freq="D")            # two W-FRI weeks: Jan 10, Jan 17
    d = pd.DataFrame({"a": 0.0, "b": 0.0}, index=days)
    d.loc["2020-01-06", "a"] = -0.10
    d.loc["2020-01-07", "a"] = 0.10
    wk = pd.DatetimeIndex(["2020-01-10", "2020-01-17"])
    W = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]}, index=wk)
    w_ex = pd.DataFrame({"a": [-0.01, 0.0], "b": [0.0, 0.0]}, index=wk)
    t = bk.intraweek_trough(d, W, w_ex)
    assert t.loc["2020-01-10", "trough"] == pytest.approx(-0.05)
    assert t.loc["2020-01-10", "week_end"] == pytest.approx(0.5 * (0.9 * 1.1 - 1))
    assert t.loc["2020-01-17", "trough"] == 0.0


def test_expanding_pct_by_date_excludes_same_day():
    t = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 2), "v": [1.0, 2.0, 3.0, 2.5, 0.5]})
    p = tr.expanding_pct_by_date(t, "v", min_n=3)
    assert p.iloc[:3].isna().all()
    assert p.iloc[3] == pytest.approx(2 / 3) and p.iloc[4] == 0.0


def test_trade_account_with_unit_multipliers_matches_live_mtm():
    from src.backtest.x80.live_mtm import simulate_mtm

    cal = pd.bdate_range("2021-01-01", periods=60)
    rng = np.random.default_rng(3)
    adj = pd.DataFrame(np.cumprod(1 + rng.normal(0, 0.02, (60, 3)), axis=0), index=cal, columns=["A", "B", "C"])
    trades = pd.DataFrame({"symbol": ["A", "B", "C", "A"], "date": cal[[2, 2, 10, 30]], "setup": ["crash_rebound", "squeeze", "squeeze", "crash_rebound"],
                           "r": [1.0, -1.0, 0.5, 2.0], "ret": [0.20, -0.16, 0.08, 0.32], "days": [5, 3, 8, 4]})   # sizes 5-6% (cap 25% never binds)
    a = adj.to_numpy()
    cols = {c: j for j, c in enumerate(adj.columns)}
    ref = simulate_mtm(trades, cal, adj, seed=0)
    mine = tr.simulate(trades, cal, a, cols, seed=0)
    assert np.allclose(ref.to_numpy(), mine.to_numpy())
    trades["m"] = 2.0
    big = tr.simulate(trades, cal, a, cols, seed=0, mult_col="m")
    assert (big.iloc[-1] - 1) == pytest.approx(2 * (mine.iloc[-1] - 1), rel=0.05)


def test_ladder_bootstrap_counts_drawdowns_and_ruin():
    from src.backtest.levscore.ladder import bootstrap

    flat = bootstrap(np.full(200, 0.002), years=2, n=50, block=13)
    assert flat["p_dd30"] == 0 and flat["p_ruin"] == 0 and flat["cagr_median"] == pytest.approx(1.002 ** 52 - 1, rel=1e-9)
    R = np.full(200, 0.002)
    R[::10] = -1.0                                                       # a wipe-out week every 10 weeks: every 13-week block holds one
    dead = bootstrap(R, years=2, n=50, block=13)
    assert dead["p_ruin"] == 1.0 and dead["p_dd80"] == 1.0 and dead["cagr_median"] == -1.0


def test_weekly_levered_matches_lever_path_without_ruin():
    from src.backtest.levscore.ladder import weekly_levered

    idx = _weeks(60)
    rng = np.random.default_rng(4)
    base = pd.Series(rng.normal(0.002, 0.02, 60), index=idx)
    fin = pd.DataFrame({"funded": 0.8, "spread": 0.01, "unfunded": 0.2, "turn": 0.05}, index=idx)
    cash = pd.Series(0.0005, index=idx)
    trough = pd.Series(0.0, index=idx)
    a = weekly_levered(base, fin, cash, 3.0, trough)
    b = bk.lever_path(base, fin, cash, pd.Series(3.0, index=idx), trough=trough)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_score_column_needs_half_the_inputs_and_uses_earlier_days():
    days = pd.bdate_range("2020-01-01", periods=400)
    t = pd.DataFrame({"date": np.repeat(days, 2), "a": np.arange(800, dtype=float), "b": np.nan})
    sc = tr.score_column(t, {"a": 1, "b": -1}, min_n=50)
    assert sc.iloc[:50].isna().all() and sc.iloc[60:].notna().all()    # 'a' alone is half of two inputs
    assert sc.iloc[100] == pytest.approx(tr.expanding_z_by_date(t, "a", 50).iloc[100])
