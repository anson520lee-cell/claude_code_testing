import numpy as np
import pandas as pd

from src.backtest import adaptive_stop as ad
from src.backtest import holding_study as hs


def _paths(n=300, hz=ad.H, seed=0):
    rng = np.random.default_rng(seed)
    a = np.full(n, 2.0)
    ref = np.full(n, 100.0)
    r = rng.normal(0, 1.5, (n, hz)).cumsum(axis=1)
    C = 100 + r
    O = np.c_[100 + rng.normal(0, 0.5, n), C[:, :-1]] + rng.normal(0, 0.3, (n, hz))
    Hh = np.maximum(O, C) + rng.uniform(0, 1.5, (n, hz))
    L = np.minimum(O, C) - rng.uniform(0, 1.5, (n, hz))
    return {"O": O, "H": Hh, "L": L, "C": C, "ref": ref, "atr": a}


def test_live_rules_match_the_holding_study_simulator():
    for kind in ("crash_rebound", "breakout", "squeeze"):
        T = _paths(seed=1)
        T["cond"] = np.full(len(T["ref"]), kind == "squeeze")
        T["ref"] = T["ref"] + (0.3 if kind == "squeeze" else 0.0)
        mine = ad.simulate(T, ad.LIVE[kind], 0.001)
        ref = hs.live_paths(T["O"], T["H"], T["L"], T["C"], T["ref"], T["atr"], kind, 0.001)
        assert np.array_equal(mine["ok"], ref["ok"])
        assert np.allclose(mine["eq"], np.where(ref["ok"], ref["ret"] * ref["size"], 0.0))


def test_two_phase_switch_tightens_only_after_the_trigger():
    T = _paths(seed=2)
    T["cond"] = np.zeros(len(T["ref"]), dtype=bool)
    wide = ad.simulate(T, dict(S0=5.0, T1=0.0, trig="never", T2=0.0, be=False, tgt=0.0), 0.0)
    two = ad.simulate(T, dict(S0=5.0, T1=0.0, trig="close>entry", T2=2.0, be=False, tgt=0.0), 0.0)
    never_up = (T["C"] <= wide["entry"][:, None]).all(axis=1)       # never closed above the entry -> no switch
    assert np.allclose(wide["eq"][never_up], two["eq"][never_up])
    assert (two["switch_day"][never_up] == -1).all()


def test_per_day_multiple_matrix_reproduces_a_fixed_trail():
    T = _paths(seed=3)
    T["cond"] = np.zeros(len(T["ref"]), dtype=bool)
    fixed = ad.simulate(T, dict(S0=1.5, T1=3.0, trig="never", T2=0.0, be=False, tgt=0.0), 0.001)
    mult = ad.simulate(T, {"tgt": 0.0}, 0.001, mult=np.full(T["O"].shape, 3.0), S0_for_mult=1.5)
    assert np.allclose(fixed["eq"], mult["eq"])


def test_continuation_values_equal_the_live_result_while_the_live_trade_is_open():
    T = _paths(seed=4)
    T["cond"] = np.zeros(len(T["ref"]), dtype=bool)
    live = ad.simulate(T, ad.LIVE["crash_rebound"], 0.001)
    V, alive = ad.cont_values(T, 1.5, 0.0, (0.0, 3.0, 5.0), 0.001)
    last = live["days"] - 1
    for i in np.flatnonzero(live["ok"]):
        for k in range(ad.H - 1):
            if alive[i, k] and k < last[i]:
                assert abs(V[i, k, 1] - live["eq"][i]) < 1e-12


def test_chip_and_earnings_features_are_causal():
    rng = np.random.default_rng(5)
    n = 400
    c = 100 + rng.normal(0, 1, n).cumsum()
    h, lo, v, a = c + 1, c - 1, rng.uniform(1e5, 2e5, n), np.full(n, 2.0)
    full = ad.chip_features(h, lo, c, v, a)
    cut = ad.chip_features(h[:300], lo[:300], c[:300], v[:300], a[:300])
    for k in full:
        assert np.allclose(full[k][:300], cut[k], equal_nan=True)
    idx = pd.bdate_range("2020-01-01", periods=120)
    e = ad.earnings_features(idx, [idx[10], idx[70]])
    assert e["earn_since"][10] == 0 and e["earn_since"][11] == 1
    assert e["earn_to_cal"][9] == (idx[10] - idx[9]).days


def test_account_with_no_trades_stays_flat():
    cal = pd.bdate_range("2015-01-01", periods=300)
    acct = ad.Account(cal)
    T = {"date": np.array([cal[0]], dtype="datetime64[ns]")}
    res = {"ok": np.array([False]), "days": np.array([1]), "size": np.array([0.1]), "path_eq": np.zeros((1, ad.H)),
           "path_cap": np.zeros((1, ad.H))}
    for s in ad.SETUPS:
        acct.set_setup(s, T, res)
    p = acct.run("2015-01-01", "2016-01-01", seeds=2)
    assert p["cagr"] == 0.0 and p["max_dd"] == 0.0
