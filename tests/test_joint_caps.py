"""Regression cases for joint caps, drift, explicit cash, and closing costs."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.optimize import constraints as co
from src.backtest.optimize import engine as en


def test_reported_counterexample_is_explicitly_infeasible():
    with pytest.raises(co.AllocationBlocked, match="unique") as err:
        co.joint_caps(np.full(4, .25), np.eye(4), np.array([1, 1, 1, 0], bool), .25, .35)
    assert err.value.code == 1


def test_feasible_joint_solution_respects_both_limits():
    # Legacy crypto scaling concentrates the two non-crypto sleeves above 25%.
    S = np.diag([25., 16., 9., 4., 1., 1.])
    mask = np.array([1, 1, 1, 1, 0, 0], bool)
    w = co.joint_caps(np.ones(6) / 6, S, mask, .25, .90)
    assert co.within_caps(w, S, mask, .25, .90)
    assert w.sum() == pytest.approx(1)
    assert w.max() <= .25 + 1e-10


def test_cap_does_not_silently_relax_or_get_stuck_on_zero_scores():
    with pytest.raises(co.AllocationBlocked):
        co.cap_weights(np.ones(3) / 3, .25)
    w = co.cap_weights(np.array([1., 0., 0., 0., 0.]), .25)
    assert w.max() <= .25 + 1e-12 and w.sum() == pytest.approx(1)
    with pytest.raises(co.AllocationBlocked, match="all eligible"):
        co.joint_caps(np.ones(4) / 4, np.eye(4), np.ones(4, bool), 1., .5)


def test_numerically_unresolved_does_not_claim_proven_infeasibility():
    # Five high-risk crypto sleeves, one capped non-crypto; no solver output qualifies.
    with pytest.raises(co.AllocationBlocked) as err:
        co.joint_caps(np.ones(6) / 6, np.diag([1e4]*5 + [1.]),
                      np.array([1]*5 + [0], bool), .25, .1)
    assert err.value.code == 2


def make_pool(crypto, seed=773):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-04", periods=230, freq="W-FRI")
    n = len(crypto)
    x = pd.DataFrame(rng.normal(.002, .02, (len(idx), n)), index=idx, columns=[str(i) for i in range(n)])
    # A sustained winner makes monthly and band weights drift through the cap.
    x.iloc[80:110, 0] = .12
    cash = pd.Series(.0004, index=idx)
    meta = pd.DataFrame({"family": ["macro"]*n, "crypto": crypto,
                         "unfunded": [False]*n, "spread": [.01]*n}, index=x.columns)
    return en.Pool(x, cash, meta, {"all": list(x.columns)})


@pytest.mark.parametrize("rebal", ["w", "m", "band"])
def test_actual_held_weights_obey_caps_after_drift(rebal):
    pool = make_pool([True, True, False, False, False, False])
    cfg = en.Config(universe="all", method="ew", look=52, cap=.20, crypto_cap=.35, rebal=rebal)
    book = en.run_book(pool, cfg, keep_weights=True)
    assert book["invested"].any()
    assert book["weights"].max() <= .20 + 1e-10
    assert np.nanmax(book["crypto_share"]) <= .35 + 1e-8
    for i in np.flatnonzero(book["invested"]):
        win = pool.X[max(0, i-cfg.look):i]
        S = en.covariance(win, cfg.cov, cfg.look)
        w = book["weights"][i]
        assert co.crypto_share_of(w, S, pool.is_c) <= .35 + 1e-8


def test_impossible_allocation_sits_in_labelled_cash():
    pool = make_pool([False, False, False])
    book = en.run_book(pool, en.Config(universe="all", cap=.25), keep_weights=True)
    assert not book["invested"].any()
    assert (book["allocation_status"][26:] == 1).all()
    assert (book["weights"] == 0).all()
    ret, lev = en.lever(book, pool.cash, .8)
    assert np.allclose(ret[26:], pool.cash[26:]) and (lev[26:] == 0).all()


def test_switch_to_cash_charges_liquidation_turnover():
    pool = make_pool([False, False, False])
    book = en.run_book(pool, en.Config(universe="all"))
    # Isolate one causal move to cash after a long invested history.
    j = 160
    book["invested"][j:] = False
    book["base"][j:] = 0.
    r, lev = en.lever(book, pool.cash, .3)
    carried = lev[j-1] * (1+book["base"][j-1]) / (1+r[j-1])
    assert carried > 0
    assert r[j] == pytest.approx(pool.cash[j] - en.COST*carried)
    assert r[j+1] == pytest.approx(pool.cash[j+1])


def test_inactive_limits_preserve_weights_exactly():
    w = np.array([.7, .2, .1])
    assert np.array_equal(co.joint_caps(w, np.eye(3), np.array([1, 0, 0], bool), 1., 1.), w)


def test_disabled_crypto_limit_preserves_a_hedged_baseline():
    S = np.array([[4., -1.8], [-1.8, 1.]])
    w = np.array([.5, .5])
    mask = np.array([True, False])
    assert co.crypto_share_of(w, S, mask) > 1.
    assert np.array_equal(co.joint_caps(w, S, mask, 1., 1.), w)


def test_constraint_code_is_part_of_checkpoint_fingerprint(tmp_path, monkeypatch):
    from pathlib import Path
    from src.backtest.optimize import checkpoint
    pool = make_pool([False, False, False])
    real = Path(checkpoint.__file__).parent
    for name in ["engine.py", "constraints.py", "search.py", "pool.py", "checkpoint.py", "__init__.py"]:
        (tmp_path/name).write_bytes((real/name).read_bytes())
    monkeypatch.setattr(checkpoint, "__file__", str(tmp_path/"checkpoint.py"))
    a = checkpoint.fingerprint(pool, [en.Config()])
    p = tmp_path/"constraints.py"
    p.write_bytes(p.read_bytes()+b"\n# changed constraint semantics\n")
    assert checkpoint.fingerprint(pool, [en.Config()]) != a
