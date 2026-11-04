"""Strategy-book optimisation (2026-11-04): the engine reproduces x80's book, the weighting methods do what they
claim, the search cannot see the holdout, and the overfitting statistics behave on known cases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.levscore import book as lbk
from src.backtest.optimize import HOLDOUT
from src.backtest.optimize import engine as en
from src.backtest.optimize import newsleeves as ns
from src.backtest.optimize import overfit as of
from src.backtest.x80 import meta


def _weeks(n: int, start: str = "2008-01-04") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="W-FRI")


def _pool(n_weeks: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = _weeks(n_weeks)
    cols = ["BTC trend", "QQQ vol target", "PutWrite", "BTC vs alts", "multi-asset trend"]
    common = rng.standard_normal(n_weeks)
    X = pd.DataFrame({c: 0.002 + 0.02 * (0.4 * common + rng.standard_normal(n_weeks)) * (1 + k) for k, c in enumerate(cols)}, index=idx)
    X.iloc[:120, 0] = np.nan                                              # a sleeve that starts late
    X.iloc[:60, 3] = np.nan
    cash = pd.Series(0.0004, index=idx)
    m = pd.DataFrame({"family": ["crypto trend", "equity long", "volatility selling", "crypto relative value", "macro"],
                      "crypto": [True, False, False, True, False], "unfunded": [False, False, False, True, False],
                      "spread": [0.04, 0.01, 0.01, 0.04, 0.01]}, index=cols)
    return X, cash, m


def test_erc_gives_equal_risk_contributions():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((200, 6)) @ rng.standard_normal((6, 6))
    S = np.cov(A, rowvar=False)
    w = en.erc(S)
    rc = w * (S @ w)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    assert np.allclose(rc / rc.sum(), 1 / 6, atol=1e-6)
    assert np.allclose(w, meta.equal_risk_weights(S.copy()), atol=1e-6)    # same answer as x80's coordinate descent


def test_other_methods_are_long_only_and_sum_to_one():
    rng = np.random.default_rng(2)
    S = np.cov(rng.standard_normal((150, 7)) @ rng.standard_normal((7, 7)), rowvar=False)
    fam = np.array(["a", "a", "b", "b", "b", "c", "c"])
    for m in ("ivol", "ew", "hrp", "maxdiv", "family", "ens", "sharpe"):
        w = en.weights(m, S, rng.standard_normal(7) * 0.01, fam)
        assert w.sum() == pytest.approx(1.0) and (w >= -1e-12).all(), m
    capped = en.cap_weights(np.array([0.7, 0.2, 0.1]), 0.4)
    assert capped.max() <= 0.4 + 1e-12 and capped.sum() == pytest.approx(1.0)


def test_family_method_gives_each_family_equal_risk():
    rng = np.random.default_rng(3)
    S = np.cov(rng.standard_normal((300, 5)) * [1, 2, 3, 1, 5], rowvar=False)
    fam = np.array(["x", "x", "x", "y", "z"])
    w = en.weights("family", S, np.zeros(5), fam)
    rc = w * (S @ w)
    per_family = [rc[fam == f].sum() / rc.sum() for f in ("x", "y", "z")]
    assert np.allclose(per_family, 1 / 3, atol=0.02)


def test_crypto_risk_cap_holds():
    rng = np.random.default_rng(4)
    S = np.cov(rng.standard_normal((300, 4)) * [5, 4, 1, 1], rowvar=False)
    is_c = np.array([True, True, False, False])
    w0 = np.full(4, 0.25)
    w = en.crypto_risk_cap(w0, S, is_c, 0.35)
    share = (w[is_c] * (S @ w)[is_c]).sum() / (w @ S @ w)
    assert share == pytest.approx(0.35, abs=1e-4) and w.sum() == pytest.approx(1.0)
    assert np.array_equal(en.crypto_risk_cap(np.array([0.5, 0.5, 0.0, 0.0]), S, is_c, 0.35), [0.5, 0.5, 0.0, 0.0])


def test_engine_reproduces_x80_book_and_leverage():
    X, cash, m = _pool()
    pool = en.Pool(X, cash, m, {"all": list(X.columns)})
    bk = en.run_book(pool, en.Config(universe="all"))
    base, fin, _ = meta.run(X, cash, "equal risk", 104, unfunded={"BTC vs alts"})
    assert np.allclose(bk["base"][26:], base.to_numpy()[26:], atol=1e-10, equal_nan=True)
    assert np.allclose(bk["turn"][26:], fin["turn"].to_numpy()[26:], atol=1e-8)
    r, _lev = en.lever(bk, pool.cash, 0.5)
    ref = lbk.lever_path(base, fin, cash, lbk.vol_leverage(base, 0.5))
    got = pd.Series(r, index=X.index).reindex(ref.index)
    assert (got - ref).abs().max() < 1e-3                                  # same up to the drift of the cost line


def test_weights_use_only_the_past():
    X, cash, m = _pool()
    pool = en.Pool(X, cash, m, {"all": list(X.columns)})
    cfg = en.Config(universe="all", select="pos", method="hrp", rebal="band")
    a = en.run_book(pool, cfg, keep_weights=True)
    Y = X.copy()
    Y.iloc[300:] = Y.iloc[300:] * -3 + 0.05                               # rewrite the future
    b = en.run_book(en.Pool(Y, cash, m, {"all": list(X.columns)}), cfg, keep_weights=True)
    assert np.allclose(a["weights"][:301], b["weights"][:301])             # weights for week 300 are set before it
    assert np.allclose(a["base"][:300], b["base"][:300], equal_nan=True)


def test_leverage_band_trades_less_and_empty_books_sit_in_cash():
    X, cash, m = _pool()
    pool = en.Pool(X, cash, m, {"all": list(X.columns), "late": ["BTC trend"]})
    bk = en.run_book(pool, en.Config(universe="all"))

    def traded(band: float) -> float:
        r, lev = en.lever(bk, pool.cash, 0.8, band)
        drifted = lev * (1 + bk["base"]) / (1 + r)                       # exposure carried into next week
        return float(np.nansum(np.abs(lev[1:] - drifted[:-1])))

    assert traded(0.2) < 0.5 * traded(0.0)
    late = en.run_book(pool, en.Config(universe="late"))
    r, lev = en.lever(late, pool.cash, 0.8)
    first = int(np.argmax(late["invested"]))
    assert first > 120 and np.allclose(r[26:first], cash.iloc[26:first]) and (lev[26:first] == 0).all()
    assert np.isfinite(lev[first + 60:]).all() and lev[first + 60:].max() < en.CAP_LEV  # vol comes from invested weeks only


def test_search_pool_is_cut_before_the_holdout(tmp_path, monkeypatch):
    from src.backtest.optimize import search as se
    idx = pd.date_range("2015-01-02", "2026-06-26", freq="W-FRI")
    X = pd.DataFrame({c: 0.001 for c in ("a", "b")}, index=idx)
    X.to_parquet(tmp_path / "pool_weekly.parquet")
    pd.DataFrame({"family": ["macro", "macro"], "crypto": [False, False], "unfunded": [False, False], "spread": [0.01, 0.01]},
                 index=["a", "b"]).to_csv(tmp_path / "pool_meta.csv")
    pd.DataFrame({"cash": 0.0}, index=idx).to_parquet(tmp_path / "cash.parquet")
    monkeypatch.setattr(se, "CORE", ["a"])
    monkeypatch.setattr(se, "NEW", ["b"])
    se._init(str(tmp_path), str(tmp_path / "cash.parquet"))
    assert se._POOL.idx.max() < pd.Timestamp(HOLDOUT)
    assert len(se._POOL.idx) == (idx < HOLDOUT).sum()


def test_turn_of_month_window():
    idx = pd.bdate_range("2020-01-01", "2020-04-30")
    px = {"SPY": pd.DataFrame({"adj_close": np.linspace(100, 110, len(idx))}, index=idx)}
    ex = ns.tom_sleeve(px, pd.Series(0.0, index=idx))
    held = ex[ex != 0].index
    feb = [d for d in held if d.month == 2]
    assert feb[:3] == list(pd.bdate_range("2020-02-03", "2020-02-05")) and pd.Timestamp("2020-02-28") in held
    assert pd.Timestamp("2020-02-10") not in held


def test_cscv_flags_pure_noise_and_spares_a_real_edge():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2010-01-01", "2024-12-27", freq="W-FRI")
    noise = pd.DataFrame(rng.standard_normal((len(idx), 200)) * 0.02, index=idx, columns=[f"c{k}" for k in range(200)])
    cash = pd.Series(0.0, index=idx)
    pure = of.cscv(noise, cash)
    assert 0.15 < pure["pbo"] < 0.85 and abs(pure["oos_winner_sharpe_median"]) < 0.35   # the winner of noise is noise
    edge = noise.copy()
    edge["c0"] += 0.006                                                   # Sharpe ~2 on top of the noise
    real = of.cscv(edge, cash)
    assert real["pbo"] < 0.05 and real["oos_winner_sharpe_median"] > 1.0


def test_walk_forward_finds_a_persistent_winner():
    rng = np.random.default_rng(6)
    idx = pd.date_range("2010-01-01", "2024-12-27", freq="W-FRI")
    R = pd.DataFrame(rng.standard_normal((len(idx), 50)) * 0.02, index=idx, columns=[f"c{k:05d}" for k in range(50)])
    R["c00007"] += 0.008
    _wf, picks, pct = of.walk_forward(R, pd.Series(0.0, index=idx), "c00000")
    best = picks[picks["rule"] == "best score so far"]
    assert (best["pick"] == "c00007").mean() > 0.8
    assert pct.loc["best score so far", "held_year_percentile"] > 0.8
