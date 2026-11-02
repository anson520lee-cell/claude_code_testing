"""Trendline / EMA retest study: signals are causal, the simulator accounts correctly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import trendline as T


def _walk(n: int = 900, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0006, 0.02, n)
    c = 50 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.008, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.008, n)))
    idx = pd.bdate_range("2015-01-01", periods=n)
    f = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)
    tr = pd.concat([f["high"] - f["low"], (f["high"] - f["close"].shift()).abs(), (f["low"] - f["close"].shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["ema200"] = f["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    f["ema250"] = f["close"].ewm(span=250, adjust=False, min_periods=250).mean()
    return f


def test_pivot_lows():
    low = np.array([5, 4, 3, 4, 5, 6, 5, 2, 5, 6, 7.0])
    assert list(T.pivot_lows(low, 2)) == [2, 7]
    assert len(T.pivot_lows(low[:3], 2)) == 0


def test_signals_are_causal():
    f = _walk()
    full = T.collect_signals("X", f)
    assert sum(len(v) for v in full.values()) > 20           # the test is meaningless without signals
    for cut in (500, 640, 777):
        part = T.collect_signals("X", f.iloc[: cut + 1])
        for k, df in full.items():
            a = df[df["d"] <= cut - 1].reset_index(drop=True)
            b = part[k][part[k]["d"] <= cut - 1].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_trendline_signal_touches_its_line():
    f = _walk()
    tl = T.collect_signals("X", f)["tl_d"]
    assert len(tl) > 5
    low, atr = f["low"].to_numpy(), f["atr"].to_numpy()
    for _, r in tl.iterrows():
        d = int(r["d"])
        assert low[d] <= r["level"] + 0.25 * atr[d] + 1e-9           # a touch
        assert r["nt"] >= 3 and r["d"] - r["pb"] > 0                   # after the second anchor
        assert r["ann"] >= -0.05 - 1e-9


def test_retest_counts_from_the_last_major_leg():
    n = 700
    c = np.concatenate([np.full(250, 100.0), 100 * 1.004 ** np.arange(150), np.linspace(100 * 1.004 ** 149, 110, 150), np.full(150, 110.0)])
    idx = pd.bdate_range("2016-01-01", periods=n)
    s = pd.Series(c, index=idx)
    ma = s.ewm(span=200, adjust=False, min_periods=200).mean().to_numpy()
    atr = np.full(n, 1.0)
    low = c - 0.3
    out = T.ema_retests(c, low, atr, ma, (0.3,))
    assert len(out) >= 1
    assert out["no30"].iloc[0] == 1
    assert (out["no30"] >= 1).all()


def test_simulator_stops_targets_trails_and_gaps():
    # 3 trades x 6 bars; entry open 100, no slippage. stop distance 2.
    O = np.array([[100, 100, 100, 100, 100, 100], [100, 100, 97, 100, 100, 100], [100, 101, 103, 110, 110, 110], [100, 101, 102, 103, 104, 105.0]])
    H = O + 1
    L = O - 1
    C = O.copy()
    L[0, 2] = 97                          # trade 0: low 97 <= stop 98 -> exit at the stop
    L[1, 2] = 90                          # trade 1: gap open 97 below the stop -> exit at the open
    P = {"O": O.astype(float), "H": H.astype(float), "L": L.astype(float), "C": C.astype(float)}
    atr = np.full(4, 1.0)
    out = T.simulate(P, 0.0, np.full(4, 2.0), 6, atr=atr)
    assert out["stopped"].tolist() == [True, True, False, False]
    np.testing.assert_allclose(out["ret"][:2], [-0.02, -0.03])
    np.testing.assert_allclose(out["r"][:2], [-1.0, -1.5])
    np.testing.assert_allclose(out["ret"][3], 0.05)                  # time exit at the last close
    # account fraction: 1% risk, stop 2% away -> position 0.5 of the account, capped at 25%
    np.testing.assert_allclose(out["eq"][3], 0.05 * 0.25)
    # target 2.5 away is hit on trade 2 (high 104 on bar 2 -> no; 111 on bar 3 -> yes) at max(open, target) = 110
    out2 = T.simulate(P, 0.0, np.full(4, 2.0), 6, atr=atr, tgtd=np.full(4, 2.5))
    assert out2["target"].tolist() == [False, False, True, True] or out2["target"][2]
    # trailing stop: highest high 111 -> trail 4 ATR = 107 > initial stop 98; trade 2 stays in (low never <= 107? low 109)
    out3 = T.simulate(P, 0.0, np.full(4, 2.0), 6, trail=4.0, atr=atr)
    assert not out3["stopped"][2]


def test_slippage_both_sides():
    O = np.full((1, 3), 100.0)
    P = {"O": O, "H": O + 1, "L": O - 1, "C": O.copy()}
    out = T.simulate(P, 0.001, np.array([5.0]), 3, atr=np.array([1.0]))
    np.testing.assert_allclose(out["ret"][0], 100 * 0.999 / (100 * 1.001) - 1)


def test_cluster_stat_matches_plain_t_for_one_obs_per_cluster():
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1, 400)
    cl = np.arange(400)
    mu, t, n = T.cluster_stat(x, cl)
    se = x.std(ddof=1) / np.sqrt(400)
    assert n == 400 and abs(mu - x.mean()) < 1e-12
    assert abs(t - mu / se) < 0.05 * abs(mu / se)


def test_gather_paths_pads_with_nan():
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = {"open": np.arange(10.0), "high": np.arange(10.0) + 1, "low": np.arange(10.0) - 1, "close": np.arange(10.0) + .5, "index": idx}
    P = T.gather_paths({"A": a}, np.array([0, 0]), np.array([2, 8]), ["A"])
    assert P["O"][0, 0] == 3.0 and P["O"][0, 6] == 9.0 and np.isnan(P["O"][0, 7])
    assert P["O"][1, 0] == 9.0 and np.isnan(P["O"][1, 1])


def test_limit_entry_override_and_same_bar_stop():
    # fill at 98 (limit) on bar 0 whose open is 100; stop 2 away -> 96; the bar's low 95 is below the stop: stopped at once
    O = np.array([[100.0, 99, 99], [100.0, 101, 103]])
    P = {"O": O, "H": O + 1, "L": np.array([[95.0, 95, 95], [97.5, 100, 102]]), "C": O.copy()}
    out = T.simulate(P, 0.0, np.array([2.0, 2.0]), 3, atr=np.array([1.0, 1.0]), entry_px=np.array([98.0, 98.0]))
    assert out["stopped"].tolist() == [True, False]
    np.testing.assert_allclose(out["ret"][0], -2 / 98)
    np.testing.assert_allclose(out["r"][0], -1.0)
    np.testing.assert_allclose(out["ret"][1], 103 / 98 - 1)


def test_lines_report_failed_touches_only_in_limit_family():
    f = _walk(1500, seed=9)
    c = f["close"].to_numpy()
    out = T.collect_signals("X", f)
    held, lim = out["tl_d"], out["tl_d_lim"]
    assert len(lim) > 0 and len(held) > 0
    # a held signal never closes more than 0.5 ATR under its line; a limit signal can
    a = f["atr"].to_numpy()
    assert (c[held["d"].to_numpy()] >= held["level"].to_numpy() - 0.5 * a[held["d"].to_numpy()] - 1e-9).all()
    low = f["low"].to_numpy()
    assert (low[lim["d"].to_numpy()] <= lim["lim_px"].to_numpy() + 1e-9).all()


def test_limit_fill_bar_high_gets_no_credit():
    # fill 98 on bar 0 (high 101 came before the fill); target 2 away = 100: must NOT be hit on bar 0
    O = np.array([[100.0, 99.0, 99.0]])
    P = {"O": O, "H": np.array([[101.0, 99.5, 99.5]]), "L": np.array([[97.5, 98.5, 98.5]]), "C": np.array([[98.5, 99.0, 99.0]])}
    out = T.simulate(P, 0.0, np.array([5.0]), 3, atr=np.array([1.0]), tgtd=np.array([2.0]), entry_px=np.array([98.0]))
    assert not out["target"][0] and not out["stopped"][0]


def test_trendline_study_shipped_copy():
    import json
    from pathlib import Path
    d = json.loads((Path(__file__).resolve().parents[1] / "config" / "trendline_study.default.json").read_text())
    assert d["n_variants"] * d["n_configs"] == 4608
    assert all(g["n_pass"] == 0 for g in d["gate"].values())           # nothing validated, so no live rule changed
    assert set(d["signal_counts"]) == {"tl_d", "tl_w", "ma200", "ma250", "tl_d_lim", "tl_w_lim", "ma200_lim", "ma250_lim"}
