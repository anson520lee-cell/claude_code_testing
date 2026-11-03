"""Edge-hunt study (2026-11-03): statistics helpers, replication universe, and the live FINRA short-volume rank."""
from __future__ import annotations

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.edge_hunt.gate import cluster_t
from src.backtest.edge_hunt.panel import membership_all
from src.backtest.edge_hunt.replicate import mean_t
from src.daily.settings import load_terminal_settings
from src.market import calendar
from src.positioning.positioning import FinraData, load_reference
from src.scanner.scoring import apply_enrichment, finalize


def test_cluster_t_recovers_delta_and_clusters():
    rng = np.random.default_rng(0)
    n = 4000
    groups = np.repeat(np.arange(200), 20)
    c = (rng.random(n) < 0.3).astype(float)
    y = 0.5 * c + rng.normal(0, 1, n)
    d, t, n1, n0 = cluster_t(y, c, groups)
    assert d == pytest.approx(y[c == 1].mean() - y[c == 0].mean())
    assert n1 + n0 == n and t > 10
    # a shock shared by every member of a month inflates the clustered standard error, not the delta
    y2 = y + np.repeat(rng.normal(0, 3, 200), 20) * c
    _, t2, _, _ = cluster_t(y2, c, groups)
    assert t2 < t


def test_mean_t_and_membership_all():
    rng = np.random.default_rng(1)
    y = rng.normal(0.2, 1, 3000)
    m, t = mean_t(y, np.repeat(np.arange(150), 20))
    assert m == pytest.approx(y.mean()) and t > 5
    assert np.isnan(mean_t(y[:5], np.arange(5))[0])
    dates = pd.bdate_range("2020-01-01", periods=60)
    px = {"BIG": pd.DataFrame({"close": 50.0, "volume": 1e6}, index=dates),     # $50M a day
          "TINY": pd.DataFrame({"close": 5.0, "volume": 1e5}, index=dates),     # $0.5M a day
          "^VIX": pd.DataFrame({"close": 20.0, "volume": 0.0}, index=dates)}
    mem = membership_all(px, dates, 5e6)
    assert set(mem) == {"BIG", "TINY"}
    assert mem["BIG"][14:].all() and not mem["BIG"][:14].any() and not mem["TINY"].any()


class _Resp:
    def __init__(self, text: str, status: int = 200):
        self.text, self.status_code = text, status


def _finra_files(session: date, n_days: int = 7) -> dict[str, str]:
    """CNMS-style files: 150 reference stocks with short ratios 0.30..0.60, LOW (0.20), HIGH (0.70), BRK.B (0.25)."""
    ref = [f"R{i:03d}" for i in range(150)]
    files, day = {}, session
    for _ in range(n_days):
        lines = ["Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"]
        for i, s in enumerate(ref):
            lines.append(f"{day:%Y%m%d}|{s}|{300 + 2 * i}|0|1000|B,Q,N")
        for s, short in (("LOW", 200), ("HIGH", 700), ("BRK.B", 250)):
            lines.append(f"{day:%Y%m%d}|{s}|{short}|0|1000|B,Q,N")
        files[f"{day:%Y%m%d}"] = "\n".join(lines)
        day = calendar.previous_trading_day(day)
    return files


def test_short_volume_rank(tmp_path):
    session = date(2026, 10, 30)
    files = _finra_files(session)
    calls = []

    def getter(url, timeout=20):
        calls.append(url)
        key = url.split("shvol")[1][:8]
        return _Resp(files[key]) if key in files and key != f"{session:%Y%m%d}" else _Resp("", 404)  # today not out yet

    fd = FinraData(tmp_path, getter=getter)
    ref = [f"R{i:03d}" for i in range(150)]
    low = fd.short_volume_rank("LOW", session, ref)
    assert low["available"] and low["ratio5"] == pytest.approx(0.2) and low["pct"] < 0.01
    assert low["files"][0] != session.isoformat() and len(low["files"]) == 5      # falls back to published days
    high = fd.short_volume_rank("HIGH", session, ref)
    assert high["pct"] > 0.99
    mid = fd.short_volume_rank("R075", session, ref)
    assert 0.45 < mid["pct"] < 0.55
    assert fd.short_volume_rank("BRK-B", session, ref)["ratio5"] == pytest.approx(0.25)   # BRK.B -> BRK-B
    n_calls = len(calls)
    fd.short_volume_rank("HIGH", session, ref)
    assert len(calls) == n_calls                                                          # cached per session
    assert not fd.short_volume_rank("ZZZ", session, ref)["available"]
    assert not fd.short_volume_rank("LOW", session, ref[:50])["available"]                # too few reference stocks
    offline = FinraData(tmp_path / "empty", offline=True)
    assert not offline.short_volume_rank("LOW", session, ref)["available"]


def test_reference_list_shipped():
    ref = load_reference()
    assert len(ref) >= 490 and "AAPL" in ref and "BRK-B" in ref


def _candidate(st):
    c = {"symbol": "AAA", "setup": "crash_rebound", "direction": "long", "components": {k: 50.0 for k in st["weights"]},
         "penalties": {}, "supporting": [], "contradicting": []}
    finalize(c, st)
    return c


def test_low_short_volume_bonus_and_optional_filter():
    st = copy.deepcopy(load_terminal_settings())
    cfg = st["scanner"]["finra_low_short_volume"]
    assert cfg["bonus"] == 3 and cfg["max_pct"] == pytest.approx(0.333) and cfg["penalty_above"] == 0
    low = {"positioning": {"short_volume_rank": {"available": True, "ratio5": 0.35, "pct": 0.2}}}
    high = {"positioning": {"short_volume_rank": {"available": True, "ratio5": 0.55, "pct": 0.8}}}
    c = _candidate(st)
    before = c["score"]
    apply_enrichment(c, low, st)
    assert c["bonuses"] == {"finra_low_short_volume": 3.0} and c["score"] == pytest.approx(before + 3)
    assert any("bottom third" in x for x in c["supporting"])
    c = _candidate(st)
    apply_enrichment(c, high, st)
    assert not c.get("bonuses") and "high_short_volume" not in c["penalties"] and c["score"] == pytest.approx(before)
    st["scanner"]["finra_low_short_volume"]["penalty_above"] = 30                         # the filter switched on
    c = _candidate(st)
    apply_enrichment(c, high, st)
    assert c["penalties"]["high_short_volume"] == 30 and c["score"] == pytest.approx(max(before - 30, 0))
    short = _candidate(st)
    short["direction"] = "short"
    apply_enrichment(short, low, st)
    assert not short.get("bonuses")                                                      # longs only
