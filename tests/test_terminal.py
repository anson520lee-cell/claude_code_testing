"""Tests for the research terminal: calendar, data update + quality, regime, setups, plans,
historical simulation, scoring, fundamentals (point-in-time), catalysts, options, the daily
pipeline end to end (fake data, no internet) and the dashboard API."""

from __future__ import annotations

import copy
import json
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.catalysts.catalysts import build_catalysts, classify_filings, earnings_info, macro_events
from src.daily.settings import load_terminal_settings, validate
from src.config import ConfigError
from src.data import market_data, quality
from src.derivatives.options import analyze_chains, max_pain
from src.fundamentals.sec_fundamentals import analyze_fundamentals, facts_from_yahoo, quarterly_series
from src.market import calendar, regime
from src.scanner.backtest import run_backtest, simulate_symbol
from src.scanner.features import build_features
from src.scanner.plan import plan_for_setup, plan_row
from src.scanner.scoring import apply_enrichment, build_candidate, cross_sectional_ranks, reward_risk_score
from src.scanner.setups import detect_setups

SESSION = date(2026, 10, 30)


# ======================================================================= synthetic data
def bars(n: int = 520, end: date = SESSION, drift: float = 0.0006, vol: float = 0.012, seed: int = 1,
         base: float = 100.0, breakout_last: bool = False, volume: float = 2e6) -> pd.DataFrame:
    """Standard price frame (open/high/low/close/adj_close/volume) on weekdays ending ``end``."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    r = rng.normal(drift, vol, n)
    if breakout_last:
        r[-25:-1] = rng.normal(0, 0.003, 24)  # tight base
        r[-1] = 0.05
    close = base * np.cumprod(1 + r)
    prev = np.concatenate([[base], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    vols = rng.lognormal(np.log(volume), 0.15, n)
    if breakout_last:
        high[-1] = close[-1] * 1.001
        vols[-1] *= 3.5
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "adj_close": close,
                         "volume": vols.round()}, index=idx.rename("date"))


@pytest.fixture
def tsettings(tmp_path: Path) -> dict:
    s = load_terminal_settings()
    s = copy.deepcopy(s)
    s["universe"] = {"AAA": "XLK", "BBB": "XLK", "CCC": "XLE", "DDD": "XLE"}
    s["market"]["industry_etfs"] = {}
    for key, folder in (("database", "t.db"), ("reports_dir", "reports"), ("cache_dir", "cache"), ("backups_dir", "bk")):
        s["paths"][key] = str(tmp_path / folder)
    s["data"]["retry_wait_seconds"] = 0
    s["data"]["sec_user_agent"] = None
    s["liquidity"]["min_avg_dollar_volume"] = 1e6
    s["universe_filter"]["enabled"] = False   # synthetic stocks are calm; the filter has its own tests
    s["news"]["enabled"] = False              # no network in tests; news has its own tests
    s["leveraged_etfs"] = []
    return s


def market_frames(settings: dict, end: date = SESSION, n: int = 520) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(["SPY", "QQQ", "IWM", "DIA", "HYG", "IEF", "TLT", *settings["market"]["sector_etfs"]]):
        frames[sym] = bars(n, end, drift=0.0005, seed=10 + i, base=100 + i)
    frames["^VIX"] = bars(n, end, drift=0.0, vol=0.03, seed=99, base=16)
    frames["^VIX3M"] = bars(n, end, drift=0.0, vol=0.02, seed=98, base=18)
    frames["AAA"] = bars(n, end, drift=0.0012, seed=2, breakout_last=True, volume=5e6)
    frames["BBB"] = bars(n, end, drift=0.0008, seed=3, volume=5e6)
    frames["CCC"] = bars(n, end, drift=-0.0012, seed=4, volume=5e6)
    frames["DDD"] = bars(n, end, drift=0.0004, seed=5, volume=5e6)
    for i, sym in enumerate(settings["market"].get("macro", {})):
        frames[sym] = bars(n, end, drift=0.0002, vol=0.01, seed=60 + i, base=50 + 10 * i)
    if "BTC-USD" in frames:   # crypto trades every day: weekend bars must not break anything
        idx = pd.date_range(end=pd.Timestamp(end), periods=n, freq="D").rename("date")
        frames["BTC-USD"] = frames["BTC-USD"].set_axis(idx)
    return frames


class FakeDownloader:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames, self.calls = frames, []

    def download(self, symbols, period=None, start=None):
        self.calls.append((tuple(symbols), period, start))
        out = {}
        for s in symbols:
            if s in self.frames:
                f = self.frames[s]
                out[s] = f[f.index >= pd.Timestamp(start)] if start else f
        return out


class FakeResponse:
    def __init__(self, text: str, status: int = 200) -> None:
        self.text, self.status_code = text, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def fake_fred(url, timeout=None):
    name = url.split("id=")[1]
    days = pd.bdate_range(end="2026-10-29", periods=300)
    level = {"DGS10": 4.2, "DGS2": 3.9, "DGS3MO": 4.0}[name]
    return FakeResponse("observation_date," + name + "\n" + "\n".join(f"{d:%Y-%m-%d},{level:.2f}" for d in days))


def option_chain(spot: float, days: int, expiry: str) -> dict:
    strikes = np.arange(round(spot * 0.8), round(spot * 1.2), 5.0)
    def side(call: bool) -> pd.DataFrame:
        intrinsic = np.clip((spot - strikes) if call else (strikes - spot), 0, None)
        mid = intrinsic + spot * 0.03
        f = pd.DataFrame({"strike": strikes, "lastPrice": mid, "bid": mid * 0.98, "ask": mid * 1.02,
                          "volume": 100.0, "openInterest": 1000.0, "impliedVolatility": 0.35})
        f["mid"] = mid
        f["iv_ok"] = True
        return f
    return {"expiry": expiry, "days": days, "calls": side(True), "puts": side(False)}


class FakeEnrichment:
    finra = None

    def __init__(self) -> None:
        self.fetched = []

    def fetch(self, symbol: str) -> dict:
        self.fetched.append(symbol)
        earnings = pd.DataFrame({"EPS Estimate": [1.0, 0.9], "Reported EPS": [np.nan, 1.0], "Surprise(%)": [np.nan, 11.1]},
                                index=pd.DatetimeIndex([pd.Timestamp("2026-11-03 16:05", tz="America/New_York"),
                                                        pd.Timestamp("2026-08-01 07:00", tz="America/New_York")]))
        return {"symbol": symbol, "sec_facts": sample_facts(), "sec_status": "ok",
                "filings": [{"form": "S-3", "filingDate": "2026-10-20", "acceptanceDateTime": "2026-10-20T20:00:00Z",
                             "items": "", "url": "https://www.sec.gov/x"}],
                "info": {"shortPercentOfFloat": 0.2, "sharesShort": 2e6, "dateShortInterest": 1791000000,
                         "shortRatio": 6.0, "longName": f"{symbol} Corp"},
                "earnings_dates": earnings, "calendar": {}, "insider": None, "institutions": None, "upgrades": None,
                "option_chains": [option_chain(100, 7, "2026-11-06"), option_chain(100, 14, "2026-11-13")]}


def fact(start, end, val, filed, form="10-Q"):
    out = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        out["start"] = start
    return out


def sample_facts() -> dict:
    rev = [fact("2025-01-01", "2025-03-31", 100, "2025-05-01"), fact("2025-01-01", "2025-06-30", 210, "2025-08-01"),
           fact("2025-04-01", "2025-06-30", 110, "2025-08-01"), fact("2025-01-01", "2025-09-30", 330, "2025-11-01"),
           fact("2025-01-01", "2025-12-31", 460, "2026-02-15", "10-K"),
           fact("2026-01-01", "2026-03-31", 150, "2026-05-01"), fact("2026-04-01", "2026-06-30", 165, "2026-08-01"),
           # restated in a later filing - must be ignored (point-in-time original wins)
           fact("2025-04-01", "2025-06-30", 999, "2026-08-01"),
           # filed AFTER the session - must be invisible
           fact("2026-07-01", "2026-09-30", 500, "2026-11-05")]
    ocf = [fact("2026-01-01", "2026-03-31", 20, "2026-05-01"), fact("2026-01-01", "2026-06-30", 50, "2026-08-01")]
    shares = [fact(None, "2025-07-25", 100e6, "2025-08-01"), fact(None, "2026-07-25", 115e6, "2026-08-01")]
    return {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": rev}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": ocf}},
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [fact(None, "2026-06-30", 80, "2026-08-01")]}},
    }, "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares}}}}}


# ======================================================================= calendar
class TestCalendar:
    def test_2026_holidays(self):
        h = calendar.holidays(2026)
        for d in ("2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
                  "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"):
            assert date.fromisoformat(d) in h, d
        assert len(h) == 10

    def test_new_year_on_saturday_not_observed(self):
        assert date(2021, 12, 31) not in calendar.holidays(2021)
        assert calendar.is_trading_day(date(2021, 12, 31))

    def test_good_friday_and_easter(self):
        assert calendar.easter_sunday(2025) == date(2025, 4, 20)
        assert date(2025, 4, 18) in calendar.holidays(2025)

    def test_early_close(self):
        assert calendar.early_close(date(2026, 11, 27))   # day after Thanksgiving
        assert calendar.early_close(date(2026, 12, 24))
        assert not calendar.early_close(date(2026, 7, 3))  # holiday itself in 2026

    @pytest.mark.parametrize("now,expected", [
        (datetime(2026, 10, 30, 17, 0), date(2026, 10, 30)),   # Friday after close
        (datetime(2026, 10, 30, 15, 0), date(2026, 10, 29)),   # Friday during session
        (datetime(2026, 10, 30, 16, 10), date(2026, 10, 29)),  # inside the settle time
        (datetime(2026, 10, 31, 12, 0), date(2026, 10, 30)),   # Saturday
        (datetime(2026, 11, 27, 13, 45), date(2026, 11, 27)),  # early close + settle
        (datetime(2026, 11, 26, 18, 0), date(2026, 11, 25)),   # Thanksgiving
    ])
    def test_latest_completed_session(self, now, expected):
        assert calendar.latest_completed_session(now) == expected

    def test_trading_days_between(self):
        assert calendar.trading_days_between(date(2026, 10, 30), date(2026, 11, 3)) == 2


# ======================================================================= settings
def test_settings_validation(tsettings):
    validate(tsettings)
    bad = copy.deepcopy(tsettings)
    bad["weights"]["trend"] += 1
    with pytest.raises(ConfigError, match="sum to 100"):
        validate(bad)
    bad = copy.deepcopy(tsettings)
    bad["universe"][True] = "XLK"
    with pytest.raises(ConfigError, match="quote tickers"):
        validate(bad)


def test_shipped_universe_has_on_semiconductor():
    assert "ON" in load_terminal_settings()["universe"]


# ======================================================================= data update + quality
def test_update_prices_full_then_incremental_and_future_bar(tsettings, tmp_path):
    from src.daily import store
    frames = {"SPY": bars(520, date(2026, 11, 2))}  # contains a bar AFTER the session
    conn = store.connect(tmp_path / "x.db", tmp_path / "bk")
    dl = FakeDownloader(frames)
    s1 = market_data.update_prices(conn, ["SPY"], SESSION, tsettings, downloader=dl)
    assert s1["full"] == ["SPY"] and s1["discarded_future_bars"] == 1
    stored = market_data.load_universe_prices(conn, ["SPY"], SESSION)["SPY"]
    assert stored.index[-1] == pd.Timestamp(SESSION)
    s2 = market_data.update_prices(conn, ["SPY"], SESSION, tsettings, downloader=dl)
    assert s2["incremental"] == ["SPY"] and not s2["readjusted"]


def test_corporate_action_triggers_full_redownload(tsettings, tmp_path):
    from src.daily import store
    conn = store.connect(tmp_path / "x.db", tmp_path / "bk")
    original = bars(520)
    market_data.update_prices(conn, ["AAA"], SESSION, tsettings, downloader=FakeDownloader({"AAA": original}))
    split = original.copy()
    split[["open", "high", "low", "close", "adj_close"]] /= 2  # provider re-adjusted history for a 2:1 split
    summary = market_data.update_prices(conn, ["AAA"], SESSION, tsettings, downloader=FakeDownloader({"AAA": split}))
    assert summary["readjusted"] == ["AAA"]
    stored = market_data.load_universe_prices(conn, ["AAA"], SESSION)["AAA"]
    assert stored["close"].iloc[0] == pytest.approx(split["close"].iloc[0])


def test_quality_checks():
    good = bars(400)
    stale = good.iloc[:-2]
    short = good.tail(100)
    rep = quality.check_universe({"G": good, "S": stale, "H": short}, ["G", "S", "H", "M"], SESSION, 260)
    status = {r["symbol"]: r["status"] for r in rep["symbols"]}
    assert status == {"G": "ok", "S": "excluded", "H": "excluded", "M": "excluded"}
    assert rep["usable"] == ["G"]


# ======================================================================= regime
def test_regime_bullish_and_bearish(tsettings):
    up = market_frames(tsettings)
    for s in ("SPY", "QQQ", "IWM", "DIA", "AAA", "BBB", "CCC", "DDD"):
        up[s] = bars(520, drift=0.003, vol=0.006, seed=hash(s) % 100)
    up["^VIX"] = bars(520, drift=0, vol=0.001, base=12)
    hist = regime.compute_regime_history(up, ["AAA", "BBB", "CCC", "DDD"], tsettings)
    assert hist["label"].iloc[-1] in ("Bullish", "Strong Bullish")
    down = {k: v.copy() for k, v in up.items()}
    for s in ("SPY", "QQQ", "IWM", "DIA", "AAA", "BBB", "CCC", "DDD"):
        down[s] = bars(520, drift=-0.003, vol=0.006, seed=hash(s) % 100)
    down["^VIX"] = bars(520, drift=0, vol=0.001, base=35)
    hist = regime.compute_regime_history(down, ["AAA", "BBB", "CCC", "DDD"], tsettings)
    assert hist["label"].iloc[-1] in ("Bearish", "Strong Bearish")
    summary = regime.summarize(hist)
    assert len(summary["components"]) == 6 and summary["guidance"]


def test_regime_label_guard():
    assert regime._label(60, spy_above_ema50=False) == "Bullish"
    assert regime._label(-60, spy_above_ema50=True) == "Bearish"
    assert regime._label(0, True) == "Neutral"


# ======================================================================= features, setups, plan
@pytest.fixture
def breakout_features():
    spy = bars(520, seed=11)["adj_close"]
    f = build_features(bars(520, seed=2, drift=0.0012, breakout_last=True), spy, spy)
    return f, detect_setups(f)


def test_breakout_detected_and_plan(breakout_features, tsettings):
    f, st = breakout_features
    assert bool(st["setup_breakout"].iloc[-1])
    assert st["primary_setup"].iloc[-1] == "breakout"
    p = plan_row(f, "breakout", tsettings["scanner"], 7)
    atr = f["atr"].iloc[-1]
    assert p["stop"] < p["entry"] < p["target1"] < p["target2"]
    assert 1.0 - 1e-9 <= (p["entry"] - p["stop"]) / atr <= 3.0 + 1e-9
    assert p["reward_risk"] == pytest.approx((p["target1"] - p["entry"]) / (p["entry"] - p["stop"]))


def test_features_are_causal():
    """Values on day t must not change when later data is appended (no look-ahead)."""
    full = bars(400, seed=7)
    spy = bars(400, seed=8)["adj_close"]
    a = build_features(full.iloc[:350], spy.iloc[:350], None)
    b = build_features(full, spy, None)
    cols = ["ema21", "rsi14", "macd_hist", "atr", "swing_high_last", "swing_low_last", "rel_20d", "hv20", "clv"]
    pd.testing.assert_frame_equal(a[cols].iloc[-1:], b[cols].iloc[349:350], check_names=False)


def test_short_plan_mirrors(breakout_features, tsettings):
    f, _ = breakout_features
    p = plan_for_setup(f, "breakdown", tsettings["scanner"]).iloc[-1]
    assert p["stop"] > p["entry"] > p["target1"]


# ======================================================================= historical simulation
def _path_frame(prices: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-01", periods=len(prices))
    o, h, lo, c = zip(*prices)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)


def test_simulation_target_and_conservative_stop(monkeypatch, tsettings):
    df = _path_frame([(100, 101, 99, 100), (100, 103, 99.5, 102), (102, 112, 101, 111)] + [(111, 112, 110, 111)] * 8)
    st = pd.DataFrame(False, index=df.index, columns=[f"setup_{s}" for s in
                                                      ("breakout", "pullback", "squeeze", "reclaim", "momentum",
                                                       "breakdown", "rally_fail", "ema21_loss", "crash_rebound")])
    st.iloc[0, 0] = True
    plan = pd.DataFrame({"entry": 100.0, "stop": 98.0, "target1": 110.0, "target2": 115.0}, index=df.index)
    monkeypatch.setattr("src.scanner.backtest.plan_for_setup", lambda d, s, c: plan)
    cfg = {**tsettings["scanner"], "slippage_bps": 0}
    trades = simulate_symbol("X", df, st, cfg)
    assert trades[0]["outcome"] == "target" and trades[0]["r"] == pytest.approx(5.0) and trades[0]["days"] == 2
    # a bar touching both the stop and the target counts as a STOP (conservative)
    df2 = _path_frame([(100, 101, 99, 100), (100, 111, 97, 105)] + [(105, 106, 104, 105)] * 9)
    trades = simulate_symbol("X", df2, st, cfg)
    assert trades[0]["outcome"] == "stop" and trades[0]["r"] == pytest.approx(-1.0)


def test_backtest_summary_structure(breakout_features, tsettings):
    f, st = breakout_features
    res = run_backtest({"AAA": f}, {"AAA": st}, tsettings["scanner"])
    assert res["n_trades"] == len(res["trades_frame"])
    assert res["baseline"]["n"] > 0
    for stats in res["by_setup"].values():
        assert 0 <= stats["win_rate"] <= 1


# ======================================================================= scoring
def test_reward_risk_score_monotonic():
    values = [reward_risk_score(x) for x in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)]
    assert values == sorted(values) and values[-1] == 100


def test_candidate_scoring_and_enrichment_penalties(breakout_features, tsettings):
    f, st = breakout_features
    latest = pd.DataFrame({"AAA": f.iloc[-1], "BBB": f.iloc[-200]}).T
    for c in ("rel_5d", "rel_20d", "rel_60d", "rel_sector_20d"):
        latest[c] = pd.to_numeric(latest[c])
    ranks = cross_sectional_ranks(latest)
    plan = plan_row(f, "breakout", tsettings["scanner"], 7)
    c = build_candidate("AAA", f.iloc[-1], "breakout", float(st["quality_breakout"].iloc[-1]), plan, ranks.loc["AAA"],
                        {"regime": "Bullish", "sector_quadrant": "Leading", "sector_etf": "XLK"}, tsettings)
    assert 0 <= c["score"] <= 100 and set(c["components"]) == set(tsettings["weights"])
    assert c["score"] == pytest.approx(max(0, c["raw_score"] - c["penalty_total"]), abs=0.01)
    before = c["score"]
    enrichment = {"catalysts": {"next_earnings": {"date": "2026-11-03", "timing": "after close", "sessions_until": 2},
                                "filings": [{"form": "S-3", "dilution_risk": True, "age_days": 10}], "analyst": []},
                  "fundamentals": {"flags": [{"kind": "risk", "severity": "high", "text": "Cash runway short"}]}}
    apply_enrichment(c, enrichment, tsettings)
    assert {"earnings_in_window", "dilution_filing", "financial_distress"} <= set(c["penalties"])
    assert c["score"] < before
    assert any("Earnings" in x for x in c["contradicting"])


# ======================================================================= fundamentals
def test_fundamentals_point_in_time_and_ytd_derivation():
    facts = sample_facts()
    q = quarterly_series(facts, "revenue", SESSION)
    by_end = dict(zip(q["end"], q["value"]))
    assert by_end["2025-06-30"] == 110            # original, not the 999 restatement
    assert by_end["2025-09-30"] == 120            # derived: 9M YTD 330 - 6M YTD 210
    assert by_end["2025-12-31"] == 130            # derived Q4: FY 460 - 9M 330
    assert "2026-09-30" not in by_end             # filed after the session
    ocf = quarterly_series(facts, "ocf", SESSION)
    assert dict(zip(ocf["end"], ocf["value"]))["2026-06-30"] == 30
    r = analyze_fundamentals(facts, SESSION, price=10.0)
    m = r["metrics"]
    assert m["latest_quarter_end"] == "2026-06-30"
    assert m["revenue_yoy"] == pytest.approx(165 / 110 - 1)
    assert m["revenue_ttm"] == 120 + 130 + 150 + 165
    assert m["shares_yoy"] == pytest.approx(0.15)
    assert any("dilution" in f["text"] for f in r["flags"])
    early = analyze_fundamentals(facts, date(2025, 8, 15), price=10.0)
    assert early["metrics"]["latest_quarter_end"] == "2025-06-30"


def test_fundamentals_unavailable_reasons():
    assert not analyze_fundamentals(None, SESSION)["available"]
    assert "IFRS" in analyze_fundamentals({"facts": {"ifrs-full": {}}}, SESSION)["reason"]


def test_yahoo_fallback_estimates_filing_dates():
    ends = [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-03-31")]
    income = pd.DataFrame({ends[0]: [165.0], ends[1]: [150.0]}, index=["TotalRevenue"])
    facts = facts_from_yahoo({"income": income, "balance": pd.DataFrame(), "cashflow": pd.DataFrame()})
    q = quarterly_series(facts, "revenue", date(2026, 8, 10))
    assert list(q["end"]) == ["2026-03-31"]          # Q2 not yet "published" (30 Jun + 45 days)
    assert list(quarterly_series(facts, "revenue", date(2026, 8, 20))["end"])[-1] == "2026-06-30"


# ======================================================================= catalysts
def test_filing_classification():
    filings = [{"form": "8-K", "filingDate": "2026-10-28", "items": "2.02,9.01", "url": "u"},
               {"form": "8-K", "filingDate": "2026-10-20", "items": "1.03"},
               {"form": "424B5", "filingDate": "2026-10-15", "items": ""},
               {"form": "4", "filingDate": "2026-10-15", "items": ""},
               {"form": "S-3", "filingDate": "2026-05-01", "items": ""}]  # too old
    out = classify_filings(filings, SESSION, "XYZ")
    types = [(o["form"], o.get("item"), o["sentiment"], o["dilution_risk"]) for o in out]
    assert ("8-K", "2.02", "Uncertain", False) in types
    assert ("8-K", "1.03", "Negative", False) in types
    assert ("424B5", None, "Negative", True) in types
    assert all(o["form"] != "4" for o in out) and len(out) == 3


def test_earnings_timing():
    dates = pd.DataFrame({"EPS Estimate": [1.0, 2.0]}, index=pd.DatetimeIndex(
        [pd.Timestamp("2026-11-05 15:00", tz="America/New_York"), pd.Timestamp("2026-11-20 16:05", tz="America/New_York")]))
    info = earnings_info(dates, None, SESSION)
    n = info["next"]
    assert n["date"] == "2026-11-05" and n["timing"] is None and n["status"] == "Estimated"  # 15:00 placeholder
    assert n["reaction_session"] == "2026-11-05" and n["sessions_until"] == 4
    later = earnings_info(dates.iloc[1:], None, SESSION)["next"]
    assert later["timing"] == "after close" and later["reaction_session"] == "2026-11-23"


def test_macro_events_rules():
    ev = macro_events(SESSION, [{"date": "2026-11-12", "event": "CPI"}], days=30)
    names = [(e["event_date"], e["event_type"]) for e in ev]
    assert ("2026-11-12", "CPI") in names
    assert ("2026-11-20", "Monthly options expiration") in names
    assert any(e["event_type"].startswith("Employment") and e["status"] == "Estimated" for e in ev)


# ======================================================================= options
def test_options_analysis():
    chains = [option_chain(100, 7, "2026-11-06"), option_chain(100, 14, "2026-11-13")]
    o = analyze_chains(chains, spot=100.0, hv20=0.25, horizon_days=7)
    assert o["available"] and o["pcr_volume"] == pytest.approx(1.0)
    assert o["atm_iv"] == pytest.approx(0.35) and o["iv_hv_ratio"] == pytest.approx(1.4)
    assert o["expected_move"] == pytest.approx(6.0, rel=0.01)   # call mid + put mid at 100 = 3 + 3
    assert o["iv_rank"] is None and "Insufficient" in o["iv_rank_note"]
    sym = pd.DataFrame({"strike": [90.0, 95, 100, 105, 110], "openInterest": [1000.0] * 5})
    assert max_pain({"calls": sym, "puts": sym}) == pytest.approx(100.0)   # symmetric OI -> middle strike
    assert not analyze_chains([], 100, 0.2)["available"]


# ======================================================================= pipeline end to end
def test_daily_pipeline_end_to_end(tsettings, tmp_path):
    from src.daily.pipeline import run_daily
    frames = market_frames(tsettings)
    enrich = FakeEnrichment()
    result = run_daily(tsettings, session=SESSION, downloader=FakeDownloader(frames), enrichment=enrich,
                       fred_getter=fake_fred, progress=lambda m: None)
    out = Path(result["output_dir"])
    for name in ("data.json", "report.md", "report.html"):
        assert (out / name).exists()
    data = json.loads((out / "data.json").read_text())
    assert data["session"]["date"] == "2026-10-30"
    assert data["regime"]["label"] in regime.LABELS
    long_symbols = [c["symbol"] for c in data["candidates"]["long"]]
    assert "AAA" in long_symbols
    aaa = next(c for c in data["candidates"]["long"] if c["symbol"] == "AAA")
    assert aaa["deep_dive"] and "earnings_in_window" in aaa["penalties"] and "dilution_filing" in aaa["penalties"]
    assert aaa["history"]["setup"] is not None and aaa["thesis"]
    report = (out / "report.md").read_text()
    assert "Market regime" in report and "AAA" in report and "not investment advice" in report
    conn = sqlite3.connect(tsettings["paths"]["database"])
    assert conn.execute("SELECT status FROM terminal_runs").fetchone()[0].startswith("success")
    assert conn.execute("SELECT COUNT(*) FROM scan_candidates").fetchone()[0] >= 1
    assert conn.execute("SELECT COUNT(*) FROM options_snapshots").fetchone()[0] >= 1
    assert conn.execute("SELECT COUNT(*) FROM regime_history").fetchone()[0] > 100
    # second run is idempotent (no duplicates) and only refreshes recent bars
    rows = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
    dl = FakeDownloader(frames)
    run_daily(tsettings, session=SESSION, downloader=dl, enrichment=enrich, fred_getter=fake_fred, progress=lambda m: None)
    assert conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0] == rows
    assert all(call[2] is not None for call in dl.calls)  # incremental (start date) downloads only
    assert conn.execute("SELECT COUNT(*) FROM scan_candidates").fetchone()[0] == len(data["candidates"]["long"]) + len(data["candidates"]["short"])


def test_pipeline_uses_latest_published_session(tsettings):
    from src.daily.pipeline import run_daily
    frames = market_frames(tsettings, end=date(2026, 10, 29))   # provider has not published Friday yet
    result = run_daily(tsettings, session=SESSION, downloader=FakeDownloader(frames), skip_deep=True,
                       fred_getter=fake_fred, progress=lambda m: None)
    assert result["session"]["date"] == "2026-10-29"
    assert any("has not published" in w for w in result["warnings"])


def test_existing_database_is_backed_up_before_upgrade(tsettings, tmp_path):
    from src.daily import store
    db_file = tmp_path / "old.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE keepme (x INTEGER)")
        conn.execute("INSERT INTO keepme VALUES (42)")
    conn = store.connect(db_file, tmp_path / "bk")
    assert conn.execute("SELECT x FROM keepme").fetchone()[0] == 42
    backups = list((tmp_path / "bk").glob("*.db"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as b:
        assert b.execute("SELECT x FROM keepme").fetchone()[0] == 42
    conn.close()
    store.connect(db_file, tmp_path / "bk").close()   # already upgraded: no second backup
    assert len(list((tmp_path / "bk").glob("*.db"))) == 1


def test_build_catalysts_events(tmp_path):
    raw = FakeEnrichment().fetch("AAA")
    cat = build_catalysts("AAA", SESSION, raw["earnings_dates"], {}, raw["filings"], None)
    assert cat["next_earnings"]["sessions_until"] == 3          # after close Tue -> reacts Wed
    assert any(e["event_type"] == "Earnings release" for e in cat["events"])
    assert cat["filings"][0]["dilution_risk"]


# ======================================================================= web API
def test_web_api(tsettings):
    from fastapi.testclient import TestClient

    from src.daily.pipeline import run_daily
    from src.web.app import create_app
    run_daily(tsettings, session=SESSION, downloader=FakeDownloader(market_frames(tsettings)), enrichment=FakeEnrichment(),
              fred_getter=fake_fred, progress=lambda m: None)
    client = TestClient(create_app(tsettings))
    assert client.get("/").status_code == 200
    assert client.get("/api/sessions").json()["sessions"] == ["2026-10-30"]
    snap = client.get("/api/snapshot").json()
    assert snap["session"]["date"] == "2026-10-30"
    chart = client.get("/api/chart/AAA?days=120").json()
    assert len(chart["candles"]) == 120 and chart["ema21"] and chart["rsi14"] and "levels" in chart
    q = chart["quote"]
    assert q["prev_close"] and q["high52"] >= q["close"] >= q["low52"] and q["hist_high"] >= q["high52"] and q["amplitude"] >= 0
    vp = chart["volume_profile"]
    assert vp and vp["val"] <= vp["poc"] <= vp["vah"]
    assert client.get("/api/chart/NOPE").status_code == 404
    assert client.get("/api/regime/history").json()["history"]
    assert client.get("/api/runs").json()["runs"][0]["status"].startswith("success")
    assert client.get("/api/candidate-history/AAA").json()["history"]
    assert "composite" in client.get("/api/methodology").json()["regime"].lower()
    assert client.get("/api/i18n").json()["wide_stop"] == "Wide stop"
    assert client.get("/api/minicharts?symbols=AAA,BBB&days=30").json()["AAA"]["c"]
    assert client.get("/reports/2026-10-30/report.html").status_code == 200
    assert client.get("/api/snapshot?session=1999-01-01").status_code == 404


# ======================================================================= intraday refresh (display only)
def test_intraday_slots_dst_holidays_and_early_close():
    from src.daily import intraday
    s = intraday.slots(date(2026, 11, 2))
    assert [t.strftime("%H:%M") for t in s] == ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00",
                                                 "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]
    # daylight saving: 9:30 New York = 13:30 UTC on Fri Oct 30 (EDT) but 14:30 UTC on Mon Nov 2 (EST)
    assert intraday.slots(date(2026, 10, 30))[0].astimezone(timezone.utc).hour == 13
    assert s[0].astimezone(timezone.utc).hour == 14
    assert intraday.slots(date(2026, 11, 1)) == [] and intraday.slots(date(2026, 11, 26)) == []   # Sunday, Thanksgiving
    assert intraday.slots(date(2026, 11, 27))[-1].strftime("%H:%M") == "13:00"                      # early close


def test_intraday_due_slot():
    from src.daily import intraday
    utc = timezone.utc
    assert intraday.due_slot(datetime(2026, 11, 2, 14, 29, tzinfo=utc), None) is None           # 9:29 ET
    first = intraday.due_slot(datetime(2026, 11, 2, 14, 31, tzinfo=utc), None)                 # 9:31 ET
    assert first.strftime("%H:%M") == "09:30"
    assert intraday.due_slot(datetime(2026, 11, 2, 14, 36, tzinfo=utc), first.isoformat()) is None   # already done
    assert intraday.due_slot(datetime(2026, 11, 2, 10, 2), first.isoformat()).strftime("%H:%M") == "10:00"  # naive = NY
    assert intraday.due_slot(datetime(2026, 11, 2, 16, 10), None).strftime("%H:%M") == "16:00"
    assert intraday.due_slot(datetime(2026, 11, 2, 16, 40), None) is None                     # after the close
    assert intraday.due_slot(datetime(2026, 11, 2, 12, 58), None) is None                     # 28 min after 12:30
    assert intraday.due_slot(datetime(2026, 11, 1, 12, 0), None) is None                      # Sunday


def _five_minute_bars(day: date, prev_day: date, start: float) -> pd.DataFrame:
    ny = calendar.NEW_YORK
    idx = [datetime.combine(prev_day, datetime.min.time(), ny).replace(hour=15, minute=55)]
    idx += [datetime.combine(day, datetime.min.time(), ny).replace(hour=9, minute=30) + timedelta(minutes=5 * i)
            for i in range(7)]
    close = [start] + [start * (1.01 + 0.002 * i) for i in range(7)]
    return pd.DataFrame({"Open": close, "High": [c * 1.003 for c in close], "Low": [c * 0.997 for c in close],
                         "Close": close, "Volume": [1000.0] * 8}, index=pd.DatetimeIndex(idx))


def test_intraday_refresh_is_display_only(tsettings):
    from fastapi.testclient import TestClient

    from src.daily import intraday
    from src.daily.pipeline import run_daily
    from src.web.app import create_app
    run_daily(tsettings, session=SESSION, downloader=FakeDownloader(market_frames(tsettings)), enrichment=FakeEnrichment(),
              fred_getter=fake_fred, progress=lambda m: None)
    db_file = Path(tsettings["paths"]["database"])
    conn = sqlite3.connect(db_file)
    before_prices = conn.execute("SELECT COUNT(*), MAX(date) FROM price_data").fetchone()
    snap_file = Path(tsettings["paths"]["reports_dir"]) / "2026-10-30" / "data.json"
    before_snap = snap_file.read_bytes()
    client = TestClient(create_app(tsettings))
    stored = client.get("/api/chart/AAA?days=60").json()
    assert "live" not in stored["quote"]
    live_day, asked = date(2026, 11, 2), []

    def fetch(symbols):
        asked.extend(symbols)
        return {sym: _five_minute_bars(live_day, SESSION, 100.0) for sym in symbols}

    now = datetime(2026, 11, 2, 10, 2)                      # 10:02 New York
    r = intraday.refresh(tsettings, now=now, fetch=fetch, conn=conn)
    assert r["status"] == "ok" and r["slot"] == "10:00"
    assert "SPY" in asked and len(asked) <= tsettings["intraday"]["max_symbols"]
    assert intraday.refresh(tsettings, now=now + timedelta(minutes=5), fetch=fetch, conn=conn)["status"] == "skipped"
    assert conn.execute("SELECT COUNT(*), MAX(date) FROM price_data").fetchone() == before_prices   # prices untouched
    conn.close()
    assert snap_file.read_bytes() == before_snap                                                      # scores untouched
    latest = client.get("/api/intraday").json()
    assert latest["session"] == "2026-11-02" and latest["quotes"]
    sym = next(iter(latest["quotes"]))
    chart = client.get(f"/api/chart/{sym}?days=60").json()
    q = chart["quote"]
    assert q["live"]["slot"] == "10:00" and q["close"] == pytest.approx(latest["quotes"][sym]["last"])
    assert q["prev_close"] == pytest.approx(stored["quote"]["close"]) if sym == "AAA" else q["prev_close"]
    assert chart["candles"][-1]["time"] == "2026-10-30"                     # no provisional candle in the chart
    assert "live" not in client.get(f"/api/chart/{sym}?days=60&session=2026-10-29").json()["quote"]   # history view


def test_run_lock_blocks_concurrent_runs(tsettings, tmp_path):
    from src.daily.pipeline import RunInProgressError, run_daily, run_lock
    from src.daily.settings import path
    with run_lock(path(tsettings, "database").parent):
        with pytest.raises(RunInProgressError):
            run_daily(tsettings, session=SESSION, downloader=FakeDownloader({}), skip_deep=True, progress=lambda m: None)
    with run_lock(path(tsettings, "database").parent):  # released after the block
        pass


def test_api_run_if_new_skips_when_current(tsettings, monkeypatch):
    from fastapi.testclient import TestClient

    from src.daily.pipeline import run_daily
    from src.web.app import create_app
    run_daily(tsettings, session=SESSION, downloader=FakeDownloader(market_frames(tsettings)), skip_deep=True,
              fred_getter=fake_fred, progress=lambda m: None)
    monkeypatch.setattr("src.web.app.calendar.latest_completed_session", lambda **kw: SESSION)
    client = TestClient(create_app(tsettings))
    body = client.post("/api/run?if_new=true").json()
    assert body["started"] is False and "up to date" in body["reason"]
    assert client.get("/api/run/status").json()["running"] is False


# ======================================================================= focus filter (high beta / high volatility)
def test_universe_filter_rules(tsettings):
    from src.scanner.universe_filter import apply_universe_filter
    spy = bars(400, seed=11, vol=0.01)["adj_close"]
    calm = build_features(bars(400, seed=3, vol=0.008, volume=5e7), spy, None)        # low vol, low beta
    wild = bars(400, seed=4, vol=0.008, volume=5e7)
    spy_ret = spy.pct_change().fillna(0).to_numpy()
    noise = np.random.default_rng(9).normal(0, 0.02, 400)
    close = 50 * np.cumprod(1 + 2.2 * spy_ret + noise)                                  # beta ~2.2, high vol
    wild["close"] = wild["adj_close"] = close
    wild["open"] = close * 0.999; wild["high"] = close * 1.01; wild["low"] = close * 0.99
    wild_f = build_features(wild, spy, None)
    s = copy.deepcopy(tsettings)
    s["universe_filter"]["enabled"] = True
    s["liquidity"]["min_avg_dollar_volume"] = 1e8
    s["leveraged_etfs"] = ["LEV"]
    out = apply_universe_filter({"CALM": calm, "WILD": wild_f, "SMALL": wild_f, "LEV": wild_f}, s,
                                {"CALM": 5e11, "WILD": 9e10, "SMALL": 5e8, "LEV": None})
    assert wild_f["beta120"].iloc[-1] > 1.6 and wild_f["hv60"].iloc[-1] > 0.35
    assert sorted(out["kept"]) == ["LEV", "WILD"]
    reasons = {x["symbol"]: x["reason"] for x in out["excluded"]}
    assert "beta" in reasons["CALM"] and "volatility" in reasons["CALM"]
    assert "market cap" in reasons["SMALL"]


def test_market_cap_cache(tmp_path):
    from src.data.profiles import market_caps
    calls = []
    fetch = lambda s: (calls.append(s), {"A": 3e9}.get(s))[1]  # noqa: E731
    first = market_caps(["A", "B"], tmp_path, SESSION, fetch=fetch)
    assert first == {"A": 3e9, "B": None} and calls == ["A", "B"]
    calls.clear()
    second = market_caps(["A", "B"], tmp_path, SESSION, fetch=fetch)
    assert second["A"] == 3e9 and calls == ["B"]          # A cached; unknown B retried
    assert market_caps(["A"], tmp_path, date(2026, 12, 31), offline=True, fetch=fetch)["A"] == 3e9


def test_pipeline_with_filter_enabled(tsettings):
    from src.daily.pipeline import run_daily
    s = copy.deepcopy(tsettings)
    s["universe_filter"]["enabled"] = True
    s["universe_filter"]["min_beta"] = -10          # let the synthetic stocks pass beta / volatility ...
    s["universe_filter"]["min_hv60"] = 0.0
    caps = {"AAA": 5e9, "BBB": 5e9, "CCC": 1e8, "DDD": 5e9}   # ... but CCC is too small
    result = run_daily(s, session=SESSION, downloader=FakeDownloader(market_frames(s)), skip_deep=True,
                       fred_getter=fake_fred, progress=lambda m: None, market_cap_fetch=caps.get)
    assert [x["symbol"] for x in result["universe_filter"]["excluded"]] == ["CCC"]
    assert "CCC" not in [u["symbol"] for u in result["universe"]]
    assert all(c["symbol"] != "CCC" for side in ("long", "short") for c in result["candidates"][side])


# ======================================================================= analyst note + techniques toolkit
def test_techniques_and_analyst_note():
    from src.analyst.analyst import analyze
    from src.analyst.techniques import run_all, volume_profile
    spy = bars(520, seed=11)["adj_close"]
    f = build_features(bars(520, seed=2, drift=0.0012, breakout_last=True), spy, spy)
    tk = run_all(f, spy)
    assert tk["count"] >= 45 and not [x for x in tk["findings"] if x["group"] == "System"]
    groups = {x["group"] for x in tk["findings"]}
    assert {"Stock behaviour", "Volume profile", "Patterns", "Indicator consensus", "Relative performance"} <= groups
    for x in tk["findings"]:
        assert x["signal"] in (-1, 0, 1) and 0 <= x["strength"] <= 1 and x["text"]
    vp = volume_profile(f)
    assert vp["val"] <= vp["poc"] <= vp["vah"] and 0 <= vp["profit_ratio"] <= 1
    assert abs(sum(b["volume"] for b in vp["bins"]) - f["volume"].tail(120).sum()) / f["volume"].tail(120).sum() < 1e-6
    note = analyze("AAA", f, None, None, {"label": "Bullish"}, {"symbol": "XLK", "name": "Technology", "quadrant": "Leading"}, spy, 80.0)
    assert note["bias"] in ("Strongly bullish", "Bullish", "Leaning bullish", "Neutral", "Leaning bearish", "Bearish", "Strongly bearish")
    assert note["confidence"] in ("high", "medium", "low") and -100 <= note["bias_score"] <= 100
    assert len(note["scorecard"]) == 14 and sum(x["weight"] for x in note["scorecard"]) == 100
    assert note["summary"] and len(note["scenarios"]) == 3 and note["watch"]
    assert any(sec["title"].startswith("Volume profile") for sec in note["sections"])


def test_level_reaction_counts_rejections():
    from src.analyst.techniques import level_reaction
    idx = pd.bdate_range("2026-01-01", periods=12)
    level = pd.Series(100.0, index=idx)
    close = pd.Series([95, 96, 97, 99.5, 98, 97, 96, 95, 96, 99.6, 97, 96], index=idx, dtype=float)
    f = pd.DataFrame({"close": close, "high": close + 0.8, "low": close - 0.8}, index=idx)
    r = level_reaction(f, level, "L", horizon=2)
    assert r["n_below"] == 2 and r["rejected"] == 1.0


# ======================================================================= SL / TP validation research
def _sig_data(paths: list[list[tuple[float, float, float, float]]], long: bool = True) -> dict:
    from src.backtest.research import H_MAX
    rows, arrs = [], {k: [] for k in "OHLC"}
    for i, p in enumerate(paths):
        p = p + [p[-1]] * (H_MAX - len(p))
        for k, j in zip("OHLC", range(4)):
            arrs[k].append([bar[j] for bar in p])
        rows.append({"symbol": f"S{i}", "setup": "breakout" if long else "breakdown", "long": long, "date": pd.Timestamp("2025-01-02") + pd.Timedelta(days=i),
                     "ref": 100.0, "atr": 2.0, "structural": 97.0 if long else 103.0, "resistance": 104.0 if long else 96.0, "planned_rr": 1.33,
                     "swing": np.nan, "chip_stop": np.nan, "chip_target": np.nan, "regime": "Bullish", "score": 50.0, "tier": "B",
                     "conditional": False, "hv20": 0.5})
    return {"signals": pd.DataFrame(rows), **{k: np.array(v, dtype=float) for k, v in arrs.items()}}


def test_research_simulation_stop_target_and_trail():
    from src.backtest.research import simulate
    data = _sig_data([
        [(100, 101, 99, 100), (100, 100.5, 96.5, 97)],                                  # stop 1.5 ATR = 97 on day 2
        [(100, 101, 99.5, 100.5), (101, 102, 100, 101.5), (102, 106.5, 101.5, 106)],    # 2R target = 106 on day 3
        [(100, 101, 99, 100), (100, 102, 96, 101)],                                     # falls through the stop on day 2
    ])
    t = simulate(data, "atr1.5", "r2", 10, 0.0)
    assert list(t["outcome"]) == ["stop", "target", "stop"]
    assert t["r"].iloc[0] == pytest.approx(-1.0) and t["r"].iloc[1] == pytest.approx(2.0)
    # trailing stop: rally to 112 then fall; the chandelier (highest high - 3 ATR = 106) locks in +6
    up = [(100, 101, 99.5, 100.5), (101, 106, 100.5, 105), (105, 112, 104.5, 111), (111, 111.5, 105.5, 106)]
    tt = simulate(_sig_data([up]), "atr1.5", "trail", 10, 0.0)
    assert tt["outcome"].iloc[0] == "stop" and tt["r"].iloc[0] == pytest.approx(6 / 3)
    # winners' heat is reported as a positive number of ATRs
    from src.backtest.research import diagnostics
    d = diagnostics(simulate(data, "atr1.5", "r2", 10, 0.0))
    assert d["winner_mae_atr"]["p50"] >= 0


def test_research_end_to_end_with_baseline():
    from src.backtest.research import collect_baseline, collect_signals, run_research
    s = load_terminal_settings()
    spy = bars(700, seed=99)["adj_close"]
    features, setups = {}, {}
    for i in range(4):
        f = build_features(bars(700, seed=10 + i, vol=0.03, drift=0.001), spy, None)
        features[f"T{i}"], setups[f"T{i}"] = f, detect_setups(f)
    data = collect_signals(features, setups, s)
    assert len(data["signals"]) > 50 and data["O"].shape[1] == 10
    assert any(c.startswith("c_") for c in data["signals"].columns) and "x_rsi14" in data["signals"].columns
    base = collect_baseline(features, step=10)
    assert set(base["signals"]["long"]) == {True, False}
    res = run_research(data, s, baseline=base)
    assert res["available"] and res["verdict"] and res["live"]["oos"]["n"] > 0
    assert len(res["grid"]) == 8 * 12 * 3 * 3
    assert "long" in res["factors"] and res["baseline"]
    key = next(k for k in res["baseline"] if not k.startswith("_"))
    assert {"baseline_oos", "edge_oos"} <= set(res["baseline"][key])


def test_backtest_api(tsettings, tmp_path):
    from fastapi.testclient import TestClient

    from src.web.app import create_app
    client = TestClient(create_app(tsettings))
    assert client.get("/api/backtest").json()["available"] is False
    out = Path(tsettings["paths"]["reports_dir"]).parent / "backtest"
    (out / "2026-10-30").mkdir(parents=True)
    g = {"n": 10, "avg_r": 0.1, "win_rate": 0.5, "t_stat": 1.0, "median_r": 0.0}
    (out / "2026-10-30" / "backtest.json").write_text(json.dumps({"available": True, "method": "x",
        "grid": [{"stop": "atr1.5", "target": "trail", "horizon": 10, "side": "long", "is": g, "oos": g}]}))
    (out / "latest.json").write_text(json.dumps({"session": "2026-10-30"}))
    b = client.get("/api/backtest").json()
    assert b["available"] and "method" not in b and "median_r" not in b["grid"][0]["is"]
    assert client.get("/api/backtest/status").json()["running"] is False


# ======================================================================= portfolio, groups, macro, earnings, FTD, alerts
def test_earnings_study_reactions_and_implied_move():
    from src.catalysts.earnings_study import earnings_study
    px = bars(300, seed=7)
    days = px.index
    # an after-close report two sessions before a +8% gap, and a pre-market report with a -6% day
    i1, i2 = 200, 260
    px.iloc[i1 + 1, px.columns.get_loc("open")] = px["close"].iat[i1] * 1.08
    px.iloc[i1 + 1, px.columns.get_loc("close")] = px["close"].iat[i1] * 1.09
    px.iloc[i2, px.columns.get_loc("open")] = px["close"].iat[i2 - 1] * 0.95
    px.iloc[i2, px.columns.get_loc("close")] = px["close"].iat[i2 - 1] * 0.94
    dates = pd.DataFrame({"Surprise(%)": [5.0, -3.0]},
                         index=pd.DatetimeIndex([days[i1] + pd.Timedelta(hours=16, minutes=5), days[i2] + pd.Timedelta(hours=7)]))
    calls = pd.DataFrame({"strike": [95.0, 100.0, 105.0], "mid": [6.0, 3.0, 1.0], "lastPrice": [6.0, 3.0, 1.0]})
    puts = pd.DataFrame({"strike": [95.0, 100.0, 105.0], "mid": [1.0, 3.0, 6.0], "lastPrice": [1.0, 3.0, 6.0]})
    nxt = {"reaction_session": "2026-11-12"}
    res = earnings_study(dates, px, SESSION, nxt, [{"expiry": "2026-11-13", "days": 14, "calls": calls, "puts": puts}])
    assert res["available"] and res["n"] == 2
    moves = {e["timing"]: e for e in res["events"]}
    assert moves["after close"]["reaction_session"] == days[i1 + 1].strftime("%Y-%m-%d")
    assert moves["before open"]["move"] == pytest.approx(-0.06, abs=0.002)
    assert moves["after close"]["gap"] == pytest.approx(0.08)
    k = min([95.0, 100.0, 105.0], key=lambda x: abs(x - px["close"].iat[-1]))
    straddle = {95.0: 7.0, 100.0: 6.0, 105.0: 7.0}[k]
    assert res["implied"]["move_pct"] == pytest.approx(straddle / px["close"].iat[-1]) and res["implied_vs_history"] > 0 and "Options imply" in res["note"]


def test_ftd_parse_and_analysis():
    import io
    import zipfile

    from src.positioning.ftd import analyze_ftd, candidate_files, load_ftd
    txt = ("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
           "20261001|111|AAA|50000|AAA INC|10.5\n20261002|111|AAA|400000|AAA INC|10.7\n20261002|222|BBB|100|BBB|5\n"
           "Trailer: record count 3\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("cnsfails202610a.txt", txt)

    class FakeSec:
        def __init__(self):
            self.urls = []

        def get_file(self, url, name):
            self.urls.append(url)
            return buf.getvalue() if name == "cnsfails202610a.zip" else None
    sec = FakeSec()
    frame = load_ftd(sec, date(2026, 10, 31), files=1)
    assert set(frame["symbol"]) == {"AAA", "BBB"} and sec.urls[0].endswith("cnsfails202610b.zip")
    assert candidate_files(date(2026, 1, 5))[2][1] == "cnsfails202512b.zip"
    vol = pd.Series(2e6, index=pd.bdate_range("2026-09-01", "2026-10-30"))
    a = analyze_ftd(frame, "AAA", vol)
    assert a["max"] == 400000 and a["max_vs_adv"] == pytest.approx(0.2) and "borrow" in a["note"]
    assert analyze_ftd(frame, "ZZZ", vol)["n"] == 0
    assert analyze_ftd(None, "AAA", vol)["available"] is False


def test_portfolio_report_sizing_correlation_and_heat():
    from src.portfolio.risk import portfolio_report, size_position
    sz = size_position(100, 95, 100_000, 0.005, 1.0, 0.25)
    assert sz["shares"] == 100 and sz["dollar_risk"] == pytest.approx(500)
    assert size_position(10, 9.99, 100_000, 0.01, 1.0, 0.25)["capped"]
    base = bars(200, seed=3)
    twin = base.copy()
    twin["close"] = base["close"] * (1 + np.random.default_rng(1).normal(0, 0.0005, 200))
    feats = {"AAA": base, "BBB": twin, "CCC": bars(200, seed=9)}
    cand = lambda s, d: {"symbol": s, "direction": d, "setup": "breakout", "score": 70,
                         "plan": {"entry": 100.0, "stop": 96.0, "atr": 2.0}}
    rep = portfolio_report(feats, {"long": [cand("AAA", "long"), cand("BBB", "long"), cand("CCC", "long")], "short": []},
                           {"account_size": 100_000, "risk_per_trade": 0.01, "max_heat": 0.02}, {"long": 1.0, "short": 0.5})
    assert rep["total_heat"] == pytest.approx(3 * 1000) and rep["total_heat_pct"] > 0.02
    assert ["AAA", "BBB"] in rep["clusters"] and rep["effective_bets"] < 3
    assert any("above the" in n and "cap" in n for n in rep["notes"])
    hedge = portfolio_report(feats, {"long": [cand("AAA", "long")], "short": [cand("BBB", "short")]}, {}, {"long": 1, "short": 1})
    assert hedge["correlation"][0][1] < -0.9   # long AAA vs short of its twin = hedge


def test_industry_groups_breadth_and_macro():
    from src.market.macro import advance_decline, group_lookup, industry_groups, macro_table
    spy = bars(400, seed=99)
    feats = {f"S{i}": build_features(bars(400, seed=20 + i, drift=0.002 if i < 3 else -0.001), spy["adj_close"], None) for i in range(6)}
    inds = {"S0": "Semiconductors", "S1": "Semiconductors", "S2": "Semiconductors", "S3": "Biotechnology", "S4": "Biotechnology", "S5": None}
    groups = industry_groups(feats, inds, {s: "XLK" for s in feats})
    assert groups[0]["group"] == "Semiconductors" and groups[0]["rank"] == 1 and len(groups) == 2   # S5 alone is dropped
    assert group_lookup(groups)["S3"]["of"] == 2
    prices = {s: bars(400, seed=30 + i) for i, s in enumerate([*[f"S{i}" for i in range(12)], "SPY", "QQQ"])}
    ad = advance_decline(prices, [f"S{i}" for i in range(12)], prices["SPY"].index)
    assert ad["available"] and ad["advancers"] + ad["decliners"] + ad["unchanged"] == 12 and len(ad["series"]["ad_line"]) == 160
    btc = bars(560, seed=5)
    btc.index = pd.date_range(end=pd.Timestamp(SESSION), periods=560, freq="D")
    prices["BTC-USD"] = btc
    m = macro_table(prices, {"BTC-USD": "Bitcoin", "NOPE": "None"}, prices["SPY"].index)
    assert m[0]["available"] and m[0]["corr_qqq_60d"] is not None and m[0]["comment"] and m[1]["available"] is False


def test_journal_and_alerts(tsettings):
    from src.daily import store as st
    from src.portfolio import journal
    conn = st.connect(Path(tsettings["paths"]["database"]), Path(tsettings["paths"]["backups_dir"]))
    from src.database import db
    db.upsert_stock(conn, "AAA")
    frame = bars(30, end=SESSION, seed=4)
    db.upsert_prices(conn, "AAA", frame, "test")
    tid = journal.add_trade(conn, {"symbol": "aaa", "direction": "long", "setup": "breakout", "entry_price": 100, "stop": 95,
                                   "shares": 10, "exit_price": 110})
    journal.add_trade(conn, {"symbol": "AAA", "direction": "short", "entry_price": 100, "stop": 105, "target": 90, "shares": 5})
    with pytest.raises(ValueError):
        journal.add_trade(conn, {"symbol": "AAA", "direction": "sideways", "entry_price": 1, "stop": 2})
    rev = journal.journal_review(conn)
    closed = next(t for t in rev["trades"] if t["id"] == tid)
    assert closed["symbol"] == "AAA" and closed["r"] == pytest.approx(2.0) and closed["pnl"] == pytest.approx(100)
    assert rev["stats"]["n_closed"] == 1 and rev["stats"]["n_open"] == 1
    assert any(t.get("mark") is not None for t in rev["trades"] if t["status"] == "open")
    # alerts trigger only on bars after creation: back-date the alert so the stored bars count
    aid = journal.add_alert(conn, "AAA", "above", float(frame["high"].iloc[-5:].max()) - 0.01, "test")
    conn.execute("UPDATE price_alerts SET created_at = ? WHERE id = ?", (f"{frame.index[-6]:%Y-%m-%d}T00:00:00Z", aid))
    conn.commit()
    far = journal.add_alert(conn, "AAA", "below", 0.01)
    fired = journal.check_alerts(conn)
    assert [a["id"] for a in fired] == [aid] and journal.check_alerts(conn) == []
    lst = {a["id"]: a for a in journal.list_alerts(conn)}
    assert lst[aid]["active"] == 0 and lst[far]["active"] == 1 and lst[far]["distance"] < 0
    with pytest.raises(ValueError):
        journal.add_alert(conn, "AAA", "sideways", 10)


def test_journal_and_alert_api(tsettings):
    from fastapi.testclient import TestClient

    from src.web.app import create_app
    client = TestClient(create_app(tsettings))
    r = client.post("/api/journal", json={"symbol": "AAA", "direction": "long", "entry_price": 10, "stop": 9})
    assert r.status_code == 200
    assert client.post("/api/journal", json={"symbol": "AAA", "direction": "long", "entry_price": 10, "stop": 10}).status_code == 400
    tid = r.json()["id"]
    assert client.put(f"/api/journal/{tid}", json={"exit_price": 12}).status_code == 200
    j = client.get("/api/journal").json()
    assert j["trades"][0]["status"] == "closed" and j["stats"]["avg_r"] == pytest.approx(2.0)
    assert client.post("/api/alerts", json={"symbol": "AAA", "kind": "above", "level": "abc"}).status_code == 400
    aid = client.post("/api/alerts", json={"symbol": "AAA", "kind": "close_above", "level": 12.5}).json()["id"]
    assert client.get("/api/alerts").json()["alerts"][0]["id"] == aid
    assert client.delete(f"/api/alerts/{aid}").status_code == 200 and client.get("/api/alerts").json()["alerts"] == []
    assert client.delete(f"/api/journal/{tid}").status_code == 200


def test_implied_earnings_move_isolates_event_variance():
    from src.catalysts.earnings_study import implied_earnings_move

    def chain(expiry, days, iv, price):
        f = pd.DataFrame({"strike": [100.0], "mid": [price], "lastPrice": [price], "impliedVolatility": [iv], "iv_ok": [True]})
        return {"expiry": expiry, "days": days, "calls": f, "puts": f.copy()}
    chains = [chain("2026-11-06", 7, 0.40, 2.0), chain("2026-12-04", 35, 0.60, 7.0)]
    imp = implied_earnings_move(chains, 100.0, date(2026, 11, 12))
    t = 35 / 365
    expected = np.sqrt(2 / np.pi) * np.sqrt(0.36 * t - 0.16 * (t - 1 / 365))
    assert imp["method"] == "event_variance" and imp["move_pct"] == pytest.approx(expected) and imp["straddle_pct"] == pytest.approx(0.14)
    near = implied_earnings_move([chain("2026-11-13", 14, 0.6, 5.0)], 100.0, date(2026, 11, 12))
    assert near["method"] == "straddle" and near["move_pct"] == pytest.approx(0.10)


def test_implied_earnings_move_without_event_premium():
    from src.catalysts.earnings_study import implied_earnings_move

    def chain(expiry, days, iv, price):
        f = pd.DataFrame({"strike": [100.0], "mid": [price], "lastPrice": [price], "impliedVolatility": [iv], "iv_ok": [True]})
        return {"expiry": expiry, "days": days, "calls": f, "puts": f.copy()}
    imp = implied_earnings_move([chain("2026-11-06", 7, 0.60, 2.0), chain("2026-12-18", 49, 0.40, 7.0)], 100.0, date(2026, 12, 18))
    assert imp["method"] == "no_event_premium" and imp["move_pct"] is None


# ======================================================================= crash rebound (buy the plunge)
def _crash_frame(n: int = 400, crash_at: int = 330, depth: float = 0.25) -> pd.DataFrame:
    px = bars(n, seed=21, drift=0.001, vol=0.006)
    k = px.columns.get_loc
    for i, day in enumerate(range(crash_at - 4, crash_at + 1)):       # five down days, -25% in total
        f = (1 - depth) ** (1 / 5)
        prev = px["close"].iat[day - 1]
        px.iloc[day, k("open")] = prev * 0.999
        px.iloc[day, k("close")] = prev * f
        px.iloc[day, k("high")] = prev
        px.iloc[day, k("low")] = prev * f * 0.995
    after = px["close"].iat[crash_at]
    for day in range(crash_at + 1, n):                                # then a rebound: +1%/day for 6 days
        prev = px["close"].iat[day - 1]
        g = 1.01 if day <= crash_at + 6 else 1.0005
        px.iloc[day, k("open")] = prev
        px.iloc[day, k("close")] = prev * g
        px.iloc[day, k("high")] = prev * g * 1.002
        px.iloc[day, k("low")] = prev * 0.998
    px["adj_close"] = px["close"]
    assert after < px["close"].iat[crash_at - 5]
    return px


def test_crash_rebound_setup_and_plan(tsettings):
    from src.scanner.plan import plan_for_setup
    spy = bars(400, seed=99)["adj_close"]
    f = build_features(_crash_frame(), spy, None)
    st = detect_setups(f)
    on = st["setup_crash_rebound"]
    first = int(np.flatnonzero(on.to_numpy())[0])
    assert on.iloc[first:first + 3].all() and not on.iloc[first + 3:first + 10].any()   # 0-2 days after the first crash only
    assert st["quality_crash_rebound"].iloc[first] > 0
    p = plan_for_setup(f, "crash_rebound", tsettings["scanner"]).iloc[first]
    a = f["atr"].iat[first]
    assert p["entry"] - p["stop"] == pytest.approx(2 * a) and p["target1"] - p["entry"] == pytest.approx(1.5 * a)
    assert p["target2"] >= p["target1"] + 0.5 * a - 1e-9


def test_crash_rebound_scoring_bonus_and_long_only(tsettings):
    spy = bars(400, seed=99)["adj_close"]
    f = build_features(_crash_frame(), spy, None)
    st = detect_setups(f)
    i = int(np.flatnonzero(st["setup_crash_rebound"].to_numpy())[0])
    from src.scanner.plan import plan_row
    s = copy.deepcopy(tsettings)
    s["scanner"]["setup_bonus"] = {"crash_rebound": 15}
    plan = plan_row(f.iloc[: i + 1], "crash_rebound", s["scanner"], 10)
    c = build_candidate("AAA", f.iloc[i], "crash_rebound", float(st["quality_crash_rebound"].iat[i]), plan,
                        pd.Series({"rs_pct": 50.0, "sector_rs_pct": 50.0}), {"regime": "Neutral"}, s)
    assert c["setup_bonus"] == 15 and "poor_reward_risk" not in c["penalties"]
    assert c["score"] == pytest.approx(min(100, max(0, c["raw_score"] - c["penalty_total"] + 15)), abs=0.01)
    assert any("Crash rebound" in x for x in c["supporting"]) and plan["horizon_days"] == 10
    assert tsettings["scanner"]["directions"] == ["long"]


def test_only_enabled_setups_make_candidates():
    from src.scanner.setups import LONG_SETUPS
    enabled = load_terminal_settings()["scanner"]["enabled_setups"]
    assert set(enabled) == {"crash_rebound", "squeeze", "breakout"} and set(enabled) <= set(LONG_SETUPS)
    src = (Path(__file__).resolve().parents[1] / "src" / "daily" / "pipeline.py").read_text()
    assert 'sc.get("enabled_setups")' in src and "n in enabled" in src


def test_crash_study_events_trades_and_history():
    from src.backtest.crash_study import _trade, analyse, baseline, collect_events, crash_history, event_masks
    spy = bars(400, seed=99)["adj_close"]
    f = build_features(_crash_frame(), spy, None)
    m = event_masks(f, 1)
    assert m["d5_20"].sum() == 1                       # consecutive crash days merged into one event
    ev = collect_events({"AAA": f}, spy, 1)
    d1 = ev[(ev["event"] == "d5_20") & (ev["entry"] == "d1")].iloc[0]
    assert d1["ret5"] > 0.03          # the event fires on day 4 of the slide; entering after the last down day pays
    # trade sim: long hits the target, short hits the stop on the same path
    o = np.array([100.0, 101, 102, 103]); hi = o + 1; lo = o - 0.5; c = o + 0.5
    assert _trade(o, hi, lo, c, 0, 100.0, 1.0, 0, "atr1", 1, horizon=3) == pytest.approx(0.5)
    assert _trade(o, hi, lo, c, 0, 100.0, 1.0, 0, "atr1", -1, horizon=3) == pytest.approx(-1.0)
    res = analyse(ev, baseline({"AAA": f}, 1))
    assert res["available"] and "delay_curve" in res and res["grid"]
    h = crash_history(f)
    assert h["summary"]["n"] >= 1 and {"d0_ret5", "d1_ret5", "d2_ret5", "d0_mae10"} <= set(h["events"][-1])
    assert h["events"][0]["date"] > h["events"][-1]["date"]          # newest first


def test_setup_bonus_from_backtest(tsettings):
    from src.scanner.scoring import backtest_setup_bonus
    bt = {"session": "2026-10-30", "by_setup": {"crash_rebound": {"n": 2700, "avg_r": 0.1}, "breakout": {"n": 5000, "avg_r": 0.03},
                                                "reclaim": {"n": 3000, "avg_r": -0.2}, "squeeze": {"n": 50, "avg_r": 0.5},
                                                "breakdown": {"n": 4000, "avg_r": -0.1}}}
    bonus, src = backtest_setup_bonus(bt, tsettings["scanner"])
    assert bonus == {"crash_rebound": 15.0, "breakout": 4.5, "reclaim": -10.0} and src.startswith("backtest")
    fb, src2 = backtest_setup_bonus(None, tsettings["scanner"])
    assert fb == {"crash_rebound": 15} and src2 == "config"



# ======================================================================= news
RSS_Y = """<?xml version="1.0"?><rss><channel>
<item><title>Acme beats estimates, raises full-year guidance</title><link>https://ex.com/a</link><pubDate>{d1}</pubDate></item>
<item><title>Acme announces $500 million convertible offering</title><link>https://ex.com/b</link><pubDate>{d2}</pubDate></item>
<item><title>Old story about Acme</title><link>https://ex.com/old</link><pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate></item>
</channel></rss>"""
RSS_G = """<?xml version="1.0"?><rss><channel>
<item><title>Acme beats estimates, raises full-year guidance - Reuters</title><link>https://g.com/1</link><pubDate>{d1}</pubDate><source url="https://reuters.com">Reuters</source></item>
<item><title>Unrelated company shares plunge - CNBC</title><link>https://g.com/2</link><pubDate>{d1}</pubDate><source>CNBC</source></item>
</channel></rss>"""


def _rss_getter(calls):
    from email.utils import format_datetime
    now = datetime.now(timezone.utc)
    d1, d2 = format_datetime(now - timedelta(hours=3)), format_datetime(now - timedelta(days=1))
    def get(url):
        calls.append(url)
        if "news.google.com" in url:
            return RSS_G.format(d1=d1)
        if "nasdaq.com" in url:
            return None
        return RSS_Y.format(d1=d1, d2=d2)
    return get


def test_news_parse_classify_dedupe_and_store(tsettings, tmp_path):
    from src.catalysts import news as nm
    calls = []
    client = nm.NewsClient(tmp_path / "cache", getter=_rss_getter(calls), max_age_days=21)
    items = client.symbol_news("ACME", "Acme Corp", google=True)
    titles = [x["title"] for x in items]
    assert len(items) == 2 and "Old story about Acme" not in titles          # dedupe across feeds + age filter
    assert all("Unrelated" not in t for t in titles)                          # Google relevance filter
    beat = next(x for x in items if x["title"].startswith("Acme beats"))
    assert {"earnings", "guidance"} <= set(beat["tags"]) and beat["sentiment"] == "positive"
    off = next(x for x in items if "offering" in x["title"])
    assert "offering" in off["tags"] and off["sentiment"] == "negative"
    n = len(calls)
    client.symbol_news("ACME", "Acme Corp", google=True)                      # served from the disk cache
    assert len(calls) == n
    summ = nm.summarize(items)
    assert summ["n_recent"] == 2 and "offering" in summ["flags"]
    from src.daily import store as st
    conn = st.connect(Path(tsettings["paths"]["database"]), Path(tsettings["paths"]["backups_dir"]))
    nm.save(conn, "ACME", items)
    nm.save(conn, "ACME", items)                                              # idempotent
    rows = nm.load(conn, "ACME")
    assert len(rows) == 2 and nm.load(conn, "ACME", tag="offering")[0]["title"].startswith("Acme announces")


def test_news_api(tsettings, tmp_path):
    from fastapi.testclient import TestClient

    from src.catalysts import news as nm
    from src.daily import store as st
    from src.web.app import create_app
    conn = st.connect(Path(tsettings["paths"]["database"]), Path(tsettings["paths"]["backups_dir"]))
    client = nm.NewsClient(tmp_path / "c2", getter=_rss_getter([]))
    nm.save(conn, "ACME", client.symbol_news("ACME", "Acme Corp"))
    nm.save(conn, "_MARKET", [{"title": "Stocks rally", "url": "https://m.com/1", "published": None, "publisher": "X", "feed": "Google News",
                               "tags": ["big move"], "sentiment": "positive", "topic": "Stock market"}])
    conn.close()
    api = TestClient(create_app(tsettings))
    assert len(api.get("/api/news?symbol=ACME").json()["items"]) == 2
    assert api.get("/api/news?symbol=_MARKET").json()["items"][0]["topic"] == "Stock market"
    assert api.get("/api/news?tag=offering").json()["items"][0]["symbol"] == "ACME"


# ======================================================================= long-term robustness
def test_provider_fallback_is_retried_not_marked_done(tsettings):
    """If the provider has not published the target session, the run analyses the latest available one,
    records THAT session, and --if-new keeps retrying the target (at most 6 times)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import daily_run
    from src.daily.pipeline import run_daily
    frames = {s: f[f.index < pd.Timestamp(SESSION)] for s, f in market_frames(tsettings).items()}   # nobody has 10-30
    res = run_daily(tsettings, session=SESSION, downloader=FakeDownloader(frames), enrichment=FakeEnrichment(),
                    fred_getter=fake_fred, progress=lambda m: None)
    assert res["session"]["date"] == "2026-10-29"
    db_file = Path(tsettings["paths"]["database"])
    conn = sqlite3.connect(db_file)
    row = conn.execute("SELECT session_date, requested_session FROM terminal_runs ORDER BY run_id DESC LIMIT 1").fetchone()
    conn.close()
    assert row == ("2026-10-29", "2026-10-30")
    done, tries = daily_run.session_status(db_file, SESSION)
    assert not done and tries == 1
    done29, _ = daily_run.session_status(db_file, date(2026, 10, 29))
    assert done29


def test_quality_ignores_unscheduled_market_closures(tsettings):
    frames = market_frames(tsettings)
    closed = pd.Timestamp("2026-06-17")                   # pretend the market shut that day: SPY has no bar
    frames = {s: f[f.index != closed] for s, f in frames.items()}
    q = quality.check_universe(frames, ["SPY", "AAA"], SESSION)
    aaa = next(r for r in q["symbols"] if r["symbol"] == "AAA")
    assert not any("missing" in i for i in aaa["issues"])
    frames["AAA"] = frames["AAA"][frames["AAA"].index != pd.Timestamp("2026-06-18")]   # a real gap is still caught
    aaa = next(r for r in quality.check_universe(frames, ["SPY", "AAA"], SESSION)["symbols"] if r["symbol"] == "AAA")
    assert any("1 expected session" in i for i in aaa["issues"])


def test_housekeeping_compresses_prunes_and_api_reads_gz(tsettings, tmp_path):
    import gzip

    from fastapi.testclient import TestClient

    from src.daily import housekeeping, store as st
    from src.web.app import create_app
    reports = Path(tsettings["paths"]["reports_dir"])
    for i in range(5):
        d = reports / f"2026-10-{20 + i:02d}"
        d.mkdir(parents=True)
        (d / "data.json").write_text(json.dumps({"session": {"date": d.name}, "n": i}))
    assert housekeeping.compress_old_reports(reports, keep_plain=2) == 3
    assert (reports / "2026-10-20" / "data.json.gz").exists() and not (reports / "2026-10-20" / "data.json").exists()
    assert st.read_snapshot(reports / "2026-10-20")["n"] == 0 and st.read_snapshot(reports / "2026-10-24")["n"] == 4
    client = TestClient(create_app(tsettings))
    assert "2026-10-20" in client.get("/api/sessions").json()["sessions"]
    assert client.get("/api/snapshot?session=2026-10-20").json()["n"] == 0
    # dated caches, backtests, logs, news
    fin = tmp_path / "finra"
    fin.mkdir()
    (fin / "regsho_20260101.txt").write_text("x")
    (fin / "regsho_20261029.txt").write_text("x")
    assert housekeeping.prune_dated_files(fin, r"regsho_(\d{8})\.txt", "%Y%m%d", 45, date(2026, 10, 30)) == 1
    bt = tmp_path / "backtest"
    for m in range(1, 10):
        (bt / f"2026-{m:02d}-01").mkdir(parents=True)
    assert housekeeping.prune_backtests(bt, keep=6) == 3 and len(list(bt.iterdir())) == 6
    log = tmp_path / "x.log"
    log.write_bytes(b"line\n" * 2_000_000)
    assert housekeeping.trim_log(log, max_bytes=5_000_000, keep_bytes=1_000_000) and log.stat().st_size < 1_100_000
    conn = st.connect(Path(tsettings["paths"]["database"]), Path(tsettings["paths"]["backups_dir"]))
    conn.execute("INSERT INTO news (symbol, url, title, published) VALUES ('A', 'u1', 't', '2020-01-01T00:00:00Z')")
    conn.execute("INSERT INTO news (symbol, url, title, published) VALUES ('A', 'u2', 't', ?)",
                 (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),))
    conn.commit()
    assert housekeeping.prune_news(conn, 180) == 1
    assert conn.execute("SELECT COUNT(*) FROM news").fetchone()[0] == 1
    assert gzip  # imported for clarity of the compressed format


def test_news_host_backoff(tmp_path, monkeypatch):
    from src.catalysts import news as nm
    calls = []

    class R:
        def __init__(self, code):
            self.status_code, self.text = code, ""
    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return R(429)
    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    client = nm.NewsClient(tmp_path)
    assert client._http("https://feeds.finance.yahoo.com/rss/x") is None
    assert client._http("https://feeds.finance.yahoo.com/rss/y") is None     # host paused: no second request
    assert len(calls) == 1 and "feeds.finance.yahoo.com" in client._blocked()
    assert nm.NewsClient(tmp_path)._blocked()                                 # remembered across processes


def test_universe_default_file_fallback(tmp_path, monkeypatch):
    from src.market import universe_builder as ub
    monkeypatch.setattr(ub, "AUTO_PATH", tmp_path / "missing.yaml")
    default = tmp_path / "default.yaml"
    default.write_text("generated_for: '2026-10-31'\nsymbols: {AAA: XLK}\n")
    monkeypatch.setattr(ub, "DEFAULT_AUTO_PATH", default)
    assert ub.load_auto(ub.AUTO_PATH)["symbols"] == {"AAA": "XLK"}
    assert ub.load_auto(tmp_path / "other.yaml") is None


@pytest.mark.parametrize("end", [date(2027, 12, 31), date(2028, 2, 29)])
def test_pipeline_runs_in_future_years(tsettings, end):
    """Year boundaries (New Year 2028 falls on a Saturday and is not observed on Friday Dec 31) and leap days."""
    from src.daily.pipeline import run_daily
    assert calendar.is_trading_day(end)
    res = run_daily(tsettings, session=end, downloader=FakeDownloader(market_frames(tsettings, end=end)),
                    enrichment=FakeEnrichment(), fred_getter=fake_fred, progress=lambda m: None)
    assert res["session"]["date"] == end.isoformat() and res["status"].startswith("success")
    assert any("FOMC dates only up to" in w for w in res["warnings"])      # the FOMC list (2027) has run out
    assert calendar.next_trading_day(date(2027, 12, 31)) == date(2028, 1, 3)


def test_universe_rebuild_refuses_to_shrink(tsettings, tmp_path):
    """A screener / download glitch that returns only a handful of names must not replace the universe."""
    from src.market import universe_builder as ub
    s = copy.deepcopy(tsettings)
    s["universe_filter"].update({"min_beta": -99.0, "min_hv60": 0.0})
    s["data"]["min_history_days"] = 100
    auto = tmp_path / "auto.yaml"
    auto.write_text("generated_for: '2026-09-01'\nsymbols: {" + ", ".join(f"S{i}: XLK" for i in range(20)) + "}\n")
    frames = {"SPY": bars(300, SESSION, seed=1)}
    for i in range(3):
        frames[f"N{i}"] = bars(300, SESSION, seed=20 + i, volume=5e6)
    quotes = [{"symbol": f"N{i}", "regularMarketPrice": 100.0, "averageDailyVolume3Month": 5e6, "marketCap": 5e9}
              for i in range(3)]

    def screener(offset):
        return {"quotes": quotes if offset == 0 else [], "total": len(quotes)}

    def profile(sym):
        return {"sector": "Technology", "industry": "Software", "name": sym}

    def build():
        return ub.build_universe(s, downloader=FakeDownloader(frames), screener=screener, profile=profile,
                                 progress=lambda m: None, path=auto)
    with pytest.raises(RuntimeError, match="data glitch"):
        build()
    assert "S19" in auto.read_text()                                   # the previous list is kept
    auto.write_text("generated_for: '2026-09-01'\nsymbols: {S0: XLK, S1: XLK}\n")
    assert set(build()["symbols"]) == {"N0", "N1", "N2"}               # a genuine small change is accepted


def test_backtest_and_universe_failures_cool_down(tsettings, monkeypatch):
    import os
    import sys
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import daily_run
    from src.daily.pipeline import DailyRunError
    from src.market import universe_builder as ub
    bt = Path(tsettings["paths"]["reports_dir"]).parent / "backtest"
    assert daily_run.backtest_stale(tsettings)                         # never run
    bt.mkdir(parents=True)
    (bt / "last_failure").write_text("x")
    assert not daily_run.backtest_stale(tsettings)                     # failed today: not retried for 3 days
    old = time.time() - 4 * 86400
    os.utime(bt / "last_failure", (old, old))
    assert daily_run.backtest_stale(tsettings)
    # universe rebuild: one failure, then quiet for 3 days (an explicit --rebuild-universe still runs)
    calls = []

    def failing_build(settings, progress=None):
        calls.append(1)
        raise RuntimeError("screener down")

    def stop(*a, **k):
        raise DailyRunError("stop here")
    monkeypatch.setattr(daily_run, "load_terminal_settings", lambda *_: copy.deepcopy(tsettings))
    monkeypatch.setattr(daily_run, "run_daily", stop)
    monkeypatch.setattr(ub, "build_universe", failing_build)
    monkeypatch.setattr(ub, "is_stale", lambda *a: True)
    assert daily_run.main(["--skip-backtest"]) == 3 and len(calls) == 1
    assert (Path(tsettings["paths"]["database"]).parent / ".universe_rebuild_failed").exists()
    assert daily_run.main(["--skip-backtest"]) == 3 and len(calls) == 1
    assert daily_run.main(["--skip-backtest", "--rebuild-universe"]) == 3 and len(calls) == 2


def test_alert_and_open_trade_symbols_keep_updating(tsettings):
    """Names that left the universe (monthly rebuild) but have an alert / open trade still get prices."""
    from src.daily import store as st
    from src.daily.pipeline import run_daily
    from src.portfolio import journal
    conn = st.connect(Path(tsettings["paths"]["database"]), Path(tsettings["paths"]["backups_dir"]))
    journal.add_alert(conn, "ZZZ", "above", 1e9)
    journal.add_trade(conn, {"symbol": "YYY", "direction": "long", "entry_price": 10, "stop": 9})
    conn.close()
    frames = market_frames(tsettings)
    frames["ZZZ"] = bars(520, SESSION, seed=77)
    frames["YYY"] = bars(520, SESSION, seed=78)
    dl = FakeDownloader(frames)
    res = run_daily(tsettings, session=SESSION, downloader=dl, enrichment=FakeEnrichment(), fred_getter=fake_fred,
                    progress=lambda m: None)
    asked = {s for call in dl.calls for s in call[0]}
    assert {"ZZZ", "YYY"} <= asked
    assert "ZZZ" not in {c["symbol"] for side in res["candidates"].values() for c in side}
    conn = sqlite3.connect(tsettings["paths"]["database"])
    assert conn.execute("SELECT COUNT(*) FROM price_data WHERE ticker = 'ZZZ'").fetchone()[0] > 0
    conn.close()


def test_alert_watch_from_uses_new_york_date(tsettings):
    """An alert set in the Hong Kong morning (= the US evening before) must watch the next US session."""
    from src.daily import store as st
    from src.data import market_data
    from src.database import db
    from src.portfolio import journal
    assert journal.watch_from("2026-10-30T02:00:00Z") == "2026-10-29"   # 22:00 ET on the 29th
    assert journal.watch_from("2026-10-30T15:00:00Z") == "2026-10-30"   # mid-session: from the next session
    assert journal.watch_from("2026-10-30T12:00:00Z") == "2026-10-29"   # pre-market: today's session counts
    conn = st.connect(Path(tsettings["paths"]["database"]), Path(tsettings["paths"]["backups_dir"]))
    db.upsert_stock(conn, "AAA")
    db.upsert_prices(conn, "AAA", bars(5, SESSION, seed=3), market_data.SOURCE)
    for created in ("2026-10-30T02:00:00Z", "2026-10-30T15:00:00Z"):
        conn.execute("INSERT INTO price_alerts (created_at, symbol, kind, level) VALUES (?, 'AAA', 'close_above', 1)",
                     (created,))
    conn.commit()
    fired = journal.check_alerts(conn)
    conn.close()
    assert [(a["created_at"], a["triggered_session"]) for a in fired] == [("2026-10-30T02:00:00Z", "2026-10-30")]


def test_news_load_dedupes_same_story_from_two_feeds(tsettings):
    from src.catalysts import news as nm
    from src.daily import store as st
    conn = st.connect(Path(tsettings["paths"]["database"]), Path(tsettings["paths"]["backups_dir"]))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    item = {"tags": [], "sentiment": "positive", "published": now}
    nm.save(conn, "AAA", [{**item, "url": "https://yahoo/1", "title": "Acme beats estimates, raises outlook"},
                          {**item, "url": "https://nasdaq/2", "title": "Acme Beats Estimates, Raises Outlook - Nasdaq"},
                          {**item, "url": "https://yahoo/3", "title": "Acme names a new CFO"}])
    nm.save(conn, "BBB", [{**item, "url": "https://yahoo/1", "title": "Acme beats estimates, raises outlook"}])
    rows = nm.load(conn)
    conn.close()
    assert sorted((r["symbol"], r["url"]) for r in rows) == [("AAA", "https://yahoo/1"), ("AAA", "https://yahoo/3"),
                                                             ("BBB", "https://yahoo/1")] or \
        sorted((r["symbol"], r["url"]) for r in rows) == [("AAA", "https://nasdaq/2"), ("AAA", "https://yahoo/3"),
                                                          ("BBB", "https://yahoo/1")]


def test_sec_cache_never_keeps_a_truncated_zip(tmp_path):
    import io
    import zipfile

    from src.data.sec import SecClient
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("cnsfails.txt", "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n")
    good = buf.getvalue()

    class Resp:
        status_code = 200
        content = good[:20]                            # connection dropped mid-download

        def raise_for_status(self):
            pass

    class Http:
        n = 0

        def get(self, url, headers=None, timeout=None):
            Http.n += 1
            return Resp()
    client = SecClient("Swing Terminal test test@example.com", tmp_path, today=date(2026, 10, 30), session=Http())
    cached = tmp_path / "sec" / "files" / "cnsfails202610a.zip"
    assert client.get_file("https://example.invalid/cnsfails202610a.zip", cached.name) is None
    assert not cached.exists()
    Resp.content = good
    assert client.get_file("https://example.invalid/cnsfails202610a.zip", cached.name) == good and cached.exists()
    cached.write_bytes(b"PK broken")                    # an older version left a broken file behind
    assert client.get_file("https://example.invalid/cnsfails202610a.zip", cached.name) == good and Http.n == 3
    assert not list(cached.parent.glob(".*.tmp"))


def test_atomic_writes_and_tmp_cleanup(tmp_path):
    import os
    import time

    from src.daily import housekeeping, store as st
    target = tmp_path / "data.json"
    st.write_atomic(target, '{"a": 1}')
    st.write_atomic(target, b'{"a": 2}')
    assert json.loads(target.read_text()) == {"a": 2} and not list(tmp_path.glob(".*.tmp"))
    stale = tmp_path / "s" / ".data.json.123.tmp"
    stale.parent.mkdir()
    stale.write_text("half")
    fresh = tmp_path / "s" / ".report.html.456.tmp"
    fresh.write_text("being written")
    old = time.time() - 2 * 86400
    os.utime(stale, (old, old))
    assert housekeeping.prune_tmp(tmp_path) == 1 and fresh.exists() and not stale.exists()


def test_yfinance_self_repair_when_every_download_fails(tsettings, monkeypatch):
    import subprocess
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import daily_run
    broken = {"quality": {"update": {"failed": [f"S{i}" for i in range(40)], "incremental": ["SPY"], "full": []}}}
    fine = {"quality": {"update": {"failed": ["S1"], "incremental": [f"S{i}" for i in range(40)], "full": []}}}
    assert daily_run.provider_broken(broken) and not daily_run.provider_broken(fine)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert daily_run.maybe_upgrade_yfinance(tsettings)
    assert calls[0][-1] == "yfinance" and "--upgrade" in calls[0]
    data_dir = Path(tsettings["paths"]["database"]).parent
    assert (data_dir / ".deps_updated").exists()                     # the launcher restarts the dashboard
    assert not daily_run.maybe_upgrade_yfinance(tsettings) and len(calls) == 1   # at most once a day


def test_user_data_export_and_restore(tsettings):
    from src.daily import housekeeping, store as st
    from src.portfolio import journal
    conn = st.connect(Path(tsettings["paths"]["database"]), Path(tsettings["paths"]["backups_dir"]))
    journal.add_trade(conn, {"symbol": "AAA", "direction": "long", "entry_price": 10, "stop": 9, "notes": "first"})
    journal.add_alert(conn, "BBB", "below", 50)
    backups = Path(tsettings["paths"]["backups_dir"])
    for day in range(1, 20):
        housekeeping.export_user_data(conn, backups, date(2026, 10, day), keep=14)
    files = sorted(backups.glob("user_data_*.json"))
    assert len(files) == 14 and files[-1].name == "user_data_20261019.json"
    conn.execute("DELETE FROM trade_journal")
    conn.execute("DELETE FROM price_alerts")
    conn.commit()
    assert housekeeping.restore_user_data(conn, files[-1]) == 2
    assert housekeeping.restore_user_data(conn, files[-1]) == 0            # idempotent
    assert conn.execute("SELECT notes FROM trade_journal").fetchone()[0] == "first"
    assert conn.execute("SELECT symbol FROM price_alerts").fetchone()[0] == "BBB"
    conn.close()


def test_delisted_symbol_gets_its_own_warning(tsettings):
    from src.daily import store as st
    from src.daily.pipeline import run_daily
    from src.data import market_data
    from src.database import db
    conn = st.connect(Path(tsettings["paths"]["database"]), Path(tsettings["paths"]["backups_dir"]))
    db.upsert_stock(conn, "DDD")
    db.upsert_prices(conn, "DDD", bars(300, date(2026, 9, 1), seed=5), market_data.SOURCE)   # last bar 2026-09-01
    conn.close()
    frames = market_frames(tsettings)
    frames.pop("DDD")                                                     # the provider no longer has it
    res = run_daily(tsettings, session=SESSION, downloader=FakeDownloader(frames), enrichment=FakeEnrichment(),
                    fred_getter=fake_fred, progress=lambda m: None)
    assert any(w.startswith("No data for 3+ weeks") and "DDD" in w for w in res["warnings"])
    assert not any(w.startswith("No fresh data") and "DDD" in w for w in res["warnings"])
    assert "DDD" not in {c["symbol"] for side in res["candidates"].values() for c in side}


def test_chart_library_is_served_locally(tsettings):
    from fastapi.testclient import TestClient

    from src.web.app import create_app
    client = TestClient(create_app(tsettings))
    page = client.get("/").text
    assert "unpkg.com" not in page and "/static/vendor/lightweight-charts-4.1.3" in page
    lib = client.get("/static/vendor/lightweight-charts-4.1.3.standalone.production.js")
    assert lib.status_code == 200 and "LightweightCharts" in lib.text[:200000]


def test_exit_rule_replaces_plan_levels(tsettings):
    from src.scanner.plan import plan_row
    f = build_features(bars(400, SESSION, seed=2, breakout_last=True, volume=5e6), bars(400, SESSION, seed=10)["adj_close"], None)
    sc = {**tsettings["scanner"], "exit_rules": {
        "breakout": {"stop": "atr1.5", "target": "trail2.5", "horizon": 15, "name": "x", "oos": {"n": 900, "avg_r": 0.21, "win_rate": 0.5}},
        "pullback": {"stop": "atr2.0", "target": "r2", "horizon": 10, "name": "y", "oos": {}}}}
    p = plan_row(f, "breakout", sc, 7)
    a, e = p["atr"], p["entry"]
    assert abs(p["stop"] - (e - 1.5 * a)) < 1e-9 and p["target1"] is None and p["target2"] is None
    assert p["trail_atr"] == 2.5 and p["horizon_days"] == 15 and p["reward_risk"] is None
    assert "+0.21R" in p["target_basis"]
    # trailing exits carry concrete prices, so nobody has to compute "highest high - 2.5 ATR" by hand
    risk = e - p["stop"]
    assert p["trail_distance"] == pytest.approx(2.5 * a)
    assert p["trail_start"] == pytest.approx(p["stop"] + 2.5 * a) and p["breakeven_high"] == pytest.approx(e + 2.5 * a)
    lad = {r["r"]: r for r in p["trail_ladder"]}
    assert set(lad) == {1, 2, 3, 4, 6}
    for n, r in lad.items():
        assert r["high"] == pytest.approx(e + n * risk) and r["stop"] == pytest.approx(max(p["stop"], r["high"] - 2.5 * a))
        assert r["locked_r"] == pytest.approx((r["stop"] - e) / risk)
    assert f"{p['trail_start']:.2f}" in p["target_basis"]
    q = plan_row(f, "pullback", sc, 7)
    assert abs(q["target1"] - (q["entry"] + 2 * 2.0 * q["atr"])) < 1e-9 and abs(q["reward_risk"] - 2.0) < 1e-9
    assert q["trail_ladder"] == [] and q["trail_start"] is None      # a fixed target needs no ladder
    r = plan_row(f, "momentum", sc, 7)                     # no rule: the structural plan is unchanged
    assert "exit_rule" not in r and r["target1"] is not None


def test_trailing_stop_multiple_in_simulation():
    from src.backtest.research import simulate
    n = 3
    sig = pd.DataFrame({"symbol": ["A"] * n, "setup": ["x"] * n, "long": [True] * n, "date": pd.to_datetime(["2025-01-02"] * n),
                        "ref": [100.0] * n, "atr": [2.0] * n, "structural": [97.0] * n, "resistance": [104.0] * n,
                        "planned_rr": [1.0] * n, "swing": [96.0] * n, "chip_stop": [np.nan] * n, "chip_target": [np.nan] * n,
                        "regime": ["n/a"] * n, "score": [50.0] * n, "tier": ["B"] * n, "conditional": [False] * n})
    # price runs to 110 then falls back to 104: a 2-ATR trail exits at 106, a 3-ATR trail at 104 (close)
    H = np.tile([101, 105, 110, 109, 107, 105], (n, 1)).astype(float)
    L = np.tile([99.5, 101, 106, 107, 105.5, 104], (n, 1)).astype(float)
    O = np.tile([100, 101, 106, 109, 107, 105], (n, 1)).astype(float)
    C = np.tile([100.5, 104, 109, 108, 106, 104], (n, 1)).astype(float)
    data = {"signals": sig, "O": O, "H": H, "L": L, "C": C}
    t2 = simulate(data, "atr1.5", "trail2", 6, 0.0)
    t3 = simulate(data, "atr1.5", "trail", 6, 0.0)
    assert abs(t2["r"].iloc[0] - (106 - 100) / 3) < 1e-9 and t2["outcome"].iloc[0] == "stop"
    assert abs(t3["r"].iloc[0] - (104 - 100) / 3) < 1e-9


def test_strategy_lab_end_to_end_small(tsettings):
    """The lab runs on synthetic data: exits chosen per setup, every trade labelled, models walk-forward tested."""
    from src.backtest import strategy_lab as lab
    from src.scanner.setups import detect_setups
    end = date(2026, 10, 30)
    spy = bars(1300, end, seed=10)
    feats, sets = {}, {}
    for i in range(6):
        f = build_features(bars(1300, end, drift=0.0006, vol=0.02, seed=40 + i, volume=5e6), spy["adj_close"], None)
        feats[f"S{i}"], sets[f"S{i}"] = f, detect_setups(f)
    regime_s = pd.Series(0.0, index=spy.index)
    data = lab.collect(feats, sets, spy["close"], regime_s, tsettings, techniques=False, workers=1)
    sig = data["signals"]
    assert (sig["setup"] == "baseline").sum() > 100 and (sig["setup"] != "baseline").sum() > 50
    assert data["O"].shape == (len(sig), lab.H_LAB)
    assert any(c.startswith("d_") for c in sig.columns) and "m_regime" in sig.columns
    res = lab.run_lab(data, tsettings)
    assert res["exits"]["chosen"]["all"]["horizon"] in lab.LAB_HORIZONS
    assert res["n_features"] > 50 and res["verdict"]
    assert set(res["walk_forward"]["models"]) <= {"ic", "ridge"}


def test_setup_bonus_from_strategy_lab_edge():
    from src.scanner.scoring import backtest_setup_bonus
    cfg = {"setup_bonus_from_backtest": {"points_per_r": 150, "min": -10, "max": 20, "min_trades": 200}}
    lab = {"chosen": {"all": {}, "breakout": {"is": {"n": 300, "avg_r": 0.30}, "oos": {"n": 200, "avg_r": 0.25},
                                              "baseline_is": {"n": 3000, "avg_r": 0.20}, "baseline_oos": {"n": 2000, "avg_r": 0.20}},
                      "momentum": {"is": {"n": 300, "avg_r": 0.18}, "oos": {"n": 200, "avg_r": 0.20},
                                   "baseline_is": {"n": 3000, "avg_r": 0.20}, "baseline_oos": {"n": 2000, "avg_r": 0.20}}}}
    bonus, src = backtest_setup_bonus({"by_setup": {"_": {}}, "session": "2026-10-30", "lab": lab}, cfg)
    # momentum's edge (-0.012R) is inside the 0.05R noise band -> 0
    assert src.startswith("strategy lab") and bonus["breakout"] == 12.0 and bonus["momentum"] == 0.0
    bonus, _ = backtest_setup_bonus({"by_setup": {"_": {}}, "session": "x", "lab": lab},
                                    {"setup_bonus_from_backtest": {**cfg["setup_bonus_from_backtest"], "min_edge_r": 0}})
    assert bonus["momentum"] == -1.8


def test_shipped_strategy_lab_bonus_zeroes_setups_without_edge(tsettings):
    from src.daily.pipeline import LAB_DEFAULT
    from src.scanner.scoring import backtest_setup_bonus
    lab = json.loads(LAB_DEFAULT.read_text())
    bonus, _ = backtest_setup_bonus({"by_setup": {"_": {}}, "session": "x", "lab": lab["exits"]},
                                    load_terminal_settings()["scanner"])
    assert bonus["crash_rebound"] == 20 and bonus["squeeze"] > 5
    assert all(bonus[s] == 0 for s in ("breakout", "reclaim", "momentum", "pullback"))


def test_strategy_lab_default_copy_is_used_until_a_local_run(tmp_path):
    from src.daily.pipeline import LAB_DEFAULT, load_strategy_lab
    shipped = load_strategy_lab(tmp_path / "none")
    assert LAB_DEFAULT.exists() and shipped["shipped_default"] and shipped["exits"]["chosen"]["all"]["horizon"]
    local = tmp_path / "2026-12-01"
    local.mkdir()
    (local / "strategy_lab.json").write_text(json.dumps({"session": "2026-12-01", "exits": {"chosen": {}}}))
    assert load_strategy_lab(tmp_path)["session"] == "2026-12-01"


def test_discovery_signals_and_turn_of_month_on_the_latest_day():
    from src.backtest import discovery as dc
    end = date(2026, 10, 30)                     # last session of October
    f = build_features(bars(400, end, seed=3, volume=5e6), bars(400, end, seed=10)["adj_close"], None)
    sig = dc.stock_signals(f)
    assert {"rsi2", "ibs", "down3", "event_drift", "gap_reversal", "high52", "nr7"} <= set(sig)
    m = dc.market_signals({}, {}, f.index)
    assert bool(m["turn_month"].iloc[-1])                     # known from the calendar, not from tomorrow's bar
    assert not bool(m["turn_month"].iloc[-2])
    # an RSI(2) signal needs a short sharp dip above EMA200
    assert sig["rsi2"].dtype == bool and sig["rsi2"].sum() >= 0


def test_discovery_study_small_and_index_timing(tsettings):
    from src.backtest import discovery as dc
    end = date(2026, 10, 30)
    spy = bars(1300, end, seed=10, volume=5e6)
    feats = {f"S{i}": build_features(bars(1300, end, drift=0.0006, vol=0.02, seed=50 + i, volume=5e6), spy["adj_close"], None)
             for i in range(5)}
    prices = {"SPY": spy, "^VIX": bars(1300, end, drift=0.0, vol=0.05, seed=99, base=18)}
    regime_s = pd.Series("Neutral", index=spy.index)
    data = dc.collect(feats, prices, regime_s)
    res = dc.run(data, feats, tsettings)
    assert res["verdict"] and res["overnight"]["out_of_sample"]["days"] > 100
    tested = [s for s in res["styles"].values() if not s.get("skipped")]
    assert tested and all("t_clustered" in s["oos"] or not s["oos"].get("n") for s in tested)
    qqq = bars(1700, end, seed=11, volume=5e6)          # the index study needs >= 1500 sessions
    study = {"index": dc.index_study({"SPY": bars(1700, end, seed=10, volume=5e6), "QQQ": qqq}, 0.001)}
    assert "QQQ" in study["index"]["instruments"]
    fake = {"name": "Turn of the month", "stop": "atr2.0", "target": "r2", "horizon": 20, "passes": True, "edge_is": 0.07,
            "edge_oos": 0.06, "is": {"n": 150}, "oos": {"n": 160, "avg_r": 0.25, "win_rate": 0.53, "t_clustered": 2.2},
            "per_year": {"2024": 0.1, "2025": -0.1}}
    study = {"index": {"instruments": {"QQQ": {"first": "1999-03-10", "last": "2026-10-30", "styles": {"turn_month": fake}}}}}
    rows = dc.index_timing_today({"SPY": spy, "QQQ": qqq}, study)
    assert len(rows) == 1 and rows[0]["active"] and rows[0]["stop"] < rows[0]["close"] < rows[0]["target"]
    assert rows[0]["years_positive"] == 1 and rows[0]["period"] == "1999-2026"


def _form345_zip(rows):
    import io
    import zipfile
    sub = ["ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERTRADINGSYMBOL"]
    tr = ["ACCESSION_NUMBER\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD"]
    own = ["ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE"]
    for k, (sym, day, owner, rel, title, code, shares, price, form) in enumerate(rows):
        acc = f"0000-{k}"
        sub.append(f"{acc}\t{day}\t{form}\t{sym}")
        tr.append(f"{acc}\t{code}\t{shares}\t{price}\tA")
        own.append(f"{acc}\t{owner}\t{rel}\t{title}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, lines in (("SUBMISSION.tsv", sub), ("NONDERIV_TRANS.tsv", tr), ("REPORTINGOWNER.tsv", own)):
            zf.writestr(name, "\n".join(lines) + "\n")
    return buf.getvalue()


def test_insider_events_from_form4_data_set():
    from src.backtest import insider
    content = _form345_zip([
        ("ABC", "02-MAR-2024", "1", "Director", "", "P", 1000, 20, "4"),
        ("ABC", "05-MAR-2024", "2", "Officer", "CFO", "P", 10000, 20, "4"),       # CFO, $200k
        ("ABC", "20-MAR-2024", "3", "Director,Officer", "COO", "P", 500, 20, "4"),
        ("ABC", "21-MAR-2024", "4", "Director", "", "S", 9000, 20, "4"),          # a sale: ignored
        ("ABC", "22-MAR-2024", "5", "TenPercentOwner", "", "P", 90000, 20, "4"),  # 10% owner: ignored
        ("XYZ", "10-APR-2024", "6", "Director", "", "P", 100, 20, "4"),           # $2,000: below the floor
        ("XYZ", "11-APR-2024", "7", "Officer", "Chief Executive Officer", "P", 3000, 50, "3"),  # Form 3: ignored
    ])
    buys = insider.parse_quarter(content)
    assert len(buys) == 5 and set(buys["sym"]) == {"ABC", "XYZ"}
    ev = insider.insider_events(buys)
    got = {(r.sym, r.style): r.filing_date.strftime("%Y-%m-%d") for r in ev.itertuples()}
    assert got == {("ABC", "ins_ceo_100k"): "2024-03-05", ("ABC", "ins_cluster3"): "2024-03-20"}
    # the CFO + director already make $220k in 30 days -> the >= 2 buyers / $250k style has not fired
    assert ("ABC", "ins_cluster2_250k") not in got


def test_insider_study_small_and_shipped_copy(tsettings, tmp_path):
    from src.backtest import insider
    from src.daily.pipeline import INSIDER_DEFAULT, load_insider
    end = date(2026, 10, 30)
    spy = bars(1300, end, seed=10, volume=5e6)
    prices = {f"S{i}": bars(1300, end, drift=0.0005, vol=0.02, seed=60 + i, base=40, volume=5e6) for i in range(4)}
    days = spy.index[300:-30:9]
    ev = pd.DataFrame([{"sym": f"S{k % 4}", "style": list(insider.STYLE_NAMES)[k % 3], "filing_date": d, "n_buyers": 3,
                        "value_30d": 3e5} for k, d in enumerate(days)])
    data = insider.collect(ev, prices, spy["adj_close"], pd.Series("Neutral", index=spy.index), {"S0"})
    sig = data["signals"]
    assert (sig["setup"] == "baseline").sum() > 100 and (sig["setup"] != "baseline").sum() > 50
    first = sig[sig["setup"] != "baseline"].iloc[0]
    assert first["date"] <= ev.loc[ev["sym"] == first["symbol"], "filing_date"].min() + pd.Timedelta(days=4)
    res = insider.run(data, 0.001)
    assert res["verdict"] and res["robustness"] and set(res["styles"]) == set(insider.STYLE_NAMES)
    pc = insider.period_check(data, 0.001, style=list(insider.STYLE_NAMES)[0], split="2024-01-01")
    assert {"rows", "passes", "edge_min", "main_exit"} <= set(pc) and isinstance(pc["passes"], bool)
    shipped = load_insider(tmp_path)
    assert INSIDER_DEFAULT.exists() and shipped["styles"] and not any(s.get("passes") for s in shipped["styles"].values())
    pc = shipped["period_check"]
    assert pc["passes"] and {r["period"] for r in pc["rows"]} == {"before", "after"} and min(r["edge"] for r in pc["rows"]) > 0


def test_regime_split_rule_and_shipped_result(tsettings):
    from src.backtest import strategy_lab as lab
    from src.daily.pipeline import LAB_DEFAULT
    from src.scanner.setups import detect_setups
    end = date(2026, 10, 30)
    spy = bars(1300, end, seed=10)
    feats, sets = {}, {}
    for i in range(5):
        f = build_features(bars(1300, end, drift=0.0006, vol=0.02, seed=70 + i, volume=5e6), spy["adj_close"], None)
        feats[f"S{i}"], sets[f"S{i}"] = f, detect_setups(f)
    data = lab.collect(feats, sets, spy["close"], pd.Series(0.0, index=spy.index), tsettings, techniques=False, workers=1)
    label = pd.Series(np.where(np.arange(len(spy)) % 2, "Bullish", "Bearish"), index=spy.index)
    rule = {"stop": "atr1.5", "target": "trail3", "horizon": 20, "name": "x"}
    rs = lab.regime_split(data, {"momentum": rule, "pullback": rule}, label, data["signals"]["date"].quantile(0.6), 0.001)
    for v in rs["setups"].values():
        assert {r["regime"] for r in v["rows"]} >= {"Bullish", "Bearish", "kept", "dropped", "all"}
        assert v["passes"] == (all(e is not None and e >= 0.05 for e in v["kept_edge"])
                               and all(e is not None and e < 0 for e in v["dropped_edge"]))
    shipped = json.loads(LAB_DEFAULT.read_text())["regime_split"]
    assert set(shipped["setups"]) == {"momentum", "pullback"} and not any(v["passes"] for v in shipped["setups"].values())


# ---------------------------------------------------------------------- scaling-in study
def _scale_data(rows, bars):
    from src.backtest import research  # noqa: F401
    sig = pd.DataFrame(rows)
    sig["date"] = pd.to_datetime(sig["date"])
    arr = np.array(bars, dtype=float)
    return {"signals": sig, "O": arr[:, 0], "H": arr[:, 1], "L": arr[:, 2], "C": arr[:, 3]}


def test_scale_in_single_matches_research_simulate():
    from src.backtest.research import simulate
    from src.backtest.scale_in import simulate_scaled
    rng = np.random.default_rng(3)
    rows, bars = [], []
    for i in range(60):
        c = 100 + rng.normal(0, 3, 21).cumsum()
        o = c + rng.normal(0, 0.5, 21)
        h, lo = np.maximum(o, c) + rng.uniform(0, 2, 21), np.minimum(o, c) - rng.uniform(0, 2, 21)
        rows.append({"symbol": f"S{i}", "setup": "crash_rebound", "date": "2024-01-02", "ref": c[0], "atr": 2.0,
                     "conditional": i % 4 == 0, "structural": c[0] - 3, "swing": c[0] - 3})
        bars.append([o[1:], h[1:], lo[1:], c[1:]])
    data = _scale_data(rows, bars)
    for target in ("trail3", "r3"):
        one = simulate_scaled(data, 1.5, target, 20, 0.001, [(1.0, "first", 0, 0)])
        d2 = {**data, "signals": data["signals"].assign(long=True, score=np.nan, tier=np.nan, planned_rr=np.nan,
                                                         regime=np.nan)}
        ref = simulate(d2, "atr1.5", target, 20, 0.001)
        assert len(one) == len(ref)
        assert np.allclose(one["r"].to_numpy(), ref["r"].to_numpy())


def test_scale_in_dip_tranche_sizing_and_budget():
    from src.backtest.scale_in import simulate_scaled
    # entry 100 (open day 0), ATR 2, stop 97; day 1 dips to 98 (limit 99 fills), then closes 110 at the time stop
    o = [100, 100, 104, 106, 108]
    h = [101, 101, 105, 107, 110]
    lo = [99.5, 98, 103, 105, 107]
    c = [100.5, 100, 104, 106, 110]
    data = _scale_data([{"symbol": "X", "setup": "squeeze", "date": "2024-01-02", "ref": 100.0, "atr": 2.0,
                         "conditional": False}], [[o, h, lo, c]])
    t = simulate_scaled(data, 1.5, "r9", 5, 0.0, [(0.5, "first", 0, 0), (0.5, "dip", 0.5, 3)])
    # tranche 1: 0.5/3 shares from 100; tranche 2: 0.5/2 shares from 99; exit 110
    assert t["tranches"].iat[0] == 2 and t["risk_used"].iat[0] == pytest.approx(1.0)
    assert t["r"].iat[0] == pytest.approx(0.5 / 3 * 10 + 0.5 / 2 * 11)
    # the same day as a stop hit: the dip tranche fills and is stopped (conservative), total loss = 1R
    lo2 = [99.5, 96, 103, 105, 107]
    data2 = _scale_data([{"symbol": "X", "setup": "squeeze", "date": "2024-01-02", "ref": 100.0, "atr": 2.0,
                          "conditional": False}], [[o, h, lo2, c]])
    t2 = simulate_scaled(data2, 1.5, "r9", 5, 0.0, [(0.5, "first", 0, 0), (0.5, "dip", 0.5, 3)])
    assert t2["r"].iat[0] == pytest.approx(-1.0)


def test_scale_in_shipped_copy():
    d = json.loads((Path(__file__).resolve().parents[1] / "config" / "scale_in.default.json").read_text())
    assert set(d["setups"]) == {"crash_rebound", "squeeze", "breakout"}
    assert all(v["passed"] is False for v in d["setups"].values())


# ---------------------------------------------------------------- 2x ETF + scaling-in study
def _lev_data(n=400, seed=3):
    """Hand-made forward paths for the leverage engine (no setups needed)."""
    rng = np.random.default_rng(seed)
    c0 = np.full(n, 100.0)
    C = c0[:, None] * np.cumprod(1 + rng.normal(0.002, 0.03, (n, 20)), axis=1)
    prev = np.concatenate([c0[:, None], C[:, :-1]], axis=1)
    O = prev * (1 + rng.normal(0, 0.01, (n, 20)))
    H = np.maximum(O, C) * (1 + np.abs(rng.normal(0, 0.01, (n, 20))))
    L = np.minimum(O, C) * (1 - np.abs(rng.normal(0, 0.01, (n, 20))))
    dates = pd.bdate_range("2023-01-02", periods=n)
    sig = pd.DataFrame({"symbol": "AAA", "setup": np.where(np.arange(n) % 3 == 0, "crash_rebound", "baseline"), "date": dates,
                        "c0": c0, "ref": c0, "atr": 3.0, "structural": c0 - 4.5, "conditional": False, "cal0": np.arange(n) + 1})
    return {"signals": sig, "O": O, "H": H, "L": L, "C": C}, pd.bdate_range("2023-01-02", periods=n + 30)


def test_leverage_engine_matches_research_simulate_for_one_entry():
    from src.backtest import leverage as lv
    from src.backtest.research import simulate
    d, _ = _lev_data()
    s = d["signals"].assign(long=True, swing=np.nan, regime=np.nan, score=np.nan, tier=np.nan, planned_rr=np.nan)
    for stop, ek, ep, tm in ((1.5, "trail", 3.0, "trail3"), (2.0, "r", 3.0, "r3")):
        old = simulate({"signals": s, **{k: d[k] for k in "OHLC"}}, f"atr{stop}", tm, 20, 0.001)
        new = lv.simulate_plan(d, stop, ek, ep, 20, slip=0.001, sizing="actual", marks=True)
        assert len(old) == len(new["trades"])
        assert np.allclose(old["r"].to_numpy(), new["trades"]["r"].to_numpy(), atol=1e-9)
        assert np.allclose(new["pnl"][:, -1], new["trades"]["r"].to_numpy())


def test_synthetic_etf_is_twice_the_daily_move_and_risk_budget_holds():
    from src.backtest import leverage as lv
    d, _ = _lev_data(60)
    e = lv.to_etf(d, 2.0)                                  # no cost: exact 2x of each day's close-to-close move
    prev = np.concatenate([d["signals"]["c0"].to_numpy()[:, None], d["C"][:, :-1]], axis=1)
    eprev = np.concatenate([d["signals"]["c0"].to_numpy()[:, None], e["C"][:, :-1]], axis=1)
    assert np.allclose(e["C"] / eprev - 1, 2 * (d["C"] / prev - 1))
    assert np.allclose(e["L"] / eprev - 1, 2 * (d["L"] / prev - 1))
    assert np.allclose(e["signals"]["atr"], 6.0)
    # volatility decay: up 10% then down 10% -> the 2x ETF loses more than 2x the stock
    assert (1.2 * 0.8) - 1 < 2 * ((1.1 * 0.9) - 1)
    # scaling in: whatever fills, the loss at the initial stop never exceeds the 1R budget (no gaps here)
    calm = {**d, "O": d["C"] * 0 + 100.0, "H": d["C"] * 0 + 100.5, "L": d["C"] * 0 + 99.5, "C": d["C"] * 0 + 100.0}
    t = lv.simulate_plan(calm, 0.1, "trail", 3.0, 20, [("now", 0, 0.5), ("down", 0.05, 0.5)], slip=0.0)["trades"]
    assert (t["r"] >= -1.0 - 1e-9).all() and t["n_fills"].max() == 2


def test_dip_add_never_fills_on_the_target_bar():
    from src.backtest import leverage as lv
    d, _ = _lev_data(1)
    one = {"signals": d["signals"], "O": np.full((1, 20), 100.0), "H": np.full((1, 20), 100.5), "L": np.full((1, 20), 99.5),
           "C": np.full((1, 20), 100.0)}
    one["H"][0, 2], one["L"][0, 2] = 120.0, 98.0           # day 2 reaches the target AND the dip level
    t = lv.simulate_plan(one, 1.0, "r", 2.0, 20, [("now", 0, 0.5), ("down", 0.5, 0.5)], slip=0.0)["trades"]
    assert t["outcome"].iat[0] == "target" and t["n_fills"].iat[0] == 1   # the dip-buy could have come after the target


def test_leverage_portfolio_and_study_small():
    from src.backtest import leverage as lv
    d, cal = _lev_data()
    a = lv.simulate_plan(d, 1.5, "trail", 3.0, 20, slip=0.001, marks=True)
    p = lv.portfolio(a["trades"], a["pnl"], a["cap"], cal, risk=0.01, max_positions=5)
    assert p["taken"] > 0 and p["max_invested"] <= 1.0 + 1e-9 and np.isfinite(p["cagr"])
    spy = pd.Series(np.linspace(100, 130, len(cal)), index=cal)
    res = lv.run_study(d, cal, spy, cost=0.15, seeds=2, max_positions=5)
    assert res["trades"]["crash_rebound"]["stops"] and res["scale"]["crash_rebound"]["rows"]
    assert {r["code"] for r in res["portfolio"]["all"]} == set(lv.PORTFOLIO_VARIANTS)
    assert res["verdict"] and res["benchmarks"]


def test_shipped_leverage_study_loads(tmp_path):
    from src.daily.pipeline import load_leverage
    d = load_leverage(tmp_path)
    assert d and d["trades"] and d["verdict"]


def test_ceo_cfo_buy_bonus(tsettings):
    from src.positioning.positioning import insider_summary
    from src.scanner.scoring import apply_enrichment, finalize
    as_of = date(2026, 10, 30)
    table = pd.DataFrame([
        {"Start Date": "2026-10-27", "Insider": "DOE JANE", "Position": "Chief Executive Officer", "Text": "Purchase at price 20.00 per share.",
         "Transaction": "", "Shares": 10000, "Value": 200000},
        {"Start Date": "2026-10-28", "Insider": "ROE RICH", "Position": "Vice President", "Text": "Purchase at price 20.00 per share.",
         "Transaction": "", "Shares": 50000, "Value": 1000000},                       # a VP: not counted
        {"Start Date": "2026-09-01", "Insider": "POE PAT", "Position": "Chief Financial Officer", "Text": "Purchase at price 20.00 per share.",
         "Transaction": "", "Shares": 50000, "Value": 1000000},                       # too old
    ])
    s = insider_summary(table, as_of)
    assert s["exec_buy"]["insider"] == "DOE JANE" and s["exec_buy"]["age_days"] == 3
    assert insider_summary(table.iloc[1:], as_of)["exec_buy"] is None
    cfg = load_terminal_settings()["scanner"]["insider_exec_buy"]
    assert cfg["bonus"] == 10 and cfg["min_value"] == 100000
    st = copy.deepcopy(tsettings)
    st["scanner"]["insider_exec_buy"] = cfg
    c = {"symbol": "AAA", "setup": "breakout", "direction": "long", "components": {k: 50.0 for k in st["weights"]},
         "penalties": {}, "supporting": [], "contradicting": []}
    finalize(c, st)
    before = c["score"]
    apply_enrichment(c, {"positioning": {"insider_90d": s}}, st)
    assert c["bonuses"] == {"insider_exec_buy": 10.0} and c["score"] == pytest.approx(before + 10)
    assert any("Chief Executive Officer bought" in x for x in c["supporting"])


def test_live_strategy_entry_rules_and_evidence(tsettings):
    from src.scanner.plan import ENTRY_RULE, EVIDENCE, plan_row
    spy = bars(400, seed=99)["adj_close"]
    f = build_features(_crash_frame(), spy, None)
    i = int(np.flatnonzero(detect_setups(f)["setup_crash_rebound"].to_numpy())[0])
    plan = plan_row(f.iloc[: i + 1], "crash_rebound", tsettings["scanner"], 10)
    assert "next open, not later" in plan["entry_rule"] and "t below 2.5" in plan["evidence"]
    assert set(ENTRY_RULE) == set(EVIDENCE) == {"crash_rebound", "squeeze", "breakout"}
    assert "Unproven" in EVIDENCE["breakout"] and "Buy-stop" in ENTRY_RULE["squeeze"]
    assert load_terminal_settings()["scanner"]["fresh_only_setups"] == ["crash_rebound"]
    src = (Path(__file__).resolve().parents[1] / "src" / "daily" / "pipeline.py").read_text()
    assert 'sc.get("fresh_only_setups")' in src and '.iloc[-2]' in src


def test_exec_buy_prefers_the_most_recent_buy():
    from datetime import date as _d
    from src.positioning.positioning import exec_buy
    buys = pd.DataFrame({"date": pd.to_datetime(["2026-10-20", "2026-10-29"]), "Position": ["Chief Executive Officer", "CFO"],
                         "Value": [5_000_000, 150_000], "Insider": ["A", "B"]})
    eb = exec_buy(buys, _d(2026, 10, 30))
    assert eb["insider"] == "B" and eb["age_days"] == 1


def test_company_profile_summary_and_ceo():
    from src.daily.pipeline import company_profile
    info = {"longName": "Acme Corp", "sector": "Technology", "industry": "Software", "city": "Austin", "country": "United States",
            "longBusinessSummary": "Acme   makes\nwidgets. " + "word " * 600,
            "companyOfficers": [{"name": "Jane Roe", "title": "CFO"}, {"name": "John Doe", "title": "Co-Founder, CEO & Director"}]}
    p = company_profile(info)
    assert p["longName"] == "Acme Corp" and p["city"] == "Austin"
    assert p["summary"].startswith("Acme makes widgets.") and p["summary"].endswith("…") and len(p["summary"]) <= 1601
    assert p["ceo"] == {"name": "John Doe", "title": "Co-Founder, CEO & Director"}
    empty = company_profile({})
    assert empty["summary"] is None and empty["ceo"] is None


# ======================================================================= dashboard login
def test_password_hash_and_cookie_token():
    from src.web import auth
    stored = auth.hash_password("correct horse", iterations=1000)
    assert auth.verify_password("correct horse", stored) and not auth.verify_password("wrong", stored)
    assert not auth.verify_password("x", "garbage")
    key = "ab" * 32
    token = auth.make_token(key, now=1_000_000)
    assert auth.check_token(key, token, now=1_000_100)
    assert not auth.check_token(key, token, now=1_000_000 + auth.IDLE_SECONDS + 1)   # idle timeout
    # renewal slides the idle window but never past the 12 h cap measured from login
    renewed = auth.make_token(key, now=1_000_000 + 25 * 60, started=1_000_000)
    assert auth.check_token(key, renewed, now=1_000_000 + 50 * 60)
    late = auth.make_token(key, now=1_000_000 + auth.MAX_SESSION_SECONDS - 60, started=1_000_000)
    assert not auth.check_token(key, late, now=1_000_000 + auth.MAX_SESSION_SECONDS + 1)
    assert not auth.check_token(key, "v1.9999999999." + "0" * 64, now=1_000_100)  # old 90-day tokens are dead
    assert not auth.check_token("cd" * 32, token, now=1_000_100)                # other key (password changed)
    assert not auth.check_token(key, token[:-1] + ("0" if token[-1] != "0" else "1"), now=1_000_100)
    assert auth.safe_next("/#/ticker/AAA") == "/#/ticker/AAA"
    assert auth.safe_next("//evil.example") == "/" and auth.safe_next("https://evil.example") == "/"
    h = lambda **kw: [(k.replace("_", "-").encode(), v.encode()) for k, v in kw.items()]  # noqa: E731
    assert auth.is_direct_local("127.0.0.1", h(host="127.0.0.1:8000"))
    assert auth.is_direct_local("::1", h(host="localhost:8000"))
    assert not auth.is_direct_local("127.0.0.1", h(host="127.0.0.1:8000", x_forwarded_for="100.64.0.2"))
    assert not auth.is_direct_local("127.0.0.1", h(host="swing.tail1234.ts.net"))
    assert not auth.is_direct_local("127.0.0.1", h(host="127.0.0.1", tailscale_user_login="a@b.c"))
    assert not auth.is_direct_local("100.64.0.2", h(host="127.0.0.1:8000"))
    assert auth.client_address("127.0.0.1", h(x_forwarded_for="1.2.3.4, 100.64.0.9")) == "100.64.0.9"
    lock = auth.Lockout()
    for i in range(auth.GLOBAL_MAX_FAILURES):
        lock.fail(f"10.0.0.{i}", now=100)
    assert lock.blocked("10.9.9.9", now=101) and not lock.blocked("10.9.9.9", now=100 + auth.LOCKOUT_SECONDS + 1)


def test_dashboard_login_guard(tsettings):
    from fastapi.testclient import TestClient

    from src.web import auth
    from src.web.app import auth_file_for, create_app
    client = TestClient(create_app(tsettings))
    assert client.get("/api/sessions").status_code == 200          # no password set: open, as before
    auth.set_password(auth_file_for(tsettings), "s3cret-pass")
    assert client.get("/api/sessions").status_code == 401           # picked up without a restart
    assert client.post("/api/journal", json={"symbol": "AAA"}).status_code == 401
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")
    assert client.get("/static/app.js").status_code == 200          # assets carry no data
    assert client.get("/login").status_code == 200
    assert client.post("/login", data={"password": "nope"}).status_code == 401
    r = client.post("/login", data={"password": "s3cret-pass", "next": "//evil.example"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert "max-age" not in cookie and "expires" not in cookie      # session cookie: not remembered by the browser
    r = client.get("/api/sessions")
    assert r.status_code == 200 and "max-age" not in r.headers["set-cookie"].lower()   # renewed, still a session cookie
    # idle timeout: a cookie that sat unused past the limit is rejected and the password is asked again
    old = auth.make_token(auth.load(auth_file_for(tsettings))["cookie_key"], now=time.time() - auth.IDLE_SECONDS - 5)
    stale = TestClient(create_app(tsettings))
    stale.cookies.set(auth.COOKIE, old)
    assert stale.get("/api/sessions").status_code == 401
    client.get("/logout")
    assert client.get("/api/sessions").status_code == 401
    # lockout after repeated wrong passwords, even for the right one
    other = TestClient(create_app(tsettings))
    for _ in range(auth.MAX_FAILURES):
        other.post("/login", data={"password": "bad"})
    assert other.post("/login", data={"password": "s3cret-pass"}).status_code == 429
    # changing the password signs everyone out
    client.post("/login", data={"password": "s3cret-pass"})
    assert client.get("/api/sessions").status_code == 200
    auth.set_password(auth_file_for(tsettings), "another-pass")
    assert client.get("/api/sessions").status_code == 401
    assert auth.clear_password(auth_file_for(tsettings))
    assert client.get("/api/sessions").status_code == 200


def test_serve_refuses_public_host_without_password():
    import serve
    assert serve.is_loopback("127.0.0.1") and serve.is_loopback("localhost") and serve.is_loopback("::1")
    assert not serve.is_loopback("0.0.0.0") and not serve.is_loopback("100.101.102.103")


# ---------------------------------------------------------------------- structure-exit study
def _structure_data(rows, bars, ser=None):
    sig = pd.DataFrame(rows)
    sig["date"] = pd.to_datetime(sig["date"])
    arr = np.array(bars, dtype=float)
    n = len(sig)
    return {"signals": sig, "O": arr[:, 0], "H": arr[:, 1], "L": arr[:, 2], "C": arr[:, 3], "V": np.ones_like(arr[:, 0]),
            "SER": ser or {}, "S": {}, "T": {}, "F": {}}


def test_structure_simulate_matches_research_simulate():
    from src.backtest import structure
    from src.backtest.research import simulate
    rng = np.random.default_rng(5)
    rows, bars = [], []
    for i in range(80):
        c = 100 + rng.normal(0, 3, 21).cumsum()
        o = c + rng.normal(0, 0.5, 21)
        h, lo = np.maximum(o, c) + rng.uniform(0, 2, 21), np.minimum(o, c) - rng.uniform(0, 2, 21)
        rows.append({"symbol": f"S{i}", "setup": "squeeze" if i % 4 == 0 else "breakout", "date": "2024-01-02",
                     "ref": c[0], "atr": 2.0})
        bars.append([o[1:], h[1:], lo[1:], c[1:]])
    d = _structure_data(rows, bars)
    sig = d["signals"]
    ref = {"signals": sig.assign(long=True, conditional=sig["setup"] == "squeeze", structural=sig["ref"] - 3,
                                 swing=sig["ref"] - 3, score=np.nan, tier=np.nan, planned_rr=np.nan, regime=np.nan),
           "O": d["O"], "H": d["H"], "L": d["L"], "C": d["C"]}
    for kw, target in (({"trail": 3.0}, "trail3"), ({"tgt_r": 3.0}, "r3")):
        a, b = sig["ref"].to_numpy(), sig["atr"].to_numpy()
        mine = structure.simulate(d, 0.001, stop=a - 1.5 * b, **kw)
        old = simulate(ref, "atr1.5", target, 20, 0.001)
        okm = mine["ok"].to_numpy()
        keep = d["O"][okm, 0] > (a - 1.5 * b)[okm]      # research.simulate exits at the stop level on a bar-0 gap through it
        assert keep.sum() > 40
        assert np.allclose(mine["r"].to_numpy()[okm][keep], old["r"].to_numpy()[keep])


def test_structure_exits_close_basis_and_ratchet():
    from src.backtest import structure
    # entry 100 at the open, ATR 2, stop 97. Day 2 closes below the close-basis level -> exit at day 3's open.
    o = [100, 101, 102, 99, 99]
    h = [101, 103, 103, 100, 100]
    lo = [99.5, 100.5, 100.5, 98.5, 98.5]
    c = [100.5, 102, 101, 99, 99]
    ser = {"x": np.array([[95, 95, 100.5, 100.5, 100.5, 100.5]], dtype=float)}   # known at the close of t..t+5
    d = _structure_data([{"symbol": "X", "setup": "breakout", "date": "2024-01-02", "ref": 100.0, "atr": 2.0}], [[o, h, lo, c]], ser)
    t = structure.simulate(d, 0.0, stop=np.array([97.0]), cser="x", hz=5)
    # bar k=2 closes at 101 > 100.5; bar k=3 closes at 99 < 100.5 -> exit at the open of bar 4 (99)
    assert t["days"].iat[0] == 5
    assert (99 - 100) / 100 == pytest.approx(t["ret"].iat[0])
    # ratchet: the series level (95 -> 100.5) lifts the stop above the later lows -> stopped at the ratcheted level
    t2 = structure.simulate(d, 0.0, stop=np.array([97.0]), sser="x", sbuf=0.0, hz=5)
    assert t2["days"].iat[0] == 3 and t2["ret"].iat[0] == pytest.approx(0.005)


def test_structure_stop_from_level_clamp_and_fallback():
    from src.backtest import structure
    sig = pd.DataFrame({"setup": ["breakout"] * 4, "ref": [100.0] * 4, "atr": [2.0] * 4})
    D = {"signals": sig}
    st, valid = structure.stop_from_level(D, np.array([99.9, 98.0, 90.0, np.nan]), 1.5)
    assert valid.tolist() == [False, True, True, False]
    # too close (< 0.5 ATR) and NaN fall back to the base 1.5 ATR; 1 ATR distance is raised to 0.75 min; 5 ATR clipped to 4
    assert st.tolist() == [97.0, 98.0, 92.0, 97.0]
    cap, ok = structure.target_cap(D, np.array([101.0, 103.0, 120.0, np.nan]))
    assert ok.tolist() == [False, True, False, False]


def test_structure_levels_are_causal():
    from src.backtest import structure
    rng = np.random.default_rng(11)
    n = 520
    idx = pd.bdate_range("2020-01-01", periods=n)

    def frame(seed_tail):
        close = 100 * np.exp(np.cumsum(np.r_[rng_part, seed_tail]))
        px = pd.DataFrame({"open": close * 0.998, "high": close * 1.01, "low": close * 0.99, "close": close,
                           "adj_close": close, "volume": 1e6}, index=idx)
        return build_features(px, px["adj_close"], None)
    rng_part = rng.normal(0.0005, 0.02, 440)
    a = frame(rng.normal(0, 0.02, n - 440))
    b = frame(rng.normal(0, 0.05, n - 440))
    la, lb = structure._psar_free_levels(a), structure._psar_free_levels(b)
    for k in la:
        assert np.allclose(la[k][:430], lb[k][:430], equal_nan=True), k
    ba, bb = structure._bounce_stats(a, la), structure._bounce_stats(b, lb)
    for k in ba:
        assert np.allclose(ba[k][:430], bb[k][:430], equal_nan=True), k


def test_structure_study_shipped_copy():
    d = json.loads((Path(__file__).resolve().parents[1] / "config" / "structure_study.default.json").read_text())
    assert set(d["setups"]) == {"crash_rebound", "squeeze", "breakout"}
    assert all(v["validated"] == [] for v in d["setups"].values())


# ======================================================================= holdings (positions you bought) + holding study
def test_candle_detector_is_causal():
    from src.technical import candles
    rng = np.random.default_rng(21)
    n = 400
    head = rng.normal(0.0005, 0.02, 360)

    def frame(tail):
        c = 100 * np.exp(np.cumsum(np.r_[head, tail]))
        o = np.r_[100, c[:-1]] * (1 + rng.normal(0, 0.004, n))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
        return o, h, lo, c
    rng2 = np.random.default_rng(5)
    a = frame(rng2.normal(0, 0.02, n - 360))
    b = frame(rng2.normal(0, 0.06, n - 360))
    # the random noise of o/h/l differs between a and b, so rebuild both from the same o/h/l except the tail
    o, h, lo, c = a
    o2, h2, lo2, c2 = (np.r_[x[:360], y[360:]] for x, y in zip(a, b))
    atr = candles.wilder_atr(h, lo, c, 14)
    atr2 = candles.wilder_atr(h2, lo2, c2, 14)
    fa, fb = candles.detect(o, h, lo, c, atr), candles.detect(o2, h2, lo2, c2, atr2)
    assert len(candles.PATTERN_IDS) >= 40 and len(fa) == len(candles.PATTERN_IDS)
    for k in fa:
        assert np.array_equal(fa[k][:360], fb[k][:360]), k
    assert sum(int(v.sum()) for v in fa.values()) > 0


def test_holding_exit_state_replays_the_live_trailing_stop():
    from src.portfolio import holdings
    idx = pd.bdate_range("2026-10-01", periods=12)
    high = np.array([101, 103, 106, 110, 109, 108, 107, 106, 105, 104, 103, 102], float)
    low = high - 1.5
    low[8] = 103.5            # day 9: the trailing stop 110 - 3 x 2 = 104 is hit
    f = pd.DataFrame({"open": high - 1, "high": high, "low": low, "close": high - 0.5, "atr": 2.0}, index=idx)
    trade = {"symbol": "AAA", "entry_price": 100.0, "stop": 97.0, "entry_date": "2026-10-01", "setup": "crash_rebound",
             "plan_json": json.dumps({"atr": 2.0, "trail_atr": 3.0, "horizon_days": 20})}
    plan = holdings.trade_plan(trade)
    assert plan["trail"] == 3.0 and plan["atr"] == 2.0 and plan["horizon"] == 20
    st = holdings.exit_state(f, trade, plan)
    assert st["status"] == "stop" and st["exit_date"] == idx[8].strftime("%Y-%m-%d") and st["exit_px"] == pytest.approx(104.0)
    open_ = holdings.exit_state(f.iloc[:6], trade, plan)
    assert open_["status"] == "open" and open_["stop_next"] == pytest.approx(104.0) and open_["trail_on"]
    assert open_["mfe_r"] == pytest.approx(10 / 3) and open_["days_held"] == 6 and open_["days_since_high"] == 2
    sq = {**trade, "setup": "squeeze", "plan_json": json.dumps({"target": 105.0, "trail_atr": None})}
    st2 = holdings.exit_state(f, sq, holdings.trade_plan(sq))
    assert st2["status"] == "target" and st2["exit_px"] == pytest.approx(105.0)


def test_holding_study_first_fire_and_paired_delta():
    from src.backtest import holding_study as hs
    # 3 trades; trade-days sorted by (sig, k)
    td = pd.DataFrame({"sig": [0, 0, 0, 1, 1, 2], "k": [1, 2, 3, 1, 2, 1], "cv_eq": [0.01, -0.02, 0.005, 0.03, 0.0, -0.01],
                       "cv_r": [0.5, -1.0, 0.25, 1.5, 0.0, -0.5]})
    trades = pd.DataFrame({"date": ["2021-01-04", "2021-01-05", "2021-01-06"]})
    mask = np.array([False, True, True, False, False, True])
    assert hs.first_fire(mask, td["sig"].to_numpy(), 3).tolist() == [1, -1, 5]
    book = hs.Book(trades, td)
    d_eq, d_r, fired = book.delta_exit(mask)
    # selling at the first firing gains minus the continuation value there; trades that never fire are unchanged
    assert d_eq.tolist() == pytest.approx([0.02, 0.0, 0.01]) and d_r.tolist() == pytest.approx([1.0, 0.0, 0.5])
    assert fired.tolist() == [True, False, True]


def test_journal_plan_json_round_trip(tsettings):
    from src.daily import store as st
    from src.portfolio import journal
    conn = st.connect(Path(tsettings["paths"]["database"]), Path(tsettings["paths"]["backups_dir"]))
    plan = {"setup": "crash_rebound", "atr": 2.5, "trail_atr": 3, "horizon_days": 20, "ref": 50.0}
    tid = journal.add_trade(conn, {"symbol": "AAA", "direction": "long", "setup": "crash_rebound", "entry_price": 50.4,
                                   "stop": 46.25, "shares": 100, "plan_json": plan})
    row = journal.open_trades(conn)[0]
    assert row["id"] == tid and json.loads(row["plan_json"]) == plan
    with pytest.raises(ValueError):
        journal.update_trade(conn, tid, {"plan_json": "{not json"})
    conn.close()
    # an older database without the column is upgraded in place
    old = tmp = Path(tsettings["paths"]["database"]).with_name("old.db")
    c = sqlite3.connect(tmp)
    c.execute("CREATE TABLE terminal_runs (id INTEGER PRIMARY KEY, session_date TEXT, started_at TEXT, status TEXT)")
    c.execute("CREATE TABLE trade_journal (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, symbol TEXT NOT NULL,"
              " direction TEXT NOT NULL, setup TEXT, entry_date TEXT, entry_price REAL NOT NULL, stop REAL NOT NULL, target REAL,"
              " shares REAL, exit_date TEXT, exit_price REAL, status TEXT NOT NULL DEFAULT 'open', notes TEXT)")
    c.commit()
    c.close()
    conn = st.connect(old, Path(tsettings["paths"]["backups_dir"]))
    assert "plan_json" in {r[1] for r in conn.execute("PRAGMA table_info(trade_journal)")}
    conn.close()


def test_hourly_bars_keep_only_completed_bars_and_session_end():
    from src.daily import intraday
    from src.portfolio import holdings
    ny = calendar.NEW_YORK
    day = date(2026, 11, 2)
    starts = [datetime.combine(day, datetime.min.time(), ny).replace(hour=9, minute=30) + timedelta(hours=i) for i in range(7)]
    fr = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1e5}, index=pd.DatetimeIndex(starts))
    rows = intraday.completed_hourly(fr, datetime(2026, 11, 2, 11, 45, tzinfo=ny))
    assert [r[0][11:16] for r in rows] == ["09:30", "10:30"]     # the 11:30 bar is still forming at 11:45
    assert len(intraday.completed_hourly(fr, datetime(2026, 11, 2, 16, 1, tzinfo=ny))) == 7
    assert holdings.session_complete(pd.Timestamp(starts[-1])) and not holdings.session_complete(pd.Timestamp(starts[3]))
    early = pd.Timestamp(datetime(2026, 11, 27, 12, 30, tzinfo=ny))  # day after Thanksgiving closes at 13:00
    assert holdings.session_complete(early)


def test_holdings_api_analyses_open_trades(tsettings):
    from fastapi.testclient import TestClient

    from src.daily import intraday
    from src.daily.pipeline import run_daily
    from src.web.app import create_app
    run_daily(tsettings, session=SESSION, downloader=FakeDownloader(market_frames(tsettings)), enrichment=FakeEnrichment(),
              fred_getter=fake_fred, progress=lambda m: None)
    client = TestClient(create_app(tsettings))
    chart = client.get("/api/chart/AAA?days=60").json()
    c5 = chart["candles"][-6]
    entry, atr = float(c5["open"]), 2.0
    body = {"symbol": "AAA", "direction": "long", "setup": "crash_rebound", "entry_date": c5["time"], "entry_price": entry,
            "stop": entry - 1.5 * atr, "shares": 50, "plan_json": {"atr": atr, "trail_atr": 3, "horizon_days": 20, "ref": entry}}
    assert client.post("/api/journal", json=body).status_code == 200
    client.post("/api/journal", json={"symbol": "BBB", "direction": "short", "entry_price": 10, "stop": 11})
    # hourly bars of the held symbol, as the intraday job stores them (fake download, no network)
    ny = calendar.NEW_YORK
    conn = sqlite3.connect(tsettings["paths"]["database"])
    assert intraday.held_symbols(conn) == ["AAA", "BBB"]

    def fake_hourly(symbols):
        idx = []
        for d in calendar.trading_days(date(2026, 9, 1), SESSION):
            idx += [datetime.combine(d, datetime.min.time(), ny).replace(hour=9, minute=30) + timedelta(hours=i) for i in range(7)]
        rng = np.random.default_rng(3)
        c = entry * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
        fr = pd.DataFrame({"Open": c * 0.999, "High": c * 1.004, "Low": c * 0.996, "Close": c, "Volume": 1e5},
                          index=pd.DatetimeIndex(idx))
        return {s: fr for s in symbols}
    r = intraday.refresh_hourly(tsettings, conn, now=datetime(2026, 10, 30, 17, 0, tzinfo=ny), fetch=fake_hourly)
    conn.close()
    assert r["status"] == "ok" and r["with_bars"] == 2
    res = client.get("/api/holdings").json()
    by = {h["symbol"]: h for h in res["holdings"]}
    assert "error" in by["BBB"]                                      # short: not analysed
    h = by["AAA"]
    assert "error" not in h, h
    ex = h["exit"]
    assert ex["days_held"] == 6 and ex["status"] in ("open", "stop", "time", "target")
    assert ex["stop_next"] >= body["stop"] - 1e-9 and h["action"]["kind"] in ("hold", "sell")
    assert isinstance(h["signals"], list) and h["hourly_session"] == "2026-10-30" and h["hourly_complete"] is True
    assert all({"rule", "name", "verdict"} <= set(s) for s in h["signals"])
    assert res["hourly"]["fetched_at"]


def test_holding_study_shipped_copy():
    d = json.loads((Path(__file__).resolve().parents[1] / "config" / "holding_study.default.json").read_text())
    assert set(d["setups"]) == {"crash_rebound", "squeeze", "breakout"}
    for v in d["setups"].values():
        assert v["lookup"] and "gate" in v and "day_profile" in v
        some = next(iter(v["lookup"].values()))
        assert "pd" in some and "pt" in some
