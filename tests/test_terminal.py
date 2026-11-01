"""Tests for the research terminal: calendar, data update + quality, regime, setups, plans,
historical simulation, scoring, fundamentals (point-in-time), catalysts, options, the daily
pipeline end to end (fake data, no internet) and the dashboard API."""

from __future__ import annotations

import copy
import json
import sqlite3
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
    q = plan_row(f, "pullback", sc, 7)
    assert abs(q["target1"] - (q["entry"] + 2 * 2.0 * q["atr"])) < 1e-9 and abs(q["reward_risk"] - 2.0) < 1e-9
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
    assert src.startswith("strategy lab") and bonus["breakout"] == 12.0 and bonus["momentum"] == -1.8


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
