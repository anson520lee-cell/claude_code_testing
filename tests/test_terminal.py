"""Tests for the research terminal: calendar, data update + quality, regime, setups, plans,
historical simulation, scoring, fundamentals (point-in-time), catalysts, options, the daily
pipeline end to end (fake data, no internet) and the dashboard API."""

from __future__ import annotations

import copy
import json
import sqlite3
from datetime import date, datetime
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
                                                       "breakdown", "rally_fail", "ema21_loss")])
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
    assert client.get("/api/chart/NOPE").status_code == 404
    assert client.get("/api/regime/history").json()["history"]
    assert client.get("/api/runs").json()["runs"][0]["status"].startswith("success")
    assert client.get("/api/candidate-history/AAA").json()["history"]
    assert "Composite" in client.get("/api/methodology").json()["regime"] or client.get("/api/methodology").status_code == 200
    assert client.get("/reports/2026-10-30/report.html").status_code == 200
    assert client.get("/api/snapshot?session=1999-01-01").status_code == 404
