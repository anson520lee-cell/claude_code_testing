"""Tests for abnormal-event detection and earnings-date evidence matching."""

import pandas as pd
import pytest

from src.analysis.statistics import add_statistics
from src.data.prices import standardize_price_frame
from src.events.detector import EVENT_COLUMNS, build_rule_flags, detect_events
from src.events.evidence import (
    AUTO_MATCH_STATUS,
    EARNINGS_CATEGORY,
    match_earnings_to_events,
    normalize_earnings_dates,
)
from tests.conftest import make_yf_prices

RULES = {
    "return_zscore_threshold": 2.5,
    "volume_ratio_threshold": 2.0,
    "volume_zscore_threshold": None,
    "abs_return_threshold": None,
}


def stats_for(shocks, n=300, seed=11):
    prices = standardize_price_frame(make_yf_prices(n=n, seed=seed, shocks=shocks))
    return add_statistics(prices, "adj_close", zscore_window=60, volume_window=20, volatility_window=20)


def test_planted_price_and_volume_shock_is_detected():
    stats = stats_for({200: (0.15, 5.0)})
    events = detect_events(stats, "TEST", RULES)
    shock_date = stats.index[200]
    assert shock_date in set(events["event_date"])
    event = events[events["event_date"] == shock_date].iloc[0]
    assert event["direction"] == "up"
    assert event["return_zscore"] > 2.5
    assert event["volume_ratio"] > 2.0
    assert set(event["trigger_rules"].split(",")) == {"return_zscore", "volume_ratio"}
    assert event["anomaly_type"] == "Abnormal price move with abnormal volume"
    assert bool(event["is_price_event"]) and bool(event["is_volume_event"])
    assert event["daily_return"] == pytest.approx(0.15)


def test_down_move_without_volume_spike():
    stats = stats_for({150: (-0.12, 1.0)})
    events = detect_events(stats, "TEST", RULES)
    event = events[events["event_date"] == stats.index[150]].iloc[0]
    assert event["direction"] == "down"
    assert event["volume_ratio"] < 2.0  # normal volume that day
    assert event["trigger_rules"] == "return_zscore"
    assert event["anomaly_type"] == "Abnormal price move (volume not flagged)"
    assert not bool(event["is_volume_event"])


def test_volume_only_event():
    stats = stats_for({180: (0.0005, 6.0)})
    events = detect_events(stats, "TEST", RULES)
    event = events[events["event_date"] == stats.index[180]].iloc[0]
    assert event["trigger_rules"] == "volume_ratio"
    assert event["anomaly_type"] == "Abnormal volume (price move not flagged)"


def test_disabling_a_rule_removes_its_events():
    stats = stats_for({180: (0.0005, 6.0)})
    rules = dict(RULES, volume_ratio_threshold=None)
    events = detect_events(stats, "TEST", rules)
    assert stats.index[180] not in set(events["event_date"])


def test_higher_threshold_gives_fewer_events():
    stats = stats_for({120: (0.08, 1.0), 200: (0.20, 4.0)})
    loose = detect_events(stats, "TEST", dict(RULES, return_zscore_threshold=2.0))
    strict = detect_events(stats, "TEST", dict(RULES, return_zscore_threshold=4.0))
    assert len(strict) <= len(loose)
    assert set(strict["event_date"]) <= set(loose["event_date"])


def test_no_rules_enabled_raises():
    stats = stats_for({})
    with pytest.raises(ValueError):
        detect_events(stats, "TEST", {k: None for k in RULES})


def test_days_without_enough_history_are_never_flagged():
    # A huge move on day 10 cannot be scored: the Z-score needs 60 previous returns
    # and the volume ratio needs 20 previous days.
    stats = stats_for({10: (0.30, 10.0)})
    flags = build_rule_flags(stats, RULES)
    assert not flags.iloc[:21].any().any()
    events = detect_events(stats, "TEST", RULES)
    assert stats.index[10] not in set(events["event_date"])


def test_research_fields_default_to_unknown():
    events = detect_events(stats_for({200: (0.15, 5.0)}), "TEST", RULES)
    assert list(events.columns) == EVENT_COLUMNS
    assert (events["event_category"] == "Unknown").all()
    assert (events["verification_status"] == "Unverified").all()
    assert events["event_description"].isna().all()
    assert events["source"].isna().all()


def test_spacing_between_events():
    # With a 5-sigma threshold only the two planted +/-20% moves qualify.
    stats = stats_for({100: (0.2, 5.0), 103: (-0.2, 5.0)})
    events = detect_events(stats, "TEST", dict(RULES, volume_ratio_threshold=None, return_zscore_threshold=5))
    assert list(events["event_date"]) == [stats.index[100], stats.index[103]]
    assert pd.isna(events.iloc[0]["trading_days_since_prev_event"])
    assert events.iloc[1]["trading_days_since_prev_event"] == 3


def test_forward_returns_copied_to_events():
    stats = stats_for({200: (0.15, 5.0)})
    events = detect_events(stats, "TEST", RULES)
    event = events[events["event_date"] == stats.index[200]].iloc[0]
    expected = stats["price"].iloc[205] / stats["price"].iloc[200] - 1
    assert event["fwd_return_5d"] == pytest.approx(expected)


# ------------------------------------------------------------------ evidence
def earnings_table(times):
    index = pd.DatetimeIndex(pd.to_datetime(times)).tz_localize("America/New_York")
    raw = pd.DataFrame({"EPS Estimate": [0.5] * len(times), "Reported EPS": [0.6] * len(times),
                        "Surprise(%)": [20.0] * len(times)}, index=index)
    raw.index.name = "Earnings Date"
    return normalize_earnings_dates(raw)


def simple_events(dates):
    return pd.DataFrame({
        "event_date": pd.to_datetime(dates), "event_category": "Unknown", "event_description": None,
        "source": None, "verification_status": "Unverified",
    })


TRADING_DAYS = pd.bdate_range("2025-03-03", "2025-03-21")


def test_after_close_release_matches_next_trading_day():
    earnings = earnings_table(["2025-03-10 16:05"])  # Monday after the close
    events = simple_events(["2025-03-10", "2025-03-11", "2025-03-12", "2025-03-13"])
    matched = match_earnings_to_events(events, TRADING_DAYS, earnings, window_days=1)
    categories = dict(zip(matched["event_date"].dt.strftime("%Y-%m-%d"), matched["event_category"]))
    assert categories["2025-03-10"] == "Unknown"  # before the release
    assert categories["2025-03-11"] == EARNINGS_CATEGORY  # first reaction day
    assert categories["2025-03-12"] == EARNINGS_CATEGORY  # within the 1-day window
    assert categories["2025-03-13"] == "Unknown"  # outside the window
    row = matched[matched["event_date"] == "2025-03-11"].iloc[0]
    assert row["verification_status"] == AUTO_MATCH_STATUS
    assert "not been verified" in row["event_description"]
    assert "0.60" in row["event_description"]


def test_pre_market_release_matches_same_day_and_weekend_release_next_monday():
    earnings = earnings_table(["2025-03-05 07:00", "2025-03-15 10:00"])  # Wednesday pre-market; Saturday
    events = simple_events(["2025-03-05", "2025-03-17"])
    matched = match_earnings_to_events(events, TRADING_DAYS, earnings, window_days=0)
    assert (matched["event_category"] == EARNINGS_CATEGORY).all()


def test_matching_never_overwrites_existing_research():
    earnings = earnings_table(["2025-03-10 08:00"])
    events = simple_events(["2025-03-10"])
    events.loc[0, "event_category"] = "Contract award"
    matched = match_earnings_to_events(events, TRADING_DAYS, earnings, window_days=1)
    assert matched.loc[0, "event_category"] == "Contract award"


def test_empty_or_missing_earnings_calendar():
    events = simple_events(["2025-03-10"])
    assert normalize_earnings_dates(None).empty
    assert normalize_earnings_dates(pd.DataFrame()).empty
    unchanged = match_earnings_to_events(events, TRADING_DAYS, normalize_earnings_dates(None))
    assert unchanged.loc[0, "event_category"] == "Unknown"
