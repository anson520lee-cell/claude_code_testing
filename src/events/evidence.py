"""Attach factual evidence to detected events.

Evidence source currently implemented: the earnings calendar.

For each earnings release we work out the first trading day on which the market
could react:

* release time before 16:00 New York time (or unknown, shown by Yahoo as
  midnight)  ->  the release date itself (or the next trading day if it was
  not a trading day)
* release at/after 16:00 (after the close)  ->  the next trading day

An event is "matched" when it falls on that reaction day or up to
``window_days`` trading days later.

A match is a DATE COINCIDENCE. It is useful evidence, but it does not prove the
earnings release caused the move, so matched events are labelled
"Unverified (automatic date match)". Events without a match keep the category
"Unknown"; this module never invents an explanation.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from src.events.detector import UNKNOWN_CATEGORY

logger = logging.getLogger(__name__)

EARNINGS_CATEGORY = "Earnings release (date match)"
AUTO_MATCH_STATUS = "Unverified (automatic date match)"
# Must equal f"{YahooFinanceProvider.name} earnings calendar" (checked by a test).
EARNINGS_SOURCE = "Yahoo Finance (via yfinance) earnings calendar"
MARKET_CLOSE_HOUR = 16


def normalize_earnings_dates(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Convert yfinance's earnings-dates table into a simple table.

    Returns a DataFrame with columns: release_time (New York time, no time zone),
    eps_estimate, reported_eps, surprise_pct. Empty if nothing usable.
    """
    columns = ["release_time", "eps_estimate", "reported_eps", "surprise_pct"]
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=columns)

    times = pd.DatetimeIndex(pd.to_datetime(raw.index, errors="coerce"))
    if times.tz is not None:
        times = times.tz_convert("America/New_York").tz_localize(None)

    def column(*names: str) -> pd.Series:
        for name in names:
            if name in raw.columns:
                return pd.to_numeric(raw[name], errors="coerce").reset_index(drop=True)
        return pd.Series([math.nan] * len(raw))

    table = pd.DataFrame({
        "release_time": times,
        "eps_estimate": column("EPS Estimate"),
        "reported_eps": column("Reported EPS"),
        "surprise_pct": column("Surprise(%)", "Surprise (%)"),
    })
    table = table.dropna(subset=["release_time"]).drop_duplicates(subset=["release_time"])
    return table.sort_values("release_time").reset_index(drop=True)


def reaction_day_position(release_time: pd.Timestamp, trading_days: pd.DatetimeIndex) -> int | None:
    """Index (in ``trading_days``) of the first trading day that could react to a release."""
    start = release_time.normalize()
    if release_time.hour >= MARKET_CLOSE_HOUR:
        start = start + pd.Timedelta(days=1)
    position = int(trading_days.searchsorted(start))
    return position if position < len(trading_days) else None


def _format_number(value: float, suffix: str = "") -> str:
    return "not available" if value is None or pd.isna(value) else f"{value:.2f}{suffix}"


def match_earnings_to_events(
    events: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    earnings: pd.DataFrame,
    window_days: int = 1,
    source: str = EARNINGS_SOURCE,
) -> pd.DataFrame:
    """Fill the research fields of events that fall in an earnings reaction window.

    Only events whose category is still "Unknown" are changed.

    Args:
        events: events table from ``detect_events``.
        trading_days: all trading days in the price data (sorted).
        earnings: output of ``normalize_earnings_dates``.
        window_days: extra trading days after the reaction day that still count.
        source: where the earnings dates came from (stored in the event's "source").

    Returns:
        A copy of ``events`` with matched rows updated.
    """
    events = events.copy()
    if events.empty or earnings.empty:
        return events

    # Map: trading-day position -> (release row, offset in trading days)
    windows: dict[int, tuple[pd.Series, int]] = {}
    for _, release in earnings.iterrows():
        start = reaction_day_position(release["release_time"], trading_days)
        if start is None:
            continue  # release is after the last day of price data
        for offset in range(window_days + 1):
            position = start + offset
            if position < len(trading_days) and position not in windows:
                windows[position] = (release, offset)

    matched = 0
    for index, event in events.iterrows():
        if event["event_category"] != UNKNOWN_CATEGORY:
            continue
        position = int(trading_days.searchsorted(pd.Timestamp(event["event_date"])))
        if position not in windows:
            continue
        release, offset = windows[position]
        timing = (
            "on the first trading session that could react to the release"
            if offset == 0
            else f"{offset} trading day(s) after the first trading session that could react to the release"
        )
        events.at[index, "event_category"] = EARNINGS_CATEGORY
        events.at[index, "event_description"] = (
            f"Earnings release dated {release['release_time']:%Y-%m-%d %H:%M} (New York time); "
            f"this event occurred {timing}. Reported EPS {_format_number(release['reported_eps'])}, "
            f"EPS estimate {_format_number(release['eps_estimate'])}, "
            f"surprise {_format_number(release['surprise_pct'], '%')}. "
            "Date match only - the cause of the move has not been verified."
        )
        events.at[index, "source"] = source
        events.at[index, "verification_status"] = AUTO_MATCH_STATUS
        matched += 1

    logger.info("Matched %d of %d event(s) to earnings release dates.", matched, len(events))
    return events
