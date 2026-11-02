"""US equity (NYSE / Nasdaq) trading calendar.

Rules (NYSE holiday schedule, valid since 2022 when Juneteenth was added):

* New Year's Day (Jan 1; Sunday -> Monday. Saturday -> NOT observed on Friday Dec 31,
  per NYSE Rule 7.2, because that would close the market on the last day of the year)
* Martin Luther King Jr. Day (3rd Monday of January)
* Washington's Birthday / Presidents' Day (3rd Monday of February)
* Good Friday (two days before Western Easter Sunday)
* Memorial Day (last Monday of May)
* Juneteenth (Jun 19; Saturday -> Friday, Sunday -> Monday)  -- from 2022
* Independence Day (Jul 4; Saturday -> Friday, Sunday -> Monday)
* Labor Day (1st Monday of September)
* Thanksgiving (4th Thursday of November)
* Christmas (Dec 25; Saturday -> Friday, Sunday -> Monday)

Early closes (13:00 New York time): Jul 3, the day after Thanksgiving and Dec 24 -
each only when that day is itself a trading day (e.g. Jul 3, 2026 is the observed
Independence Day holiday, so there is no early close that year).

One-off closures (national days of mourning, weather) cannot be derived from rules;
they are listed in ``SPECIAL_CLOSURES`` and can be extended in config/terminal.yaml.

The calendar is used to decide which session is the "latest completed" one and to
check that downloaded data is complete. It is cross-checked against the dates the
data provider actually returns (see ``src/data/quality.py``).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
# Daily bars from free providers are usually final ~15-30 minutes after the close.
DEFAULT_SETTLE_MINUTES = 30

SPECIAL_CLOSURES = {
    date(2018, 12, 5): "National Day of Mourning (George H. W. Bush)",
    date(2025, 1, 9): "National Day of Mourning (Jimmy Carter)",
}


def easter_sunday(year: int) -> date:
    """Western (Gregorian) Easter Sunday - anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) given weekday (Mon=0) of a month."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=64)
def holidays(year: int) -> dict[date, str]:
    """Full-day market holidays of one calendar year."""
    out: dict[date, str] = {}
    new_year = date(year, 1, 1)
    if new_year.weekday() == 6:
        out[new_year + timedelta(days=1)] = "New Year's Day (observed)"
    elif new_year.weekday() != 5:  # Saturday: not observed (NYSE Rule 7.2)
        out[new_year] = "New Year's Day"
    out[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    out[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    out[easter_sunday(year) - timedelta(days=2)] = "Good Friday"
    out[_last_weekday(year, 5, 0)] = "Memorial Day"
    if year >= 2022:
        out[_observed(date(year, 6, 19))] = "Juneteenth"
    out[_observed(date(year, 7, 4))] = "Independence Day"
    out[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    out[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    out[_observed(date(year, 12, 25))] = "Christmas Day"
    for day, name in SPECIAL_CLOSURES.items():
        if day.year == year:
            out[day] = name
    return out


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in holidays(day.year)


def early_close(day: date) -> bool:
    """13:00 close: Jul 3, the day after Thanksgiving, Dec 24 (each only when it is a trading day)."""
    if not is_trading_day(day):
        return False
    thanksgiving = _nth_weekday(day.year, 11, 3, 4)
    return day in (date(day.year, 7, 3), thanksgiving + timedelta(days=1), date(day.year, 12, 24))


def close_time(day: date) -> time:
    return EARLY_CLOSE if early_close(day) else REGULAR_CLOSE


def previous_trading_day(day: date) -> date:
    day -= timedelta(days=1)
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day


def next_trading_day(day: date) -> date:
    day += timedelta(days=1)
    while not is_trading_day(day):
        day += timedelta(days=1)
    return day


def trading_days(start: date, end: date) -> list[date]:
    """All trading days in [start, end]."""
    out, day = [], start
    while day <= end:
        if is_trading_day(day):
            out.append(day)
        day += timedelta(days=1)
    return out


def trading_days_between(start: date, end: date) -> int:
    """Number of trading days d with start < d <= end (0 if end <= start)."""
    if end <= start:
        return 0
    return len(trading_days(start + timedelta(days=1), end))


def add_trading_days(day: date, n: int) -> date:
    for _ in range(n):
        day = next_trading_day(day)
    return day


def latest_completed_session(now: datetime | None = None, settle_minutes: int = DEFAULT_SETTLE_MINUTES) -> date:
    """The most recent trading session whose close (+ settle time for data) has passed.

    ``now`` may be naive (interpreted as New York time) or timezone-aware.
    Example: Friday 17:00 ET -> that Friday; Friday 15:00 ET -> Thursday; Saturday -> Friday.
    """
    now = datetime.now(NEW_YORK) if now is None else now
    now = now.replace(tzinfo=NEW_YORK) if now.tzinfo is None else now.astimezone(NEW_YORK)
    today = now.date()
    if is_trading_day(today):
        settle = datetime.combine(today, close_time(today), NEW_YORK) + timedelta(minutes=settle_minutes)
        if now >= settle:
            return today
    return previous_trading_day(today)


def session_info(day: date) -> dict:
    """Description of a session for reports."""
    return {
        "date": day.isoformat(),
        "weekday": day.strftime("%A"),
        "early_close": early_close(day),
        "close_time_et": close_time(day).strftime("%H:%M"),
        "next_session": next_trading_day(day).isoformat(),
        "previous_session": previous_trading_day(day).isoformat(),
    }


def upcoming_holidays(start: date, days: int = 45) -> list[dict]:
    end = start + timedelta(days=days)
    out = []
    for year in {start.year, end.year}:
        for day, name in holidays(year).items():
            if start <= day <= end:
                out.append({"date": day.isoformat(), "name": name})
    return sorted(out, key=lambda item: item["date"])
