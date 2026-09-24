"""Standardise, clean and validate daily price data.

Every price table inside this project uses one standard layout (a "price frame"):

    index   : DatetimeIndex named "date" (one row per trading day, no time zone)
    columns : open, high, low, close, adj_close, volume

``adj_close`` is the close adjusted for dividends and splits. It may be missing
(all NaN) when the data source does not provide it.

Cleaning never silently "fixes" suspicious values. Rows that cannot be used
(e.g. a missing close price) are removed, and every change or suspicious
pattern is reported as a warning so it can appear in the report's
"Data Limitations" section.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

# Maps lower-case column names from different sources to our standard names.
_COLUMN_ALIASES = {
    "date": "date",
    "datetime": "date",
    "timestamp": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "close/last": "close",  # nasdaq.com download format
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "adjclose": "adj_close",
    "adjusted close": "adj_close",
    "volume": "volume",
}

# A gap longer than this between two consecutive trading days is unusual
# (weekends + holidays are at most 4-5 calendar days) and may mean missing data.
MAX_NORMAL_GAP_DAYS = 6


class DataUnavailableError(Exception):
    """Raised when no usable price data can be obtained for a ticker."""


def standardize_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw price table (yfinance or CSV) into the standard price frame.

    The input may have the dates either as the index or as a "Date" column.
    Column names are matched case-insensitively (e.g. "Adj Close" -> "adj_close").

    Raises:
        DataUnavailableError: if the table is empty or has no close prices.
    """
    if raw is None or raw.empty:
        raise DataUnavailableError("The price table is empty.")

    df = raw.copy()
    rename = {}
    for column in df.columns:
        key = str(column).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[column] = _COLUMN_ALIASES[key]
    df = df.rename(columns=rename)

    if "date" in df.columns:
        df = df.set_index("date")

    index = pd.to_datetime(df.index, errors="coerce")
    # yfinance returns time-zone-aware timestamps (e.g. America/New_York).
    # We only need the calendar date of each trading day.
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    df.index = pd.DatetimeIndex(index).normalize()
    df.index.name = "date"
    df = df[df.index.notna()]

    if "close" not in df.columns:
        raise DataUnavailableError("The price table has no 'Close' column.")

    for column in PRICE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = _to_number(df[column])

    return df[PRICE_COLUMNS]


def _to_number(series: pd.Series) -> pd.Series:
    """Convert a column to floats, stripping '$' and thousands separators if present."""
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        series = series.astype(str).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(series, errors="coerce").astype(float)


def clean_price_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Sort, de-duplicate and sanity-check a standard price frame.

    Returns:
        (cleaned_frame, warnings) where ``warnings`` is a list of human-readable
        notes about anything that was removed or looks suspicious.

    Raises:
        DataUnavailableError: if fewer than 2 usable rows remain.
    """
    warnings: list[str] = []
    df = df.sort_index()

    duplicated = df.index.duplicated(keep="last")
    if duplicated.any():
        warnings.append(f"Removed {int(duplicated.sum())} duplicate date row(s) (kept the last copy).")
        df = df[~duplicated]

    bad_close = df["close"].isna() | (df["close"] <= 0)
    if bad_close.any():
        warnings.append(f"Removed {int(bad_close.sum())} row(s) with a missing or non-positive close price.")
        df = df[~bad_close]

    if len(df) < 2:
        raise DataUnavailableError("Fewer than 2 usable trading days of price data.")

    negative_volume = df["volume"] < 0
    if negative_volume.any():
        warnings.append(f"Set {int(negative_volume.sum())} negative volume value(s) to missing.")
        df.loc[negative_volume, "volume"] = np.nan

    missing_volume = int(df["volume"].isna().sum())
    if missing_volume:
        warnings.append(f"{missing_volume} day(s) have no volume data; volume statistics skip those days.")

    zero_volume = int((df["volume"] == 0).sum())
    if zero_volume:
        warnings.append(f"{zero_volume} day(s) report zero volume (possible stale or illiquid quotes).")

    # OHLC consistency checks: reported, not modified.
    has_ohlc = df[["open", "high", "low"]].notna().all(axis=1)
    inconsistent = has_ohlc & (
        (df["high"] < df["low"])
        | (df["close"] > df["high"] * 1.001)
        | (df["close"] < df["low"] * 0.999)
    )
    if inconsistent.any():
        warnings.append(
            f"{int(inconsistent.sum())} day(s) have inconsistent OHLC values "
            "(close outside the high-low range); values were kept as reported."
        )

    gaps = df.index.to_series().diff().dt.days
    long_gaps = gaps[gaps > MAX_NORMAL_GAP_DAYS]
    if not long_gaps.empty:
        examples = ", ".join(d.strftime("%Y-%m-%d") for d in long_gaps.index[:3])
        warnings.append(
            f"{len(long_gaps)} gap(s) longer than {MAX_NORMAL_GAP_DAYS} calendar days between "
            f"trading days (e.g. before {examples}); data may be missing for those periods."
        )

    return df, warnings


def choose_price_column(df: pd.DataFrame) -> tuple[str, str | None]:
    """Decide which price column to use for return calculations.

    The adjusted close is preferred because it includes dividends and splits
    (a "total return" series). If it is missing on any day, the plain close is
    used for every day instead, so adjusted and unadjusted prices are never mixed.

    Returns:
        (column_name, warning_or_None)
    """
    adj = df["adj_close"]
    if adj.notna().all() and (adj > 0).all():
        return "adj_close", None
    if adj.notna().any():
        return "close", (
            "Adjusted close was missing on some days, so returns use the unadjusted close; "
            "dividends are not included in returns."
        )
    return "close", "No adjusted close available; returns use the close price (dividends not included)."


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load daily prices from a CSV file (e.g. downloaded manually from a broker or website).

    The file needs a Date column and at least a Close column. Open, High, Low,
    Adj Close and Volume are used when present. Returns a standard price frame.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataUnavailableError(f"CSV file not found: {csv_path}")
    try:
        raw = pd.read_csv(csv_path)
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
        raise DataUnavailableError(f"Could not read CSV file {csv_path}: {exc}") from exc
    if not any(str(c).strip().lower() in ("date", "datetime", "timestamp") for c in raw.columns):
        raise DataUnavailableError(f"CSV file {csv_path} needs a 'Date' column.")
    logger.info("Loaded %d rows from %s", len(raw), csv_path)
    return standardize_price_frame(raw)
