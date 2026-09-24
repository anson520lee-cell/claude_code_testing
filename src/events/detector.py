"""Detect abnormal price/volume days ("events") from the statistics frame.

A day is an event when ANY enabled rule in settings.yaml ("events" section) is true:

    return_zscore   |Return Z-score| >= return_zscore_threshold
    abs_return      |Daily return|   >= abs_return_threshold
    volume_ratio    Volume ratio     >= volume_ratio_threshold
    volume_zscore   Volume Z-score   >= volume_zscore_threshold

The detector only describes WHAT happened in the data (direction, which rules
fired). It never guesses WHY. The research fields (category, description,
source, verification status, notes) start as "Unknown"/"Unverified"/empty and
are only filled by real evidence (see evidence.py) or by the researcher.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.analysis.statistics import EXCURSION_HORIZONS, FORWARD_HORIZONS

PRICE_RULES = ("return_zscore", "abs_return")
VOLUME_RULES = ("volume_ratio", "volume_zscore")

UNKNOWN_CATEGORY = "Unknown"
UNVERIFIED = "Unverified"

# What happened after the event: forward returns, forward returns relative to the
# benchmark, and maximum favourable / adverse excursions (MFE / MAE).
OUTCOME_COLUMNS = (
    [f"fwd_return_{h}d" for h in FORWARD_HORIZONS]
    + [f"fwd_abnormal_{h}d" for h in FORWARD_HORIZONS]
    + [f"mfe_{h}d" for h in EXCURSION_HORIZONS]
    + [f"mae_{h}d" for h in EXCURSION_HORIZONS]
)

# Columns of the events table, in the order they are saved to events.csv.
EVENT_COLUMNS = [
    "ticker", "event_date", "close", "daily_return", "return_zscore",
    "volume", "avg_volume", "volume_ratio", "volume_zscore",
    "benchmark", "benchmark_return", "abnormal_return",
    *OUTCOME_COLUMNS,
    "direction", "is_price_event", "is_volume_event", "trigger_rules", "anomaly_type",
    "trading_days_since_prev_event",
    "event_category", "event_description", "source", "verification_status", "notes",
]


def build_rule_flags(df: pd.DataFrame, event_settings: dict[str, Any]) -> pd.DataFrame:
    """Return a True/False table with one column per ENABLED rule and one row per day.

    Days where a rule cannot be evaluated (NaN, e.g. not enough history yet)
    are False for that rule.
    """
    rules: dict[str, pd.Series] = {}
    threshold = event_settings.get("return_zscore_threshold")
    if threshold is not None:
        rules["return_zscore"] = df["return_zscore"].abs() >= threshold
    threshold = event_settings.get("abs_return_threshold")
    if threshold is not None:
        rules["abs_return"] = df["daily_return"].abs() >= threshold
    threshold = event_settings.get("volume_ratio_threshold")
    if threshold is not None:
        rules["volume_ratio"] = df["volume_ratio"] >= threshold
    threshold = event_settings.get("volume_zscore_threshold")
    if threshold is not None:
        rules["volume_zscore"] = df["volume_zscore"] >= threshold
    if not rules:
        raise ValueError("No event rules are enabled - check the 'events' section of settings.yaml.")
    # Comparisons with NaN are already False; fillna protects against pandas NA values.
    return pd.DataFrame(rules, index=df.index).fillna(False).astype(bool)


def _anomaly_type(is_price: bool, is_volume: bool) -> str:
    if is_price and is_volume:
        return "Abnormal price move with abnormal volume"
    if is_price:
        return "Abnormal price move (volume not flagged)"
    return "Abnormal volume (price move not flagged)"


def detect_events(
    df: pd.DataFrame, ticker: str, event_settings: dict[str, Any], benchmark: str | None = None
) -> pd.DataFrame:
    """Find abnormal days and return them as an events table (one row per event day).

    Args:
        df: statistics frame (output of add_statistics, optionally with benchmark columns).
        ticker: ticker symbol stored with every event.
        event_settings: the "events" section of the settings.
        benchmark: benchmark symbol used for the abnormal-return columns (or None).

    Returns:
        DataFrame with the columns in ``EVENT_COLUMNS`` (empty if no events).
    """
    flags = build_rule_flags(df, event_settings)
    is_event = flags.any(axis=1)
    if not is_event.any():
        return pd.DataFrame(columns=EVENT_COLUMNS)

    # Trading-day position of every row, used to measure spacing between events.
    positions = pd.Series(range(len(df)), index=df.index)
    event_positions = positions[is_event]

    rows = []
    previous_position: int | None = None
    for date in event_positions.index:
        row = df.loc[date]
        fired = [name for name in flags.columns if flags.at[date, name]]
        is_price = any(name in PRICE_RULES for name in fired)
        is_volume = any(name in VOLUME_RULES for name in fired)
        daily_return = row["daily_return"]
        if pd.isna(daily_return) or daily_return == 0:
            direction = "flat"
        else:
            direction = "up" if daily_return > 0 else "down"

        position = int(event_positions.loc[date])
        record = {
            "ticker": ticker,
            "event_date": date,
            "close": row["close"],
            "daily_return": daily_return,
            "return_zscore": row.get("return_zscore"),
            "volume": row.get("volume"),
            "avg_volume": row.get("avg_volume"),
            "volume_ratio": row.get("volume_ratio"),
            "volume_zscore": row.get("volume_zscore"),
            "benchmark": benchmark if pd.notna(row.get("benchmark_return", float("nan"))) else None,
            "benchmark_return": row.get("benchmark_return"),
            "abnormal_return": row.get("abnormal_return"),
            "direction": direction,
            "is_price_event": is_price,
            "is_volume_event": is_volume,
            "trigger_rules": ",".join(fired),
            "anomaly_type": _anomaly_type(is_price, is_volume),
            "trading_days_since_prev_event": (
                position - previous_position if previous_position is not None else None
            ),
            # Research fields: unknown until real evidence is attached.
            "event_category": UNKNOWN_CATEGORY,
            "event_description": None,
            "source": None,
            "verification_status": UNVERIFIED,
            "notes": None,
        }
        for column in OUTCOME_COLUMNS:
            record[column] = row.get(column)
        rows.append(record)
        previous_position = position

    events = pd.DataFrame(rows)
    return events.reindex(columns=EVENT_COLUMNS)
