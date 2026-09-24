"""Summarise what happened AFTER abnormal events ("historical event outcomes").

For groups of events (e.g. "up moves with abnormal volume") this calculates
the average / median forward return over 1, 5 and 20 trading days and the
share of events where the forward return was positive.

A baseline row, "All trading days", shows the same numbers for every day in
the sample. Comparing a group with the baseline answers "is this better or
worse than a random day?".

Reading the "% positive" columns:
* for UP-move events, % positive = how often the move CONTINUED;
* for DOWN-move events, % positive = how often the move REVERSED.

Caveats (repeated in the report): event counts are often small, windows of
events that are close together overlap, and no significance test is applied.
These are descriptive statistics, not predictions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.analysis.statistics import FORWARD_HORIZONS


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def event_groups(events: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks selecting each event group (keys are the labels shown in reports)."""
    price = _as_bool(events["is_price_event"])
    volume = _as_bool(events["is_volume_event"])
    up = events["direction"] == "up"
    down = events["direction"] == "down"
    return {
        "All events": pd.Series(True, index=events.index),
        "Up moves (price rule fired)": price & up,
        "Down moves (price rule fired)": price & down,
        "Up moves with abnormal volume": price & up & volume,
        "Up moves without abnormal volume": price & up & ~volume,
        "Down moves with abnormal volume": price & down & volume,
        "Down moves without abnormal volume": price & down & ~volume,
        "Volume-only events": volume & ~price,
    }


def _describe(frame: pd.DataFrame, label: str, n_events: int) -> dict[str, Any]:
    row: dict[str, Any] = {"group": label, "n_events": n_events}
    for h in FORWARD_HORIZONS:
        fwd = frame[f"fwd_return_{h}d"].dropna() if f"fwd_return_{h}d" in frame else pd.Series(dtype=float)
        row[f"n_{h}d"] = int(fwd.size)
        row[f"mean_{h}d"] = float(fwd.mean()) if fwd.size else None
        row[f"median_{h}d"] = float(fwd.median()) if fwd.size else None
        row[f"pct_positive_{h}d"] = float((fwd > 0).mean()) if fwd.size else None
        column = f"fwd_abnormal_{h}d"
        abnormal = frame[column].dropna() if column in frame else pd.Series(dtype=float)
        row[f"mean_abnormal_{h}d"] = float(abnormal.mean()) if abnormal.size else None
    return row


def summarize_event_outcomes(events: pd.DataFrame, all_days: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forward-return statistics per event group, plus an "All trading days" baseline row.

    Args:
        events: events table (from detect_events or the database).
        all_days: optional statistics frame for the baseline row.

    Returns:
        DataFrame with one row per non-empty group.
    """
    rows = []
    if not events.empty:
        for label, mask in event_groups(events).items():
            subset = events[mask]
            if len(subset):
                rows.append(_describe(subset, label, len(subset)))
    if all_days is not None and not all_days.empty:
        rows.append(_describe(all_days, "All trading days (baseline)", int(all_days["daily_return"].notna().sum())))
    return pd.DataFrame(rows)
