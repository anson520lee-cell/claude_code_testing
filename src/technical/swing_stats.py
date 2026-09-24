"""Historical 2-7 day outcomes after conditions comparable to today's.

For a condition (e.g. "Pullback to EMA9", or "Short-term uptrend with RSI6 in
positive momentum") every past trading day that met it is found, and what happened next
is summarised:

* forward return after 2, 3, 5 and 7 trading days: average, median, % positive
* forward return minus the benchmark's over the same days (average)
* maximum favourable / adverse excursion within 3, 5 and 7 days (average, median)

All days in the sample must have a complete 7-day forward window, so every
horizon uses the same days and N is the same for every column.

Days in a row that meet the same condition have overlapping forward windows,
so N overstates the number of independent observations; the number of
separate episodes is shown next to N. These are descriptive statistics only:
no significance test is applied and nothing here is a prediction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

HORIZONS = (2, 3, 5, 7)
EXCURSIONS = (3, 5, 7)
MIN_DISPLAY_SAMPLES = 5


def _value(series: pd.Series, how: str) -> float | None:
    series = series.dropna()
    if series.empty:
        return None
    if how == "mean":
        return float(series.mean())
    if how == "median":
        return float(series.median())
    return float((series > 0).mean())


def count_episodes(mask: pd.Series) -> int:
    """Number of separate runs of consecutive True values."""
    mask = mask.fillna(False).astype(bool)
    return int((mask & ~mask.shift(1, fill_value=False)).sum())


def condition_outcomes(frame: pd.DataFrame, mask: pd.Series, label: str,
                       min_samples: int = 20) -> dict[str, Any]:
    """Outcome statistics for the days where ``mask`` is True (and a 7-day future exists)."""
    usable = mask.fillna(False).astype(bool) & frame["fwd_return_7d"].notna()
    sample = frame[usable]
    result: dict[str, Any] = {
        "label": label,
        "n": int(len(sample)),
        "episodes": count_episodes(usable),
        "small_sample": len(sample) < min_samples,
        "insufficient": len(sample) < MIN_DISPLAY_SAMPLES,
        "rows": [],
    }
    for h in HORIZONS:
        row: dict[str, Any] = {
            "horizon": h,
            "avg": _value(sample[f"fwd_return_{h}d"], "mean"),
            "median": _value(sample[f"fwd_return_{h}d"], "median"),
            "pct_positive": _value(sample[f"fwd_return_{h}d"], "share"),
            "avg_vs_benchmark": (_value(sample[f"fwd_abnormal_{h}d"], "mean")
                                 if f"fwd_abnormal_{h}d" in sample else None),
        }
        for kind in ("mfe", "mae"):
            column = f"{kind}_{h}d"
            has = h in EXCURSIONS and column in sample
            row[f"{kind}_avg"] = _value(sample[column], "mean") if has else None
            row[f"{kind}_median"] = _value(sample[column], "median") if has else None
        result["rows"].append(row)
    return result


def historical_context(frame: pd.DataFrame, min_samples: int = 20) -> list[dict[str, Any]]:
    """Statistics for the conditions that describe TODAY, plus an all-days baseline.

    Groups: (1) today's primary structure state, if there is one; (2) today's
    swing trend (EMA9 / EMA21) together with today's RSI6 zone; (3) all trading days.
    """
    today = frame.iloc[-1]
    groups: list[dict[str, Any]] = []
    structure = today.get("structure")
    if isinstance(structure, str) and structure in frame.columns:
        groups.append(condition_outcomes(frame, frame[structure], f"Same structure: {structure}", min_samples))
    trend, zone = today.get("trend"), today.get("rsi_zone")
    if trend not in (None, "n/a") and zone not in (None, "n/a"):
        mask = (frame["trend"] == trend) & (frame["rsi_zone"] == zone)
        groups.append(condition_outcomes(frame, mask, f"Same swing trend and RSI6 zone: {trend}, RSI6 {zone}",
                                         min_samples))
    baseline = condition_outcomes(frame, pd.Series(True, index=frame.index), "All trading days (baseline)",
                                  min_samples)
    baseline["baseline"] = True
    groups.append(baseline)
    return groups
