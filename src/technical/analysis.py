"""Swing technical analysis (2-7 trading-day horizon) for one ticker.

Uses the trader's chart settings: EMA 9/21/50/200/250, RSI 6/14, MACD 12/26/9,
MAVOL20 and BOLL(20, 1.8) (see ``indicators.py``).

Puts the pieces together:
    indicators -> classifications -> structure states -> overall condition
    -> support / resistance -> historical 2-7 day context -> earnings event-risk overlay

Nothing here predicts prices or recommends trades; every output is a
description of the current technical state or of past outcomes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.technical.classification import classify_all, swing_condition
from src.technical.indicators import adjusted_ohlcv, compute_indicators
from src.technical.structure import (
    NO_STRUCTURE,
    active_states,
    proximity_notes,
    structure_states,
    support_resistance,
)
from src.technical.swing_stats import historical_context

MARKET_CLOSE_HOUR = 16
OUTCOME_PREFIXES = ("fwd_return_", "fwd_abnormal_", "mfe_", "mae_")

DEFAULT_TECHNICAL_SETTINGS: dict[str, Any] = {
    "chart_days": 126,
    "structure_lookback_days": 60,
    "pivot_bars": 3,
    "event_risk_days": 7,
    "min_history_samples": 20,
}


def next_earnings(earnings: pd.DataFrame | None, last_date: pd.Timestamp, window_days: int) -> dict[str, Any]:
    """The next earnings release listed AFTER the last trading day in the data.

    The first session that can react is the release day (release before 16:00 New
    York time or time unknown) or the next trading day (release at/after 16:00).
    Trading days are counted Monday-Friday; exchange holidays are not removed, so
    the count is approximate. Dates are never guessed: without a calendar the
    status is "unknown", and with no listed future date it is "none".
    """
    base = {"window_days": window_days, "within_window": False, "release_time": None,
            "reaction_date": None, "sessions_until": None}
    if earnings is None:
        return {**base, "status": "unknown"}
    upcoming = []
    for release in earnings["release_time"].dropna():
        start = release.normalize() + pd.Timedelta(days=1 if release.hour >= MARKET_CLOSE_HOUR else 0)
        session = pd.Timestamp(np.busday_offset(start.date(), 0, roll="forward"))
        if session > last_date:
            upcoming.append((session, release))
    if not upcoming:
        return {**base, "status": "none"}
    session, release = min(upcoming)
    sessions_until = int(np.busday_count((last_date + pd.Timedelta(days=1)).date(),
                                         (session + pd.Timedelta(days=1)).date()))
    return {**base, "status": "scheduled", "release_time": release, "reaction_date": session,
            "sessions_until": sessions_until, "within_window": sessions_until <= window_days}


def run_technical_analysis(prices: pd.DataFrame, price_column: str, stats: pd.DataFrame,
                           earnings: pd.DataFrame | None, settings: dict[str, Any] | None = None,
                           benchmark: str | None = None, ticker: str = "") -> dict[str, Any]:
    """Full swing technical analysis.

    Args:
        prices: cleaned standard price frame (open/high/low/close/adj_close/volume).
        price_column: "adj_close" or "close" (the series used for returns).
        stats: output of add_statistics (+ benchmark columns): forward returns,
            MFE/MAE and, if available, the aligned benchmark price.
        earnings: normalised earnings calendar, or None if it is not available.
        settings: the "technical" section of settings.yaml.
        benchmark: benchmark symbol (None if not used).
    """
    cfg = {**DEFAULT_TECHNICAL_SETTINGS, **(settings or {})}
    ohlcv = adjusted_ohlcv(prices, price_column)
    benchmark_close = stats["benchmark_price"] if benchmark and "benchmark_price" in stats else None
    frame = classify_all(compute_indicators(ohlcv, benchmark_close))
    frame = frame.join(structure_states(frame))
    frame = frame.join(swing_condition(frame))
    outcome_columns = [c for c in stats.columns if c.startswith(OUTCOME_PREFIXES)]
    frame = frame.join(stats[outcome_columns])

    latest = frame.iloc[-1]
    notes: list[str] = []
    filled = ohlcv.attrs.get("filled_high_low_days", 0)
    if filled:
        notes.append(f"{filled} day(s) had no High/Low; the close was used for those days in ATR, VWAP, "
                     "MFE/MAE and swing-point calculations.")
    if len(frame) < 250:
        notes.append(f"Only {len(frame)} trading days of data: EMA250 needs 250 sessions, EMA200 200 and the ATR "
                     "percentile 126, so some values show n/a.")
    elif len(frame) < 500:
        notes.append(f"EMA200 / EMA250 rest on {len(frame)} sessions (fewer than 500), so they still depend a "
                     "little on the first price in the data; they are context only.")
    if latest.get("swing_condition") == "n/a":
        notes.append("Not enough history to classify the overall Swing Technical Condition.")

    states = active_states(latest)
    return {
        "ticker": ticker,
        "benchmark": benchmark if benchmark_close is not None else None,
        "as_of": frame.index[-1],
        "frame": frame,
        "latest": latest,
        "levels": support_resistance(frame, cfg["structure_lookback_days"], cfg["pivot_bars"]),
        "structure": states[0] if states else NO_STRUCTURE,
        "states": states,
        "proximity": proximity_notes(latest),
        "history": historical_context(frame, cfg["min_history_samples"]),
        "event_risk": next_earnings(earnings, frame.index[-1], cfg["event_risk_days"]),
        "condition": latest.get("swing_condition", "n/a"),
        "score": latest.get("swing_score"),
        "settings": cfg,
        "notes": notes,
    }
