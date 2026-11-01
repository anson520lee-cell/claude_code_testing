"""Batch daily-bar download for the whole universe, stored in the shared SQLite database.

Strategy (cheap and idempotent - running twice on the same day changes nothing):

1. A symbol with no (or too little) stored history gets a full download
   (``history_period``, default 2 years).
2. Otherwise only the last ``refresh_days`` calendar days are downloaded and upserted.
3. Corporate-action check: Yahoo re-adjusts past prices after splits (Close) and
   dividends (Adj Close). If the freshly downloaded overlap disagrees with the stored
   rows by more than 0.5%, the stored history is stale -> the symbol is re-downloaded
   in full, so adjusted and unadjusted prices are never mixed.
4. Bars dated AFTER the latest completed session are discarded (a run during market
   hours would otherwise store an incomplete, still-changing bar).

All network access goes through yfinance (``yf.download``, many symbols per request).
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from src.data.prices import standardize_price_frame
from src.database import db

logger = logging.getLogger(__name__)

SOURCE = "Yahoo Finance (via yfinance)"
OVERLAP_TOLERANCE = 0.005
CHUNK = 60


def _split_download(raw: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """yf.download(group_by='ticker') -> {symbol: standard price frame} (empty frames dropped)."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({symbols[0]: raw}, axis=1)
    level0 = set(raw.columns.get_level_values(0))
    for symbol in symbols:
        if symbol not in level0:
            continue
        frame = raw[symbol].dropna(how="all")
        if frame.empty or frame.get("Close") is None or frame["Close"].dropna().empty:
            continue
        try:
            out[symbol] = standardize_price_frame(frame).dropna(subset=["close"])
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the batch
            logger.warning("Could not parse prices for %s: %s", symbol, exc)
    return out


class YahooBatchDownloader:
    """Thin wrapper around yfinance.download with retries (injectable for tests)."""

    def __init__(self, attempts: int = 3, retry_wait_seconds: float = 2.0) -> None:
        import yfinance

        self._yf = yfinance
        self.attempts = max(1, attempts)
        self.wait = retry_wait_seconds
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    def download(self, symbols: list[str], period: str | None = None,
                 start: date | None = None) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), CHUNK):
            chunk = symbols[i:i + CHUNK]
            missing = list(chunk)
            for attempt in range(1, self.attempts + 1):
                kwargs: dict[str, Any] = dict(interval="1d", auto_adjust=False, actions=False, group_by="ticker",
                                              progress=False, threads=True)
                if start is not None:
                    kwargs["start"] = start.isoformat()
                else:
                    kwargs["period"] = period or "2y"
                try:
                    raw = self._yf.download(missing, **kwargs)
                    got = _split_download(raw, missing)
                    out.update(got)
                    missing = [s for s in missing if s not in got]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Batch download failed (attempt %d): %s", attempt, exc)
                if not missing:
                    break
                if attempt < self.attempts:
                    time.sleep(self.wait * 2 ** (attempt - 1))
            if missing:
                logger.warning("No data returned for: %s", ", ".join(missing))
        return out


def _overlap_mismatch(stored: pd.DataFrame, fresh: pd.DataFrame) -> float:
    """Largest relative difference of close / adj_close on dates present in both frames."""
    common = stored.index.intersection(fresh.index)
    if len(common) == 0:
        return 0.0
    worst = 0.0
    for column in ("close", "adj_close"):
        a, b = stored.loc[common, column], fresh.loc[common, column]
        valid = a.notna() & b.notna() & (a > 0)
        if valid.any():
            worst = max(worst, float(((b[valid] / a[valid]) - 1).abs().max()))
    return worst


def update_prices(conn, symbols: list[str], session: date, settings: dict[str, Any],
                  downloader: Any | None = None, offline: bool = False,
                  progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Bring stored daily bars up to ``session`` for every symbol. Returns a summary dict."""
    cfg = settings["data"]
    min_days = int(cfg.get("min_history_days", 260))
    summary: dict[str, Any] = {"full": [], "incremental": [], "readjusted": [], "failed": [], "offline": offline,
                               "rows_written": 0, "discarded_future_bars": 0}
    if offline:
        return summary
    downloader = downloader or YahooBatchDownloader(cfg.get("download_attempts", 3), cfg.get("retry_wait_seconds", 2))

    stored_info: dict[str, tuple[int, str | None]] = {}
    for row in conn.execute("SELECT ticker, COUNT(*), MAX(date) FROM price_data GROUP BY ticker"):
        stored_info[row[0]] = (int(row[1]), row[2])
    full = [s for s in symbols if stored_info.get(s, (0, None))[0] < min_days]
    incremental = [s for s in symbols if s not in full]

    def save(symbol: str, frame: pd.DataFrame) -> None:
        future = frame.index > pd.Timestamp(session)
        if future.any():
            summary["discarded_future_bars"] += int(future.sum())
            frame = frame[~future]
        if frame.empty:
            return
        db.upsert_stock(conn, symbol)
        summary["rows_written"] += db.upsert_prices(conn, symbol, frame, SOURCE)

    if incremental:
        start = session - timedelta(days=int(cfg.get("refresh_days", 15)))
        if progress:
            progress(f"Refreshing {len(incremental)} symbols from {start}")
        fresh = downloader.download(incremental, start=start)
        for symbol in incremental:
            frame = fresh.get(symbol)
            if frame is None:
                summary["failed"].append(symbol)
                continue
            stored = db.load_prices(conn, symbol)
            if _overlap_mismatch(stored, frame) > OVERLAP_TOLERANCE:
                summary["readjusted"].append(symbol)
                full.append(symbol)
                continue
            save(symbol, frame)
            summary["incremental"].append(symbol)

    if full:
        if progress:
            progress(f"Downloading full history ({cfg.get('history_period', '2y')}) for {len(full)} symbols")
        fresh = downloader.download(full, period=cfg.get("history_period", "2y"))
        for symbol in full:
            frame = fresh.get(symbol)
            if frame is None:
                if symbol not in summary["failed"]:
                    summary["failed"].append(symbol)
                continue
            if symbol in summary["readjusted"]:
                with conn:  # stale (pre-adjustment) history is replaced, not mixed
                    conn.execute("DELETE FROM price_data WHERE ticker = ? AND date < ?",
                                 (symbol, frame.index[0].strftime("%Y-%m-%d")))
            save(symbol, frame)
            summary["full"].append(symbol)
    return summary


def load_universe_prices(conn, symbols: list[str], session: date, max_days: int = 600) -> dict[str, pd.DataFrame]:
    """Stored bars up to and including ``session`` (latest ``max_days`` sessions per symbol)."""
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = db.load_prices(conn, symbol)
        if frame.empty:
            continue
        frame = frame[frame.index <= pd.Timestamp(session)].tail(max_days)
        if not frame.empty:
            out[symbol] = frame
    return out


# ------------------------------------------------------------------------ FRED
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def fetch_fred(series: list[str], session: date, timeout: float = 20.0,
               getter: Callable[..., Any] | None = None) -> pd.DataFrame:
    """Daily Treasury constant-maturity yields (percent) from FRED, one column per series.

    FRED publishes H.15 yields with a lag of ~1 business day, so the latest value may
    be from the previous session; the date of every value is kept.
    """
    getter = getter or requests.get
    frames = []
    for name in series:
        try:
            response = getter(FRED_URL.format(series=name), timeout=timeout)
            response.raise_for_status()
            frame = pd.read_csv(io.StringIO(response.text))
            frame.columns = ["date", name]
            frame["date"] = pd.to_datetime(frame["date"])
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
            frames.append(frame.set_index("date"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("FRED series %s unavailable: %s", name, exc)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out = out[out.index <= pd.Timestamp(session)]
    return out[out.index >= pd.Timestamp(session) - pd.Timedelta(days=400)].dropna(how="all")


def dollar_volume(frame: pd.DataFrame, window: int = 20) -> float:
    tail = frame.tail(window)
    value = (tail["close"] * tail["volume"]).mean()
    return float(value) if np.isfinite(value) else float("nan")
