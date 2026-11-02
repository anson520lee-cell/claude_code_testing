"""Intraday price refresh: every 30 minutes while the US market is open (9:30, 10:00, ... 16:00 New York).

What it does - and what it deliberately does NOT do:

* It downloads today's 5-minute bars for the watch list (the latest report's candidates, open journal
  trades, active price alerts, the index and leveraged ETFs, then the rest of the universe), condenses them into one quote per symbol
  (open / high / low / last / volume so far) and writes ``data/intraday/latest.json``.
* The dashboard shows those quotes as "Live hh:mm ET" next to the stored close.
* Nothing is written to the price database, and scores, setups, plans and alerts are untouched: the
  validated setups are after-close signals entered at the next open, so they are still computed once
  per session by the after-close run (``daily_run.py --if-new``) from the final daily bar.

The background job calls ``intraday_run.py`` every 5 minutes; ``due_slot`` decides (in New York time,
so daylight saving and the user's own time zone do not matter) whether a new half-hour slot has started
on a trading day, and the run exits immediately otherwise. Early-close days (13:00) stop at 13:00.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.daily import store
from src.daily.settings import path
from src.market import calendar

logger = logging.getLogger(__name__)

OPEN = time(9, 30)
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "every_minutes": 30,
    "grace_minutes": 25,         # a slot missed (Mac asleep) is still fetched up to this long after it started
    "max_symbols": 300,          # one batch per slot (~30 s for 250 symbols): candidates first, then the whole universe
    "min_coverage": 0.8,         # a slot counts as fetched only when this share of the symbols returned a quote
    "extra_symbols": ["SPY", "QQQ", "IWM"],
}


def intraday_settings(settings: dict) -> dict[str, Any]:
    return {**DEFAULTS, **(settings.get("intraday") or {})}


def slots(day: date, every_minutes: int = 30) -> list[datetime]:
    """Refresh times (New York, tz-aware) of a trading day: 9:30, 10:00, ... the close (16:00 or 13:00)."""
    if not calendar.is_trading_day(day):
        return []
    start = datetime.combine(day, OPEN, calendar.NEW_YORK)
    end = datetime.combine(day, calendar.close_time(day), calendar.NEW_YORK)
    out, t = [], start
    while t <= end:
        out.append(t)
        t += timedelta(minutes=every_minutes)
    if out[-1] != end:
        out.append(end)
    return out


def due_slot(now: datetime, last_slot: str | None, every_minutes: int = 30,
             grace_minutes: int = 25) -> datetime | None:
    """The slot to fetch now, or None. ``now`` may be naive (New York) or tz-aware (any zone)."""
    now = now.replace(tzinfo=calendar.NEW_YORK) if now.tzinfo is None else now.astimezone(calendar.NEW_YORK)
    current = [s for s in slots(now.date(), every_minutes) if s <= now]
    if not current:
        return None
    slot = current[-1]
    if now - slot > timedelta(minutes=grace_minutes):
        return None
    if last_slot and last_slot >= slot.isoformat():
        return None
    return slot


def latest_file(settings: dict) -> Path:
    return path(settings, "database").parent / "intraday" / "latest.json"


def load_latest(settings: dict) -> dict[str, Any] | None:
    try:
        return json.loads(latest_file(settings).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def watchlist(settings: dict, conn=None) -> list[str]:
    """Latest report's candidates, open journal trades, active alerts, index ETFs, leveraged ETFs, then the
    rest of the universe (capped; the order decides what is dropped first). Every stock page can show a live price."""
    cfg = intraday_settings(settings)
    out: list[str] = []

    def add(sym: Any) -> None:
        if isinstance(sym, str) and sym and sym.upper() not in out:
            out.append(sym.upper())

    reports = path(settings, "reports_dir")
    latest = sorted((p for p in reports.iterdir() if (p / "data.json").exists()), reverse=True) if reports.exists() else []
    if latest:
        try:
            snap = json.loads((latest[0] / "data.json").read_text(encoding="utf-8"))
            for side in (snap.get("candidates") or {}).values():
                for c in side:
                    add(c.get("symbol"))
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    if conn is not None:
        for sql in ("SELECT DISTINCT symbol FROM trade_journal WHERE status = 'open'",
                    "SELECT DISTINCT symbol FROM price_alerts WHERE active = 1"):
            try:
                for (sym,) in conn.execute(sql):
                    add(sym)
            except Exception:  # noqa: BLE001 - tables appear with the first daily run
                pass
    for sym in cfg.get("extra_symbols") or []:
        add(sym)
    for sym in settings.get("leveraged_etfs") or []:
        add(sym)
    for sym in settings.get("universe") or {}:
        add(sym)
    return out[: int(cfg["max_symbols"])]


def summarize(bars: pd.DataFrame, session: date) -> dict[str, Any] | None:
    """5-minute bars (New York index, regular hours) -> today's quote so far + previous session's last price."""
    if bars is None or bars.empty:
        return None
    bars = bars.dropna(subset=["Close"])
    if bars.empty:
        return None
    idx = bars.index
    idx = idx.tz_convert(calendar.NEW_YORK) if idx.tz is not None else idx.tz_localize(calendar.NEW_YORK)
    bars = bars.set_axis(idx)
    today = bars[bars.index.date == session]
    before = bars[bars.index.date < session]
    if today.empty:
        return None
    vol = today["Volume"].fillna(0) if "Volume" in today else None
    return {
        "open": float(today["Open"].iloc[0]), "high": float(today["High"].max()), "low": float(today["Low"].min()),
        "last": float(today["Close"].iloc[-1]), "volume": float(vol.sum()) if vol is not None else None,
        "prev_close": float(before["Close"].iloc[-1]) if not before.empty else None,
        "bar_time": today.index[-1].strftime("%H:%M"),
    }


def yahoo_bars(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """One yfinance batch request: 5-minute bars for today and the previous session, regular hours."""
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    raw = yf.download(symbols, period="2d", interval="5m", prepost=False, auto_adjust=False, actions=False,
                      group_by="ticker", progress=False, threads=4)
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for sym in symbols:
        try:
            frame = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        frame = frame.dropna(how="all")
        if not frame.empty:
            out[sym] = frame
    return out


def refresh(settings: dict, now: datetime | None = None, force: bool = False,
            fetch=yahoo_bars, conn=None) -> dict[str, Any]:
    """Fetch the due slot (or nothing). Returns {"status": "skipped"|"ok"|"empty", ...}."""
    cfg = intraday_settings(settings)
    if not cfg.get("enabled", True) and not force:
        return {"status": "skipped", "reason": "disabled (intraday.enabled: false)"}
    now = datetime.now(calendar.NEW_YORK) if now is None else now
    now = now.replace(tzinfo=calendar.NEW_YORK) if now.tzinfo is None else now.astimezone(calendar.NEW_YORK)
    previous = load_latest(settings) or {}
    slot = due_slot(now, previous.get("slot"), int(cfg["every_minutes"]), int(cfg["grace_minutes"]))
    if slot is None and force and calendar.is_trading_day(now.date()):
        past = [s for s in slots(now.date(), int(cfg["every_minutes"])) if s <= now]
        slot = past[-1] if past else None
    if slot is None:
        return {"status": "skipped", "reason": "no new half-hour slot (market closed or already fetched)"}
    symbols = watchlist(settings, conn)
    bars = fetch(symbols) if symbols else {}
    quotes = {}
    for sym, frame in bars.items():
        q = summarize(frame, now.date())
        if q:
            quotes[sym] = q
    label = slot.strftime("%H:%M")
    if not quotes:
        # Yahoo often has no bar yet at 9:30-9:35 (or hiccups): write nothing and do not mark the slot as done,
        # so the next 5-minute run asks again (instead of showing the previous close for another 30 minutes)
        return {"status": "empty", "slot": label, "symbols": len(symbols), "quotes": 0}
    same_session = previous.get("session") == now.date().isoformat()
    merged = {**(previous.get("quotes") or {}), **quotes} if same_session else quotes
    complete = len(quotes) >= math.ceil(float(cfg["min_coverage"]) * len(symbols))
    payload = {"session": now.date().isoformat(),
               # a partly filled slot keeps the previous slot id, so due_slot() asks again at the next run
               "slot": slot.isoformat() if complete else (previous.get("slot") if same_session else None),
               "slot_label": label, "complete": complete,
               "fetched_at": store.utc_now(), "requested": len(symbols), "quotes": merged,
               "note": "Display only - scores, setups and plans use the after-close daily bar."}
    target = latest_file(settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    store.write_atomic(target, json.dumps(store.jsonable(payload)))
    return {"status": "ok" if complete else "partial", "slot": label, "symbols": len(symbols), "quotes": len(quotes)}


def overlay(quote: dict[str, Any], stored_as_of: str, latest: dict[str, Any] | None, symbol: str) -> dict[str, Any] | None:
    """Live quote for the chart API when the intraday file is from a session after the stored daily bar."""
    if not latest or latest.get("session", "") <= stored_as_of:
        return None
    q = (latest.get("quotes") or {}).get(symbol)
    if not q:
        return None
    prev = quote.get("close")              # last stored (final) close = the live session's previous close...
    if calendar.previous_trading_day(date.fromisoformat(latest["session"])).isoformat() != stored_as_of:
        prev = q.get("prev_close")         # ...unless the after-close run has not stored that session yet
    last = q["last"]
    return {**quote, "open": q["open"], "high": q["high"], "low": q["low"], "close": last, "volume": q["volume"],
            "prev_close": prev, "change": last - prev if prev else None,
            "change_pct": last / prev - 1 if prev else None,
            "amplitude": (q["high"] - q["low"]) / prev if prev else None,
            "dollar_volume": (q["volume"] or 0) * (q["high"] + q["low"] + last) / 3,
            "avg_price": (q["high"] + q["low"] + last) / 3, "volume_ratio": None,
            "live": {"session": latest["session"], "slot": latest.get("slot_label"), "bar_time": q.get("bar_time"),
                     "fetched_at": latest.get("fetched_at")}}
