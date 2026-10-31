"""Terminal tables in the shared SQLite database (added next to the analyze.py tables).

Upgrade safety: before the terminal tables are created in an EXISTING database for the first
time, the file is copied to data/backups/ - nothing is ever dropped or rewritten.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.database import db

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS terminal_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date  TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,          -- running / success / success_with_warnings / failed
    regime_label  TEXT,
    regime_score  REAL,
    n_universe    INTEGER,
    n_usable      INTEGER,
    n_candidates  INTEGER,
    report_path   TEXT,
    warnings      TEXT,                   -- JSON list
    error         TEXT
);
CREATE TABLE IF NOT EXISTS regime_history (
    date        TEXT PRIMARY KEY,
    composite   REAL,
    label       TEXT,
    components  TEXT,                     -- JSON {component: score}
    updated_at  TEXT
);
CREATE TABLE IF NOT EXISTS scan_candidates (
    session_date TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    direction    TEXT NOT NULL,
    setup        TEXT,
    score        REAL,
    tier         TEXT,
    rank         INTEGER,
    entry        REAL,
    stop         REAL,
    target1      REAL,
    reward_risk  REAL,
    payload      TEXT,                    -- JSON (full candidate)
    PRIMARY KEY (session_date, symbol, direction)
);
CREATE TABLE IF NOT EXISTS options_snapshots (
    symbol            TEXT NOT NULL,
    date              TEXT NOT NULL,
    atm_iv            REAL,
    atm_iv_expiry     TEXT,
    pcr_volume        REAL,
    pcr_oi            REAL,
    call_volume       REAL,
    put_volume        REAL,
    call_oi           REAL,
    put_oi            REAL,
    expected_move_pct REAL,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS catalyst_events (
    symbol        TEXT NOT NULL,
    event_date    TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    form          TEXT,
    sentiment     TEXT,
    status        TEXT,
    source        TEXT,
    url           TEXT,
    published_at  TEXT,
    description   TEXT,
    first_seen    TEXT,
    PRIMARY KEY (symbol, event_date, event_type)
);
CREATE TABLE IF NOT EXISTS data_quality (
    session_date TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    status       TEXT,
    issues       TEXT,
    PRIMARY KEY (session_date, symbol)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonable(value: Any) -> Any:
    """Recursively convert numpy / pandas values to JSON-safe Python (NaN / inf -> None)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def connect(db_path: Path, backups_dir: Path) -> sqlite3.Connection:
    existed = db_path.exists()
    conn = db.connect(db_path)
    has = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='terminal_runs'").fetchone()
    if existed and not has:
        backups_dir.mkdir(parents=True, exist_ok=True)
        target = backups_dir / f"{db_path.stem}_before_terminal_{datetime.now():%Y%m%d_%H%M%S}.db"
        conn.commit()
        shutil.copy2(db_path, target)
        logger.info("Database backed up to %s before adding terminal tables", target)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def start_run(conn: sqlite3.Connection, session: date) -> int:
    with conn:
        cur = conn.execute("INSERT INTO terminal_runs (session_date, started_at, status) VALUES (?, ?, 'running')",
                           (session.isoformat(), utc_now()))
    return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, status: str, **fields: Any) -> None:
    fields = {k: (json.dumps(jsonable(v)) if isinstance(v, (list, dict)) else v) for k, v in fields.items()}
    sets = ", ".join(f"{k} = ?" for k in fields)
    with conn:
        conn.execute(f"UPDATE terminal_runs SET status = ?, finished_at = ?{', ' + sets if sets else ''} "
                     "WHERE run_id = ?", (status, utc_now(), *fields.values(), run_id))


def save_regime(conn: sqlite3.Connection, history: pd.DataFrame, components: list[str]) -> None:
    rows = []
    for d, r in history.tail(400).iterrows():
        rows.append((d.strftime("%Y-%m-%d"), jsonable(r.get("composite")), r.get("label"),
                     json.dumps({c: jsonable(r.get(c)) for c in components}), utc_now()))
    with conn:
        conn.executemany("INSERT INTO regime_history VALUES (?, ?, ?, ?, ?) ON CONFLICT(date) DO UPDATE SET "
                         "composite=excluded.composite, label=excluded.label, components=excluded.components, "
                         "updated_at=excluded.updated_at", rows)


def save_candidates(conn: sqlite3.Connection, session: date, candidates: list[dict[str, Any]]) -> None:
    with conn:
        conn.execute("DELETE FROM scan_candidates WHERE session_date = ?", (session.isoformat(),))
        conn.executemany(
            "INSERT INTO scan_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(session.isoformat(), c["symbol"], c["direction"], c["setup"], c["score"], c["tier"], c.get("rank"),
              jsonable(c["plan"].get("entry")), jsonable(c["plan"].get("stop")), jsonable(c["plan"].get("target1")),
              jsonable(c["plan"].get("reward_risk")), json.dumps(jsonable(c))) for c in candidates])


def save_options(conn: sqlite3.Connection, symbol: str, session: date, o: dict[str, Any]) -> None:
    if not o.get("available") or not o.get("atm_iv"):
        return
    with conn:
        conn.execute("INSERT INTO options_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(symbol, date) "
                     "DO UPDATE SET atm_iv=excluded.atm_iv, atm_iv_expiry=excluded.atm_iv_expiry, "
                     "pcr_volume=excluded.pcr_volume, pcr_oi=excluded.pcr_oi, call_volume=excluded.call_volume, "
                     "put_volume=excluded.put_volume, call_oi=excluded.call_oi, put_oi=excluded.put_oi, "
                     "expected_move_pct=excluded.expected_move_pct",
                     (symbol, session.isoformat(), *(jsonable(o.get(k)) for k in (
                         "atm_iv", "atm_iv_expiry", "pcr_volume", "pcr_oi", "call_volume", "put_volume", "call_oi",
                         "put_oi", "expected_move_pct"))))


def iv_history(conn: sqlite3.Connection, symbol: str, before: date) -> pd.Series:
    rows = conn.execute("SELECT date, atm_iv FROM options_snapshots WHERE symbol = ? AND date < ? ORDER BY date",
                        (symbol, before.isoformat())).fetchall()
    return pd.Series([r[1] for r in rows], index=[r[0] for r in rows], dtype=float)


def save_catalysts(conn: sqlite3.Connection, symbol: str, events: list[dict[str, Any]]) -> None:
    now = utc_now()
    with conn:
        conn.executemany(
            "INSERT INTO catalyst_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(symbol, event_date, "
            "event_type) DO UPDATE SET sentiment=excluded.sentiment, status=excluded.status, url=excluded.url, "
            "description=excluded.description",
            [(symbol, e.get("event_date"), e.get("event_type"), e.get("form"), e.get("sentiment"), e.get("status"),
              e.get("source"), e.get("url"), e.get("published_at"), e.get("description"), now) for e in events])


def save_quality(conn: sqlite3.Connection, session: date, report: dict[str, Any]) -> None:
    with conn:
        conn.executemany("INSERT INTO data_quality VALUES (?, ?, ?, ?) ON CONFLICT(session_date, symbol) DO UPDATE "
                         "SET status=excluded.status, issues=excluded.issues",
                         [(session.isoformat(), r["symbol"], r["status"], json.dumps(r["issues"]))
                          for r in report["symbols"]])
