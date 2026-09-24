"""SQLite database access: create tables, save and load data.

Why SQLite? It is built into Python (no server, no extra install), stores
everything in one file (data/stock_research.db) and is fast enough for
millions of rows. It can be opened with any SQLite viewer, e.g. "DB Browser
for SQLite", or from Python/pandas with ``pd.read_sql``.

Duplicate handling: every table has a natural unique key, e.g. (ticker, date)
for prices and (ticker, event_date) for events. Saving uses SQLite "upserts"
(INSERT ... ON CONFLICT DO UPDATE), so running the analysis twice updates
existing rows instead of creating duplicates.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.events.detector import EVENT_COLUMNS, OUTCOME_COLUMNS, UNKNOWN_CATEGORY, UNVERIFIED
from src.events.evidence import AUTO_MATCH_STATUS, EARNINGS_CATEGORY, EARNINGS_SOURCE

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Research values the program writes itself. A stored event's research fields
# are only refreshed automatically while ALL of them still have such values;
# as soon as you change the category, source or verification status by hand,
# the program never touches that event's research fields again.
AUTOMATIC_STATUSES = (UNVERIFIED, AUTO_MATCH_STATUS)
AUTOMATIC_CATEGORIES = (UNKNOWN_CATEGORY, EARNINGS_CATEGORY)
AUTOMATIC_SOURCES = (EARNINGS_SOURCE,)

# The events table stores every column of the detector's events table plus the
# detection settings used (kept in the database only, for provenance).
_SETTINGS_POSITION = EVENT_COLUMNS.index("trading_days_since_prev_event") + 1
EVENT_DB_COLUMNS = [*EVENT_COLUMNS[:_SETTINGS_POSITION], "detection_settings", *EVENT_COLUMNS[_SETTINGS_POSITION:]]
# Columns recalculated from market data on every run.
EVENT_QUANT_COLUMNS = [
    c for c in EVENT_DB_COLUMNS
    if c not in ("ticker", "event_date", "event_category", "event_description",
                 "source", "verification_status", "notes")
]
RESEARCH_COLUMNS = ("event_category", "event_description", "source", "verification_status")
BENCHMARK_COLUMNS = (
    "benchmark", "benchmark_return", "abnormal_return",
    *(c for c in OUTCOME_COLUMNS if c.startswith("fwd_abnormal")),
)
# Outcome columns that only depend on the stock's own prices (fill in as time passes).
PRICE_OUTCOME_COLUMNS = tuple(c for c in OUTCOME_COLUMNS if not c.startswith("fwd_abnormal"))


def utc_now() -> str:
    """Current UTC time as ISO text, e.g. '2026-09-24T16:30:00'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def to_db_value(value: Any) -> Any:
    """Convert pandas/numpy values to plain Python values SQLite understands (NaN -> NULL)."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(float(value)) or math.isinf(float(value)) else float(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (and if needed create) the database and make sure all tables exist."""
    path = Path(db_path)
    if str(db_path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    _add_missing_event_columns(conn)
    return conn


def _add_missing_event_columns(conn: sqlite3.Connection) -> None:
    """Safe migration for databases created by an older version of this program.

    New outcome columns (e.g. 2/3/7-day forward returns, MFE/MAE) are added to an
    existing ``events`` table; existing rows and columns are left untouched (the
    new columns start empty and are filled by the next run of that ticker).
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    missing = [c for c in OUTCOME_COLUMNS if c not in existing]
    if missing:
        with conn:
            for column in missing:
                conn.execute(f"ALTER TABLE events ADD COLUMN {column} REAL")
        logger.info("Database upgraded: added event columns %s", ", ".join(missing))


# --------------------------------------------------------------------- stocks
def upsert_stock(conn: sqlite3.Connection, ticker: str, profile: dict[str, Any] | None = None) -> None:
    """Create the stock row if needed and update any profile fields that are known."""
    profile = profile or {}
    now = utc_now()
    with conn:
        conn.execute(
            """
            INSERT INTO stocks (ticker, name, exchange, sector, industry, currency, country,
                                quote_type, first_analyzed, last_analyzed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                name       = COALESCE(excluded.name, stocks.name),
                exchange   = COALESCE(excluded.exchange, stocks.exchange),
                sector     = COALESCE(excluded.sector, stocks.sector),
                industry   = COALESCE(excluded.industry, stocks.industry),
                currency   = COALESCE(excluded.currency, stocks.currency),
                country    = COALESCE(excluded.country, stocks.country),
                quote_type = COALESCE(excluded.quote_type, stocks.quote_type),
                last_analyzed = excluded.last_analyzed
            """,
            (
                ticker, profile.get("name"), profile.get("exchange"), profile.get("sector"),
                profile.get("industry"), profile.get("currency"), profile.get("country"),
                profile.get("quote_type"), now, now,
            ),
        )


def get_stock(conn: sqlite3.Connection, ticker: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM stocks WHERE ticker = ?", (ticker,)).fetchone()
    return dict(row) if row else None


# ----------------------------------------------------------------- price_data
def upsert_prices(conn: sqlite3.Connection, ticker: str, prices: pd.DataFrame, source: str) -> int:
    """Save daily prices. Existing (ticker, date) rows are updated. Returns rows written."""
    now = utc_now()
    rows = [
        (
            ticker, date.strftime("%Y-%m-%d"),
            to_db_value(r.open), to_db_value(r.high), to_db_value(r.low), to_db_value(r.close),
            to_db_value(r.adj_close), to_db_value(r.volume), source, now,
        )
        for date, r in prices.iterrows()
    ]
    with conn:
        conn.executemany(
            """
            INSERT INTO price_data (ticker, date, open, high, low, close, adj_close, volume, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                open = excluded.open, high = excluded.high, low = excluded.low,
                close = excluded.close, adj_close = excluded.adj_close, volume = excluded.volume,
                source = excluded.source, updated_at = excluded.updated_at
            """,
            rows,
        )
    return len(rows)


def load_prices(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """Load stored daily prices for a ticker as a standard price frame (may be empty)."""
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, adj_close, volume FROM price_data "
        "WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,),
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df.astype(float)


def latest_price_update(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute("SELECT MAX(updated_at) FROM price_data WHERE ticker = ?", (ticker,)).fetchone()
    return row[0] if row else None


# --------------------------------------------------------------------- events
def upsert_events(
    conn: sqlite3.Connection, events: pd.DataFrame, run_id: int | None = None
) -> dict[str, int]:
    """Save events without creating duplicates.

    Rules for an event that is already stored (same ticker and date):

    * quantitative columns (returns, Z-scores, forward returns ...) are updated;
    * research columns (category, description, source, verification status)
      are only refreshed while the stored category, source and status are all
      values the program writes itself AND the new data actually contains
      evidence - so existing evidence is never replaced by "Unknown", and
      anything you changed by hand is never overwritten;
    * notes are never overwritten once they contain text.

    Returns:
        {"inserted": n_new_rows, "updated": n_existing_rows}
    """
    if events.empty:
        return {"inserted": 0, "updated": 0}

    now = utc_now()
    tickers = events["ticker"].unique().tolist()
    existing: set[tuple[str, str]] = set()
    for ticker in tickers:
        for row in conn.execute("SELECT event_date FROM events WHERE ticker = ?", (ticker,)):
            existing.add((ticker, row[0]))

    columns = EVENT_DB_COLUMNS + ["first_run_id", "last_run_id", "created_at", "updated_at"]
    placeholders = ", ".join("?" for _ in columns)

    def quant_update(column: str) -> str:
        if column in BENCHMARK_COLUMNS:
            # A run without a benchmark keeps the previously stored benchmark figures.
            return f"{column} = CASE WHEN excluded.benchmark IS NULL THEN events.{column} ELSE excluded.{column} END"
        if column in PRICE_OUTCOME_COLUMNS:
            # A known forward return / excursion is never replaced by "unknown".
            return f"{column} = COALESCE(excluded.{column}, events.{column})"
        return f"{column} = excluded.{column}"

    quant_updates = ",\n                ".join(quant_update(c) for c in EVENT_QUANT_COLUMNS)
    def placeholders_for(values: tuple[str, ...]) -> str:
        return ", ".join("?" for _ in values)

    refresh = (
        f"events.verification_status IN ({placeholders_for(AUTOMATIC_STATUSES)}) "
        f"AND events.event_category IN ({placeholders_for(AUTOMATIC_CATEGORIES)}) "
        f"AND (events.source IS NULL OR events.source IN ({placeholders_for(AUTOMATIC_SOURCES)})) "
        "AND excluded.event_category <> ?"
    )
    refresh_params = [*AUTOMATIC_STATUSES, *AUTOMATIC_CATEGORIES, *AUTOMATIC_SOURCES, UNKNOWN_CATEGORY]
    research_updates = ",\n                ".join(
        f"{c} = CASE WHEN {refresh} THEN excluded.{c} ELSE events.{c} END" for c in RESEARCH_COLUMNS
    )
    sql = f"""
        INSERT INTO events ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(ticker, event_date) DO UPDATE SET
                {quant_updates},
                {research_updates},
                notes = COALESCE(events.notes, excluded.notes),
                last_run_id = excluded.last_run_id,
                updated_at = excluded.updated_at
    """
    # Each research CASE expression uses the refresh placeholders once.
    extra_params = refresh_params * len(RESEARCH_COLUMNS)

    inserted = updated = 0
    with conn:
        for record in events.to_dict(orient="records"):
            values = [to_db_value(record.get(c)) for c in EVENT_DB_COLUMNS]
            values += [run_id, run_id, now, now]
            conn.execute(sql, values + extra_params)
            key = (record["ticker"], to_db_value(record["event_date"]))
            if key in existing:
                updated += 1
            else:
                inserted += 1
                existing.add(key)
    return {"inserted": inserted, "updated": updated}


FORWARD_COLUMNS = list(OUTCOME_COLUMNS)


def refresh_forward_returns(
    conn: sqlite3.Connection, ticker: str, stats: pd.DataFrame, benchmark: str | None = None
) -> int:
    """Update forward returns of ALL stored events of a ticker from fresh statistics.

    Events stored by earlier runs (possibly with other thresholds) have forward
    returns that were unknown at the time (e.g. an event 3 days before the
    last run has no 20-day forward return yet). This fills them in.
    A stored value is only replaced by a known (non-missing) value, and
    forward ABNORMAL returns are only refreshed when the stored event used the
    same benchmark as this run.

    Returns:
        Number of event rows updated.
    """
    raw_columns = [c for c in PRICE_OUTCOME_COLUMNS if c in stats.columns]
    abnormal_columns = [c for c in FORWARD_COLUMNS if c.startswith("fwd_abnormal") and c in stats.columns]
    stored = conn.execute("SELECT event_date, benchmark FROM events WHERE ticker = ?", (ticker,)).fetchall()
    count = 0
    with conn:
        for event_date, stored_benchmark in stored:
            date = pd.Timestamp(event_date)
            if date not in stats.index:
                continue
            columns = list(raw_columns)
            if benchmark and stored_benchmark == benchmark:
                columns += abnormal_columns
            if not columns:
                continue
            assignments = ", ".join(f"{c} = COALESCE(?, {c})" for c in columns)
            values = [to_db_value(stats.at[date, c]) for c in columns]
            conn.execute(
                f"UPDATE events SET {assignments} WHERE ticker = ? AND event_date = ?",
                [*values, ticker, event_date],
            )
            count += 1
    return count


def load_events(
    conn: sqlite3.Connection, ticker: str | None = None, tickers: Iterable[str] | None = None
) -> pd.DataFrame:
    """Load stored events (all tickers, one ticker, or a list of tickers), oldest first."""
    query = "SELECT * FROM events"
    params: list[Any] = []
    if ticker:
        query += " WHERE ticker = ?"
        params.append(ticker)
    elif tickers:
        tickers = list(tickers)
        query += f" WHERE ticker IN ({', '.join('?' for _ in tickers)})"
        params.extend(tickers)
    query += " ORDER BY ticker, event_date"
    df = pd.read_sql_query(query, conn, params=params)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


# --------------------------------------------------------------- fundamentals
def upsert_fundamentals(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
    """Save normalised fundamental records. Same (ticker, period_type, period_end, metric) -> update."""
    rows = [
        (
            r["ticker"], r["period_type"], r["period_end"], r["metric"], to_db_value(r.get("value")),
            r.get("unit"), r.get("source"), r.get("retrieved_at"),
        )
        for r in records
    ]
    with conn:
        conn.executemany(
            """
            INSERT INTO fundamentals (ticker, period_type, period_end, metric, value, unit, source, retrieved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, period_type, period_end, metric) DO UPDATE SET
                value = excluded.value, unit = excluded.unit,
                source = excluded.source, retrieved_at = excluded.retrieved_at
            """,
            rows,
        )
    return len(rows)


def load_fundamentals(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM fundamentals WHERE ticker = ? ORDER BY period_type, period_end, metric",
        conn, params=(ticker,),
    )


# -------------------------------------------------------------- analysis_runs
def start_run(conn: sqlite3.Connection, ticker: str, period: str, benchmark: str | None,
              parameters: dict[str, Any]) -> int:
    """Record the start of an analysis run and return its run_id."""
    with conn:
        cursor = conn.execute(
            "INSERT INTO analysis_runs (ticker, started_at, status, period, benchmark, parameters) "
            "VALUES (?, ?, 'running', ?, ?, ?)",
            (ticker, utc_now(), period, benchmark, json.dumps(parameters, default=str)),
        )
    return int(cursor.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, status: str, **fields: Any) -> None:
    """Record the outcome of a run. Accepted fields: price_source, start_date, end_date,
    trading_days, n_events, warnings (list), error, report_path."""
    allowed = {"price_source", "start_date", "end_date", "trading_days", "n_events",
               "warnings", "error", "report_path"}
    updates = {"status": status, "finished_at": utc_now()}
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"Unknown analysis_runs field: {key}")
        updates[key] = json.dumps(value) if key == "warnings" else to_db_value(value)
    assignments = ", ".join(f"{key} = ?" for key in updates)
    with conn:
        conn.execute(f"UPDATE analysis_runs SET {assignments} WHERE run_id = ?", [*updates.values(), run_id])


def get_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM analysis_runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


# ------------------------------------------------------- manual research import
MANUAL_STATUS = "Researched manually"
IMPORTABLE_COLUMNS = ("event_category", "event_description", "source", "verification_status", "notes")


def import_research_fields(conn: sqlite3.Connection, table: pd.DataFrame, dry_run: bool = False) -> dict[str, Any]:
    """Copy research fields from an edited events table (e.g. events.csv) into the database.

    Safety rules:
    * only the research columns (category, description, source, verification
      status, notes) are imported - never prices or statistics;
    * empty cells are ignored, so existing research is never deleted;
    * rows that do not match a stored event (same ticker and date) are skipped
      and reported;
    * if you add research but leave the verification status automatic, the
      status becomes "Researched manually" so later runs will not overwrite it;
      an automatic status in the file never replaces a manual one, so
      importing the same file twice changes nothing the second time.

    Returns:
        {"updated_events": int, "changes": [(ticker, date, column, old, new)],
         "unmatched": [(ticker, date)], "status_set_manual": int}
    """
    missing = {"ticker", "event_date"} - set(table.columns)
    if missing:
        raise ValueError(f"The file needs the columns {sorted(missing)}.")
    columns = [c for c in IMPORTABLE_COLUMNS if c in table.columns]
    result: dict[str, Any] = {"updated_events": 0, "changes": [], "unmatched": [], "status_set_manual": 0}

    for record in table.to_dict(orient="records"):
        ticker = str(record["ticker"]).strip().upper()
        event_date = pd.Timestamp(record["event_date"]).strftime("%Y-%m-%d")
        row = conn.execute("SELECT * FROM events WHERE ticker = ? AND event_date = ?", (ticker, event_date)).fetchone()
        if row is None:
            result["unmatched"].append((ticker, event_date))
            continue
        updates: dict[str, Any] = {}
        for column in columns:
            value = record.get(column)
            if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
                continue
            value = str(value).strip()
            if (column == "verification_status" and value in AUTOMATIC_STATUSES
                    and row[column] not in AUTOMATIC_STATUSES):
                continue  # an old automatic status in the file never downgrades your manual status
            if value != row[column]:
                updates[column] = value
                result["changes"].append((ticker, event_date, column, row[column], value))
        if not updates:
            continue
        research_changed = any(c in updates for c in ("event_category", "event_description", "source"))
        final_status = updates.get("verification_status", row["verification_status"])
        if research_changed and final_status in AUTOMATIC_STATUSES:
            updates["verification_status"] = MANUAL_STATUS
            result["status_set_manual"] += 1
        result["updated_events"] += 1
        if not dry_run:
            assignments = ", ".join(f"{c} = ?" for c in updates)
            with conn:
                conn.execute(
                    f"UPDATE events SET {assignments}, updated_at = ? WHERE ticker = ? AND event_date = ?",
                    [*updates.values(), utc_now(), ticker, event_date],
                )
    return result
