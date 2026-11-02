"""Keeps disk use flat over years of daily runs (called at the end of every daily run).

* daily reports: the newest ``keep_plain_sessions`` keep a plain data.json; older ones are gzip-compressed
  (~8x smaller; the dashboard reads both). report.html / report.md are kept as they are.
* backtest folders: the newest ``keep_backtests`` monthly runs are kept.
* caches: FINRA daily short-volume files older than 45 days, SEC fails-to-deliver files older than
  200 days and news feed cache files older than 7 days are deleted (they are re-downloadable).
* news table: headlines older than ``news_days`` are deleted.
* logs: data/daily_run.log and data/server.log are trimmed to their last 1 MB once they pass 5 MB.
Nothing here touches prices, the journal, alerts, candidate history or the regime history.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.daily import store
from src.daily.settings import path

logger = logging.getLogger(__name__)


def compress_old_reports(reports_dir: Path, keep_plain: int = 30) -> int:
    if not reports_dir.exists():
        return 0
    sessions = sorted((p for p in reports_dir.iterdir() if (p / "data.json").exists()), key=lambda p: p.name, reverse=True)
    done = 0
    for folder in sessions[keep_plain:]:
        plain = folder / "data.json"
        packed = folder / "data.json.gz"
        try:
            store.write_atomic(packed, gzip.compress(plain.read_bytes(), compresslevel=6))
            plain.unlink()
        except OSError as exc:           # disk full etc.: leave the plain file, try again next run
            logger.warning("could not compress %s: %s", plain, exc)
            continue
        done += 1
    return done


def prune_backtests(backtest_dir: Path, keep: int = 6) -> int:
    if not backtest_dir.exists():
        return 0
    runs = sorted((p for p in backtest_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)),
                  key=lambda p: p.name, reverse=True)
    for folder in runs[keep:]:
        shutil.rmtree(folder, ignore_errors=True)
    return max(0, len(runs) - keep)


def prune_dated_files(folder: Path, pattern: str, fmt: str, max_age_days: int, today: date) -> int:
    """Delete files whose name carries a date (``pattern`` with one group, parsed with ``fmt``) older than N days."""
    if not folder.exists():
        return 0
    n = 0
    rx = re.compile(pattern)
    for f in folder.iterdir():
        m = rx.fullmatch(f.name)
        if not m:
            continue
        try:
            d = datetime.strptime(m.group(1), fmt).date()
        except ValueError:
            continue
        if (today - d).days > max_age_days:
            f.unlink(missing_ok=True)
            n += 1
    return n


def prune_old_files(folder: Path, max_age_days: int) -> int:
    if not folder.exists():
        return 0
    cutoff = time.time() - max_age_days * 86400
    n = 0
    for f in folder.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            n += 1
    return n


def prune_tmp(folder: Path, max_age_hours: float = 12) -> int:
    """Leftovers of an atomic write that was killed half-way (``.name.<pid>.tmp``)."""
    if not folder.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    n = 0
    for f in folder.rglob(".*.tmp"):
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                n += 1
        except OSError:
            continue
    return n


def trim_log(file: Path, max_bytes: int = 5_000_000, keep_bytes: int = 1_000_000) -> bool:
    if not file.exists() or file.stat().st_size <= max_bytes:
        return False
    with open(file, "rb") as fh:
        fh.seek(-keep_bytes, 2)
        tail = fh.read()
    tail = tail[tail.find(b"\n") + 1:]
    with open(file, "r+b") as fh:          # in place: the running process keeps appending to the same file
        fh.seek(0)
        fh.write(b"[log trimmed]\n" + tail)
        fh.truncate()
    return True


def prune_news(conn, days: int = 180) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with conn:
        cur = conn.execute("DELETE FROM news WHERE COALESCE(published, fetched_at) < ?", (cutoff,))
    return cur.rowcount or 0


def export_user_data(conn, folder: Path, today: date, keep: int = 14) -> int:
    """Daily JSON copy of what cannot be re-downloaded - the trade journal and the price alerts - so a
    damaged or deleted database never loses them (restore: python daily_run.py --restore-user-data FILE)."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for table in ("trade_journal", "price_alerts"):
        cur = conn.execute(f"SELECT * FROM {table} ORDER BY id")
        cols = [c[0] for c in cur.description]
        rows[table] = [dict(zip(cols, r)) for r in cur.fetchall()]
    if not any(rows.values()):
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    store.write_atomic(folder / f"user_data_{today:%Y%m%d}.json",
                       json.dumps({"exported_at": datetime.now(timezone.utc).isoformat(), **rows}, indent=1))
    for old in sorted(folder.glob("user_data_*.json"))[:-keep]:
        old.unlink(missing_ok=True)
    return 1


def restore_user_data(conn, file: Path) -> int:
    """Re-insert journal trades and alerts from an export (rows that still exist are left alone)."""
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    n = 0
    with conn:
        for table in ("trade_journal", "price_alerts"):
            valid = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            for row in data.get(table) or []:
                cols = [c for c in row if c in valid]
                cur = conn.execute(f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                                   [row[c] for c in cols])
                n += cur.rowcount
    return n


def run(settings: dict[str, Any], conn=None, today: date | None = None) -> dict[str, int]:
    cfg = settings.get("housekeeping") or {}
    today = today or date.today()
    reports = path(settings, "reports_dir")
    cache = path(settings, "cache_dir")
    data_dir = path(settings, "database").parent
    out = {
        "reports_compressed": compress_old_reports(reports, int(cfg.get("keep_plain_sessions", 30))),
        "backtests_removed": prune_backtests(reports.parent / "backtest", int(cfg.get("keep_backtests", 6))),
        "finra_files": prune_dated_files(cache / "finra", r"regsho_(\d{8})\.txt", "%Y%m%d", 45, today),
        "ftd_files": prune_dated_files(cache / "sec" / "files", r"cnsfails(\d{6})[ab]\.zip(?:\.missing)?", "%Y%m", 200, today),
        "news_cache_files": prune_old_files(cache / "news", 7),
        # SEC filings / XBRL facts of names deep-dived once and never again (a few MB each)
        "sec_cache_files": prune_old_files(cache / "sec", 30),
        "logs_trimmed": int(trim_log(data_dir / "daily_run.log")) + int(trim_log(data_dir / "server.log")),
        "tmp_files": sum(prune_tmp(d) for d in (reports, reports.parent / "backtest", cache)),
    }
    if conn is not None:
        try:
            out["news_rows"] = prune_news(conn, int(cfg.get("news_days", 180)))
        except Exception:  # noqa: BLE001 - table may not exist in very old databases
            out["news_rows"] = 0
        try:
            export_user_data(conn, path(settings, "backups_dir"), today)
        except Exception as exc:  # noqa: BLE001
            logger.warning("user data export failed: %s", exc)
    return out
