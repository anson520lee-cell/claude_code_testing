"""Intraday price refresh (display only) - run by the background job every 5 minutes.

    python intraday_run.py           # fetch if a new half-hour slot (9:30 ... 16:00 New York) has started
    python intraday_run.py --force   # fetch the latest slot of today now (trading days only)

Exits at once outside US market hours, on weekends / holidays and when the slot was already fetched.
Scores, setups and plans are NOT recomputed: they come from the after-close run (daily_run.py --if-new).
Research / decision support only - this program never places orders or connects to a broker.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from src.config import ConfigError
from src.daily.settings import load_terminal_settings, path
from src.market import calendar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Intraday quote refresh every 30 minutes (display only).")
    parser.add_argument("--config", help="terminal settings file (default config/terminal.yaml)")
    parser.add_argument("--force", action="store_true", help="fetch the latest slot of today even if already fetched")
    args = parser.parse_args(argv)
    try:
        settings = load_terminal_settings(args.config)
    except ConfigError as exc:
        print(f"Settings error: {exc}", file=sys.stderr)
        return 2
    from src.daily import intraday
    cfg = intraday.intraday_settings(settings)
    now = datetime.now(calendar.NEW_YORK)
    previous = intraday.load_latest(settings) or {}
    if not args.force and (not cfg.get("enabled", True) or intraday.due_slot(
            now, previous.get("slot"), int(cfg["every_minutes"]), int(cfg["grace_minutes"])) is None):
        return 0                                  # quiet: this runs every 5 minutes
    conn = None
    db_file = path(settings, "database")
    if db_file.exists():
        import sqlite3
        conn = sqlite3.connect(db_file, timeout=30)
    try:
        result = intraday.refresh(settings, now=now, force=args.force, conn=conn)
    except Exception as exc:  # noqa: BLE001 - a Yahoo hiccup must not crash the background job
        print(f"{now:%Y-%m-%d %H:%M} ET intraday refresh failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()
    if result["status"] == "skipped":
        print(f"Intraday refresh skipped: {result['reason']}")
    elif result["status"] in ("empty", "partial"):
        print(f"{now:%Y-%m-%d %H:%M} ET intraday slot {result['slot']}: only {result['quotes']}/{result['symbols']} "
              f"quotes - will retry at the next run.")
    else:
        print(f"{now:%Y-%m-%d %H:%M} ET intraday slot {result['slot']}: {result['quotes']}/{result['symbols']} quotes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
