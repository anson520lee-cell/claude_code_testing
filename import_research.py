"""Import your manual research notes from an edited events CSV into the database.

Typical workflow:
    1. python analyze.py AVAV
    2. Open reports/AVAV/<date>/events.csv in Excel / LibreOffice / Google Sheets.
    3. Fill in event_category, event_description, source, verification_status
       and notes for the events you researched. Save as CSV.
    4. python import_research.py reports/AVAV/<date>/events.csv --dry-run   (preview)
       python import_research.py reports/AVAV/<date>/events.csv             (apply)

Only research columns are imported; prices and statistics in the file are
ignored, and empty cells never delete existing text. The next report for the
ticker shows your research.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import load_settings, resolve_path
from src.database import db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import research fields from an edited events CSV.")
    parser.add_argument("csv", help="Edited events CSV (same columns as events.csv)")
    parser.add_argument("--db", help="SQLite database (default: from config/settings.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without saving")
    args = parser.parse_args(argv)

    try:
        table = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"Could not read {args.csv}: {exc}")
        return 2
    conn = db.connect(args.db or resolve_path(load_settings()["paths"]["database"]))
    try:
        result = db.import_research_fields(conn, table, dry_run=args.dry_run)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2
    finally:
        conn.close()

    for ticker, date, column, old, new in result["changes"]:
        print(f"{ticker} {date} {column}: {old!r} -> {new!r}")
    verb = "Would update" if args.dry_run else "Updated"
    print(f"\n{verb} {result['updated_events']} event(s).")
    if result["status_set_manual"]:
        print(f"{result['status_set_manual']} event(s) had research added without a verification status; "
              f"status set to '{db.MANUAL_STATUS}' so automatic runs will not overwrite it.")
    if result["unmatched"]:
        print(f"Skipped {len(result['unmatched'])} row(s) that match no stored event, e.g. "
              + ", ".join(f"{t} {d}" for t, d in result["unmatched"][:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
