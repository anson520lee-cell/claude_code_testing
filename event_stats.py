"""Pool events stored in the database (across tickers) and summarise what happened afterwards.

This answers questions such as "what usually happens after a >2.5 sigma
positive move?" using every event saved by previous ``analyze.py`` runs.

Examples:
    python event_stats.py                          # every stored event, all tickers
    python event_stats.py --min-sigma 2.5          # only events with |Return Z-score| >= 2.5
    python event_stats.py --tickers AVAV KTOS      # selected tickers only
    python event_stats.py --min-sigma 3 --csv out.csv

The output is descriptive: counts, averages, medians and hit rates. It is not
a forecast and includes no significance testing.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.analysis.event_study import summarize_event_outcomes
from src.analysis.statistics import add_statistics
from src.config import load_settings, resolve_path
from src.data.prices import choose_price_column
from src.database import db
from src.events.detector import OUTCOME_COLUMNS


def baseline_days(conn, tickers: list[str]) -> pd.DataFrame:
    """Forward returns for every stored trading day of the given tickers (the comparison baseline)."""
    frames = []
    for ticker in tickers:
        prices = db.load_prices(conn, ticker)
        if len(prices) < 2:
            continue
        column, _ = choose_price_column(prices)
        stats = add_statistics(prices, column)
        frames.append(stats[["daily_return"] + [c for c in OUTCOME_COLUMNS if c in stats.columns]])
    return pd.concat(frames) if frames else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise forward returns after stored abnormal events.")
    parser.add_argument("--db", help="SQLite database (default: from config/settings.yaml)")
    parser.add_argument("--tickers", nargs="+", help="Only these tickers")
    parser.add_argument("--min-sigma", type=float, help="Only events with |Return Z-score| >= this value")
    parser.add_argument("--min-volume-ratio", type=float, help="Only events with volume ratio >= this value")
    parser.add_argument("--csv", help="Also save the summary table to this CSV file")
    args = parser.parse_args(argv)

    db_path = args.db or resolve_path(load_settings()["paths"]["database"])
    conn = db.connect(db_path)
    tickers = [t.upper() for t in args.tickers] if args.tickers else None
    events = db.load_events(conn, tickers=tickers)
    if args.min_sigma is not None:
        events = events[events["return_zscore"].abs() >= args.min_sigma]
    if args.min_volume_ratio is not None:
        events = events[events["volume_ratio"] >= args.min_volume_ratio]
    if events.empty:
        print("No stored events match these filters. Run analyze.py for some tickers first.")
        return 0

    used_tickers = sorted(events["ticker"].unique())
    summary = summarize_event_outcomes(events, baseline_days(conn, used_tickers))
    conn.close()

    print(f"Events: {len(events)} from {len(used_tickers)} ticker(s): {', '.join(used_tickers)}")
    print(f"Date range: {events['event_date'].min():%Y-%m-%d} to {events['event_date'].max():%Y-%m-%d}\n")
    table = summary[["group", "n_events", "mean_2d", "mean_3d", "mean_5d", "mean_7d", "median_5d",
                     "pct_positive_5d", "pct_positive_7d", "median_mfe_5d", "median_mae_5d", "mean_20d",
                     "n_7d"]].copy()
    for column in table.columns:
        if column.startswith(("mean", "median")):
            table[column] = table[column].map(lambda v: "n/a" if pd.isna(v) else f"{v * 100:+.2f}%")
        elif column.startswith("pct"):
            table[column] = table[column].map(lambda v: "n/a" if pd.isna(v) else f"{v * 100:.0f}%")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    print("\nFor up-move groups '% positive' = continuation rate; for down-move groups = reversal rate.")
    print("Descriptive statistics only: overlapping windows and small samples limit what they show.")
    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"Saved {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
