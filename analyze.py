"""Event-driven stock research - command-line entry point.

Examples:
    python analyze.py AVAV
    python analyze.py AVAV --period 10y --benchmark QQQ --sigma 3
    python analyze.py AVAV --no-benchmark --no-fundamentals
    python analyze.py AVAV --offline                 # reuse data stored in the local database
    python analyze.py AVAV --prices-csv my_avav.csv  # analyse prices from your own CSV file

Run ``python analyze.py --help`` for every option.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import VALID_PERIODS, ConfigError, apply_overrides, load_settings
from src.data.prices import DataUnavailableError
from src.pipeline import AnalysisError, normalize_ticker, run_analysis

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_BAD_INPUT = 2
EXIT_NO_DATA = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download market data, detect abnormal price/volume events, and write a research report. "
                    "Descriptive research only - no buy/sell signals.",
    )
    parser.add_argument("ticker", help="Stock ticker, e.g. AVAV")
    parser.add_argument("--period", choices=sorted(VALID_PERIODS), help="Price history to analyse (default: 5y)")
    parser.add_argument("--benchmark", help="Benchmark ticker (default: SPY)")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip the benchmark comparison")
    parser.add_argument("--sigma", type=float, help="Return Z-score threshold for events (default: 2.5)")
    parser.add_argument("--volume-ratio", type=float,
                        help="Volume ratio threshold for events (default: 2.0 = twice the average volume)")
    parser.add_argument("--config", help="Path to a settings YAML file (default: config/settings.yaml)")
    parser.add_argument("--db", help="Path to the SQLite database (default: data/stock_research.db)")
    parser.add_argument("--reports-dir", help="Folder for reports (default: reports)")
    parser.add_argument("--prices-csv", help="Read the stock's daily prices from this CSV file instead of downloading")
    parser.add_argument("--benchmark-csv", help="Read the benchmark's daily prices from this CSV file")
    parser.add_argument("--offline", action="store_true",
                        help="Do not access the internet; use data previously stored in the database")
    parser.add_argument("--no-fundamentals", action="store_true", help="Skip fundamental data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed log messages")
    return parser


def setup_console_logging(verbose: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # the per-run log file receives everything
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    root.addHandler(handler)
    # Third-party libraries are noisy at DEBUG level.
    for name in ("matplotlib", "PIL", "urllib3", "peewee", "yfinance"):
        logging.getLogger(name).setLevel(logging.WARNING if name != "yfinance" else logging.CRITICAL)


def make_provider(settings: dict, offline: bool):
    """Create the Yahoo Finance provider, or None when offline / yfinance is not installed."""
    if offline:
        return None
    from src.data.provider import YahooFinanceProvider

    try:
        return YahooFinanceProvider(
            attempts=settings["data"]["download_attempts"],
            retry_wait_seconds=settings["data"]["retry_wait_seconds"],
        )
    except ImportError:  # yfinance missing
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_console_logging(args.verbose)
    log = logging.getLogger("analyze")

    try:
        ticker = normalize_ticker(args.ticker)
        settings = load_settings(args.config)
        settings = apply_overrides(
            settings,
            period=args.period,
            benchmark="" if args.no_benchmark else args.benchmark,
            sigma=args.sigma,
            volume_ratio=args.volume_ratio,
            database=args.db,
            reports_dir=args.reports_dir,
        )
    except (ValueError, ConfigError) as exc:
        log.error("%s", exc)
        return EXIT_BAD_INPUT

    provider = make_provider(settings, args.offline)
    if provider is None and not args.offline and not args.prices_csv:
        log.error("yfinance is not installed. Run: pip install -r requirements.txt")
        return EXIT_BAD_INPUT

    try:
        result = run_analysis(
            ticker, settings, provider=provider, prices_csv=args.prices_csv, benchmark_csv=args.benchmark_csv,
            offline=args.offline, skip_fundamentals=args.no_fundamentals,
        )
    except (AnalysisError, DataUnavailableError) as exc:
        log.error("Analysis of %s stopped: %s", ticker, exc)
        return EXIT_NO_DATA
    except Exception:  # noqa: BLE001 - show a full traceback for unexpected bugs
        log.exception("Unexpected error while analysing %s", ticker)
        return EXIT_UNEXPECTED

    print()
    print(f"Finished {result.ticker}: {result.n_events} abnormal event(s) detected (run #{result.run_id}).")
    print(f"Report folder: {result.report_dir}")
    if result.warnings:
        print(f"{len(result.warnings)} data note(s) - see 'Data Limitations' in summary.md.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
