"""Run the complete research workflow for one ticker.

    Market data -> Detect anomalies -> Identify events -> Collect evidence
    -> Quantitative analysis -> Fundamental context -> Historical comparison
    -> Research report

Every step that is not essential (benchmark, fundamentals, earnings dates,
individual charts) is allowed to fail: the problem is logged, added to the
report's "Data Limitations" section, and the run continues. Only a missing
price history stops the run, because nothing meaningful can be calculated
without it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.benchmark import add_benchmark_columns, summarize_benchmark
from src.analysis.event_study import summarize_event_outcomes
from src.analysis.statistics import add_statistics, summarize_statistics
from src.config import resolve_path
from src.data.prices import (
    DataUnavailableError,
    choose_price_column,
    clean_price_frame,
    load_prices_csv,
)
from src.data.provider import ProviderError
from src.database import db
from src.events.detector import detect_events
from src.events.evidence import match_earnings_to_events, normalize_earnings_dates
from src.fundamentals.fundamentals import collect_fundamentals, fundamentals_frame, metric_labels
from src.reporting import report
from src.visualization import charts

logger = logging.getLogger(__name__)

TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,14}$")
# Approximate calendar length of each download period (used to trim cached data).
PERIOD_DAYS = {"1y": 366, "2y": 731, "5y": 1827, "10y": 3653}
STALE_AFTER_DAYS = 7
RESEARCH_FIELDS = ["event_category", "event_description", "source", "verification_status", "notes"]


class AnalysisError(Exception):
    """A problem that stops the analysis (e.g. no price data at all)."""


@dataclass
class RunResult:
    """What ``run_analysis`` returns to the command-line interface."""

    ticker: str
    status: str
    report_dir: Path
    run_id: int
    n_events: int
    warnings: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)


def normalize_ticker(ticker: str) -> str:
    """Upper-case and validate a ticker symbol (e.g. ' avav ' -> 'AVAV').

    Raises:
        ValueError: for empty or clearly invalid symbols.
    """
    symbol = (ticker or "").strip().upper()
    if not TICKER_PATTERN.match(symbol):
        raise ValueError(
            f"'{ticker}' is not a valid ticker symbol. Use letters/digits, optionally with '.', '-', "
            "'=' or a leading '^' (e.g. AVAV, BRK-B, ^GSPC)."
        )
    return symbol


# ------------------------------------------------------------------ prices
def _trim_to_period(prices: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "ytd":
        return prices[prices.index.year == prices.index[-1].year]
    days = PERIOD_DAYS.get(period)
    if days is None or prices.empty:
        return prices
    return prices[prices.index >= prices.index[-1] - pd.Timedelta(days=days)]


def load_price_history(
    conn, symbol: str, period: str, provider: Any, csv_path: str | Path | None,
    offline: bool, warnings: list[str], label: str = "",
) -> tuple[pd.DataFrame, str]:
    """Get prices from a CSV file, the data provider, or the local database cache.

    Order of preference:
      1. ``csv_path`` if given;
      2. ``offline`` -> database cache only;
      3. live download; if it fails, fall back to the database cache with a warning.

    Returns:
        (price_frame, description_of_source)

    Raises:
        AnalysisError: when no source provides data.
    """
    prefix = f"{label}: " if label else ""
    if csv_path:
        prices = load_prices_csv(csv_path)
        return prices, f"Local CSV file ({Path(csv_path).name})"

    def from_cache(reason: str) -> tuple[pd.DataFrame, str]:
        cached = db.load_prices(conn, symbol)
        if cached.empty:
            raise AnalysisError(f"{prefix}{reason} No cached prices for {symbol} exist in the database either.")
        updated = db.latest_price_update(conn, symbol)
        warnings.append(
            f"{prefix}{reason} Using {len(cached):,} days of prices for {symbol} cached in the local "
            f"database (last saved {updated} UTC); they may be out of date."
        )
        return cached, f"Local database cache (originally downloaded; last saved {updated} UTC)"

    if offline:
        return from_cache("Offline mode.")
    if provider is None:
        return from_cache("No data provider available (is yfinance installed?).")
    try:
        prices = provider.get_price_history(symbol, period)
        return prices, getattr(provider, "name", "data provider")
    except (ProviderError, DataUnavailableError) as exc:
        logger.debug("%sPrice download for %s failed: %s", prefix, symbol, exc)
        return from_cache(f"Live price download failed ({exc}).")


def _stale_data_warning(prices: pd.DataFrame, today: date, label: str) -> str | None:
    last = prices.index[-1].date()
    age = (today - last).days
    if age > STALE_AFTER_DAYS:
        return f"{label}latest price is from {last} ({age} days before this run); the data may be stale."
    return None


# ------------------------------------------------------------------ helpers
def _merge_stored_research(events: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    """Show research fields from the database (e.g. your manual notes) in this run's report."""
    if events.empty or stored.empty:
        return events
    stored = stored.set_index(pd.to_datetime(stored["event_date"]))[RESEARCH_FIELDS]
    merged = events.copy()
    for index, event in merged.iterrows():
        key = pd.Timestamp(event["event_date"])
        if key in stored.index:
            for column in RESEARCH_FIELDS:
                merged.at[index, column] = stored.at[key, column]
    return merged


def _collect_evidence(events: pd.DataFrame, trading_days: pd.DatetimeIndex, provider: Any, ticker: str,
                      window_days: int, warnings: list[str]) -> pd.DataFrame:
    try:
        raw = provider.get_earnings_dates(ticker)
    except Exception as exc:  # noqa: BLE001 - evidence is optional
        warnings.append(f"Earnings calendar unavailable, so no events were linked to earnings dates ({exc}).")
        return events
    earnings = normalize_earnings_dates(raw)
    if earnings.empty:
        warnings.append("The earnings calendar returned no dates; no events were linked to earnings releases.")
        return events
    covered_from = earnings["release_time"].min()
    if covered_from > trading_days[0]:
        warnings.append(
            f"The earnings calendar only covers releases from {covered_from:%Y-%m-%d}; earlier events cannot "
            "be linked to earnings dates."
        )
    source = f"{getattr(provider, 'name', 'data provider')} earnings calendar"
    return match_earnings_to_events(events, trading_days, earnings, window_days, source=source)


def _fundamentals_with_fallback(conn, provider: Any, ticker: str, run_date: date, offline: bool,
                                skip: bool) -> dict[str, Any]:
    """Fetch fundamentals; for any part that is unavailable, fall back to the copy stored earlier.

    Returns {"profile", "records", "warnings", "fresh_records"} where "fresh_records"
    are the newly downloaded records (the only ones saved to the database).
    """
    result: dict[str, Any] = {"profile": {}, "records": [], "warnings": [], "fresh_records": []}
    if skip:
        result["warnings"].append("Fundamental data was skipped (--no-fundamentals).")
        return result
    if not offline and provider is not None:
        live = collect_fundamentals(provider, ticker, run_date)
        result.update(profile=live["profile"], records=list(live["records"]), warnings=list(live["warnings"]),
                      fresh_records=list(live["records"]))

    stored = db.load_fundamentals(conn, ticker)
    labels = metric_labels()
    for period_type in ("snapshot", "annual"):
        if any(r["period_type"] == period_type for r in result["records"]):
            continue
        rows = stored[stored["period_type"] == period_type]
        if rows.empty:
            if offline:
                result["warnings"].append(f"Offline mode: no {period_type} fundamentals are stored for this ticker.")
            continue
        if period_type == "snapshot":
            latest = rows["period_end"].max()
            rows = rows[rows["period_end"] == latest]
            note = f"snapshot fundamentals retrieved on {latest}"
        else:
            note = "annual statement data"
        for record in rows.to_dict(orient="records"):
            record["label"] = labels.get(record["metric"], record["metric"])
            result["records"].append(record)
        result["warnings"].append(f"Live data unavailable: showing {note} stored in the local database.")

    profile_known = any(value for value in result["profile"].values())
    if not profile_known:  # e.g. company info download failed: use the stored profile
        stock = db.get_stock(conn, ticker) or {}
        result["profile"] = {k: stock.get(k) for k in ("name", "exchange", "sector", "industry", "currency",
                                                       "country", "quote_type")}
    return result


def _make_charts(stats: pd.DataFrame, events: pd.DataFrame, ticker: str, benchmark: str | None,
                 report_dir: Path, settings: dict[str, Any], warnings: list[str]) -> dict[str, str]:
    e = settings["events"]
    jobs = {
        "price_chart": lambda p: charts.price_chart(stats, events, ticker, p),
        "returns_distribution": lambda p: charts.returns_distribution_chart(stats, ticker, p),
        "volume_chart": lambda p: charts.volume_chart(stats, events, ticker, p,
                                                      settings["statistics"]["volume_window"]),
        "event_chart": lambda p: charts.event_chart(stats, events, ticker, p, e.get("return_zscore_threshold"),
                                                    e.get("volume_ratio_threshold")),
    }
    if benchmark and "relative_performance" in stats and stats["relative_performance"].notna().any():
        jobs["benchmark_chart"] = lambda p: charts.benchmark_chart(stats, ticker, benchmark, p)
    files: dict[str, str] = {}
    for name, job in jobs.items():
        filename = f"{name}.png"
        try:
            job(report_dir / filename)
            files[name] = filename
        except Exception as exc:  # noqa: BLE001 - one broken chart must not stop the report
            logger.exception("Chart %s failed", name)
            warnings.append(f"Chart '{filename}' could not be created: {exc}")
    return files


def _remove_stale_charts(report_dir: Path, files: dict[str, str]) -> None:
    """Delete charts left by an earlier run on the same day that this run did not produce
    (e.g. benchmark_chart.png after re-running with --no-benchmark), so the folder never
    contains a chart that contradicts its summary.md."""
    for path in report_dir.glob("*.png"):
        if path.name not in files.values():
            path.unlink()


def _remove_empty_report_dir(report_dir: Path) -> None:
    """Delete a report folder that contains nothing but run.log (and its empty ticker folder)."""
    contents = [p.name for p in report_dir.iterdir()] if report_dir.exists() else []
    if set(contents) <= {"run.log"}:
        for name in contents:
            (report_dir / name).unlink()
        report_dir.rmdir()
        if report_dir.parent.exists() and not any(report_dir.parent.iterdir()):
            report_dir.parent.rmdir()


# ------------------------------------------------------------------ main entry
def run_analysis(
    ticker: str,
    settings: dict[str, Any],
    provider: Any = None,
    prices_csv: str | Path | None = None,
    benchmark_csv: str | Path | None = None,
    offline: bool = False,
    skip_fundamentals: bool = False,
    run_date: date | None = None,
) -> RunResult:
    """Run the full workflow for ``ticker`` and write the report folder.

    Args:
        ticker: stock symbol, e.g. "AVAV".
        settings: loaded settings (see src/config.py).
        provider: data provider object (normally YahooFinanceProvider).
        prices_csv / benchmark_csv: optional CSV files to use instead of downloading.
        offline: use only data cached in the database (no internet access).
        skip_fundamentals: do not fetch fundamentals.
        run_date: date used for the report folder name (default: today).

    Raises:
        AnalysisError / DataUnavailableError: when no price data can be obtained.
    """
    ticker = normalize_ticker(ticker)
    run_date = run_date or date.today()
    data_settings, stat_settings = settings["data"], settings["statistics"]
    period = data_settings["period"]
    benchmark = data_settings.get("benchmark")
    benchmark = normalize_ticker(benchmark) if benchmark else None

    report_dir = resolve_path(settings["paths"]["reports_dir"]) / ticker / run_date.isoformat()
    report_dir.mkdir(parents=True, exist_ok=True)
    # Append: a later run on the same day (even a failed one) never erases an earlier run's log.
    log_handler = logging.FileHandler(report_dir / "run.log", mode="a", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    log_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(log_handler)

    db_path = resolve_path(settings["paths"]["database"])
    conn = db.connect(db_path)
    run_id = db.start_run(conn, ticker, period, benchmark, settings)
    warnings: list[str] = []
    have_prices = False
    logger.info("Run #%d: analysing %s (period %s, benchmark %s)", run_id, ticker, period, benchmark or "none")

    try:
        # ---- 1. Market data -------------------------------------------------
        raw_prices, price_source = load_price_history(conn, ticker, period, provider, prices_csv, offline, warnings)
        if price_source.startswith(("Local CSV", "Local database cache")):
            # A live download already covers exactly the requested period; a CSV file or the
            # cache may hold more history, so keep only the requested period (--period max keeps all).
            trimmed = _trim_to_period(raw_prices.sort_index(), period)
            if len(trimmed) < len(raw_prices):
                warnings.append(
                    f"Analysed the last {period} of the available prices ({len(trimmed):,} of {len(raw_prices):,} "
                    "rows); use --period max to analyse all of them."
                )
            raw_prices = trimmed
        prices, cleaning_notes = clean_price_frame(raw_prices)
        have_prices = True
        db.upsert_stock(conn, ticker)
        warnings.extend(cleaning_notes)
        stale = _stale_data_warning(prices, run_date, "Price data: ")
        if stale:
            warnings.append(stale)
        if not price_source.startswith("Local database cache"):
            saved = db.upsert_prices(conn, ticker, prices, price_source)
            logger.info("Saved %d price rows for %s", saved, ticker)
        price_column, price_note = choose_price_column(prices)
        if price_note:
            warnings.append(price_note)
        if len(prices) <= stat_settings["zscore_window"]:
            warnings.append(
                f"Only {len(prices)} trading days of data: return Z-scores need more than "
                f"{stat_settings['zscore_window']} days, so price-based events cannot be detected."
            )

        # ---- 2. Quantitative statistics -----------------------------------
        stats = add_statistics(
            prices, price_column,
            zscore_window=stat_settings["zscore_window"],
            volume_window=stat_settings["volume_window"],
            volatility_window=stat_settings["volatility_window"],
            trading_days_per_year=stat_settings["trading_days_per_year"],
        )

        # ---- 3. Benchmark comparison (optional) ---------------------------
        benchmark_used: str | None = None
        benchmark_source = "Not used"
        if benchmark and benchmark == ticker:
            warnings.append(f"The benchmark ({benchmark}) is the analysed ticker itself; comparison skipped.")
        elif benchmark:
            try:
                raw_bench, benchmark_source = load_price_history(
                    conn, benchmark, period, provider, benchmark_csv, offline, warnings, label="Benchmark"
                )
                bench_prices, bench_notes = clean_price_frame(raw_bench)
                db.upsert_stock(conn, benchmark)
                warnings.extend(f"Benchmark {benchmark}: {note}" for note in bench_notes)
                if not benchmark_source.startswith("Local database cache"):
                    db.upsert_prices(conn, benchmark, bench_prices, benchmark_source)
                bench_column, _ = choose_price_column(bench_prices)
                stats = add_benchmark_columns(stats, bench_prices[bench_column])
                benchmark_used = benchmark
            except (AnalysisError, DataUnavailableError) as exc:
                warnings.append(f"Benchmark comparison unavailable: {exc}")

        # ---- 4. Detect anomalies / identify events ------------------------
        events = detect_events(stats, ticker, settings["events"], benchmark_used)
        logger.info("Detected %d event(s)", len(events))

        # ---- 5. Collect evidence (earnings dates) --------------------------
        if not events.empty and provider is not None and not offline:
            events = _collect_evidence(events, stats.index, provider, ticker,
                                       settings["events"]["earnings_match_window_days"], warnings)
        elif not events.empty:
            reason = "Offline mode" if offline else "No data provider available"
            warnings.append(f"{reason}: events were not checked against the earnings calendar.")

        # ---- 6. Fundamental context -----------------------------------------
        fundamentals = _fundamentals_with_fallback(conn, provider, ticker, run_date, offline, skip_fundamentals)
        warnings.extend(fundamentals["warnings"])
        if fundamentals["profile"]:
            db.upsert_stock(conn, ticker, fundamentals["profile"])
        if fundamentals["fresh_records"]:
            db.upsert_fundamentals(conn, fundamentals["fresh_records"])

        # ---- 7. Store events (no duplicates) --------------------------------
        detection_settings = json.dumps({"events": settings["events"], "statistics": stat_settings})
        events["detection_settings"] = detection_settings
        counts = db.upsert_events(conn, events, run_id)
        refreshed = db.refresh_forward_returns(conn, ticker, stats, benchmark_used)
        events = _merge_stored_research(events, db.load_events(conn, ticker))
        logger.info("Events saved: %d new, %d updated; forward returns refreshed for %d stored event(s)",
                    counts["inserted"], counts["updated"], refreshed)

        # ---- 8. Historical comparison ---------------------------------------
        outcomes = summarize_event_outcomes(events, stats)
        stats_summary = summarize_statistics(stats, settings["events"].get("return_zscore_threshold"),
                                             stat_settings["trading_days_per_year"])
        benchmark_summary = (summarize_benchmark(stats, benchmark_used, stat_settings["trading_days_per_year"])
                             if benchmark_used else None)

        # ---- 9. Charts and report -----------------------------------------
        files = _make_charts(stats, events, ticker, benchmark_used, report_dir, settings, warnings)
        price_column_description = (
            "Adjusted close (includes dividends and splits)" if price_column == "adj_close"
            else "Close price (not adjusted for dividends)"
        )
        results = {
            "ticker": ticker,
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "settings": settings,
            "profile": fundamentals["profile"],
            "data_sources": {
                "prices": price_source,
                "benchmark": benchmark_source if benchmark_used else "Not used",
                "fundamentals": (
                    getattr(provider, "name", "data provider") if fundamentals["fresh_records"]
                    else "the local database (copy of an earlier download)" if fundamentals["records"]
                    else "Not available"
                ),
                "price_column": price_column,
                "price_column_description": price_column_description,
            },
            "stats": stats_summary,
            "benchmark_summary": benchmark_summary,
            "events_frame": events,
            "events": events,
            "event_outcomes_frame": outcomes,
            "event_outcomes": outcomes,
            "fundamentals": fundamentals,
            "warnings": warnings,
            "files": files,
            "database": {"path": str(db_path), "events_inserted": counts["inserted"],
                         "events_updated": counts["updated"]},
        }
        report.write_events_csv(events, report_dir / "events.csv")
        report.write_fundamentals_csv(fundamentals_frame(fundamentals["records"]), report_dir / "fundamentals.csv")
        report.write_analysis_json(results, report_dir / "analysis.json")
        report.write_summary_markdown(results, report_dir / "summary.md")
        _remove_stale_charts(report_dir, files)

        status = "success_with_warnings" if warnings else "success"
        db.finish_run(
            conn, run_id, status, price_source=price_source, start_date=stats.index[0],
            end_date=stats.index[-1], trading_days=len(stats), n_events=len(events),
            warnings=warnings, report_path=str(report_dir),
        )
        for warning in warnings:
            logger.warning("%s", warning)
        logger.info("Report written to %s", report_dir)
        return RunResult(ticker, status, report_dir, run_id, len(events), warnings, files)
    except Exception as exc:
        # The caller (analyze.py) shows the error to the user; keep the details in the run log.
        logger.debug("Run #%d failed: %s", run_id, exc, exc_info=True)
        db.finish_run(conn, run_id, "failed", error=f"{type(exc).__name__}: {exc}", warnings=warnings)
        raise
    finally:
        conn.close()
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
        if not have_prices:
            # Nothing was produced (e.g. a mistyped ticker): don't leave an empty report folder behind.
            # The error is shown on screen and stored in the analysis_runs table.
            _remove_empty_report_dir(report_dir)
