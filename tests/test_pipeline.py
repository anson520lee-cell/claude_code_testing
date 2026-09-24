"""End-to-end tests: the full workflow with a fake data provider (no internet needed)."""

import json
import sqlite3
from datetime import date

import pandas as pd
import pytest

from src.data.prices import DataUnavailableError
from src.pipeline import AnalysisError, normalize_ticker, run_analysis
from tests.conftest import FakeProvider, make_yf_prices, sample_info, sample_statements

RUN_DATE = date(2024, 7, 31)
EXPECTED_FILES = {
    "summary.md", "events.csv", "fundamentals.csv", "price_chart.png", "returns_distribution.png",
    "volume_chart.png", "event_chart.png", "benchmark_chart.png", "analysis.json", "run.log",
    "technical_analysis.md", "technical_chart.png",
}
SECTIONS = ["Company / Ticker", "Market Overview", "Recent Performance", "Volatility", "Abnormal Events",
            "Fundamental Snapshot", "Benchmark Comparison", "Historical Event Outcomes", "Data Limitations"]


def make_provider(**kwargs):
    # 400 business days from 2023-01-03 end on 2024-07-16.
    stock = make_yf_prices(n=400, seed=21, shocks={250: (0.18, 6.0), 300: (-0.14, 4.0), 330: (0.001, 5.0)})
    spy = make_yf_prices(n=400, seed=99, base_price=400, daily_vol=0.008)
    release = stock.index[249].tz_convert("America/New_York").normalize() + pd.Timedelta(hours=16, minutes=5)
    earnings = pd.DataFrame({"EPS Estimate": [0.40], "Reported EPS": [0.55], "Surprise(%)": [37.5]},
                            index=pd.DatetimeIndex([release], name="Earnings Date"))
    defaults = {"prices": {"TEST": stock, "SPY": spy}, "info": sample_info(), "statements": sample_statements(),
                "earnings": earnings}
    defaults.update(kwargs)
    return FakeProvider(**defaults), stock


def test_full_workflow_creates_every_output(settings):
    provider, stock = make_provider()
    result = run_analysis("test", settings, provider=provider, run_date=RUN_DATE)

    assert result.ticker == "TEST"
    assert result.report_dir.name == "2024-07-31"
    assert result.report_dir.parent.name == "TEST"
    assert EXPECTED_FILES <= {p.name for p in result.report_dir.iterdir()}
    for png in ("price_chart.png", "returns_distribution.png", "volume_chart.png", "event_chart.png"):
        assert (result.report_dir / png).stat().st_size > 10_000

    summary = (result.report_dir / "summary.md").read_text()
    for section in SECTIONS:
        assert section in summary, section
    assert "no investment recommendations" in summary.lower()
    assert "Test Robotics Inc." in summary

    # The planted events were found; the one right after the earnings release is linked to it.
    events = pd.read_csv(result.report_dir / "events.csv", parse_dates=["event_date"])
    dates = set(events["event_date"])
    planted = [stock.index[i].tz_localize(None).normalize() for i in (250, 300, 330)]
    assert set(planted) <= dates
    earnings_event = events[events["event_date"] == planted[0]].iloc[0]
    assert earnings_event["event_category"] == "Earnings release (date match)"
    assert earnings_event["verification_status"] == "Unverified (automatic date match)"
    other = events[events["event_date"] == planted[1]].iloc[0]
    assert other["event_category"] == "Unknown"  # no evidence -> no explanation invented
    assert pd.isna(other["event_description"])
    volume_only = events[events["event_date"] == planted[2]].iloc[0]
    assert volume_only["trigger_rules"] == "volume_ratio"

    # analysis.json is valid JSON with the same events
    analysis = json.loads((result.report_dir / "analysis.json").read_text())
    assert analysis["ticker"] == "TEST"
    assert len(analysis["events"]) == len(events) == result.n_events
    assert analysis["benchmark"]["benchmark"] == "SPY"
    assert analysis["statistics"]["trading_days"] == 400

    fundamentals = pd.read_csv(result.report_dir / "fundamentals.csv")
    assert {"snapshot", "annual"} <= set(fundamentals["period_type"])

    # Database contents
    conn = sqlite3.connect(settings["paths"]["database"])
    assert conn.execute("SELECT COUNT(*) FROM events WHERE ticker='TEST'").fetchone()[0] == len(events)
    assert conn.execute("SELECT COUNT(*) FROM price_data WHERE ticker='TEST'").fetchone()[0] == 400
    assert conn.execute("SELECT COUNT(*) FROM price_data WHERE ticker='SPY'").fetchone()[0] == 400
    assert conn.execute("SELECT name FROM stocks WHERE ticker='TEST'").fetchone()[0] == "Test Robotics Inc."
    assert conn.execute("SELECT COUNT(*) FROM fundamentals WHERE ticker='TEST'").fetchone()[0] == len(fundamentals)
    run = conn.execute("SELECT status, n_events FROM analysis_runs").fetchone()
    assert run[0] in ("success", "success_with_warnings") and run[1] == len(events)
    conn.close()


def test_forward_returns_in_report_match_prices(settings):
    provider, stock = make_provider()
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    events = pd.read_csv(result.report_dir / "events.csv", parse_dates=["event_date"])
    close = stock["Adj Close"].to_numpy()
    event = events[events["event_date"] == stock.index[300].tz_localize(None).normalize()].iloc[0]
    assert event["fwd_return_1d"] == pytest.approx(close[301] / close[300] - 1)
    assert event["fwd_return_5d"] == pytest.approx(close[305] / close[300] - 1)
    assert event["fwd_return_20d"] == pytest.approx(close[320] / close[300] - 1)
    assert event["daily_return"] == pytest.approx(-0.14)


def test_second_run_creates_no_duplicates(settings):
    provider, _ = make_provider()
    first = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    second = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    assert first.n_events == second.n_events
    conn = sqlite3.connect(settings["paths"]["database"])
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == first.n_events
    assert conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0] == 800
    assert conn.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 2
    duplicates = conn.execute(
        "SELECT COUNT(*) FROM (SELECT ticker, event_date FROM events GROUP BY 1, 2 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert duplicates == 0
    conn.close()


def test_manual_notes_appear_in_next_report(settings):
    provider, stock = make_provider()
    run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    day = stock.index[300].strftime("%Y-%m-%d")
    conn = sqlite3.connect(settings["paths"]["database"])
    with conn:
        conn.execute("UPDATE events SET event_category='Guidance cut', verification_status='Verified', "
                     "source='Company press release', event_description='Lowered FY guidance' "
                     "WHERE event_date=?", (day,))
    conn.close()
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    summary = (result.report_dir / "summary.md").read_text()
    assert "Guidance cut" in summary


def test_invalid_ticker_fails_cleanly(settings):
    provider, _ = make_provider()
    with pytest.raises(AnalysisError):
        run_analysis("NOSUCH", settings, provider=provider, run_date=RUN_DATE)
    # No empty report folder is left behind, the failure is recorded, no stock row is created.
    assert not (settings_reports(settings) / "NOSUCH").exists()
    conn = sqlite3.connect(settings["paths"]["database"])
    status, error = conn.execute("SELECT status, error FROM analysis_runs").fetchone()
    assert status == "failed" and "No price data" in error
    assert conn.execute("SELECT COUNT(*) FROM stocks WHERE ticker='NOSUCH'").fetchone()[0] == 0
    conn.close()


def settings_reports(settings):
    from pathlib import Path
    return Path(settings["paths"]["reports_dir"])


def test_bad_ticker_symbols_rejected():
    assert normalize_ticker(" avav ") == "AVAV"
    assert normalize_ticker("brk-b") == "BRK-B"
    assert normalize_ticker("^gspc") == "^GSPC"
    for bad in ("", "   ", "AV AV", "../etc", "A" * 20, "AVAV;DROP"):
        with pytest.raises(ValueError):
            normalize_ticker(bad)


def test_network_failure_uses_cached_prices(settings):
    provider, _ = make_provider()
    run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)  # fills the cache
    offline_provider, _ = make_provider(failing={"TEST", "SPY"})
    result = run_analysis("TEST", settings, provider=offline_provider, run_date=RUN_DATE)
    assert result.status == "success_with_warnings"
    assert any("Live price download failed" in w and "cached" in w for w in result.warnings)
    summary = (result.report_dir / "summary.md").read_text()
    assert "Local database cache" in summary
    # Fundamentals fall back to the stored copy as well.
    assert any("stored in the local database" in w for w in result.warnings)
    assert "Test Robotics Inc." in summary


def test_network_failure_without_cache_raises(settings):
    provider, _ = make_provider(failing={"TEST"})
    with pytest.raises(AnalysisError, match="Live price download failed"):
        run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)


def test_benchmark_failure_does_not_stop_the_run(settings):
    provider, _ = make_provider(failing={"SPY"})
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    assert any("Benchmark comparison unavailable" in w for w in result.warnings)
    assert not (result.report_dir / "benchmark_chart.png").exists()
    summary = (result.report_dir / "summary.md").read_text()
    assert "## 7. Benchmark Comparison" in summary and "Not available" in summary


def test_missing_fundamentals_are_reported_not_invented(settings):
    provider, _ = make_provider(info={}, statements={})
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    summary = (result.report_dir / "summary.md").read_text()
    assert "Not available" in summary
    assert any("No annual financial-statement data" in w for w in result.warnings)


def test_offline_mode_uses_database_only(settings):
    provider, _ = make_provider()
    run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    result = run_analysis("TEST", settings, provider=None, offline=True, run_date=RUN_DATE)
    assert result.n_events > 0
    assert any("Offline mode" in w for w in result.warnings)


def test_no_events_is_handled(settings):
    provider, _ = make_provider()
    settings["events"]["return_zscore_threshold"] = 50.0
    settings["events"]["volume_ratio_threshold"] = 50.0
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    assert result.n_events == 0
    summary = (result.report_dir / "summary.md").read_text()
    assert "No abnormal events were detected" in summary
    assert (result.report_dir / "events.csv").exists()


def test_cli_with_csv_input(settings, tmp_path):
    import analyze

    stock = make_yf_prices(n=300, seed=4, shocks={200: (0.2, 5.0)})
    csv_path = tmp_path / "prices.csv"
    stock.tz_localize(None).to_csv(csv_path)
    exit_code = analyze.main([
        "TEST", "--prices-csv", str(csv_path), "--no-benchmark", "--no-fundamentals", "--offline",
        "--db", settings["paths"]["database"], "--reports-dir", settings["paths"]["reports_dir"],
    ])
    assert exit_code == 0
    folders = list((settings_reports(settings) / "TEST").iterdir())
    assert len(folders) == 1 and (folders[0] / "summary.md").exists()


def test_cli_rejects_bad_input(settings):
    import analyze

    assert analyze.main(["AV AV"]) == analyze.EXIT_BAD_INPUT
    assert analyze.main(["TEST", "--sigma", "-1", "--db", settings["paths"]["database"]]) == analyze.EXIT_BAD_INPUT


def test_cli_reports_missing_data(settings):
    import analyze

    code = analyze.main(["NODATA", "--offline", "--db", settings["paths"]["database"],
                         "--reports-dir", settings["paths"]["reports_dir"]])
    assert code == analyze.EXIT_NO_DATA


def test_data_unavailable_error_type_for_empty_csv(settings, tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("Date,Close\n")
    with pytest.raises(DataUnavailableError):
        run_analysis("TEST", settings, prices_csv=empty, offline=True, run_date=RUN_DATE)


def test_event_stats_script_reads_the_database(settings, capsys):
    import event_stats

    provider, _ = make_provider()
    run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    assert event_stats.main(["--db", settings["paths"]["database"], "--min-sigma", "2.5"]) == 0
    output = capsys.readouterr().out
    assert "All trading days (baseline)" in output
    assert "Up moves" in output


def test_analysis_json_is_strict_json(settings):
    provider, _ = make_provider()
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    text = (result.report_dir / "analysis.json").read_text()
    # NaN / Infinity are not valid JSON; missing values must be written as null.
    json.loads(text, parse_constant=lambda name: pytest.fail(f"invalid JSON constant {name}"))
    assert "NaN" not in text


def test_period_applies_to_csv_prices_and_sources_are_named(settings, tmp_path):
    stock = make_yf_prices(n=600, seed=4)
    csv_path = tmp_path / "prices.csv"
    stock.tz_localize(None).iloc[::-1].to_csv(csv_path)  # newest first, like nasdaq.com downloads
    settings["data"]["period"] = "1y"
    result = run_analysis("TEST", settings, prices_csv=csv_path, offline=True, run_date=RUN_DATE,
                          skip_fundamentals=True)
    analysis = json.loads((result.report_dir / "analysis.json").read_text())
    last = pd.Timestamp(analysis["statistics"]["end_date"])
    first = pd.Timestamp(analysis["statistics"]["start_date"])
    assert last == stock.index[-1].tz_localize(None).normalize()
    assert (last - first).days <= 366 and analysis["statistics"]["trading_days"] < 600
    assert any("use --period max" in w for w in result.warnings)
    summary = (result.report_dir / "summary.md").read_text()
    assert "Prices were read from a local CSV file (prices.csv)" in summary
    assert "Market data comes from Yahoo" not in summary

    settings["data"]["period"] = "max"
    result = run_analysis("TEST", settings, prices_csv=csv_path, offline=True, run_date=RUN_DATE,
                          skip_fundamentals=True)
    assert json.loads((result.report_dir / "analysis.json").read_text())["statistics"]["trading_days"] == 600


def test_evidence_and_fundamentals_name_the_real_source(settings):
    provider, _ = make_provider()
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    summary = (result.report_dir / "summary.md").read_text()
    assert "[Reported] by Fake test provider" in summary
    assert "Fake test provider earnings calendar" in summary
    events = pd.read_csv(result.report_dir / "events.csv")
    matched = events[events["event_category"] == "Earnings release (date match)"]
    assert (matched["source"] == "Fake test provider earnings calendar").all()


def test_annual_table_lists_metrics_the_provider_lacks(settings):
    statements = sample_statements()
    statements["cashflow"] = pd.DataFrame()  # provider has no cash-flow statement
    provider, _ = make_provider(statements=statements)
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    summary = (result.report_dir / "summary.md").read_text()
    row = next(line for line in summary.splitlines() if line.startswith("| Free cash flow |"))
    assert row.count("Not available") == 3  # shown for all three fiscal years, not dropped


def test_rerun_same_day_leaves_no_stale_charts_and_keeps_logs(settings, tmp_path):
    provider, _ = make_provider()
    first = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    assert (first.report_dir / "benchmark_chart.png").exists()
    settings["data"]["benchmark"] = None
    second = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    assert second.report_dir == first.report_dir
    assert not (second.report_dir / "benchmark_chart.png").exists()  # would contradict summary.md
    # A failed run the same day keeps the last good report and does not erase its log.
    settings["paths"]["database"] = str(tmp_path / "empty.db")
    with pytest.raises(AnalysisError):
        run_analysis("TEST", settings, provider=None, offline=True, run_date=RUN_DATE)
    assert (first.report_dir / "summary.md").exists()
    log = (first.report_dir / "run.log").read_text()
    assert log.count("analysing TEST") == 3



def test_swing_technical_report_is_part_of_every_run(settings):
    provider, stock = make_provider()
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    folder = result.report_dir
    assert (folder / "technical_chart.png").stat().st_size > 10_000
    technical = (folder / "technical_analysis.md").read_text()
    assert technical.startswith("# Swing Technical Analysis — TEST")
    assert "## 8. Swing Summary" in technical and "**Swing Technical Condition: " in technical

    for indicator in ("| EMA9 |", "| EMA21 |", "| EMA50 |", "| EMA200 |", "| EMA250 |", "| RSI6 |", "| RSI14 |",
                      "| MACD 12/26/9 |", "× MAVOL20", "| BOLL (20, 1.8) |"):
        assert indicator in technical, indicator

    summary = (folder / "summary.md").read_text()
    snapshot = summary.index("## Swing Technical Snapshot")
    assert snapshot < summary.index("## 1. Company / Ticker")  # compact section near the top
    assert "Full report: [technical_analysis.md](technical_analysis.md)" in summary
    for indicator in ("| EMA9 |", "| RSI6 |", "| MACD 12/26/9 |", "× MAVOL20", "| BOLL (20, 1.8) |"):
        assert indicator in summary[snapshot:], indicator
    assert "No upcoming earnings date is listed." in technical

    analysis = json.loads((folder / "analysis.json").read_text())
    block = analysis["technical"]
    assert block["swing_condition"] in {"Strong", "Improving", "Mixed", "Weakening", "Weak"}
    assert block["values"]["close"] == pytest.approx(stock["Adj Close"].iloc[-1])
    assert block["parameters"] == {"ema": [9, 21, 50, 200, 250], "rsi": [6, 14], "macd": [12, 26, 9], "mavol": 20,
                                   "boll": {"period": 20, "std": 1.8}, "atr": 14}
    assert block["event_risk"]["status"] == "none"
    assert f"**{block['swing_condition']}**" in summary[snapshot:]


def test_upcoming_earnings_inside_the_holding_window_are_flagged(settings):
    stock = make_yf_prices(n=400, seed=21)
    last = stock.index[-1].tz_convert("America/New_York").normalize()
    release = last + pd.offsets.BDay(2) + pd.Timedelta(hours=16, minutes=5)  # after the close, 2 sessions out
    earnings = pd.DataFrame({"EPS Estimate": [0.40], "Reported EPS": [None], "Surprise(%)": [None]},
                            index=pd.DatetimeIndex([release], name="Earnings Date"))
    provider, _ = make_provider(prices={"TEST": stock, "SPY": make_yf_prices(n=400, seed=99, base_price=400)},
                                earnings=earnings)
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    technical = (result.report_dir / "technical_analysis.md").read_text()
    summary = (result.report_dir / "summary.md").read_text()
    assert "EVENT RISK WITHIN TYPICAL HOLDING WINDOW" in technical
    assert "EVENT RISK WITHIN TYPICAL HOLDING WINDOW" in summary
    assert "3 trading days ahead" in technical  # released after the close -> the next session reacts


def test_offline_run_marks_upcoming_earnings_as_unknown(settings):
    provider, _ = make_provider()
    run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    result = run_analysis("TEST", settings, provider=None, offline=True, run_date=RUN_DATE)
    technical = (result.report_dir / "technical_analysis.md").read_text()
    assert "Earnings date unknown" in technical
    assert "EVENT RISK WITHIN TYPICAL HOLDING WINDOW" not in technical


def test_technical_failure_does_not_stop_the_main_report(settings, monkeypatch):
    import src.pipeline as pipeline

    def broken(*args, **kwargs):
        raise RuntimeError("simulated failure")

    provider, _ = make_provider()
    first = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    assert (first.report_dir / "technical_analysis.md").exists()
    # a failed technical step on a same-day re-run must not leave the earlier technical files behind
    monkeypatch.setattr(pipeline, "run_technical_analysis", broken)
    result = run_analysis("TEST", settings, provider=provider, run_date=RUN_DATE)
    assert any("Swing technical analysis could not be completed" in w for w in result.warnings)
    assert not (result.report_dir / "technical_analysis.md").exists()
    assert not (result.report_dir / "technical_chart.png").exists()
    summary = (result.report_dir / "summary.md").read_text()
    assert "## Swing Technical Snapshot" not in summary and "## 5. Abnormal Events" in summary
    assert json.loads((result.report_dir / "analysis.json").read_text())["technical"] is None
