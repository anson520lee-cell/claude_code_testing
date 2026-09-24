"""Tests for the SQLite layer: inserts, duplicate handling and protection of research notes."""

import json

import pandas as pd
import pytest

from src.database import db
from src.events.evidence import EARNINGS_SOURCE


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    db.upsert_stock(connection, "TEST")
    yield connection
    connection.close()


def make_events(dates, **overrides):
    rows = []
    for i, date in enumerate(dates):
        row = {
            "ticker": "TEST", "event_date": pd.Timestamp(date), "close": 10.0 + i, "daily_return": 0.1,
            "return_zscore": 3.0, "volume": 5000.0, "avg_volume": 1000.0, "volume_ratio": 5.0,
            "volume_zscore": 4.0, "benchmark": "SPY", "benchmark_return": 0.01, "abnormal_return": 0.09,
            "fwd_return_1d": 0.01, "fwd_return_5d": None, "fwd_return_20d": None,
            "fwd_abnormal_1d": 0.0, "fwd_abnormal_5d": None, "fwd_abnormal_20d": None,
            "direction": "up", "is_price_event": True, "is_volume_event": True,
            "trigger_rules": "return_zscore,volume_ratio", "anomaly_type": "x",
            "trading_days_since_prev_event": None, "detection_settings": json.dumps({"sigma": 2.5}),
            "event_category": "Unknown", "event_description": None, "source": None,
            "verification_status": "Unverified", "notes": None,
        }
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_schema_creates_all_tables(conn):
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"stocks", "price_data", "events", "fundamentals", "analysis_runs"} <= tables


def test_event_insertion(conn):
    result = db.upsert_events(conn, make_events(["2024-01-05", "2024-02-06"]))
    assert result == {"inserted": 2, "updated": 0}
    stored = db.load_events(conn, "TEST")
    assert len(stored) == 2
    assert stored.iloc[0]["event_date"] == pd.Timestamp("2024-01-05")
    assert stored.iloc[0]["is_price_event"] == 1
    assert pd.isna(stored.iloc[0]["fwd_return_5d"])  # NaN/None stored as NULL


def test_duplicate_events_are_updated_not_duplicated(conn):
    db.upsert_events(conn, make_events(["2024-01-05", "2024-02-06"]))
    result = db.upsert_events(conn, make_events(["2024-01-05", "2024-02-06", "2024-03-07"], fwd_return_5d=0.05))
    assert result == {"inserted": 1, "updated": 2}
    assert count(conn, "events") == 3
    stored = db.load_events(conn, "TEST")
    assert (stored["fwd_return_5d"] == 0.05).all()


def test_manual_research_is_never_overwritten(conn):
    db.upsert_events(conn, make_events(["2024-01-05"]))
    with conn:
        conn.execute(
            "UPDATE events SET event_category='Contract award', event_description='Army contract announced', "
            "source='Company press release', verification_status='Verified', notes='checked 8-K' "
            "WHERE event_date='2024-01-05'"
        )
    # A later automatic run finds an earnings date match for the same day.
    db.upsert_events(conn, make_events(["2024-01-05"], event_category="Earnings release (date match)",
                                       event_description="auto", source="Yahoo",
                                       verification_status="Unverified (automatic date match)"))
    row = db.load_events(conn, "TEST").iloc[0]
    assert row["event_category"] == "Contract award"
    assert row["event_description"] == "Army contract announced"
    assert row["verification_status"] == "Verified"
    assert row["notes"] == "checked 8-K"


def test_changed_category_is_protected_even_if_status_unchanged(conn):
    db.upsert_events(conn, make_events(["2024-01-05"]))
    with conn:
        conn.execute("UPDATE events SET event_category='Contract award' WHERE event_date='2024-01-05'")
    db.upsert_events(conn, make_events(["2024-01-05"], event_category="Earnings release (date match)",
                                       event_description="auto", source=EARNINGS_SOURCE,
                                       verification_status="Unverified (automatic date match)"))
    row = db.load_events(conn, "TEST").iloc[0]
    assert row["event_category"] == "Contract award"
    assert pd.isna(row["event_description"])


def test_automatic_evidence_not_replaced_by_unknown(conn):
    db.upsert_events(conn, make_events(["2024-01-05"], event_category="Earnings release (date match)",
                                       event_description="auto", source="Yahoo",
                                       verification_status="Unverified (automatic date match)"))
    db.upsert_events(conn, make_events(["2024-01-05"]))  # e.g. earnings calendar failed this time
    row = db.load_events(conn, "TEST").iloc[0]
    assert row["event_category"] == "Earnings release (date match)"
    assert row["event_description"] == "auto"


def test_unknown_event_gets_evidence_later(conn):
    db.upsert_events(conn, make_events(["2024-01-05"]))
    db.upsert_events(conn, make_events(["2024-01-05"], event_category="Earnings release (date match)",
                                       event_description="auto", source="Yahoo",
                                       verification_status="Unverified (automatic date match)"))
    row = db.load_events(conn, "TEST").iloc[0]
    assert row["event_category"] == "Earnings release (date match)"


def test_run_without_benchmark_keeps_stored_benchmark_values(conn):
    db.upsert_events(conn, make_events(["2024-01-05"]))
    db.upsert_events(conn, make_events(["2024-01-05"], benchmark=None, benchmark_return=None,
                                       abnormal_return=None, fwd_abnormal_1d=None))
    row = db.load_events(conn, "TEST").iloc[0]
    assert row["benchmark"] == "SPY"
    assert row["abnormal_return"] == pytest.approx(0.09)


def test_price_upsert_handles_duplicates(conn):
    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0], "low": [1.0, 2.0, 3.0],
                           "close": [1.0, 2.0, 3.0], "adj_close": [1.0, 2.0, 3.0], "volume": [10.0, 20.0, 30.0]},
                          index=dates)
    db.upsert_prices(conn, "TEST", prices, "test")
    revised = prices.copy()
    revised.loc[dates[2], "close"] = 3.5  # provider revised a value
    db.upsert_prices(conn, "TEST", revised, "test")
    assert count(conn, "price_data") == 3
    loaded = db.load_prices(conn, "TEST")
    assert loaded.loc[dates[2], "close"] == 3.5
    assert list(loaded.columns) == ["open", "high", "low", "close", "adj_close", "volume"]


def test_fundamentals_upsert_handles_duplicates(conn):
    record = {"ticker": "TEST", "period_type": "annual", "period_end": "2025-04-30", "metric": "revenue",
              "value": 100.0, "unit": "USD", "source": "test", "retrieved_at": "2026-09-24"}
    db.upsert_fundamentals(conn, [record])
    db.upsert_fundamentals(conn, [dict(record, value=110.0)])
    assert count(conn, "fundamentals") == 1
    assert db.load_fundamentals(conn, "TEST").iloc[0]["value"] == 110.0


def test_refresh_forward_returns_fills_missing_values(conn):
    db.upsert_events(conn, make_events(["2024-01-05"]))
    stats = pd.DataFrame({"fwd_return_1d": [0.02], "fwd_return_5d": [0.07], "fwd_return_20d": [float("nan")],
                          "fwd_abnormal_5d": [0.03]}, index=[pd.Timestamp("2024-01-05")])
    assert db.refresh_forward_returns(conn, "TEST", stats, benchmark="SPY") == 1
    row = db.load_events(conn, "TEST").iloc[0]
    assert row["fwd_return_5d"] == pytest.approx(0.07)
    assert row["fwd_return_1d"] == pytest.approx(0.02)
    assert pd.isna(row["fwd_return_20d"])  # still unknown
    assert row["fwd_abnormal_5d"] == pytest.approx(0.03)
    # A different benchmark must not overwrite abnormal returns measured against SPY.
    db.refresh_forward_returns(conn, "TEST", stats.assign(fwd_abnormal_5d=0.99), benchmark="QQQ")
    assert db.load_events(conn, "TEST").iloc[0]["fwd_abnormal_5d"] == pytest.approx(0.03)


def test_analysis_run_lifecycle(conn):
    run_id = db.start_run(conn, "TEST", "5y", "SPY", {"a": 1})
    assert db.get_run(conn, run_id)["status"] == "running"
    db.finish_run(conn, run_id, "success", n_events=4, warnings=["note"], start_date=pd.Timestamp("2024-01-02"))
    run = db.get_run(conn, run_id)
    assert run["status"] == "success"
    assert run["n_events"] == 4
    assert json.loads(run["warnings"]) == ["note"]
    assert run["start_date"] == "2024-01-02"
    with pytest.raises(ValueError):
        db.finish_run(conn, run_id, "success", not_a_column=1)


def test_import_research_from_edited_csv(conn):
    db.upsert_events(conn, make_events(["2024-01-05", "2024-02-06"]))
    edited = pd.DataFrame({
        "ticker": ["test", "TEST", "TEST"],
        "event_date": ["2024-01-05", "2024-02-06", "2030-01-01"],
        "close": ["999", "999", "999"],  # never imported
        "event_category": ["Contract award", "", "x"],
        "event_description": ["Army contract", "", ""],
        "source": ["Press release", "", ""],
        "verification_status": ["Unverified", "", ""],
        "notes": ["", "check 10-Q", ""],
    })
    preview = db.import_research_fields(conn, edited, dry_run=True)
    assert preview["updated_events"] == 2
    assert db.load_events(conn, "TEST").iloc[0]["event_category"] == "Unknown"  # dry run saved nothing

    result = db.import_research_fields(conn, edited)
    assert result["unmatched"] == [("TEST", "2030-01-01")]
    assert result["status_set_manual"] == 1
    stored = db.load_events(conn, "TEST").set_index("event_date")
    first, second = stored.loc["2024-01-05"], stored.loc["2024-02-06"]
    assert first["event_category"] == "Contract award"
    assert first["verification_status"] == db.MANUAL_STATUS
    assert first["close"] == 10.0  # statistics untouched
    assert second["notes"] == "check 10-Q"
    assert second["event_category"] == "Unknown"  # blank cell did not erase anything
    # Importing the same file again changes nothing (the file's old "Unverified" is ignored).
    assert db.import_research_fields(conn, edited)["updated_events"] == 0
    # A later automatic run does not overwrite the imported research.
    db.upsert_events(conn, make_events(["2024-01-05"], event_category="Earnings release (date match)",
                                       event_description="auto", source=EARNINGS_SOURCE,
                                       verification_status="Unverified (automatic date match)"))
    assert db.load_events(conn, "TEST").iloc[0]["event_description"] == "Army contract"


def test_import_requires_key_columns(conn):
    with pytest.raises(ValueError):
        db.import_research_fields(conn, pd.DataFrame({"notes": ["x"]}))
