"""Tests for standardising, cleaning and validating price data (including CSV import)."""

import numpy as np
import pandas as pd
import pytest

from src.data.prices import (
    DataUnavailableError,
    choose_price_column,
    clean_price_frame,
    load_prices_csv,
    standardize_price_frame,
)
from tests.conftest import make_yf_prices


def test_standardize_yfinance_frame():
    raw = make_yf_prices(n=5)
    df = standardize_price_frame(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert df.index.tz is None  # time zone removed
    assert df.index.name == "date"
    assert df.index[0] == pd.Timestamp("2023-01-03")
    assert df["close"].iloc[-1] == pytest.approx(raw["Close"].iloc[-1])


def test_missing_adjusted_close_becomes_nan_column():
    raw = make_yf_prices(n=5).drop(columns=["Adj Close"])
    df = standardize_price_frame(raw)
    assert df["adj_close"].isna().all()
    assert choose_price_column(df)[0] == "close"


def test_empty_or_closeless_frames_raise():
    with pytest.raises(DataUnavailableError):
        standardize_price_frame(pd.DataFrame())
    with pytest.raises(DataUnavailableError):
        standardize_price_frame(pd.DataFrame({"Open": [1.0]}, index=pd.to_datetime(["2024-01-02"])))


def test_cleaning_removes_bad_rows_and_reports_them():
    df = standardize_price_frame(make_yf_prices(n=10))
    df.iloc[3, df.columns.get_loc("close")] = np.nan
    df.iloc[4, df.columns.get_loc("close")] = -1.0
    df = pd.concat([df, df.iloc[[7]]])  # duplicate date
    df.iloc[5, df.columns.get_loc("volume")] = -100
    cleaned, warnings = clean_price_frame(df)
    assert len(cleaned) == 8
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.duplicated().any()
    text = " ".join(warnings)
    assert "duplicate" in text
    assert "missing or non-positive close" in text
    assert "negative volume" in text


def test_cleaning_flags_but_keeps_inconsistent_ohlc_and_gaps():
    df = standardize_price_frame(make_yf_prices(n=30))
    df.iloc[2, df.columns.get_loc("high")] = df["low"].iloc[2] * 0.5  # high below low
    df = df.drop(df.index[10:20])  # a two-week hole
    cleaned, warnings = clean_price_frame(df)
    assert len(cleaned) == 20  # nothing silently removed
    assert any("inconsistent OHLC" in w for w in warnings)
    assert any("gap" in w for w in warnings)


def test_too_little_data_raises():
    df = standardize_price_frame(make_yf_prices(n=1))
    with pytest.raises(DataUnavailableError):
        clean_price_frame(df)


def test_partial_adjusted_close_falls_back_to_close():
    df = standardize_price_frame(make_yf_prices(n=5))
    df.iloc[2, df.columns.get_loc("adj_close")] = np.nan
    column, note = choose_price_column(df)
    assert column == "close"
    assert "missing" in note


def test_load_nasdaq_style_csv(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "Date,Close/Last,Volume,Open,High,Low\n"
        "01/05/2024,$101.50,\"1,200,000\",$100.00,$102.00,$99.50\n"
        "01/04/2024,$100.00,\"1,000,000\",$99.00,$100.50,$98.00\n"
    )
    df = load_prices_csv(path)
    cleaned, _ = clean_price_frame(df)
    assert cleaned.index[0] == pd.Timestamp("2024-01-04")
    assert cleaned["close"].iloc[-1] == pytest.approx(101.5)
    assert cleaned["volume"].iloc[-1] == pytest.approx(1_200_000)


def test_load_csv_errors(tmp_path):
    with pytest.raises(DataUnavailableError):
        load_prices_csv(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("Close\n1\n2\n")
    with pytest.raises(DataUnavailableError):
        load_prices_csv(bad)
