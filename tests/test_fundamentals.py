"""Tests for fundamental-data normalisation (no fabricated values, graceful failures)."""

from datetime import date

import pytest

from src.data.provider import ProviderError
from src.fundamentals.fundamentals import (
    annual_records,
    annual_table,
    collect_fundamentals,
    extract_profile,
    snapshot_records,
    to_finite_float,
)
from tests.conftest import FakeProvider, sample_info, sample_statements

RETRIEVED = date(2026, 9, 24)


def by_metric(records, period_type, period_end=None):
    return {r["metric"]: r for r in records
            if r["period_type"] == period_type and (period_end is None or r["period_end"] == period_end)}


def test_to_finite_float():
    assert to_finite_float("12.5") == 12.5
    assert to_finite_float(None) is None
    assert to_finite_float("Infinity") is None
    assert to_finite_float(float("nan")) is None
    assert to_finite_float("n/a") is None
    assert to_finite_float(True) is None


def test_snapshot_records_skip_missing_and_invalid_values():
    records = by_metric(snapshot_records("TEST", sample_info(), RETRIEVED), "snapshot")
    assert records["market_cap"]["value"] == 5.0e9
    assert records["market_cap"]["unit"] == "USD"
    assert records["pe_trailing"]["value"] == 150.0
    assert "pe_forward" not in records  # "Infinity" is not a real value
    assert "ev_to_ebitda" not in records  # provider gave None - not invented
    assert records["market_cap"]["period_end"] == "2026-09-24"
    assert "marketCap" in records["market_cap"]["source"]


def test_annual_records_and_calculated_ratios():
    records = annual_records("TEST", sample_statements(), sample_info(), RETRIEVED)
    fy25 = by_metric(records, "annual", "2025-04-30")
    fy24 = by_metric(records, "annual", "2024-04-30")
    fy23 = by_metric(records, "annual", "2023-04-30")
    assert fy25["revenue"]["value"] == 800.0
    assert fy25["gross_margin"]["value"] == pytest.approx(320 / 800)
    assert fy25["operating_margin"]["value"] == pytest.approx(40 / 800)
    assert fy25["net_margin"]["value"] == pytest.approx(30 / 800)
    assert fy25["revenue_growth"]["value"] == pytest.approx(800 / 700 - 1)
    assert fy24["revenue_growth"]["value"] == pytest.approx(700 / 500 - 1)
    assert "revenue_growth" not in fy23  # no earlier year to compare with
    assert fy25["gross_margin"]["source"].startswith("Calculated")
    assert fy25["free_cash_flow"]["value"] == 20.0
    assert "free_cash_flow" not in fy23  # not in the statement -> absent, not zero
    assert fy25["shares_outstanding"]["unit"] == "shares"


def test_growth_not_calculated_across_missing_years():
    statements = sample_statements()
    statements["income"] = statements["income"].drop(columns=[statements["income"].columns[1]])  # drop FY2024
    records = annual_records("TEST", statements, sample_info(), RETRIEVED)
    assert "revenue_growth" not in by_metric(records, "annual", "2025-04-30")


def test_annual_table_layout():
    table = annual_table(annual_records("TEST", sample_statements(), sample_info(), RETRIEVED))
    assert list(table.columns) == ["2023-04-30", "2024-04-30", "2025-04-30"]
    assert table.loc["revenue", "2025-04-30"] == 800.0


def test_profile_extraction():
    profile = extract_profile(sample_info())
    assert profile["name"] == "Test Robotics Inc."
    assert profile["exchange"] == "NasdaqGS"
    assert extract_profile({})["name"] is None


def test_collect_fundamentals_survives_provider_failure():
    provider = FakeProvider(prices={}, failing={"TEST"})
    result = collect_fundamentals(provider, "TEST", RETRIEVED)
    assert result["records"] == []
    assert any("unavailable" in w for w in result["warnings"])


def test_collect_fundamentals_with_statement_exception():
    class BrokenStatements(FakeProvider):
        def get_financial_statements(self, symbol):
            raise ProviderError("timeout")

    result = collect_fundamentals(BrokenStatements(prices={}, info=sample_info()), "TEST", RETRIEVED)
    assert any(r["period_type"] == "snapshot" for r in result["records"])
    assert any("statements unavailable" in w for w in result["warnings"])
