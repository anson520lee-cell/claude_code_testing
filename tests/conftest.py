"""Shared test helpers: synthetic price data in yfinance's format and a fake data provider.

The synthetic data is only used by the tests. It lets us plant known events
(e.g. "+15% on day 200 with 5x volume") and check that the program finds them.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_SETTINGS
from src.data.prices import DataUnavailableError, standardize_price_frame
from src.data.provider import ProviderError


def make_yf_prices(
    n: int = 400,
    start: str = "2023-01-03",
    seed: int = 1,
    shocks: dict[int, tuple[float, float]] | None = None,
    base_price: float = 100.0,
    daily_vol: float = 0.015,
) -> pd.DataFrame:
    """Random-walk prices shaped exactly like ``yfinance.Ticker.history(auto_adjust=False)``.

    Args:
        shocks: {day_index: (daily_return, volume_multiplier)} to plant known events.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n, tz="America/New_York")
    returns = rng.normal(0.0004, daily_vol, n)
    returns[0] = 0.0
    volume = rng.lognormal(mean=np.log(1_000_000), sigma=0.2, size=n)
    for day, (ret, multiplier) in (shocks or {}).items():
        returns[day] = ret
        volume[day] *= multiplier
    close = base_price * np.cumprod(1 + returns)
    previous = np.concatenate([[base_price], close[:-1]])
    open_ = previous * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    frame = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": volume.round()},
        index=dates,
    )
    frame.index.name = "Date"
    return frame


class FakeProvider:
    """Stands in for YahooFinanceProvider in tests (same method names, no internet)."""

    name = "Fake test provider"

    def __init__(self, prices: dict[str, pd.DataFrame], info: dict | None = None,
                 statements: dict | None = None, earnings: pd.DataFrame | None = None,
                 failing: set[str] | None = None) -> None:
        self.prices = prices
        self.info = info or {}
        self.statements = statements or {}
        self.earnings = earnings
        self.failing = failing or set()
        self.calls: list[str] = []

    def get_price_history(self, symbol: str, period: str) -> pd.DataFrame:
        self.calls.append(f"prices:{symbol}")
        if symbol in self.failing:
            raise ProviderError(f"Price download for {symbol} failed after 3 attempt(s). Last error: timeout")
        if symbol not in self.prices:
            raise DataUnavailableError(f"No price data was returned for '{symbol}'.")
        return standardize_price_frame(self.prices[symbol])

    def get_company_info(self, symbol: str) -> dict:
        self.calls.append(f"info:{symbol}")
        if symbol in self.failing:
            raise ProviderError("Company info download failed")
        return dict(self.info)

    def get_financial_statements(self, symbol: str) -> dict:
        self.calls.append(f"statements:{symbol}")
        return dict(self.statements)

    def get_earnings_dates(self, symbol: str, limit: int = 100):
        self.calls.append(f"earnings:{symbol}")
        return self.earnings


def sample_info() -> dict:
    return {
        "longName": "Test Robotics Inc.", "fullExchangeName": "NasdaqGS", "sector": "Industrials",
        "industry": "Aerospace & Defense", "currency": "USD", "financialCurrency": "USD", "country": "United States",
        "quoteType": "EQUITY", "marketCap": 5.0e9, "enterpriseValue": 4.8e9, "totalRevenue": 8.0e8,
        "revenueGrowth": 0.12, "grossMargins": 0.40, "operatingMargins": 0.05, "netIncomeToCommon": 3.0e7,
        "trailingEps": 1.10, "freeCashflow": 2.0e7, "ebitda": 9.0e7, "totalCash": 7.0e7, "totalDebt": 2.0e7,
        "sharesOutstanding": 2.8e7, "trailingPE": 150.0, "forwardPE": "Infinity",  # invalid on purpose
        "enterpriseToRevenue": 6.0, "enterpriseToEbitda": None,  # missing on purpose
    }


def sample_statements() -> dict:
    years = [pd.Timestamp("2025-04-30"), pd.Timestamp("2024-04-30"), pd.Timestamp("2023-04-30")]
    income = pd.DataFrame(
        {years[0]: [800.0, 320.0, 40.0, 30.0, 1.1, 90.0],
         years[1]: [700.0, 266.0, 21.0, 60.0, 2.2, 80.0],
         years[2]: [500.0, 200.0, -10.0, -5.0, -0.2, 20.0]},
        index=["TotalRevenue", "GrossProfit", "OperatingIncome", "NetIncomeCommonStockholders", "DilutedEPS",
               "EBITDA"],
    )
    cashflow = pd.DataFrame({years[0]: [50.0, -30.0, 20.0], years[1]: [40.0, -20.0, 20.0]},
                            index=["OperatingCashFlow", "CapitalExpenditure", "FreeCashFlow"])
    balance = pd.DataFrame({years[0]: [70.0, 20.0, 28.0]},
                           index=["CashAndCashEquivalents", "TotalDebt", "OrdinarySharesNumber"])
    return {"income": income, "balance": balance, "cashflow": cashflow}


@pytest.fixture
def settings(tmp_path: Path) -> dict:
    """Default settings pointing the database and reports at a temporary folder."""
    s = copy.deepcopy(DEFAULT_SETTINGS)
    s["paths"]["database"] = str(tmp_path / "test.db")
    s["paths"]["reports_dir"] = str(tmp_path / "reports")
    return s
