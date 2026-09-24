"""All internet access in the project goes through this module.

``YahooFinanceProvider`` wraps the free (unofficial) Yahoo Finance API through
the ``yfinance`` library. The rest of the program only calls the four methods
below, which makes it easy to:

* switch to another data source later (write a class with the same methods), and
* test the whole pipeline without internet access (the tests use a fake
  provider that returns prepared data).

Error handling:
* Network / service errors are retried with an increasing wait ("exponential
  backoff"). If every attempt fails, a ``ProviderError`` is raised that
  includes the original error message.
* An unknown or delisted ticker is reported as ``DataUnavailableError`` right
  away (retrying would not help). Yahoo signals this either with an explicit
  "no data" answer or with HTTP 404 "Not Found" (for example when yfinance
  looks up the ticker's time zone). A network failure is never reported as an
  unknown ticker.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import pandas as pd

from src.data.prices import DataUnavailableError, standardize_price_frame

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Raised when the data service cannot be reached or keeps failing."""


class YahooFinanceProvider:
    """Downloads prices, company information, statements and earnings dates from Yahoo Finance."""

    name = "Yahoo Finance (via yfinance)"

    def __init__(self, attempts: int = 3, retry_wait_seconds: float = 2.0) -> None:
        # Imported here so the rest of the project (and the tests) can run even
        # if yfinance is not installed.
        import yfinance

        self._yf = yfinance
        self.attempts = max(1, int(attempts))
        self.retry_wait_seconds = float(retry_wait_seconds)
        # By default yfinance hides errors and returns an empty table, which makes a
        # network failure look exactly like an invalid ticker. Ask it to raise instead.
        try:
            yfinance.config.debug.hide_exceptions = False
        except AttributeError:  # older yfinance versions have no such setting
            logger.debug("yfinance.config.debug.hide_exceptions not available")
        exceptions = getattr(yfinance, "exceptions", None)
        missing = getattr(exceptions, "YFTickerMissingError", None)
        self._ticker_missing_errors: tuple[type[BaseException], ...] = (missing,) if missing else ()
        # yfinance prints its own (often repetitive) messages; ours include the
        # underlying error, so keep yfinance quiet on the console.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    # ------------------------------------------------------------------ helpers
    def _ticker(self, symbol: str) -> Any:
        # A fresh object per request: a failed request can leave yfinance's
        # cached internal state half-initialised.
        return self._yf.Ticker(symbol)

    def _is_not_found(self, exc: BaseException) -> bool:
        """True when Yahoo answered that the symbol does not exist (retrying cannot help)."""
        if self._ticker_missing_errors and isinstance(exc, self._ticker_missing_errors):
            return True
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None) == 404

    def _with_retries(self, description: str, func: Callable[[], Any]) -> Any:
        """Call ``func``; retry on errors, waiting longer each time.

        Raises:
            DataUnavailableError: Yahoo answered that the symbol does not exist (not retried).
            ProviderError: every attempt failed for another reason (network, rate limit, outage).
        """
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001 - yfinance/curl raise many different types
                if self._is_not_found(exc):
                    raise DataUnavailableError(f"{type(exc).__name__}: {exc}") from exc
                last_error = exc
                logger.warning(
                    "%s failed (attempt %d/%d): %s: %s",
                    description, attempt, self.attempts, type(exc).__name__, exc,
                )
                if attempt < self.attempts:
                    time.sleep(self.retry_wait_seconds * 2 ** (attempt - 1))
        raise ProviderError(
            f"{description} failed after {self.attempts} attempt(s) - check the internet connection. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )

    # --------------------------------------------------------------- public API
    def get_price_history(self, symbol: str, period: str) -> pd.DataFrame:
        """Download daily OHLCV prices and return a standard price frame.

        Raises:
            ProviderError: network/API failure after all retries.
            DataUnavailableError: the service answered but has no data for the
                symbol (usually an invalid or delisted ticker).
        """
        try:
            raw = self._with_retries(
                f"Price download for {symbol}",
                lambda: self._ticker(symbol).history(
                    period=period, interval="1d", auto_adjust=False, actions=False
                ),
            )
        except DataUnavailableError as exc:
            raise DataUnavailableError(
                f"Yahoo Finance has no price data for '{symbol}' ({exc}). "
                "Check the ticker symbol - it may be invalid or delisted."
            ) from exc
        if raw is None or raw.empty:
            raise DataUnavailableError(
                f"No price data was returned for '{symbol}'. The ticker may be invalid or delisted."
            )
        return standardize_price_frame(raw)

    def get_company_info(self, symbol: str) -> dict[str, Any]:
        """Return Yahoo's company profile / key statistics dictionary (may be incomplete)."""
        try:
            info = self._with_retries(f"Company info download for {symbol}", lambda: self._ticker(symbol).get_info())
        except DataUnavailableError as exc:
            raise ProviderError(f"No company information for {symbol}: {exc}") from exc
        return dict(info or {})

    def get_financial_statements(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Return annual income statement, balance sheet and cash-flow statement.

        Each table has Yahoo's raw row names (e.g. "TotalRevenue") as the index and
        one column per fiscal year end (most recent first). A statement that cannot
        be downloaded is returned as an empty DataFrame.
        """
        requests = {
            "income": lambda: self._ticker(symbol).get_income_stmt(pretty=False, freq="yearly"),
            "balance": lambda: self._ticker(symbol).get_balance_sheet(pretty=False, freq="yearly"),
            "cashflow": lambda: self._ticker(symbol).get_cash_flow(pretty=False, freq="yearly"),
        }
        statements: dict[str, pd.DataFrame] = {}
        for name, request in requests.items():
            try:
                table = self._with_retries(f"{name} statement download for {symbol}", request)
                statements[name] = table if isinstance(table, pd.DataFrame) else pd.DataFrame()
            except (ProviderError, DataUnavailableError) as exc:
                logger.warning("%s statement for %s unavailable: %s", name, symbol, exc)
                statements[name] = pd.DataFrame()
        return statements

    def get_earnings_dates(self, symbol: str, limit: int = 100) -> pd.DataFrame | None:
        """Return past and upcoming earnings dates (index = earnings date/time), or None."""
        try:
            return self._with_retries(
                f"Earnings calendar download for {symbol}",
                lambda: self._ticker(symbol).get_earnings_dates(limit=limit),
            )
        except DataUnavailableError as exc:
            raise ProviderError(f"No earnings calendar for {symbol}: {exc}") from exc


__all__ = ["YahooFinanceProvider", "ProviderError", "DataUnavailableError"]
