"""Tests for YahooFinanceProvider error handling, using a stand-in for the yfinance module.

No internet access is needed: the fake ``Ticker`` objects raise or return
whatever each test needs, exactly like yfinance would.
"""

import pandas as pd
import pytest

pytest.importorskip("yfinance")

from yfinance.exceptions import YFPricesMissingError  # noqa: E402

from src.data.prices import DataUnavailableError  # noqa: E402
from src.data.provider import ProviderError, YahooFinanceProvider  # noqa: E402
from tests.conftest import make_yf_prices  # noqa: E402


class FakeYF:
    """Replaces the yfinance module: every Ticker call returns (or raises) the next scripted outcome."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def next_outcome(self):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def Ticker(self, symbol):  # noqa: N802 - same name as in yfinance
        fake = self

        class _Ticker:
            def history(self, **kwargs):
                return fake.next_outcome()

            def get_info(self):
                return fake.next_outcome()

        return _Ticker()


def provider_with(outcomes, attempts=3):
    provider = YahooFinanceProvider(attempts=attempts, retry_wait_seconds=0)
    provider._yf = FakeYF(outcomes)
    return provider


def test_success_after_a_network_error_is_retried():
    provider = provider_with([ConnectionError("reset by peer"), make_yf_prices(n=5)])
    prices = provider.get_price_history("AVAV", "5y")
    assert len(prices) == 5
    assert provider._yf.calls == 2


def test_repeated_network_errors_raise_provider_error():
    provider = provider_with([OSError("403 tunnel")] * 3)
    with pytest.raises(ProviderError, match="failed after 3 attempt"):
        provider.get_price_history("AVAV", "5y")
    assert provider._yf.calls == 3


def test_unknown_ticker_is_not_retried():
    provider = provider_with([YFPricesMissingError("ZZZZ", "")])
    with pytest.raises(DataUnavailableError, match="invalid or delisted"):
        provider.get_price_history("ZZZZ", "5y")
    assert provider._yf.calls == 1


def test_empty_answer_means_no_data():
    provider = provider_with([pd.DataFrame()])
    with pytest.raises(DataUnavailableError):
        provider.get_price_history("ZZZZ", "5y")


def test_company_info_failure_raises_provider_error():
    provider = provider_with([TimeoutError("slow")] * 2, attempts=2)
    with pytest.raises(ProviderError):
        provider.get_company_info("AVAV")
