"""Deep-dive data for shortlisted candidates (fundamentals, options, positioning, catalysts).

``YahooSecEnrichment.fetch(symbol)`` returns the RAW inputs; analysis happens in the
domain modules. Tests replace this class with a fake that returns prepared data.
"""

from __future__ import annotations

import logging
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.sec import SecClient, recent_filings
from src.derivatives.options import fetch_chains
from src.positioning.positioning import FinraData

logger = logging.getLogger(__name__)


class YahooSecEnrichment:
    name = "Yahoo Finance + SEC EDGAR + FINRA"

    def __init__(self, user_agent: str, cache_dir: Path, session: date, offline: bool = False) -> None:
        import yfinance

        self._yf = yfinance
        self.session = session
        self.offline = offline
        self.sec_enabled = bool(user_agent)
        self.sec = SecClient(user_agent or "", cache_dir, today=session, offline=offline)
        self.finra = FinraData(cache_dir, offline=offline)
        self._sec_lock = threading.Lock()

    def _safe(self, label: str, func, default=None):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - every source is optional
            logger.warning("%s unavailable: %s: %s", label, type(exc).__name__, exc)
            return default

    def fetch(self, symbol: str) -> dict[str, Any]:
        raw: dict[str, Any] = {"symbol": symbol}
        raw["sec_submissions"] = raw["sec_facts"] = None
        if self.sec_enabled and not self.sec.blocked:
            with self._sec_lock:  # SEC fair-access rate limit is global
                raw["sec_submissions"] = self._safe(f"SEC submissions {symbol}", lambda: self.sec.submissions(symbol))
                raw["sec_facts"] = self._safe(f"SEC facts {symbol}", lambda: self.sec.company_facts(symbol))
        raw["sec_status"] = ("not_configured" if not self.sec_enabled else "blocked" if self.sec.blocked else "ok")
        raw["filings"] = recent_filings(raw["sec_submissions"])
        if self.offline:
            return raw
        t = self._yf.Ticker(symbol)
        if raw["sec_facts"] is None:
            raw["yahoo_statements"] = {
                "income": self._safe(f"quarterly income {symbol}", lambda: t.get_income_stmt(pretty=False, freq="quarterly")),
                "balance": self._safe(f"quarterly balance {symbol}", lambda: t.get_balance_sheet(pretty=False, freq="quarterly")),
                "cashflow": self._safe(f"quarterly cash flow {symbol}", lambda: t.get_cash_flow(pretty=False, freq="quarterly")),
            }
        raw["info"] = self._safe(f"info {symbol}", t.get_info, {}) or {}
        raw["earnings_dates"] = self._safe(f"earnings dates {symbol}", lambda: t.get_earnings_dates(limit=12))
        raw["calendar"] = self._safe(f"calendar {symbol}", lambda: t.calendar, {})
        raw["insider"] = self._safe(f"insider {symbol}", lambda: t.insider_transactions)
        raw["institutions"] = self._safe(f"institutions {symbol}", lambda: t.institutional_holders)
        raw["upgrades"] = self._safe(f"analyst actions {symbol}", lambda: t.upgrades_downgrades)
        raw["option_chains"] = self._safe(f"options {symbol}", lambda: fetch_chains(t, self.session), [])
        return raw


def is_frame(value: Any) -> bool:
    return isinstance(value, pd.DataFrame) and not value.empty
