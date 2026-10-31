"""SEC EDGAR client (free, official): ticker -> CIK map, filing index, XBRL company facts.

* https://www.sec.gov/files/company_tickers.json          ticker -> CIK
* https://data.sec.gov/submissions/CIK##########.json     recent filings (form, filing date, acceptance time, 8-K items)
* https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json   every XBRL fact with period end AND filing date

SEC fair-access rules: a descriptive User-Agent with contact details and <= 10 requests/second.
Responses are cached on disk per calendar day (data/cache/sec/), so a re-run is free.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
MIN_INTERVAL = 0.12  # seconds between requests (< 10 per second)


class SecClient:
    def __init__(self, user_agent: str, cache_dir: Path, today: date | None = None, session: Any = None,
                 offline: bool = False) -> None:
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.cache_dir = Path(cache_dir) / "sec"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.today = today or date.today()
        self.http = session or requests.Session()
        self.offline = offline
        self._last = 0.0
        self._tickers: dict[str, int] | None = None
        self.blocked = False          # True after SEC answered 403 (User-Agent policy)
        self.last_error: str | None = None

    def _get_json(self, url: str, cache_name: str, max_age_days: int = 0) -> Any | None:
        stamp = self.today.isoformat()
        path = self.cache_dir / f"{cache_name}.json"
        meta = self.cache_dir / f"{cache_name}.date"
        if path.exists() and meta.exists():
            cached = date.fromisoformat(meta.read_text().strip())
            if self.offline or (self.today - cached).days <= max_age_days:
                try:
                    return json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass
        if self.offline:
            return None
        wait = MIN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            response = self.http.get(url, headers=self.headers, timeout=30)
            self._last = time.monotonic()
            if response.status_code == 404:
                return None
            if response.status_code == 403:
                self.blocked = True
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("SEC request failed (%s): %s", url, exc)
            if path.exists():
                return json.loads(path.read_text())  # stale copy is better than nothing (dates are kept)
            return None
        path.write_text(json.dumps(data))
        meta.write_text(stamp)
        return data

    def cik(self, ticker: str) -> int | None:
        if self._tickers is None:
            data = self._get_json(TICKERS_URL, "company_tickers", max_age_days=7) or {}
            self._tickers = {str(v["ticker"]).upper(): int(v["cik_str"]) for v in data.values()}
        return self._tickers.get(ticker.upper().replace(".", "-"))

    def submissions(self, ticker: str) -> dict | None:
        cik = self.cik(ticker)
        return self._get_json(SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik}") if cik else None

    def company_facts(self, ticker: str) -> dict | None:
        cik = self.cik(ticker)
        return self._get_json(FACTS_URL.format(cik=cik), f"facts_{cik}") if cik else None


def recent_filings(submissions: dict | None, limit: int = 200) -> list[dict[str, Any]]:
    """Flatten the 'recent' filing arrays into dicts (newest first)."""
    if not submissions:
        return []
    recent = submissions.get("filings", {}).get("recent", {})
    keys = ("accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "primaryDocument",
            "primaryDocDescription", "reportDate")
    n = len(recent.get("form", []))
    cik = int(submissions.get("cik", 0) or 0)
    out = []
    for i in range(min(n, limit)):
        row = {k: (recent.get(k) or [None] * n)[i] for k in keys}
        accession = (row["accessionNumber"] or "").replace("-", "")
        row["url"] = (FILING_URL.format(cik=cik, accession=accession, document=row["primaryDocument"])
                      if accession and row["primaryDocument"] else None)
        out.append(row)
    return out
