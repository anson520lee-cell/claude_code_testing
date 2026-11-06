"""US-listed sector/industry equities, ranked by estimated daily dollar turnover."""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from typing import Any

# These are industry peers, not an ETF's holdings (which would omit many peers).
GROUPS = {
    "XLK": ("Technology", "sector", ["Technology"]),
    "XLC": ("Communication Services", "sector", ["Communication Services"]),
    "XLY": ("Consumer Discretionary", "sector", ["Consumer Cyclical"]),
    "XLP": ("Consumer Staples", "sector", ["Consumer Defensive"]),
    "XLE": ("Energy", "sector", ["Energy"]),
    "XLF": ("Financials", "sector", ["Financial Services"]),
    "XLV": ("Health Care", "sector", ["Healthcare"]),
    "XLI": ("Industrials", "sector", ["Industrials"]),
    "XLB": ("Materials", "sector", ["Basic Materials"]),
    "XLRE": ("Real Estate", "sector", ["Real Estate"]),
    "XLU": ("Utilities", "sector", ["Utilities"]),
    "SMH": ("Semiconductors", "industry", ["Semiconductors", "Semiconductor Equipment & Materials"]),
    "XBI": ("Biotech", "industry", ["Biotechnology"]),
    "KRE": ("Regional Banks", "industry", ["Banks—Regional"]),
    "ITA": ("Aerospace & Defense", "industry", ["Aerospace & Defense"]),
}


def number(value: Any, *, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and (not nonnegative or result >= 0) else None


def normalize(quotes: list[dict]) -> list[dict]:
    rows = {}
    for q in quotes:
        symbol = str(q.get("symbol") or "").strip().upper()
        if not symbol or q.get("quoteType", "EQUITY") != "EQUITY":
            continue
        # Keep zero volume distinct from unavailable volume; never rank by share volume.
        price = number(q.get("regularMarketPrice"), nonnegative=True)
        volume = number(q.get("regularMarketVolume"), nonnegative=True)
        change = number(q.get("regularMarketChangePercent"))
        timestamp = number(q.get("regularMarketTime"), nonnegative=True)
        try:
            quoted_at = datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp is not None else None
        except (ValueError, OSError, OverflowError):
            quoted_at = None
        rows.setdefault(symbol, {
            "symbol": symbol, "name": q.get("longName") or q.get("shortName") or symbol,
            "price": price, "change_pct": change / 100 if change is not None else None,
            "volume": volume, "turnover": number(price * volume, nonnegative=True)
            if price is not None and volume is not None else None,
            "quoted_at": quoted_at, "currency": q.get("currency") or "USD",
        })
    # All requested listings are US equities; exclude non-USD quotes from a USD ranking.
    out = [r for r in rows.values() if r["currency"] == "USD"]
    out.sort(key=lambda r: (r["turnover"] is None, -(r["turnover"] or 0), r["symbol"]))
    return [{**r, "rank": i} for i, r in enumerate(out, 1)]


class SectorStocks:
    """Bounded in-memory cache; failed refreshes may reuse a labelled last good answer."""
    def __init__(self, ttl: float = 300):
        self.ttl = ttl
        self.cache: dict[str, tuple[float, dict]] = {}
        self.locks = {symbol: threading.Lock() for symbol in GROUPS}

    def get(self, symbol: str, screener=None) -> dict:
        symbol = symbol.upper()
        if symbol not in GROUPS:
            raise ValueError("Unknown sector or industry ETF")
        with self.locks[symbol]:
            cached = self.cache.get(symbol)
            if cached and time.monotonic() - cached[0] < self.ttl:
                return dict(cached[1])
            try:
                result = self.fetch(symbol, screener)
            except Exception:
                if cached:
                    return {**cached[1], "stale": True}
                raise
            self.cache[symbol] = (time.monotonic(), result)
            return dict(result)

    @staticmethod
    def fetch(symbol: str, screener=None) -> dict:
        name, field, values = GROUPS[symbol]
        if screener is None:
            import yfinance as yf
            from yfinance import EquityQuery as Q
            group = [Q("eq", [field, value]) for value in values]
            query = Q("and", [Q("eq", ["region", "us"]),
                              group[0] if len(group) == 1 else Q("or", group)])

            def screener(offset):
                return yf.screen(query, size=250, offset=offset, sortField="dayvolume", sortAsc=False)

        quotes, offset, total = [], 0, 0
        while True:
            page = screener(offset)
            if not isinstance(page, dict) or not isinstance(page.get("quotes"), list) or "total" not in page:
                raise RuntimeError("Sector quote source returned an invalid response")
            total = int(page["total"])
            batch = page["quotes"]
            if not batch:
                if offset < total:
                    raise RuntimeError("Sector quote source returned an incomplete response")
                break
            quotes.extend(batch)
            offset += len(batch)
            if offset >= total or offset >= 5000:
                break
        return {"symbol": symbol, "name": name, "stocks": normalize(quotes),
                "source": "Yahoo Finance", "fetched_at": datetime.now(timezone.utc).isoformat(),
                "total": total, "truncated": offset < total, "stale": False,
                "basis": "Latest regular-session price × share volume; estimated USD turnover. "
                         "US-listed industry peers, not ETF holdings. Quotes may be delayed."}
