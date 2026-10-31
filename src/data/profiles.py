"""Market capitalisation for the universe (Yahoo Finance quote data), cached on disk.

Market caps change slowly and only gate a coarse rule (>= US$2B), so they are refreshed every
``refresh_days`` days. A symbol whose cap cannot be fetched is reported as unknown (never guessed)
and is not excluded for that reason.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _yahoo_cap(symbol: str) -> float | None:
    import yfinance

    try:
        value = yfinance.Ticker(symbol).fast_info.get("marketCap")
        return float(value) if value else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("market cap %s unavailable: %s", symbol, exc)
        return None


def market_caps(symbols: list[str], cache_dir: Path, today: date, refresh_days: int = 7, offline: bool = False,
                fetch: Callable[[str], float | None] | None = None) -> dict[str, Any]:
    """{symbol: market cap in USD or None}. Uses the cache when it is fresh enough."""
    path = Path(cache_dir) / "market_caps.json"
    cache: dict[str, Any] = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text())
        except json.JSONDecodeError:
            cache = {}
    fresh = {s: v for s, v in cache.items()
             if isinstance(v, dict) and (today - date.fromisoformat(v["as_of"])).days <= refresh_days}
    missing = [s for s in symbols if s not in fresh]
    if missing and not offline:
        fetch = fetch or _yahoo_cap
        with ThreadPoolExecutor(max_workers=8) as pool:
            for symbol, value in zip(missing, pool.map(fetch, missing)):
                if value is not None:
                    cache[symbol] = fresh[symbol] = {"cap": value, "as_of": today.isoformat()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    return {s: (fresh.get(s) or cache.get(s) or {}).get("cap") for s in symbols}
