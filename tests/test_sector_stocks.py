"""Sector browsing: turnover ranking, missing data, pagination and API failures."""
import copy
import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.daily.settings import load_terminal_settings
from src.web.app import create_app
from src.web.sector_stocks import GROUPS, SectorStocks, normalize


def quote(symbol, price=10, volume=100, **kw):
    return {"symbol": symbol, "regularMarketPrice": price, "regularMarketVolume": volume,
            "quoteType": "EQUITY", "regularMarketChangePercent": 2.5, **kw}


def test_turnover_is_not_share_volume_and_missing_is_last():
    rows = normalize([quote("AMD", 1, 1000), quote("NVDA", 100, 100),
                      quote("MISSING", volume=None), quote("ZERO", volume=0),
                      quote("BAD", price=float("inf")), quote("ETF", quoteType="ETF"),
                      quote("FOREIGN", currency="EUR"), quote("NVDA", 1, 1)])
    assert [r["symbol"] for r in rows] == ["NVDA", "AMD", "ZERO", "BAD", "MISSING"]
    assert rows[0]["turnover"] == 10000
    assert rows[0]["change_pct"] == .025
    assert rows[-1]["turnover"] is None
    assert rows[2]["turnover"] == 0
    assert [r["rank"] for r in rows] == list(range(1, 6))


def test_pagination_ranks_all_pages_and_cache_avoids_repeated_calls():
    calls = []
    def screen(offset):
        calls.append(offset)
        return {"total": 2, "quotes": [quote("AAA", 10)] if offset == 0 else [quote("BBB", 100)]}
    cache = SectorStocks()
    answer = cache.get("smh", screen)
    assert calls == [0, 1]
    assert [r["symbol"] for r in answer["stocks"]] == ["BBB", "AAA"]
    assert cache.get("SMH", screen) == answer
    assert calls == [0, 1]
    assert not answer["truncated"]


def test_failed_refresh_keeps_original_timestamp_and_marks_stale():
    cache = SectorStocks(ttl=0)
    first = cache.get("SMH", lambda _: {"total": 1, "quotes": [quote("NVDA")]})
    def failed(_):
        raise RuntimeError("source down")
    stale = cache.get("SMH", failed)
    assert stale["stale"]
    assert stale["fetched_at"] == first["fetched_at"]
    assert stale["stocks"] == first["stocks"]
    with pytest.raises(RuntimeError):
        SectorStocks().get("SMH", failed)


@pytest.mark.parametrize("response", [{}, {"total": 3, "quotes": []}, None])
def test_bad_or_incomplete_response_is_not_a_successful_empty_list(response):
    with pytest.raises(RuntimeError):
        SectorStocks().get("SMH", lambda _: response)


def test_empty_sector_and_unknown_sector():
    assert SectorStocks().get("XLP", lambda _: {"total": 0, "quotes": []})["stocks"] == []
    with pytest.raises(ValueError):
        SectorStocks().get("BOGUS")
    assert len(GROUPS) == 15


def test_all_sector_queries_match_yahoo_supported_values():
    from yfinance import EquityQuery as Q
    for _, field, values in GROUPS.values():
        group = [Q("eq", [field, value]) for value in values]
        Q("and", [Q("eq", ["region", "us"]), group[0] if len(group) == 1 else Q("or", group)])


def test_api_has_chart_links_only_when_history_is_available(tmp_path, monkeypatch):
    settings = copy.deepcopy(load_terminal_settings())
    for key in ("database", "reports_dir", "cache_dir", "backups_dir"):
        settings["paths"][key] = str(tmp_path / key)
    with sqlite3.connect(settings["paths"]["database"]) as conn:
        conn.execute("CREATE TABLE price_data (ticker TEXT, date TEXT)")
        conn.executemany("INSERT INTO price_data VALUES (?, ?)", [("NVDA", str(i)) for i in range(60)])
    monkeypatch.setattr(SectorStocks, "get", lambda self, symbol: {
        "stocks": normalize([quote("NVDA"), quote("AMD")]), "symbol": symbol})
    client = TestClient(create_app(settings))
    result = client.get("/api/sector-stocks/smh")
    assert result.status_code == 200
    assert {r["symbol"]: r["chart_available"] for r in result.json()["stocks"]} == {"NVDA": True, "AMD": False}
    assert client.get("/api/sector-stocks/INVALID").status_code == 404
    def failed(*_):
        raise RuntimeError("private provider failure")
    monkeypatch.setattr(SectorStocks, "get", failed)
    response = client.get("/api/sector-stocks/SMH")
    assert response.status_code == 503
    assert "private provider" not in response.text
