"""Synthetic archive and HTTP fixtures; no network or trading results."""
import hashlib
import io
import json
import sys
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from scripts.research import binance_collect as bc


def zipped(content):
    out = io.BytesIO()
    with zipfile.ZipFile(out,"w") as z:
        z.writestr("data.csv",content)
    return out.getvalue()


def test_microseconds_scientific_prices_and_asof():
    a = int(pd.Timestamp("2025-01-01").timestamp()*1e6)
    b = int(pd.Timestamp("2025-01-02").timestamp()*1e6)
    rows = "\n".join(f"{t},1e-8,2e-8,1e-9,1e-8,3,0,4,1,0,0,0" for t in [a,b])
    out = bc.parse_archive(zipped(rows),"spot","SMALLUSDT","2025-01-02")
    assert len(out) == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2025-01-01")
    assert out.iloc[0]["open"] == pytest.approx(1e-8)


def test_midnight_funding_belongs_to_the_completed_day():
    t = int(pd.Timestamp("2025-01-02").timestamp()*1000)
    out = bc.parse_archive(zipped(f"calc_time,funding_interval_hours,last_funding_rate\n{t},8,0.0001\n"),
                           "funding","BTCUSDT","2025-01-02")
    assert len(out) == 1 and out.iloc[0]["rate"] == pytest.approx(.0001)


def test_cached_source_is_verified_without_network(tmp_path, monkeypatch):
    archive = bc.Archive(tmp_path)
    url = "https://data.binance.vision/test.zip"
    key = hashlib.sha256(url.encode()).hexdigest()
    data = archive.cache/(key+".bin")
    meta = archive.cache/(key+".json")
    data.write_bytes(b"verified")
    meta.write_text(json.dumps({"sha256":hashlib.sha256(b"verified").hexdigest()}))
    monkeypatch.setattr(requests,"get",lambda *a,**k: pytest.fail("must reuse source cache"))
    assert archive.get(url) == b"verified"
    data.write_bytes(b"changed")
    with pytest.raises(ValueError,match="hash changed"):
        archive.get(url)


def test_429_stops_without_unbounded_retries(tmp_path,monkeypatch):
    calls = []
    def rate_limited(*args,**kwargs):
        calls.append(1)
        return SimpleNamespace(status_code=429,headers={"Retry-After":"3600"})
    monkeypatch.setattr(requests,"get",rate_limited)
    with pytest.raises(RuntimeError,match="Retry-After=3600"):
        bc.Archive(tmp_path).get("https://data.binance.vision/test.zip")
    assert len(calls) == 1


def test_timeouts_are_bounded_to_two_attempts(tmp_path,monkeypatch):
    calls = []
    def timeout(*args,**kwargs):
        calls.append(kwargs["timeout"])
        raise requests.Timeout("fixture")
    monkeypatch.setattr(requests,"get",timeout)
    with pytest.raises(requests.Timeout):
        bc.Archive(tmp_path).get("https://data.binance.vision/test.zip")
    assert calls == [(3,6),(3,6)]


def test_restart_reuses_completed_symbols_and_explicitly_retries_blocked(tmp_path,monkeypatch):
    calls = []
    def listing(self,prefix,tags):
        return [prefix+"BTCUSDT/",prefix+"DEADUSDT/"]
    def worker(archive,kind,symbol,asof):
        calls.append(symbol)
        if symbol == "BTCUSDT" and calls.count(symbol) == 1:
            raise requests.Timeout("fixture")
        return {"status":"empty_before_asof","rows":0,"tail_missing":[]}
    monkeypatch.setattr(bc.Archive,"listing",listing)
    monkeypatch.setattr(bc,"one_symbol",worker)
    base = ["binance-collect",str(tmp_path),"--kind","spot","--asof","2026-11-04","--workers","1","--max-new","1"]
    monkeypatch.setattr(sys,"argv",base)
    bc.main()
    bc.main()
    assert calls == ["BTCUSDT","DEADUSDT"]
    bc.main()
    assert calls == ["BTCUSDT","DEADUSDT"]
    monkeypatch.setattr(sys,"argv",base+["--retry-blocked"])
    bc.main()
    assert calls == ["BTCUSDT","DEADUSDT","BTCUSDT"]
    assert json.loads((tmp_path/"collection_spot.json").read_text())["complete"]
