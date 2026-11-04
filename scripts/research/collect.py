"""Resumable public-data collection in an isolated research directory. Never touches the live terminal."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = ["SPY", "QQQ", "^GSPC", "^NDX", "^IXIC", "^IRX", "^VIX", "^VIX3M", "TLT", "IEF", "GLD", "SHY",
        "TQQQ", "UPRO", "SSO", "QLD", "SOXL", "TECL", "SVXY", "VXX", "UVXY", "TMF",
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "LTC-USD", "ADA-USD"]
EXTRA = ["SQQQ", "SPXU", "EFA", "EEM", "IWM", "DBC", "UUP", "VNQ", "HYG", "LQD", "SLV", "USO", "FXE", "FXY",
         "TIP", "EWJ", "DIA", "XLE", "XLK", "XLF"]
CBOE = ["PUT", "BXM", "BXMD", "BFLY", "CNDR", "CLL", "PPUT", "WPUT", "PUTY", "BXMW"]


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def freeze_index_history(content, asof):
    frame = pd.read_csv(io.BytesIO(content))
    dates = pd.to_datetime(frame.iloc[:, 0], format="mixed", errors="raise")
    frame = frame.loc[dates < pd.Timestamp(asof)]
    if len(frame) < 252 or len(frame.columns) < 2:
        raise ValueError("Invalid or insufficient index history")
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--asof", required=True, help="Exclusive end date; use one frozen date for a study")
    args = p.parse_args()
    import yfinance as yf
    root = args.root.resolve()
    for sub in ("raw/yahoo", "x80/raw/cboe", "x80/raw/deribit", "x80/out", "opt/raw"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    manifest = root / "collection.json"
    state = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {"asof_exclusive": args.asof, "files": {}}
    if state["asof_exclusive"] != args.asof:
        raise ValueError("Use a separate data root for another as-of date; do not silently rewrite a study")

    def record(key, path, source, error=None):
        row = {"source": source, "checked_utc": datetime.now(timezone.utc).isoformat(), "ok": error is None}
        if error is None:
            row.update(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            row["error"] = str(error)
        state["files"][key] = row
        atomic_json(manifest, state)
        print(key, "OK" if error is None else str(error), flush=True)

    for sym in BASE + EXTRA:
        name = sym.replace("^", "INDEX_").replace("/", "_")
        dest = root / "raw/yahoo" / (name + ".parquet")
        key, source = "yahoo:" + sym, "Yahoo Finance via yfinance: " + sym
        if dest.exists() and state["files"].get(key, {}).get("ok"):
            if hashlib.sha256(dest.read_bytes()).hexdigest() != state["files"][key]["sha256"]:
                raise ValueError("Cached data hash changed: " + str(dest))
            continue
        try:
            d = yf.download(sym, start="1985-01-01", end=args.asof, auto_adjust=False, progress=False, threads=False)
            if d is None or d.empty:
                raise ValueError("No data returned; no substitute or synthetic prices used")
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
            d.index = pd.to_datetime(d.index).tz_localize(None)
            d = d.loc[d.index < args.asof]
            d["symbol"] = sym
            d = d.reset_index().rename(columns={"Date": "date", "index": "date"})
            temp = dest.with_suffix(".tmp")
            d.to_parquet(temp, index=False)
            temp.replace(dest)
            record(key, dest, source)
        except Exception as exc:
            record(key, dest, source, exc)

    for names, filename in ((BASE, "prices.parquet"), (EXTRA, "prices_extra.parquet")):
        keys = ["yahoo:" + sym for sym in names]
        missing = [k for k in keys if not state["files"].get(k, {}).get("ok")]
        if missing:
            print(filename, "NOT BUILT: missing", missing, flush=True)
            continue
        panels = [pd.read_parquet(root / state["files"][k]["path"]) for k in keys]
        dest = root / "x80/raw" / filename
        temp = dest.with_suffix(".tmp")
        pd.concat(panels, ignore_index=True).to_parquet(temp, index=False)
        temp.replace(dest)
        record(filename, dest, "Per-symbol Yahoo shards; complete requested symbol set")

    for sym in CBOE:
        key, dest = "cboe:" + sym, root / "x80/raw/cboe" / (sym + ".csv")
        url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
        if dest.exists() and state["files"].get(key, {}).get("ok"):
            if hashlib.sha256(dest.read_bytes()).hexdigest() != state["files"][key]["sha256"]:
                raise ValueError("Cached data hash changed: " + str(dest))
            continue
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            frame = freeze_index_history(response.content, args.asof)
            temp = dest.with_suffix(".tmp")
            frame.to_csv(temp, index=False)
            temp.replace(dest)
            record(key, dest, url)
        except Exception as exc:
            record(key, dest, url, exc)
    print("Collection manifest:", manifest, flush=True)
    print("Option indexes are benchmark histories, not executable option quotes or net retail returns.", flush=True)


if __name__ == "__main__":
    main()
