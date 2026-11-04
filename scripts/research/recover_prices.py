"""Recover ONLY public historical OHLCV from a terminal cache, without reading account or alert tables."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    source, out = args.database.resolve(), args.out.resolve()
    wal = Path(str(source) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Active WAL exists; use a consistent SQLite backup first")
    source_hash = digest(source)
    if (out / "manifest.json").exists():
        old = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if old["source_sha256"] != source_hash:
            raise ValueError("Source changed; use a new output directory to preserve the earlier snapshot")
        if any(digest(out / name) != sha for name, sha in old["files"].items()):
            raise ValueError("Recovered snapshot hash changed")
        print("Existing verified snapshot retained; no files rewritten", flush=True)
        return
    with sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        data = pd.read_sql_query("SELECT ticker AS symbol, date, open, high, low, close, adj_close, volume, source "
                                 "FROM price_data ORDER BY ticker,date", connection)
    if digest(source) != source_hash:
        raise ValueError("Database changed during recovery; no snapshot published")
    data["date"] = pd.to_datetime(data["date"].str[:10], format="%Y-%m-%d")
    if data.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate ticker/date keys")
    out.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out / "prices.parquet", index=False)
    inventory = data.groupby("symbol").agg(rows=("date", "size"), first=("date", "min"), last=("date", "max"))
    inventory.to_csv(out / "inventory.csv")
    meta = {"source_table": "price_data only", "source_sha256": source_hash,
            "rows": len(data), "symbols": len(inventory), "first": str(data.date.min().date()),
            "last": str(data.date.max().date()), "point_in_time_membership": False,
            "known_limitations": ["Current terminal universe; survivorship bias remains", "Does not reconstruct the original 30-sleeve research pool"],
            "files": {name: digest(out / name) for name in ("prices.parquet", "inventory.csv")}}
    temp = out / "manifest.json.tmp"
    temp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    temp.replace(out / "manifest.json")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
