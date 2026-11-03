"""python -m src.backtest.edge_hunt.grid_sources --raw RAW --dir V2 : FINRA / SEC / events / market features for
every grid stock-day (written to V2/grid_src.parquet)."""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from src.backtest.edge_hunt import sources
from src.backtest.edge_hunt.panel import load_prices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    t0 = time.time()
    raw = a.raw
    keys = pd.read_parquet(os.path.join(a.dir, "grid.parquet"), columns=["symbol", "date"])
    px = load_prices(os.path.join(raw, "prices.parquet"))
    cal = px["SPY"].index
    fin = sources.finra_features(pd.read_parquet(os.path.join(raw, "finra_shvol.parquet")), px, keys[keys["date"] >= "2009-08-01"])
    print("finra", fin.shape, round(time.time() - t0), flush=True)
    sec = sources.sec_features(pd.read_parquet(os.path.join(raw, "sec_facts.parquet")), pd.read_parquet(os.path.join(raw, "splits.parquet")), px, keys)
    print("sec", sec.shape, round(time.time() - t0), flush=True)
    ev = sources.event_features(keys, cal, pd.read_parquet(os.path.join(raw, "sec_filings.parquet")),
                                pd.read_parquet(os.path.join(raw, "yahoo_earnings.parquet")), pd.read_parquet(os.path.join(raw, "yahoo_ratings.parquet")))
    print("events", ev.shape, round(time.time() - t0), flush=True)
    out = keys
    for df in (fin, sec, ev):
        out = out.merge(df.drop_duplicates(["symbol", "date"]), on=["symbol", "date"], how="left")
    m = sources.market_features(px, os.path.join(raw, "cboe_equitypc.csv"))
    out = out.merge(m, left_on="date", right_index=True, how="left")
    out["close"] = [None] * len(out)
    close = pd.concat({s: x["close"] for s, x in px.items() if not s.startswith("^")}, axis=1)
    cs = close.stack().rename("close").reset_index().rename(columns={"level_0": "date", "level_1": "symbol"})
    out = out.drop(columns=["close"]).merge(cs, on=["date", "symbol"], how="left")
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    out.to_parquet(os.path.join(a.dir, "grid_src.parquet"))
    print("saved", out.shape, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
