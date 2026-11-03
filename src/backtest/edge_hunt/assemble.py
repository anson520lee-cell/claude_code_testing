"""python -m src.backtest.edge_hunt.assemble --raw <dir> --signals signals.parquet --out features.parquet"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.backtest.edge_hunt.panel import load_prices
from src.backtest.edge_hunt import sources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="finra,sec,market,events")
    a = ap.parse_args()
    t0 = time.time()
    raw = a.raw
    sig = pd.read_parquet(a.signals)
    sig = sig[sig["date"] >= "2008-06-01"].reset_index(drop=True)
    keys = sig[["symbol", "date"]].drop_duplicates()
    px = load_prices(os.path.join(raw, "prices.parquet"))
    cal = px["SPY"].index
    parts = {}
    only = set(a.only.split(","))
    if "finra" in only:
        sh = pd.read_parquet(os.path.join(raw, "finra_shvol.parquet"))
        parts["finra"] = sources.finra_features(sh, px, keys[keys["date"] >= "2009-08-01"])
        print("finra", len(parts["finra"]), round(time.time() - t0), flush=True)
    if "sec" in only:
        facts = pd.read_parquet(os.path.join(raw, "sec_facts.parquet"))
        splits = pd.read_parquet(os.path.join(raw, "splits.parquet"))
        parts["sec"] = sources.sec_features(facts, splits, px, keys)
        print("sec", len(parts["sec"]), round(time.time() - t0), flush=True)
    if "events" in only:
        fil = pd.read_parquet(os.path.join(raw, "sec_filings.parquet"))
        ep, rp = os.path.join(raw, "yahoo_earnings.parquet"), os.path.join(raw, "yahoo_ratings.parquet")
        earn = pd.read_parquet(ep) if os.path.exists(ep) else pd.DataFrame(columns=["symbol", "Earnings Date", "Reported EPS", "Surprise(%)"])
        rat = pd.read_parquet(rp) if os.path.exists(rp) else pd.DataFrame(columns=["symbol", "GradeDate", "Action", "Firm"])
        parts["events"] = sources.event_features(keys, cal, fil, earn, rat)
        print("events", len(parts["events"]), round(time.time() - t0), flush=True)
    out = sig
    for name, df in parts.items():
        if df is None or df.empty:
            continue
        df = df.drop_duplicates(["symbol", "date"])
        out = out.merge(df, on=["symbol", "date"], how="left")
    if "market" in only:
        m = sources.market_features(px, os.path.join(raw, "cboe_equitypc.csv"))
        out = out.merge(m, left_on="date", right_index=True, how="left")
    if os.path.exists(a.out) and a.only != "finra,sec,market,events":
        old = pd.read_parquet(a.out)
        new_cols = [c for c in out.columns if c not in sig.columns]
        old = old.drop(columns=[c for c in new_cols if c in old.columns])
        out = old.merge(out[["symbol", "date", "setup"] + new_cols].drop_duplicates(["symbol", "date", "setup"]),
                        on=["symbol", "date", "setup"], how="left")
    out.to_parquet(a.out)
    print("saved", out.shape, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
