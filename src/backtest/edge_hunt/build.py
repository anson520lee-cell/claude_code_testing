"""python -m src.backtest.edge_hunt.build --prices prices.parquet --hist sp500_hist.csv --out signals.parquet"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

from src.backtest.edge_hunt.panel import load_prices, run_symbol, universe

NON_STOCK = {"SPY", "QQQ", "IWM", "DIA", "MDY", "RSP", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
             "XLRE", "XLC", "SMH", "XBI", "KRE", "XHB", "XRT", "IGV", "TLT", "IEF", "SHY", "HYG", "LQD", "GLD", "USO",
             "UUP", "SPHB", "SPLV", "MTUM", "QUAL"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    px = load_prices(a.prices)
    spy = px["SPY"]["adj_close"]
    cal = spy.index
    mem = universe(a.hist, px, cal)
    cfg = yaml.safe_load(open("config/terminal.yaml"))
    live = set(map(str, (cfg.get("universe") or {}).keys()))
    jobs = []
    for sym, x in px.items():
        if sym.startswith("^") or sym in NON_STOCK:
            continue
        m = mem.get(sym)
        m = pd.Series(m, index=cal).reindex(x.index).fillna(False).to_numpy(bool) if m is not None else None
        if m is None and sym not in live:
            continue
        jobs.append((sym, x, spy, m, sym in live))
    print("jobs", len(jobs), flush=True)
    out = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, df in enumerate(ex.map(run_symbol, jobs, chunksize=4)):
            if df is not None:
                out.append(df)
            if (i + 1) % 100 == 0:
                print(i + 1, round(time.time() - t0), flush=True)
    sig = pd.concat(out, ignore_index=True)
    sig.to_parquet(a.out)
    print("saved", len(sig), sig["setup"].value_counts().to_dict(), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
