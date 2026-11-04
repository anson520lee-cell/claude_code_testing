"""python -m src.backtest.levscore.sp_grid RAW OUT_DIR : the point-in-time S&P 500 inputs that the x80 equity sleeves need,
without the edge-hunt zoo / confluence parts (which the sleeves never read).

Writes OUT_DIR/
  grid.parquet        every eligible stock-day 2009-06..: S&P 500 member that day, unadjusted close >= $5, >= 260 sessions,
                      with the price features of src.backtest.edge_hunt.grid.wide_features (same code, same definitions)
  grid_src.parquet    FINRA 5-day short-volume ratio (svr5) and SEC XBRL fundamentals (gpa, asset growth, issuance,
                      book/market, earnings/price, accruals) from src.backtest.edge_hunt.sources (same code)
  flow_flags.parquet  svr5_good = 5-day short-volume ratio in the bottom third of that day's grid (the live filter)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from src.backtest.edge_hunt import confluence as cf
from src.backtest.edge_hunt import sources
from src.backtest.edge_hunt.grid import START, wide_features
from src.backtest.edge_hunt.panel import load_prices, universe

KEEP = ("beta", "hv60", "ret1", "ret5", "ivol60", "max20", "mom12_1")


def build_grid(raw: str) -> tuple[pd.DataFrame, dict]:
    px = load_prices(os.path.join(raw, "prices.parquet"))
    cal = px["SPY"].index
    cal = cal[cal >= "2006-01-01"]
    mem = {s: m for s, m in universe(os.path.join(raw, "sp500_hist.csv"), px, cal).items() if s in px}
    p = cf.panel_from(px, mem, cal)
    first_valid = np.argmax(np.isfinite(p.C), axis=0)
    p.elig &= np.arange(p.D)[:, None] >= first_valid[None, :] + 260
    W = wide_features(p, px["SPY"]["adj_close"])
    E = p.elig.copy()
    E[: int(np.searchsorted(cal.values, np.datetime64(START)))] = False
    di, sj = np.nonzero(E)
    g = pd.DataFrame({"symbol": np.array(p.tickers)[sj], "date": cal[di]})
    for k in KEEP:
        g[k] = np.asarray(W[k], dtype=float)[di, sj].astype("float32")
    return g, px


def main() -> None:
    raw, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    g, px = build_grid(raw)
    g.to_parquet(os.path.join(out, "grid.parquet"))
    print("grid", g.shape, g["symbol"].nunique(), "symbols", g["date"].min().date(), g["date"].max().date(), round(time.time() - t0), flush=True)
    keys = g[["symbol", "date"]]
    fin = sources.finra_features(pd.read_parquet(os.path.join(raw, "finra_shvol.parquet")), px, keys[keys["date"] >= "2009-08-01"])
    print("finra", fin.shape, round(time.time() - t0), flush=True)
    sec = sources.sec_features(pd.read_parquet(os.path.join(raw, "sec_facts.parquet")), pd.read_parquet(os.path.join(raw, "splits.parquet")), px, keys)
    print("sec", sec.shape, round(time.time() - t0), flush=True)
    src = keys.merge(fin[["symbol", "date", "svr5"]].drop_duplicates(["symbol", "date"]), on=["symbol", "date"], how="left")
    cols = [c for c in ("gpa", "asset_growth", "issuance", "bm", "ep", "accruals") if c in sec.columns]
    src = src.merge(sec[["symbol", "date"] + cols].drop_duplicates(["symbol", "date"]), on=["symbol", "date"], how="left")
    src.to_parquet(os.path.join(out, "grid_src.parquet"))
    pct = src.groupby("date")["svr5"].rank(pct=True)
    flags = src[["symbol", "date"]].assign(svr5_pct=pct, svr5_good=(pct <= 1 / 3).astype(float).where(pct.notna()))
    flags.to_parquet(os.path.join(out, "flow_flags.parquet"))
    print("grid_src", src.shape, "coverage", src[["svr5"] + cols].notna().mean().round(3).to_dict(), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
