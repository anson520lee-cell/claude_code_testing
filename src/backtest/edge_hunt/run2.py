"""python -m src.backtest.edge_hunt.run2 --dir V2 --signals signals_v2.parquet --out reports/edge_hunt"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.backtest.edge_hunt import gate2
from src.backtest.edge_hunt.gate import IS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prices", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    g = gate2.build_grid(a.dir)
    print("grid", g.shape, round(time.time() - t0), flush=True)
    R = gate2.build_rows(a.dir, a.signals)
    print("rows", R.shape, R["strategy"].nunique(), round(time.time() - t0), flush=True)
    crash = R[R["strategy"] == "crash_rebound"][["symbol", "date"]].merge(g[["symbol", "date", "sys_share", "on_share5", "speed"]], on=["symbol", "date"], how="left")
    conds = gate2.conditions(g, crash)
    keep = ["symbol", "date", "beta", "hv60"] + conds
    R = R.merge(g[keep], on=["symbol", "date"], how="left")
    R["live_like"] = ((R["beta"] >= 1.3) & (R["hv60"] >= 0.35)).astype(float)
    # daily regime labels for the diagnostics (VIX terciles from the in-sample years, SPY vs its 200-day average)
    px = pd.read_parquet(a.prices, filters=[("symbol", "in", ["SPY", "^VIX"])])
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["symbol"] == "SPY"].set_index("date")["adj_close"].sort_index()
    vix = px[px["symbol"] == "^VIX"].set_index("date")["close"].sort_index()
    vis = vix[(vix.index >= IS[0]) & (vix.index <= IS[1])]
    lo, hi = vis.quantile(1 / 3), vis.quantile(2 / 3)
    daily = pd.DataFrame({"vix_t": np.where(vix >= hi, 2, np.where(vix <= lo, 0, 1)).astype(float)}, index=vix.index)
    daily["spy_bull"] = (spy > spy.rolling(200).mean()).astype(float).reindex(daily.index)
    diag = gate2.diagnostics(R, daily)
    diag.to_csv(os.path.join(a.out, "diagnostics.csv"), index=False)
    print("diagnostics", round(time.time() - t0), flush=True)
    res = gate2.run_tests(R, conds)
    res.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    R[["strategy", "side", "tier", "symbol", "date", "y", "exc", "live_like"] + conds].to_parquet(os.path.join(a.out, "rows.parquet"))
    summ = {}
    for tier in (1, 2):
        m = res["tier"] == tier
        summ[f"tier{tier}"] = {"n_tests": int(m.sum()), "is_pass": int((m & res["is_pass"]).sum()),
                               "z_needed": float(res.loc[m, "z_needed"].iloc[0]), "oos_pass": int((m & res["oos_pass"]).sum()),
                               "pass": int((m & res["pass"]).sum())}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
