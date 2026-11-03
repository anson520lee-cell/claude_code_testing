"""python -m src.backtest.edge_hunt.conf_build --prices prices.parquet --hist sp500_hist.csv --signals signals.parquet --out confluence.parquet"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.backtest.edge_hunt import confluence as cf
from src.backtest.edge_hunt.gate import IS, cluster_t
from src.backtest.edge_hunt.panel import EXITS, load_prices, membership, trade
from src.backtest.swing42 import core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chosen", default="reports/swing42/results_A.json")
    a = ap.parse_args()
    t0 = time.time()
    px = load_prices(a.prices)
    cal = px["SPY"].index
    cal = cal[cal >= "2006-01-01"]
    mem = {s: m for s, m in membership(a.hist, cal).items() if s in px}
    p = cf.panel_from(px, mem, cal)
    print("panel", p.D, p.S, round(time.time() - t0), flush=True)
    sig = cf.signals(p, cf.chosen_params(a.chosen))
    print("signals", round(time.time() - t0), flush=True)
    # in-sample weights: each strategy's edge over same-day random entries with the live crash exit, 2010-2017
    rows = pd.read_parquet(a.signals, columns=["symbol", "date", "setup", "member", "r_crash"])
    rnd = rows[(rows["setup"] == "random") & rows["member"]]
    rd = rnd.groupby("date")["r_crash"].mean()
    A = core.atr(p, 14)
    is_rows = np.flatnonzero((cal >= IS[0]) & (cal <= IS[1]))
    weights, wt_detail = {}, {}
    for sid, s in sig.items():
        dd, jj = np.nonzero(s[is_rows])
        dd = is_rows[dd]
        rs, ds = [], []
        for d, j in zip(dd, jj):
            r, _, _ = trade(p.O[:, j], p.H[:, j], p.L[:, j], p.C[:, j], d, p.C[d, j], A[d, j], *EXITS["crash"])
            if np.isfinite(r):
                rs.append(r); ds.append(cal[d])
        if len(rs) < 30:
            weights[sid] = 0.0
            continue
        x = pd.DataFrame({"date": ds, "r": rs})
        x["exc"] = x["r"] - x["date"].map(rd)
        x = x.dropna()
        g = x["date"].dt.to_period("M").astype(str).to_numpy()
        # t of the mean excess, clustered by month (regress on a constant)
        m = x.groupby(g)["exc"].sum()
        n = len(x)
        mean = x["exc"].mean()
        resid = (x["exc"] - mean).groupby(g).sum()
        se = np.sqrt((resid ** 2).sum()) / n
        t = float(mean / se) if se > 0 else 0.0
        weights[sid] = max(t, 0.0)
        wt_detail[sid] = {"n": int(n), "edge": float(mean), "t": t}
        print("  weight", sid, wt_detail[sid], flush=True)
    feats = cf.features(p, sig, weights)
    keys = rows[["symbol", "date"]].drop_duplicates()
    di = np.searchsorted(cal.values, keys["date"].values.astype("datetime64[ns]"))
    sj = {s: j for j, s in enumerate(p.tickers)}
    jj = keys["symbol"].map(sj)
    ok = jj.notna().to_numpy() & (di < len(cal))
    out = keys[ok].copy()
    di, jj = di[ok], jj[ok].astype(int).to_numpy()
    for k, v in feats.items():
        out[k] = np.asarray(v)[di, jj].astype(float)
    out.to_parquet(a.out)
    json.dump({"weights": weights, "detail": wt_detail}, open(a.out.replace(".parquet", "_weights.json"), "w"), indent=1)
    print("saved", out.shape, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
