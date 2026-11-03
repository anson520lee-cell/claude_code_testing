"""python -m src.backtest.edge_hunt.replicate --rep REP --raw REP/raw --out REP/out

Out-of-universe replication of the edge-hunt leads (pre-registration: docs/EDGE_HUNT.md, section "Replication").
REP holds grid.parquet / zoo_rows.parquet / sig_fwd.parquet (edge_hunt.grid run with --hist ALL:5e6) and
signals.parquet (edge_hunt.build, same universe). Every hypothesis is one-sided in the discovery direction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.backtest.edge_hunt import gate2, sources
from src.backtest.edge_hunt.gate import cluster_t
from src.backtest.edge_hunt.panel import load_prices

PERIOD = ("2010-01-01", "2026-10-31")
SPLITS = {"2010-17": ("2010-01-01", "2017-12-31"), "2018-26": ("2018-01-01", "2026-10-31")}
ENS_HI = 1.8216  # S&P 500 in-sample (2010-2017) 75th percentile of ens5 > 0, fixed before replication
SHORTS = ("breakdown", "rally_fail", "ema21_loss", "s28") + tuple(f"x{i:02d}" for i in range(1, 10))
MAIN = [  # id, strategy (or "SHORTS"), feature, kind
    ("H1", "standalone_long", "svr5", "bottom"),
    ("H2", "standalone_long", "offex", "bottom"),
    ("H3", "standalone_long", "flow", "bottom"),
    ("H4", "breakout", "ens_hi", "true"),
    ("H5", "L10", "offex", "bottom"),
    ("H6", "crash_rebound", "svr5", "bottom"),
    ("H7", "breakdown", "mom12_1", "bottom"),
    ("H8", "standalone_short", "offex", "top"),
]
ADDENDUM = [("A1", "SHORTS", "ivol60", "top")]
EDGE_TESTS = [("A2", "x02"), ("A3", "L26"), ("A4", "L19")]


def mean_t(y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """mean and month-clustered t of y."""
    if len(y) < 10:
        return math.nan, math.nan
    e = y - y.mean()
    s = pd.Series(e).groupby(groups).sum().to_numpy()
    g = len(s)
    se = math.sqrt((s ** 2).sum() * g / max(g - 1, 1)) / len(y)
    return float(y.mean()), float(y.mean() / se) if se > 0 else math.nan


def build(rep: str, raw: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    g = pd.read_parquet(os.path.join(rep, "grid.parquet"), columns=["symbol", "date", "beta", "hv60", "mom12_1", "ivol60", "ens5"])
    px = load_prices(os.path.join(raw, "prices.parquet"))
    fin = sources.finra_features(pd.read_parquet(os.path.join(raw, "finra_shvol.parquet")), px, g.loc[g["date"] >= "2009-08-01", ["symbol", "date"]])
    print("finra", fin.shape, round(time.time() - t0), flush=True)
    g = g.merge(fin[["symbol", "date", "svr5", "offex"]].drop_duplicates(["symbol", "date"]), on=["symbol", "date"], how="left")
    for f in ("svr5", "offex", "mom12_1", "ivol60"):
        g[f"pct_{f}"] = g.groupby("date")[f].rank(pct=True)
    g["flow_raw"] = g[["pct_svr5", "pct_offex"]].mean(axis=1, skipna=False)
    g["pct_flow"] = g.groupby("date")["flow_raw"].rank(pct=True)
    for f in ("svr5", "offex", "flow", "mom12_1", "ivol60"):
        p = g[f"pct_{f}"]
        g[f"cond__{f}__bottom"] = (p <= 1 / 3).astype(float).where(p.notna())
        g[f"cond__{f}__top"] = (p >= 2 / 3).astype(float).where(p.notna())
        g[f"q5__{f}"] = np.ceil(p * 5).clip(1, 5).where(p.notna())
    g["cond__ens_hi__true"] = (g["ens5"] >= ENS_HI).astype(float).where(g["ens5"].notna())
    R = gate2.build_rows(rep, os.path.join(rep, "signals.parquet"))
    keep = ["symbol", "date", "beta", "hv60"] + [c for c in g.columns if c.startswith(("cond__", "q5__"))]
    R = R.merge(g[keep], on=["symbol", "date"], how="left")
    R = R[(R["date"] >= PERIOD[0]) & (R["date"] <= PERIOD[1])].copy()
    R["month"] = R["date"].dt.to_period("M").astype(str)
    R["live_like"] = ((R["beta"] >= 1.3) & (R["hv60"] >= 0.35)).astype(float)
    print("rows", R.shape, round(time.time() - t0), flush=True)
    return g, R


def cond_test(x: pd.DataFrame, col: str) -> dict:
    x = x[x[col].notna()]
    c = x[col].to_numpy(float)
    d, t, n1, n0 = cluster_t(x["y"].to_numpy(float), c, x["month"].to_numpy())
    de, te, _, _ = cluster_t(x["exc"].to_numpy(float), c, x["month"].to_numpy())
    sel = x[c == 1]
    return {"d": d, "t": t, "n1": n1, "n0": n0, "de": de, "te": te, "r_c": float(sel["y"].mean()), "r_rest": float(x.loc[c == 0, "y"].mean()),
            "edge_c": float(sel["exc"].mean())}


def run(R: pd.DataFrame) -> dict:
    out = {"main": [], "addendum": [], "edge": [], "desc": []}
    z_main, z_add = norm.ppf(1 - 0.05 / len(MAIN)), norm.ppf(1 - 0.05 / (len(ADDENDUM) + len(EDGE_TESTS)))
    for fam, items, z in (("main", MAIN, z_main), ("addendum", ADDENDUM, z_add)):
        for hid, strat, f, kind in items:
            x = R[R["strategy"].isin(SHORTS)] if strat == "SHORTS" else R[R["strategy"] == strat]
            col = f"cond__{f}__{kind}"
            row = {"id": hid, "strategy": strat, "feature": f, "kind": kind, "z_needed": z, **cond_test(x, col)}
            for lab, (a, b) in SPLITS.items():
                s = cond_test(x[(x["date"] >= a) & (x["date"] <= b)], col)
                row[f"{lab}_d"], row[f"{lab}_t"] = s["d"], s["t"]
            ll = x[x["live_like"] >= 0.5]
            if len(ll) > 60:
                s = cond_test(ll, col)
                row["ll_d"], row["ll_t"] = s["d"], s["t"]
            row["pass"] = bool(row["t"] >= z and row["te"] >= 1.0)
            out[fam].append(row)
    for hid, strat in EDGE_TESTS:
        x = R[R["strategy"] == strat]
        m, t = mean_t(x["exc"].to_numpy(float), x["month"].to_numpy())
        row = {"id": hid, "strategy": strat, "n": len(x), "edge": m, "t": t, "r": float(x["y"].mean()), "z_needed": z_add,
               "pass": bool(t >= z_add)}
        for lab, (a, b) in SPLITS.items():
            xs = x[(x["date"] >= a) & (x["date"] <= b)]
            row[f"{lab}_edge"], row[f"{lab}_t"] = mean_t(xs["exc"].to_numpy(float), xs["month"].to_numpy())
        out["edge"].append(row)
    # descriptive: quintiles of the flow features for standalone long / short (mean R and excess per quintile)
    for strat in ("standalone_long", "standalone_short"):
        x = R[R["strategy"] == strat]
        for f in ("svr5", "offex", "flow"):
            q = x.groupby(f"q5__{f}").agg(n=("y", "size"), r=("y", "mean"), exc=("exc", "mean"), fx20=("fx20", "mean"))
            for k, v in q.iterrows():
                out["desc"].append({"strategy": strat, "feature": f, "q": int(k), **{a: float(b) for a, b in v.items()}})
    # strategy-level edge of every strategy in this universe (for the "why it fails" table)
    lev = []
    for (strat, side), x in R.groupby(["strategy", "side"]):
        m, t = mean_t(x["exc"].to_numpy(float), x["month"].to_numpy())
        lev.append({"strategy": strat, "side": int(side), "n": len(x), "r": float(x["y"].mean()), "edge": m, "t": t})
    out["levels"] = lev
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _, R = build(a.rep, a.raw)
    res = run(R)
    for k in ("main", "addendum", "edge", "desc", "levels"):
        pd.DataFrame(res[k]).to_csv(os.path.join(a.out, f"{k}.csv"), index=False)
    R.to_parquet(os.path.join(a.out, "rows.parquet"))
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for k in ("main", "addendum", "edge"):
            print(pd.DataFrame(res[k]).round(3).to_string(), flush=True)
    json.dump({k: res[k] for k in ("main", "addendum", "edge")}, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
