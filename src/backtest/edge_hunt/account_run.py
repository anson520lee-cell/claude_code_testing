"""python -m src.backtest.edge_hunt.account_run --signals signals_v2.parquet --prices prices.parquet --out OUT [--flags rows.parquet]

Account-level check of the live long rules on point-in-time S&P 500 members, 2010-01 .. 2026-10, next to SPY:
10 slots, 1% risk, 25% cap, priority crash > squeeze > breakout, 5 seeds for the random tie-break.
Variants: every member / live-like members only (beta >= 1.3, hv60 >= 0.35) and, with --flags, the same rules with
a flag tie-break (same-day flagged candidates first inside their setup = what a score bonus does) and a flag filter.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.backtest.edge_hunt.account import simulate, stats

START, END = "2010-01-01", "2026-10-31"
PERIODS = {"2010-2026": (START, END), "2010-2017": (START, "2017-12-31"), "2018-2026": ("2018-01-01", END)}


def run_variant(t: pd.DataFrame, cal: pd.DatetimeIndex, seeds: int = 5) -> tuple[pd.Series, dict]:
    curves = [simulate(t, cal, seed=s) for s in range(seeds)]
    mean_curve = pd.concat(curves, axis=1).mean(axis=1)
    st = {p: {k: float(np.mean([stats(c, a, b)[k] for c in curves])) for k in ("cagr", "max_dd", "sharpe")} for p, (a, b) in PERIODS.items()}
    st["trades_per_year"] = float(len(t) / ((pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25))
    return mean_curve, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flags", default="")
    ap.add_argument("--flag-col", default="svr5_good")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = pd.read_parquet(a.signals, columns=["symbol", "date", "setup", "member", "live_like", "r", "ret", "days"])
    s = s[s["member"] & s["setup"].isin(["crash_rebound", "squeeze", "breakout"]) & (s["date"] >= START) & (s["date"] <= END)]
    s = s.dropna(subset=["r", "ret", "days"])
    px = pd.read_parquet(a.prices, filters=[("symbol", "==", "SPY")])
    px["date"] = pd.to_datetime(px["date"])
    spy = px.set_index("date")["adj_close"].sort_index()
    cal = spy.index[(spy.index >= START) & (spy.index <= END)]
    variants = {"live_all_members": s, "live_like_only": s[s["live_like"]]}
    if a.flags:
        fl = pd.read_parquet(a.flags)
        s2 = s.merge(fl[["symbol", "date", a.flag_col]], on=["symbol", "date"], how="left")
        good = s2[a.flag_col] == 1
        s2["prio_adj"] = np.where(good, 0.0, 0.5)     # flagged candidates first inside their setup (= a score bonus)
        variants["live_all_members_bonus"] = s2
        variants["live_like_bonus"] = s2[s2["live_like"]]
        variants["live_all_members_filter"] = s2[good]
        variants["live_like_filter"] = s2[good & s2["live_like"]]
    out, curves = {}, {}
    for name, t in variants.items():
        c, st = run_variant(t, cal)
        out[name], curves[name] = st, c
    spy_c = spy.reindex(cal)
    out["SPY"] = {p: stats(spy_c, a_, b_) for p, (a_, b_) in PERIODS.items()}
    curves["SPY"] = spy_c / spy_c.iloc[0]
    pd.DataFrame(curves).to_parquet(os.path.join(a.out, "account_curves.parquet"))
    json.dump(out, open(os.path.join(a.out, "account.json"), "w"), indent=1)
    for k, v in out.items():
        print(k, {p: {m: round(x, 3) for m, x in d.items()} for p, d in v.items() if isinstance(d, dict)}, v.get("trades_per_year", ""))


if __name__ == "__main__":
    main()
