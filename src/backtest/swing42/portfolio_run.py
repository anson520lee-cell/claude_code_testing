"""Portfolio check of the shortlisted strategies (python -m src.backtest.swing42.portfolio_run ...same args as run).

For every strategy whose verdict (vs random, or vs same-day random) is 有效 / 邊緣, plus the live crash rebound as
reference: rebuild the chosen cell's signals, run a 10-slot account (portfolio.simulate), report CAGR / Sharpe /
max drawdown in and out of sample, beta and alpha against SPY, then blend the streams with in-sample
inverse-variance weights (portfolio.combine)."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import data, events, portfolio
from .core import IS_END, fresh, shift
from .run import CATALOGUE, clean, make_exits

KEEP = ("有效", "有效(強)", "邊緣")


def _alpha(m: pd.Series, spy: pd.Series) -> dict:
    out = {}
    for name, mask in (("IS", m.index <= IS_END), ("OOS", m.index > IS_END)):
        x, y = m[mask], spy.reindex(m.index)[mask]
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        if len(x) < 12:
            continue
        b = float(np.cov(x, y)[0, 1] / np.var(y))
        res = x - b * y
        out[name] = {"beta": b, "alpha_ann": float(res.mean() * 12),
                     "alpha_t": float(res.mean() / (res.std() / np.sqrt(len(res))))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--earn", default="")
    ap.add_argument("--cache", default="panel.pkl")
    ap.add_argument("--out", default="reports/swing42")
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--universe", default="A")
    a = ap.parse_args()
    d = data.build(a.px, a.hist, a.cache)
    A, X = d["A"], d["X"]
    if a.universe == "B":
        import yaml
        from .core import Panel
        ub = set(yaml.safe_load(open("config/universe_auto.default.yaml"))["symbols"])
        inb = np.array([t in ub for t in A.tickers])
        A = Panel(A.dates, A.tickers, A.O, A.H, A.L, A.C, A.V, inb[None, :] & np.isfinite(A.O) & (np.nan_to_num(A.C) >= 5))
    res = json.load(open(os.path.join(a.out, "results.json")))
    get = make_exits(A)
    spy_m = X.df("C")["SPY"].resample("ME").last().pct_change()
    streams, report = {}, {}
    cat = {str(c[0]): c for c in CATALOGUE}
    for sid, r in res.items():
        if sid.startswith("_") or "chosen" not in r:
            continue
        if not (r.get("verdict") in KEEP or r.get("verdict_dm") in KEEP or sid == "0"):
            continue
        prm, ek = r["chosen"]["params"], r["chosen"]["exit"]
        if sid in cat:
            _, zh, fn, _, _, side, cd, _ = cat[sid]
            sig = fn(A, None, **prm)
            sig = fresh(np.asarray(sig, bool), cd) if cd > 1 else np.asarray(sig, bool)
        elif sid in ("30b", "37"):
            ev = events.load_earnings(a.earn, A.tickers)
            j, react, last_pre, ok = events._earn_rows(A, ev)
            sig = np.zeros((A.D, A.S), bool)
            if sid == "30b":
                ret = A.C / shift(A.C, 1) - 1
                m = ok & (react < A.D - 1) & (ev["surprise"].to_numpy() > prm["surprise_min_%"])
                rr, jj = react[m], j[m]
                keep = ret[rr, jj] > prm["reaction_min"]
                sig[rr[keep], jj[keep]] = True
            else:
                rows = last_pre - prm["days_before"]
                m = ok & (rows > 0)
                if prm["prior"] == "beat_last":
                    evs = ev.assign(j=j, rows=rows, ok=m).sort_values(["sym", "date"])
                    pb = evs.groupby("sym")["surprise"].shift(1) > 0
                    mm = evs["ok"] & pb
                    sig[evs.loc[mm, "rows"].to_numpy(), evs.loc[mm, "j"].to_numpy()] = True
                else:
                    sig[rows[m], j[m]] = True
            zh = r["zh"]
        else:
            continue
        out, unit = get(ek)
        if unit is not None:  # R exits: back to % for the account
            pass
        held = get.held(ek)
        m, info = portfolio.simulate(A, sig, out.astype(float), held.astype(float), max_pos=a.max_pos)
        if len(m) == 0:
            continue
        streams[f"{sid} {zh}"] = m
        report[sid] = {"zh": zh, "exit": ek, **info, "perf": portfolio.perf(m), "vs_spy": _alpha(m, spy_m)}
        print(sid, zh, report[sid]["perf"].get("OOS"), report[sid]["vs_spy"].get("OOS"), flush=True)
    spy = spy_m[spy_m.index >= "2000-01-01"].dropna()
    report["SPY"] = {"perf": portfolio.perf(spy)}
    ex0 = {k: v for k, v in streams.items() if not k.startswith("0 ")}
    if ex0:
        comb = portfolio.combine(ex0)
        ivw = sum(pd.DataFrame(ex0).fillna(0)[k] * w for k, w in comb["weights"].items())
        comb["vs_spy"] = _alpha(ivw, spy_m)
        report["_combined"] = comb
    with open(os.path.join(a.out, "portfolio.json"), "w") as f:
        json.dump(clean(report), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
