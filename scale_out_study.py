"""Partial take-profit (scale-out) and stock-specific target study (research only) - see src/backtest/scale_out.py.

    python scale_out_study.py --prices prices.pkl [--cache signals.pkl]
Prices come from a pickled {symbol: standard price frame} (Yahoo, from 2003) so the 2010-2019 holdout exists; the
local database only keeps 5 years. Writes reports/backtest/<session>/scale_out_study.json (+ _compact.json;
config/scale_out.default.json is the 2026-11-02 compact copy).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import leverage, scale_out
from src.daily import store
from src.daily.settings import load_terminal_settings, path
from src.market import calendar
from structure_study import load


def accounts(data: dict, slip: float, spy: pd.DataFrame, res: dict, seeds: int = 12) -> dict:
    """Illustration: a 10-position cash account trading all three live setups, live exits vs the best in-sample
    choice of each family (one per setup), vs SPY buy & hold. Medians over random orders of same-day signals."""
    cal = spy.index
    cats = {s: {m["name"]: m for m in scale_out.catalog(s)} for s in scale_out.SETUPS}

    def best_is(setup: str, family: str | None) -> dict | None:
        ms = res["setups"][setup]["methods"]
        names = [n for n, v in ms.items() if (family is None or v["family"] == family) and v["family"] != "retrace_stock"]
        if not names:
            return None
        nm = max(names, key=lambda n: (ms[n]["setup"]["is"].get("t") or -9))
        return cats[setup][nm]
    variants = {"live": {s: None for s in scale_out.SETUPS}}
    for fam in ("partial_R", "ladder", "stock_hist", "model"):
        variants[f"best_is_{fam}"] = {s: best_is(s, fam) for s in scale_out.SETUPS}
    variants["best_is_any"] = {s: best_is(s, None) for s in scale_out.SETUPS}
    variants["half_at_2R_all"] = {s: cats[s]["partial_2R_50%"] for s in scale_out.SETUPS}
    out = {"variants": {k: {s: (m["name"] if m else "live") for s, m in v.items()} for k, v in variants.items()}, "results": {}}
    windows = {"hold": ("2010-01-01", "2019-12-31"), "is": ("2020-01-01", "2024-06-30"), "oos": ("2024-07-01", "2100-01-01"),
               "all_2010": ("2010-01-01", "2100-01-01")}
    for name, choice in variants.items():
        tr = scale_out.account_paths(data, choice, slip, cal)
        pnl, cap = tr.attrs["pnl"], tr.attrs["cap"]
        out["results"][name] = {}
        for w, (lo, hi) in windows.items():
            runs = [leverage.portfolio(tr, pnl, cap, cal, start=pd.Timestamp(lo), end=pd.Timestamp(hi), seed=s) for s in range(seeds)]
            out["results"][name][w] = {k: float(np.median([r[k] for r in runs])) for k in ("cagr", "max_dd", "sharpe", "vol", "taken")}
    for w, (lo, hi) in windows.items():
        px = spy["adj_close"].loc[lo:hi]
        out["results"].setdefault("SPY_buy_hold", {})[w] = leverage.perf(px / px.iloc[0])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--no-accounts", action="store_true")
    ap.add_argument("--indicator", action="store_true", help="indicator exit signals (RSI / MACD histogram) instead")
    args = ap.parse_args(argv)
    t0 = time.time()
    settings = load_terminal_settings()
    prices = pickle.loads(Path(args.prices).read_bytes())
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        data = pickle.loads(cache.read_bytes())
    else:
        features, setups = load(prices, settings)
        print(f"{len(features)} stocks; collecting signals and run-up histories...", flush=True)
        data = scale_out.collect(features, setups, settings, spy_close=prices["SPY"]["adj_close"],
                                 progress=lambda m: print(m, flush=True))
        if cache:
            cache.write_bytes(pickle.dumps(data))
    sig = data["signals"]
    print(sig["setup"].value_counts().to_string(), flush=True)
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    name = "scale_out_study"
    if args.indicator:
        name = "indicator_exit_study"
        features, _ = load(prices, settings)
        data["IND"] = scale_out.indicator_series(features, sig)
        res = scale_out.run(data, slip, progress=lambda m: print(m, flush=True), n_perm=args.perm, cat_fn=scale_out.indicator_catalog)
        args.no_accounts = True
    else:
        res = scale_out.run(data, slip, progress=lambda m: print(m, flush=True), n_perm=args.perm)
    if not args.no_accounts:
        print("account simulation...", flush=True)
        res["accounts"] = accounts(data, slip, prices["SPY"], res)
    session = calendar.latest_completed_session()
    res.update({"generated_at": store.utc_now(), "session": session.isoformat(), "elapsed_seconds": round(time.time() - t0, 1),
                "counts": sig["setup"].value_counts().to_dict(),
                "survivorship_note": "Only stocks in today's universe are tested (survivorship bias); comparisons are paired."})
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / f"{name}.json", json.dumps(store.jsonable(res), ensure_ascii=False, indent=1))
    comp = scale_out.compact(res)
    if "accounts" in res:
        comp["accounts"] = res["accounts"]
    store.write_atomic(out / f"{name}_compact.json", json.dumps(store.jsonable(comp), ensure_ascii=False, indent=1))
    print(f"Done in {res['elapsed_seconds']} s -> {out}/{name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
