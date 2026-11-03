"""Trendline-touch and EMA200/250-retest long entries (research only) - see src/backtest/trendline.py.

    python trendline_study.py --prices prices.pkl [--cache arrays.pkl] [--limit 30]
Prices: pickled {symbol: standard price frame} from Yahoo (2003-06 on, so the 2010-2019 holdout exists).
Writes reports/backtest/<session>/trendline_study.json and trendline_study_compact.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from src.backtest import trendline_study as ts
from src.daily import store
from src.daily.settings import load_terminal_settings, path
from src.market import calendar


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--cache", help="pickle for the per-stock arrays (development)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N stocks (development)")
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--out", help="output directory (default reports/backtest/<session>)")
    args = ap.parse_args(argv)
    t0 = time.time()
    settings = load_terminal_settings()
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        arrs, last = pickle.loads(cache.read_bytes())
    else:
        import structure_study
        prices = pickle.loads(Path(args.prices).read_bytes())
        feats, _ = structure_study.load(prices, settings)
        session = calendar.latest_completed_session()
        last = session.isoformat()
        arrs = ts.prepare(feats, last)
        if cache:
            cache.write_bytes(pickle.dumps((arrs, last)))
    if args.limit:
        arrs = dict(list(arrs.items())[: args.limit])
    syms = list(arrs)
    print(f"{len(syms)} stocks, last completed session {last}", flush=True)
    fam = ts.collect(arrs, progress=print)
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    res = ts.run(arrs, syms, fam, slip, progress=lambda m: print(m, flush=True), n_perm=args.perm)
    res.update({"generated_at": store.utc_now(), "session": last, "elapsed_seconds": round(time.time() - t0, 1), "n_stocks": len(syms),
                "survivorship_note": "Only stocks in today's universe are tested (survivorship bias); every comparison is against the same universe."})
    out = Path(args.out) if args.out else path(settings, "reports_dir").parent / "backtest" / last
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / "trendline_study.json", json.dumps(store.jsonable(res), ensure_ascii=False))
    store.write_atomic(out / "trendline_study_compact.json", json.dumps(store.jsonable(ts.compact(res)), ensure_ascii=False))
    print(f"Done in {res['elapsed_seconds']} s -> {out}/trendline_study.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
