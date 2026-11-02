"""Structure-based stop / target / filter study (research only) - see src/backtest/structure.py.

    python structure_study.py [--prices prices.pkl] [--cache data.pkl]
Prices come from a pickled {symbol: standard price frame} (Yahoo, from 2004) so the 2010-2019 holdout exists;
the local database only keeps 5 years.  Writes reports/backtest/<session>/structure_study.json (+ structure_study_compact.json; config/structure_study.default.json
is the 2026-11-02 compact copy).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from src.backtest import structure
from src.daily import store
from src.daily.settings import load_terminal_settings, path
from src.market import calendar
from src.scanner.features import build_features
from src.scanner.setups import detect_setups
from src.scanner.universe_filter import apply_universe_filter


def load(prices: dict, settings: dict):
    spy = prices["SPY"]["adj_close"]
    features, setups = {}, {}
    for sym, etf in settings["universe"].items():
        if sym not in prices or len(prices[sym]) < 300:
            continue
        f = build_features(prices[sym], spy, prices[etf]["adj_close"] if etf in prices else None)
        features[sym], setups[sym] = f, detect_setups(f)
    kept = apply_universe_filter(features, settings, {})["kept"]
    return {s: features[s] for s in kept}, {s: setups[s] for s in kept}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True, help="pickle of {symbol: price frame} from 2004")
    ap.add_argument("--cache", help="pickle for the collected signals (development)")
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args(argv)
    t0 = time.time()
    settings = load_terminal_settings()
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        data = pickle.loads(cache.read_bytes())
    else:
        prices = pickle.loads(Path(args.prices).read_bytes())
        features, setups = load(prices, settings)
        print(f"{len(features)} stocks; collecting signals and structure levels...", flush=True)
        data = structure.collect(features, setups, settings, progress=lambda m: print(m, flush=True))
        if cache:
            cache.write_bytes(pickle.dumps(data))
    sig = data["signals"]
    print(sig["setup"].value_counts().to_string(), flush=True)
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    res = structure.run(data, slip, progress=lambda m: print(m, flush=True), n_perm=args.perm)
    session = calendar.latest_completed_session()
    res.update({"generated_at": store.utc_now(), "session": session.isoformat(), "elapsed_seconds": round(time.time() - t0, 1),
                "counts": sig["setup"].value_counts().to_dict(),
                "survivorship_note": "Only stocks in today's universe are tested (survivorship bias); comparisons are paired."})
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / "structure_study.json", json.dumps(store.jsonable(res), ensure_ascii=False, indent=1))
    store.write_atomic(out / "structure_study_compact.json", json.dumps(store.jsonable(structure.compact(res)), ensure_ascii=False, indent=1))
    print(f"Done in {res['elapsed_seconds']} s -> {out}/structure_study.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
