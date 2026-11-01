"""Strategy lab (research only) - see src/backtest/strategy_lab.py.

    python strategy_lab.py                 # ~20-40 minutes: 5 years, every long setup + random entries
    python strategy_lab.py --cache F.pkl   # reuse / store the collected feature vectors (development)
Writes reports/backtest/<session>/strategy_lab.json; the live run reads the chosen exits and (if one
passed the out-of-sample test) the indicator model from it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

from src.backtest import strategy_lab as lab
from src.daily import store
from src.daily.settings import all_symbols, load_terminal_settings, path
from src.data import market_data
from src.market import calendar, regime
from src.scanner.features import build_features
from src.scanner.setups import detect_setups
from src.scanner.universe_filter import apply_universe_filter


def load(settings, session, years: int = 5):
    conn = store.connect(path(settings, "database"), path(settings, "backups_dir"))
    prices = market_data.load_universe_prices(conn, all_symbols(settings), session, max_days=300 * years)
    conn.close()
    universe = settings["universe"]
    hist = regime.compute_regime_history(prices, [s for s in universe if s in prices], settings)
    spy = prices["SPY"]["adj_close"]
    features, setups = {}, {}
    for sym, etf in universe.items():
        if sym not in prices or len(prices[sym]) < 300:
            continue
        f = build_features(prices[sym], spy, prices[etf]["adj_close"] if etf in prices else None)
        features[sym], setups[sym] = f, detect_setups(f)
    kept = apply_universe_filter(features, settings, {})["kept"]
    return {s: features[s] for s in kept}, {s: setups[s] for s in kept}, prices["SPY"]["close"], hist


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Exit and indicator-weight study (research only)")
    ap.add_argument("--cache", help="pickle file for the collected feature vectors")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2))))
    args = ap.parse_args(argv)
    t0 = time.time()
    settings = load_terminal_settings()
    session = calendar.latest_completed_session()
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        data = pickle.loads(cache.read_bytes())
    else:
        print("Loading 5 years of prices, indicators and setups...")
        features, setups, spy, hist = load(settings, session)
        print(f"{len(features)} stocks; computing ~170 indicators for every setup episode and random entry "
              f"({args.workers} processes, about 20-40 minutes)...")
        data = lab.collect(features, setups, spy, hist["composite"], settings, workers=args.workers, progress=print)
        if cache:
            cache.write_bytes(pickle.dumps(data))
    sig = data["signals"]
    print(f"{(sig['setup'] != 'baseline').sum()} setup episodes, {(sig['setup'] == 'baseline').sum()} random entries")
    res = lab.run_lab(data, settings, progress=print)
    res.update({"generated_at": store.utc_now(), "session": session.isoformat(), "elapsed_seconds": round(time.time() - t0, 1),
                "survivorship_note": "Only stocks in today's universe are tested (survivorship bias)."})
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / "strategy_lab.json", json.dumps(store.jsonable(res), ensure_ascii=False, indent=1))
    store.write_atomic(out.parent / "latest.json", json.dumps({"session": session.isoformat()}))
    print("\n".join(["", "= Verdict =", *res["verdict"]]))
    print(f"\nDone in {res['elapsed_seconds']} s -> {out}/strategy_lab.json")
    return 0


if __name__ == "__main__":
    from src.platform_compat import ensure_utf8
    ensure_utf8()
    sys.exit(main())
