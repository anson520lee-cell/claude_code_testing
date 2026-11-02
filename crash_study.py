"""Crash-rebound / spike-fade event study (research only) - see src/backtest/crash_study.py.

    python crash_study.py            # uses the stored 5-year history (run backtest.py once first)
Writes reports/backtest/<session>/crash_study.json, shown on the Research page.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

from src.backtest.crash_study import EVENTS, SPIKES, analyse, baseline, collect_events
from src.daily import store
from src.daily.settings import all_symbols, load_terminal_settings, path
from src.data import market_data
from src.market import calendar
from src.scanner.features import build_features
from src.scanner.universe_filter import apply_universe_filter


def load_features(settings, session, cache: Path | None = None):
    if cache and cache.exists():
        return pickle.loads(cache.read_bytes())
    conn = store.connect(path(settings, "database"), path(settings, "backups_dir"))
    prices = market_data.load_universe_prices(conn, all_symbols(settings), session, max_days=1500)
    conn.close()
    spy = prices["SPY"]["adj_close"]
    feats = {}
    for sym, etf in settings["universe"].items():
        if sym in prices and len(prices[sym]) >= 300:
            feats[sym] = build_features(prices[sym], spy, prices[etf]["adj_close"] if etf in prices else None)
    kept = apply_universe_filter(feats, settings, {})["kept"]
    out = ({s: feats[s] for s in kept}, prices["SPY"]["close"])
    if cache:
        cache.write_bytes(pickle.dumps(out))
    return out


def main(argv: list[str] | None = None) -> int:
    t0 = time.time()
    settings = load_terminal_settings()
    session = calendar.latest_completed_session()
    cache = Path(argv[0]) if argv else None
    features, spy = load_features(settings, session, cache)
    print(f"{len(features)} stocks; collecting crash and spike events...")
    res = {"session": session.isoformat(), "n_stocks": len(features), "event_names": EVENTS, "spike_names": SPIKES}
    for side, key in ((1, "crash"), (-1, "spike")):
        ev = collect_events(features, spy, side)
        res[key] = analyse(ev, baseline(features, side))
        print(f"{key}: {res[key].get('n_events', 0)} events")
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / "crash_study.json", json.dumps(store.jsonable(res), ensure_ascii=False, indent=1))
    print(f"Done in {res['elapsed_seconds']} s -> {out}/crash_study.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
