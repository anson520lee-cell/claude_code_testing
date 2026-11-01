"""Strategy discovery (research only) - see src/backtest/discovery.py.

    python discover.py        # ~2-5 minutes: 14 trading styles the terminal does not use yet (5 years of the
                              # universe) plus the same styles on SPY / QQQ since 1993 / 1999
Writes reports/backtest/<session>/discovery.json (shown on the Research page).
"""

from __future__ import annotations

import json
import sys
import time

from src.backtest import discovery
from src.daily import store
from src.daily.settings import all_symbols, load_terminal_settings, path
from src.data import market_data
from src.market import calendar, regime
from src.scanner.features import build_features
from src.scanner.universe_filter import apply_universe_filter


def main(argv: list[str] | None = None) -> int:
    t0 = time.time()
    settings = load_terminal_settings()
    session = calendar.latest_completed_session()
    conn = store.connect(path(settings, "database"), path(settings, "backups_dir"))
    prices = market_data.load_universe_prices(conn, all_symbols(settings), session, max_days=1500)
    conn.close()
    universe = settings["universe"]
    hist = regime.compute_regime_history(prices, [s for s in universe if s in prices], settings)
    spy = prices["SPY"]["adj_close"]
    features = {}
    for sym, etf in universe.items():
        if sym in prices and len(prices[sym]) >= 300:
            features[sym] = build_features(prices[sym], spy, prices[etf]["adj_close"] if etf in prices else None)
    kept = apply_universe_filter(features, settings, {})["kept"]
    features = {s: features[s] for s in kept}
    print(f"{len(features)} stocks; finding signals for {len(discovery.STYLE_NAMES)} styles...")
    data = discovery.collect(features, prices, hist["label"])
    print(f"{len(data['signals'])} signals (incl. random entries); testing exits...")
    res = discovery.run(data, features, settings, progress=print)
    print("Index study: SPY and QQQ over their whole history...")
    try:
        frames = market_data.YahooBatchDownloader().download(["SPY", "QQQ", "^VIX", "^VIX3M"], period="max")
        slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
        res["index"] = discovery.index_study(frames, slip, progress=print)
        res["verdict"] += ["Index study (SPY / QQQ, whole history, out of sample from 2013):", *discovery.index_verdict(res["index"])]
    except Exception as exc:  # noqa: BLE001 - the stock study is still written
        print(f"Index study failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    res.update({"generated_at": store.utc_now(), "session": session.isoformat(), "n_stocks": len(features),
                "elapsed_seconds": round(time.time() - t0, 1)})
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / "discovery.json", json.dumps(store.jsonable(res), ensure_ascii=False, indent=1))
    print("\n".join(["", "= Verdict =", *res["verdict"]]))
    print(f"\nDone in {res['elapsed_seconds']} s -> {out}/discovery.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
