"""2x leveraged ETF + scaling-in study (research only) - see src/backtest/leverage.py.

    python leverage_study.py                # ~10-20 minutes: 5 years of setups, real 2x ETFs for the check
    python leverage_study.py --cache F.pkl  # reuse / store the collected trades (development)
    python leverage_study.py --no-etf-check # skip downloading the real 2x ETFs (cost falls back to 15%/year)
Writes reports/backtest/<session>/leverage.json (Research page). Changes nothing in the live scoring.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import strategy_lab as lab_cli
from src.backtest import leverage as lv
from src.daily import store
from src.daily.settings import load_terminal_settings, path
from src.data import market_data
from src.market import calendar


def real_etf_pairs() -> dict:
    """Real 2x ETF and its stock, both split/dividend-adjusted (adj_close)."""
    syms = sorted(set(lv.ETF_UNDERLYINGS.values()) | set(lv.ETF_UNDERLYINGS))
    frames = market_data.YahooBatchDownloader().download(syms, period="6y")
    pairs = {}
    for under, etf in lv.ETF_UNDERLYINGS.items():
        if etf in frames and under in frames:
            u, e = frames[under].copy(), frames[etf].copy()
            u["close"] = u.get("adj_close", u["close"])
            e["close"] = e.get("adj_close", e["close"])
            pairs[etf] = (u, e)
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="2x ETF + scaling-in study (research only)")
    ap.add_argument("--cache", help="pickle file for the collected trades")
    ap.add_argument("--no-etf-check", action="store_true")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--max-positions", type=int, default=10)
    args = ap.parse_args(argv)
    t0 = time.time()
    settings = load_terminal_settings()
    session = calendar.latest_completed_session()
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        data = pickle.loads(cache.read_bytes())
    else:
        print("Loading 5 years of prices and setups...")
        features, setups, spy, _hist = lab_cli.load(settings, session)
        data = lv.collect(features, setups, settings["scanner"], spy.index)
        data["spy"], data["calendar"] = spy, spy.index
        if cache:
            cache.write_bytes(pickle.dumps(data))
    validation = None
    if not args.no_etf_check:
        print("Checking the synthetic 2x ETF against real 2x single-stock ETFs...")
        validation = lv.validate_synthetic(real_etf_pairs())
    sig = data["signals"]
    print(f"{(sig['setup'] != 'baseline').sum()} setup trades, {(sig['setup'] == 'baseline').sum()} random entries")
    res = lv.run_study(data, data["calendar"], data["spy"], validation=validation, seeds=args.seeds,
                       max_positions=args.max_positions, progress=print)
    res.update({"generated_at": store.utc_now(), "session": session.isoformat(), "elapsed_seconds": round(time.time() - t0, 1),
                "survivorship_note": "Only stocks in today's universe are tested (survivorship bias); 2021-2026 was mostly a bull market."})
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / "leverage.json", json.dumps(store.jsonable(res), ensure_ascii=False, indent=1))
    print("\n".join(["", "= Verdict =", *res["verdict"]]))
    print(f"\nDone in {res['elapsed_seconds']} s -> {out}/leverage.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
