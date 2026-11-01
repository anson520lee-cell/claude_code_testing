"""Scaling-in study (research only) - see src/backtest/scale_in.py.

    python scale_in_study.py               # a few minutes: 5 years, crash_rebound / squeeze / breakout + random
Writes reports/backtest/<session>/scale_in.json (+ scale_in_compact.json; config/scale_in.default.json is the
2026-10-30 compact copy).
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from src.backtest import scale_in
from src.daily import store
from src.daily.settings import load_terminal_settings, path
from src.market import calendar
from strategy_lab import load


def main(argv: list[str] | None = None) -> int:
    t0 = time.time()
    settings = load_terminal_settings()
    session = calendar.latest_completed_session()
    with open("config/strategy_lab.default.json", encoding="utf-8") as fh:
        lab = json.load(fh)
    chosen = lab["exits"]["chosen"]
    print("Loading 5 years of prices and setups...")
    features, setups, _spy, _hist = load(settings, session)
    data = scale_in.collect(features, setups, settings)
    sig = data["signals"]
    print(sig["setup"].value_counts().to_string())
    cutoff = pd.Timestamp(lab["exits"]["cutoff"])
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    res = scale_in.run(data, chosen, slip, cutoff, progress=print)
    res.update({"generated_at": store.utc_now(), "session": session.isoformat(),
                "elapsed_seconds": round(time.time() - t0, 1), "counts": sig["setup"].value_counts().to_dict(),
                "survivorship_note": "Only stocks in today's universe are tested (survivorship bias)."})
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / "scale_in.json", json.dumps(store.jsonable(res), ensure_ascii=False, indent=1))
    store.write_atomic(out / "scale_in_compact.json",
                       json.dumps(store.jsonable(scale_in.compact(res)), ensure_ascii=False, indent=1))
    print(f"Done in {res['elapsed_seconds']} s -> {out}/scale_in.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
