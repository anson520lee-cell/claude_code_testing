"""SL / TP validation backtest:  python backtest.py [--years 5] [--offline]

1. Makes sure the focus universe (and SPY, VIX, credit, sector ETFs) has `--years` of daily bars.
2. Rebuilds features, setups and the market-regime history.
3. Collects every historical setup signal and simulates the live stop / target plus 191 alternative
   SL / TP / time-stop combinations; chooses on the first 60% of dates, judges on the last 40%.
4. Writes reports/backtest/<date>/backtest.json + backtest.md (shown on the dashboard's Research page).
Research only - no orders.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from src.backtest.research import collect_baseline, collect_signals, run_research
from src.daily import store
from src.daily.settings import all_symbols, load_terminal_settings, path
from src.data import market_data
from src.market import calendar, regime
from src.scanner.features import build_features
from src.scanner.setups import detect_setups
from src.scanner.universe_filter import apply_universe_filter


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SL / TP validation backtest (research only)")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--offline", action="store_true", help="use stored bars only")
    args = ap.parse_args(argv)
    t0 = time.time()
    settings = load_terminal_settings()
    session = calendar.latest_completed_session()
    conn = store.connect(path(settings, "database"), path(settings, "backups_dir"))
    symbols = all_symbols(settings)
    if not args.offline:
        cutoff = pd.Timestamp(session) - pd.DateOffset(years=args.years) + pd.Timedelta(days=30)
        first = dict(conn.execute("SELECT ticker, MIN(date) FROM price_data GROUP BY ticker").fetchall())
        need = [s for s in symbols if s not in first or pd.Timestamp(first[s]) > cutoff]
        if need:
            print(f"下載 {len(need)} 個代號的 {args.years} 年歷史（只需一次）…")
            frames = market_data.YahooBatchDownloader().download(need, period=f"{args.years}y")
            for sym, frame in frames.items():
                frame = frame[frame.index <= pd.Timestamp(session)]
                if not frame.empty:
                    from src.database import db
                    db.upsert_stock(conn, sym)
                    db.upsert_prices(conn, sym, frame, market_data.SOURCE)
    prices = market_data.load_universe_prices(conn, symbols, session, max_days=300 * args.years)
    if "SPY" not in prices:
        print("沒有 SPY 數據，無法回測。", file=sys.stderr)
        return 3
    universe = settings["universe"]
    print("計算市場狀態歷史…")
    hist = regime.compute_regime_history(prices, [s for s in universe if s in prices], settings)
    spy = prices["SPY"]["adj_close"]
    features, setups = {}, {}
    print(f"計算 {len(universe)} 檔股票的指標和形態…")
    for sym, etf in universe.items():
        if sym not in prices or len(prices[sym]) < 300:
            continue
        sector = prices[etf]["adj_close"] if etf in prices else None
        f = build_features(prices[sym], spy, sector)
        features[sym], setups[sym] = f, detect_setups(f)
    focus = apply_universe_filter(features, settings, {})
    features = {s: features[s] for s in focus["kept"]}
    setups = {s: setups[s] for s in focus["kept"]}
    print(f"高 Beta／高波動篩選後 {len(features)} 檔；收集歷史信號…")
    data = collect_signals(features, setups, settings, hist["label"], progress=print, regime_scores=hist["composite"])
    print(f"共 {len(data['signals'])} 個信號；模擬 {8 * 8 * 3} 種止損／目標／時間止損組合…")
    base = collect_baseline(features, hist["label"])
    print(f"隨機進場基準 {len(base['signals'])} 筆；開始模擬…")
    res = run_research(data, settings, baseline=base)
    res.update({"generated_at": store.utc_now(), "session": session.isoformat(), "years": args.years,
                "universe_size": len(features), "elapsed_seconds": round(time.time() - t0, 1),
                "survivorship_note": "回測只用今日仍在股票池的股票（倖存者偏差），結果可能比真實情況好。"})
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest.json").write_text(json.dumps(store.jsonable(res), ensure_ascii=False, indent=1), encoding="utf-8")
    (out.parent / "latest.json").write_text(json.dumps({"session": session.isoformat()}), encoding="utf-8")
    print("\n".join(["", "＝ 結論 ＝", *res.get("verdict", [])]))
    print(f"\n完成，用時 {res['elapsed_seconds']} 秒 → {out}/backtest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
