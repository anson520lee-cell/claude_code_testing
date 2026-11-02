"""Adaptive ATR stop study (research only) - see src/backtest/adaptive_stop.py for the design and docs 7n.

    python adaptive_stop_study.py collect --prices prices.pkl --earnings earnings.pkl --cache DIR [--workers 4]
    python adaptive_stop_study.py grid    --prices prices.pkl --cache DIR          # (A) two-phase grid + account
    python adaptive_stop_study.py learn   --prices prices.pkl --cache DIR          # (B) learned ATR multiple
    python adaptive_stop_study.py report  --cache DIR                              # compact JSON

prices.pkl   = pickled {symbol: standard daily price frame} from Yahoo since 2003 (universe, sector ETFs, SPY, ^VIX,
               QQQ, IWM, TLT, HYG, GLD, UUP, ^TNX).
earnings.pkl = pickled {symbol: [report dates]} (yfinance get_earnings_dates, optional).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import adaptive_stop as ad
from src.daily.settings import load_terminal_settings

_G: dict = {}


def _init(prices_file: str, earn_file: str | None) -> None:
    _G["prices"] = pickle.loads(Path(prices_file).read_bytes())
    _G["earn"] = pickle.loads(Path(earn_file).read_bytes()) if earn_file and Path(earn_file).exists() else {}
    _G["settings"] = load_terminal_settings()
    syms = [s for s in _G["settings"]["universe"] if s in _G["prices"]]
    _G["mkt"] = ad.market_features(_G["prices"], syms)


def _features(sym: str):
    from src.scanner.features import build_features
    from src.scanner.setups import detect_setups
    p, settings = _G["prices"], _G["settings"]
    etf = settings["universe"].get(sym)
    f = build_features(p[sym], p["SPY"]["adj_close"], p[etf]["adj_close"] if etf in p else None)
    return f, detect_setups(f)


def _filter_row(sym: str):
    try:
        f, _ = _features(sym)
    except Exception:  # noqa: BLE001
        return sym, None
    return sym, f.iloc[[-1]]


def _collect(args):
    sym, si = args
    f, st = _features(sym)
    X = ad.stock_feature_frame(f, _G["earn"].get(sym), _G["mkt"])
    part = ad.collect_stock(sym, f, st, _G["settings"], X, si)
    part["cols"] = list(X.columns)
    return part


def collect(args) -> int:
    t0 = time.time()
    settings = load_terminal_settings()
    prices = pickle.loads(Path(args.prices).read_bytes())
    syms = [s for s in settings["universe"] if s in prices and len(prices[s]) >= 300]
    init = (args.prices, args.earnings)
    from src.scanner.universe_filter import apply_universe_filter
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as ex:
        last = dict(ex.map(_filter_row, syms, chunksize=4))
    kept = apply_universe_filter({s: f for s, f in last.items() if f is not None}, settings, {})["kept"]
    kept = [s for s in syms if s in kept][: args.limit or None]
    print(f"{len(kept)} stocks after the focus filter; collecting...", flush=True)
    parts = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as ex:
        for i, part in enumerate(ex.map(_collect, [(s, i) for i, s in enumerate(kept)], chunksize=1)):
            if "date" in part:
                parts.append(part)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(kept)} stocks, {time.time() - t0:.0f}s", flush=True)
    cols = parts[0]["cols"]
    for p in parts:                      # align columns (categorical one-hots differ between stocks)
        if p["cols"] != cols:
            cols = sorted(set(cols) | set(p["cols"]), key=lambda c: (c not in parts[0]["cols"], c))
    out = {"cols": cols}
    pos = {c: j for j, c in enumerate(cols)}
    meta = []
    Xe, Xd, OHLC = [], [], {k: [] for k in "OHLC"}
    for p in parts:
        jj = np.array([pos[c] for c in p["cols"]])
        xe = np.full((len(p["date"]), len(cols)), np.nan, dtype=np.float32)
        xe[:, jj] = p["Xe"]
        Xe.append(xe)
        if "Xd" in p:
            xd = np.full(p["Xd"].shape[:2] + (len(cols),), np.nan, dtype=np.float32)
            xd[:, :, jj] = p["Xd"]
            Xd.append(xd)
        for k in "OHLC":
            OHLC[k].append(p[k])
        meta.append(pd.DataFrame({"symbol": p["symbol"], "date": p["date"], "setup": p["setup"], "ref": p["ref"],
                                  "atr": p["atr"], "t": p["t"]}))
    sig = pd.concat(meta, ignore_index=True)
    sig["date"] = pd.to_datetime(sig["date"])
    out["signals"] = sig
    out["Xe"] = np.concatenate(Xe)
    out["Xd"] = np.concatenate(Xd)
    for k in "OHLC":
        out[k] = np.concatenate(OHLC[k])
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "Xe.npy", out.pop("Xe"))
    np.save(cache / "Xd.npy", out.pop("Xd"))
    (cache / "collect.pkl").write_bytes(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"{len(sig)} signals ({sig['setup'].value_counts().to_dict()}), {len(cols)} features, {time.time() - t0:.0f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "grid", "learn", "final", "families", "report"])
    ap.add_argument("--prices")
    ap.add_argument("--earnings")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--setups")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.stage == "collect":
        return collect(args)
    import adaptive_stop_stages as stages
    return getattr(stages, args.stage)(args)


if __name__ == "__main__":
    sys.exit(main())
