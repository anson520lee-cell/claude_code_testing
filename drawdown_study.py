"""Drawdown-reduction study (research only) - see src/backtest/drawdown.py and docs/DRAWDOWN_STUDY.md.

    python drawdown_study.py --prices prices.pkl [--cache data.pkl] [--workers 4]

prices.pkl = pickled {symbol: standard daily price frame} from Yahoo since 2003 (same file as structure_study.py).
cache      = pickle of scale_out.collect() (optional, saves ~2 minutes on a re-run).

Stages (all on the account of the three live setups with their live exits; a second copy uses a 5 ATR trail for the
breakout setup):
  1. singles   ~90 single mechanisms, ranked on the in-sample window only (2020-01..2024-06);
  2. plateau   54 settings of the adaptive heat cap (is the result a plateau or a spike?);
  3. final     the short list + 5 sub-periods, 24 random same-day orders per cell.
Writes reports/backtest/<session>/drawdown_study.json and config/drawdown_study.default.json (compact copy).
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import drawdown as dd
from src.daily import store
from src.daily.settings import load_terminal_settings, path
from src.market import calendar

V = dd.Variant
NON_CRASH = ("breakout", "squeeze")
TIERS_10_20 = ("tiers", -0.10, 0.5, -0.20, 0.25, None, None)
TIERS_15_30 = ("tiers", -0.15, 0.5, -0.30, 0.25, None, None)
WIN = dict(dd.WINDOWS)
WIN.update({"p1_2010_13": ("2010-01-01", "2013-12-31"), "p2_2014_17": ("2014-01-01", "2017-12-31"),
            "p3_2018_21": ("2018-01-01", "2021-12-31"), "p4_2022_24": ("2022-01-01", "2024-12-31"),
            "p5_2025_26": ("2025-01-01", "2100-01-01")})
MAIN = ("all", "hold", "is", "oos")

_G: dict = {}


def _init(prices_file: str, trades_file: str) -> None:
    prices = pickle.loads(Path(prices_file).read_bytes())
    tr, pnl, cap = pickle.loads(Path(trades_file).read_bytes())
    syms = list(load_terminal_settings()["universe"])
    cal = prices["SPY"].index
    _G.update(trades=tr, pnl=pnl, cap=cap, cal=cal, ctx=dd.build_context(prices, cal, syms))
    _G["rets"], _G["si"] = dd.returns_matrix(prices, cal, syms)


def _eval(args):
    v, seeds, windows = args
    return v.name, dd.evaluate(_G["trades"], _G["pnl"], _G["cap"], _G["cal"], v, _G["ctx"], _G["rets"], _G["si"],
                               seeds=seeds, windows=windows)


def singles() -> list[dd.Variant]:
    g: list[dd.Variant] = [V("base")]
    g += [V(f"risk_{r * 100:g}%", risk=r) for r in (0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.0125, 0.015)]
    g += [V(f"slots_{s}", slots=s) for s in (4, 5, 6, 8, 15)]
    g += [V(f"heat_{h * 100:g}%", heat=h) for h in (0.03, 0.04, 0.05, 0.06, 0.08)]
    g += [V(f"sector_cap_{c}", sector_cap=c) for c in (1, 2, 3)]
    g += [V(f"corr_cap_{c}", corr_cap=c) for c in (0.4, 0.5, 0.6)]
    regs = {
        "spy200_pause": ("spy_ma", "spy_vs200", 0.0), "spy200_half": ("spy_ma", "spy_vs200", 0.5),
        "spy50_pause": ("spy_ma", "spy_vs50", 0.0), "spy50_half": ("spy_ma", "spy_vs50", 0.5),
        "spy_2ma": ("spy_two_ma", 0.5, 0.0),
        "vix25_half": ("vix", 25.0, 0.5), "vix30_half": ("vix", 30.0, 0.5), "vix30_pause": ("vix", 30.0, 0.0),
        "vix_tiers_20_30": ("vix_tiers", 20.0, 0.5, 30.0, 0.0),
        "spydd10_half": ("spy_dd", -0.10, 0.5), "spydd10_pause": ("spy_dd", -0.10, 0.0), "spydd15_pause": ("spy_dd", -0.15, 0.0),
        "breadth50_40_half": ("breadth", "breadth50", 0.40, 0.5), "breadth50_30_pause": ("breadth", "breadth50", 0.30, 0.0),
        "breadth200_40_half": ("breadth", "breadth200", 0.40, 0.5), "breadth200_30_pause": ("breadth", "breadth200", 0.30, 0.0),
    }
    for nm, spec in regs.items():          # applied to all setups, and to breakout + squeeze only (crash rebound buys panics)
        g.append(V(f"reg_{nm}", regime=spec))
        g.append(V(f"reg_{nm}_noCR", regime=spec, regime_setups=NON_CRASH))
    g += [V(f"vol_spy_{t * 100:g}", vol=("spy", t)) for t in (0.15, 0.20, 0.25)]
    g += [V(f"vol_vix_{t * 100:g}", vol=("vix", t)) for t in (0.18, 0.22, 0.28)]
    g += [V(f"vol_acct_{t * 100:g}", vol=("acct", t)) for t in (0.20, 0.30, 0.40)]
    g += [V("brk_tiers_10_20", breaker=TIERS_10_20), V("brk_tiers_15_30", breaker=TIERS_15_30),
          V("brk_tiers_10_20_pause30", breaker=("tiers", -0.10, 0.5, -0.20, 0.25, -0.30, -0.15)),
          V("brk_pause25_resume12", breaker=("tiers", -1.0, 1.0, -1.0, 1.0, -0.25, -0.12)),
          V("brk_pause20_resume10", breaker=("tiers", -1.0, 1.0, -1.0, 1.0, -0.20, -0.10)),
          V("brk_half_below_20", breaker=("tiers", -0.20, 0.5, -1.0, 0.5, None, None))]
    for n in (50, 100):
        g += [V(f"brk_eqtrend_{n}_half", breaker=("trend", n, 0.5)), V(f"brk_eqtrend_{n}_pause", breaker=("trend", n, 0.0))]
    g += [V(f"bo_risk_x{m}", setup_risk={"breakout": m}) for m in (0.25, 0.5)]
    for n in (2, 3, 4, 6):
        g += [V(f"bo_slots_{n}", setup_slots={"breakout": n}), V(f"bo_slots_{n}_prio", setup_slots={"breakout": n}, priority=True)]
    g += [V("prio_only", priority=True), V("no_breakout", setup_risk={"breakout": 0.0}),
          V("bo_spy200_pause", regime=("spy_ma", "spy_vs200", 0.0), regime_setups=("breakout",)),
          V("bo_spy200_half", regime=("spy_ma", "spy_vs200", 0.5), regime_setups=("breakout",))]
    return g


def plateau() -> list[dd.Variant]:
    g = [V("base")]
    for hi, lo, trig, use in itertools.product((0.06, 0.08, 0.10), (0.03, 0.04, 0.05), (0.08, 0.10, 0.15), (True, False)):
        if lo < hi:
            g.append(V(f"ah_{hi * 100:g}/{lo * 100:g}_dd{trig * 100:g}_{'spy' if use else 'nospy'}", heat_adapt=(hi, lo, trig, use)))
    return g


def final() -> list[dd.Variant]:
    return [V("base")] + [V(f"risk_{r * 100:g}%", risk=r) for r in (0.003, 0.004, 0.005, 0.006, 0.007, 0.008)] + [
        V("reg_spy200_half(non-crash)", regime=("spy_ma", "spy_vs200", 0.5), regime_setups=NON_CRASH),
        V("reg_spy200_pause(all)", regime=("spy_ma", "spy_vs200", 0.0)),
        V("reg_breadth200<40%_half", regime=("breadth", "breadth200", 0.40, 0.5)),
        V("reg_vix>30_half(non-crash)", regime=("vix", 30.0, 0.5), regime_setups=NON_CRASH),
        V("vol_target_spy20", vol=("spy", 0.20)), V("vol_target_acct30", vol=("acct", 0.30)),
        V("breaker_tiers_10/20", breaker=TIERS_10_20), V("breaker_eqtrend100_half", breaker=("trend", 100, 0.5)),
        V("slots_6", slots=6), V("sector_cap_2", sector_cap=2), V("corr_cap_0.5", corr_cap=0.5),
        V("heat_4%(terminal)", heat=0.04), V("heat_5%", heat=0.05), V("heat_6%", heat=0.06),
        V("adaptive_heat_8/4_dd15", heat_adapt=(0.08, 0.04, 0.15, False)),
        V("adaptive_heat_8/4_dd10", heat_adapt=(0.08, 0.04, 0.10, False)),
        V("adaptive_heat_8/4_dd10+SPY200", heat_adapt=(0.08, 0.04, 0.10, True)),
        V("adaptive_heat_8/4_dd15+bo_x0.5", heat_adapt=(0.08, 0.04, 0.15, False), setup_risk={"breakout": 0.5}),
    ]


def run_stage(pool: ProcessPoolExecutor, variants: list[dd.Variant], seeds: int, windows: dict) -> dict:
    return dict(pool.map(_eval, [(v, seeds, windows) for v in variants]))


def compact(res: dict, windows: tuple[str, ...] | None = None, keys=("cagr", "max_dd", "calmar", "sharpe", "ulcer")) -> dict:
    return {name: {w: {k: round(m[k], 4) for k in keys if k in m} for w, m in per.items() if windows is None or w in windows}
            for name, per in res.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--cache", help="pickle of scale_out.collect()")
    ap.add_argument("--workdir", default=".", help="where the trade pickles are written")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    t0 = time.time()
    from src.backtest import scale_out
    from structure_study import load
    settings = load_terminal_settings()
    prices = pickle.loads(Path(args.prices).read_bytes())
    cal = prices["SPY"].index
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        data = pickle.loads(cache.read_bytes())
    else:
        features, setups = load(prices, settings)
        print(f"{len(features)} stocks; collecting signals...", flush=True)
        data = scale_out.collect(features, setups, settings, spy_close=prices["SPY"]["adj_close"], progress=lambda m: print(m, flush=True))
        if cache:
            cache.write_bytes(pickle.dumps(data))
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    work = Path(args.workdir)
    out: dict = {"windows": {k: list(v) for k, v in WIN.items()}}
    for tag, trail in (("live", None), ("breakout_trail_5atr", 5.0)):
        tr = dd.build_trades(data, slip, cal, settings["universe"], breakout_trail=trail)
        f = work / f"drawdown_trades_{tag}.pkl"
        f.write_bytes(pickle.dumps((tr, tr.attrs["pnl"], tr.attrs["cap"])))
        print(f"{tag}: {len(tr)} trades {tr.groupby('setup').size().to_dict()}", flush=True)
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.prices, str(f))) as pool:
            res: dict = {}
            if tag == "live":
                res["singles"] = run_stage(pool, singles(), 16, {k: WIN[k] for k in MAIN})
                res["plateau"] = run_stage(pool, plateau(), 16, {k: WIN[k] for k in MAIN})
            res["final"] = run_stage(pool, final(), 24, WIN)
        out[tag] = res
        print(f"{tag} done {time.time() - t0:.0f}s", flush=True)
    spy = prices["SPY"]["adj_close"]
    out["spy_buy_hold"] = {w: dd.perf(spy.loc[lo:hi] / spy.loc[lo:hi].iloc[0]) for w, (lo, hi) in WIN.items()}
    session = calendar.latest_completed_session()
    rep = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    rep.mkdir(parents=True, exist_ok=True)
    store.write_atomic(rep / "drawdown_study.json", json.dumps(store.jsonable(out), ensure_ascii=False, indent=1))
    comp = {"session": session.isoformat(), "windows": out["windows"], "spy_buy_hold": out["spy_buy_hold"],
            "survivorship_note": "Only stocks in today's universe are tested (survivorship bias); all variants share the same trades.",
            "singles": compact(out["live"]["singles"]), "plateau": compact(out["live"]["plateau"]),
            "final_live": compact(out["live"]["final"]), "final_breakout_trail_5atr": compact(out["breakout_trail_5atr"]["final"])}
    store.write_atomic(Path("config") / "drawdown_study.default.json", json.dumps(store.jsonable(comp), ensure_ascii=False, indent=1))
    print(f"Done in {time.time() - t0:.0f} s -> {rep}/drawdown_study.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
