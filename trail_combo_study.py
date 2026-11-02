"""Indicator x ATR trailing stop study, single conditions and combinations (docs 7m). Run by hand:

    python trail_combo_study.py --prices prices.pkl --cache <dir with holding_study's collect.pkl / tech.pkl>

Re-uses the trade collection of ``holding_study.py`` (run that first with --cache) and adds the 300-series indicator
library (``src.technical.indicator_library``) to every trade-day. See ``src/backtest/trail_combo.py`` for the method.
Writes reports/backtest/<session>/trail_combo.json and the compact copy config/trail_combo.default.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import holding_study as runner
from src.backtest import holding_study as hs
from src.backtest import trail_combo as tc
from src.daily import store
from src.market import calendar
from src.daily.settings import load_terminal_settings, path
from src.technical import indicator_library as il

_P: dict = {}


def _lib_init(prices_file: str) -> None:
    _P["prices"] = pickle.loads(Path(prices_file).read_bytes())


def _lib(sym: str):
    warnings.simplefilter("ignore")
    f = _P["prices"][sym][["open", "high", "low", "close", "volume"]]
    L = il.compute(f)
    return sym, (list(L.columns), L.to_numpy(np.float32))


def _r(x, nd: int = 4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def rule_masks(book: hs.Book, cols: list[str]) -> dict[str, np.ndarray]:
    td = book.td
    masks = {}
    masks.update({k: v for k, v in hs.pattern_masks(td).items()})
    masks.update(hs.classic_masks(td))
    masks.update(hs.state_masks(td))
    _, ind, _ = hs.indicator_scan(book, cols)
    masks.update(ind)
    k = td["k"].to_numpy()
    for day in (1, 2, 3, 5, 10):   # controls: the same action on a fixed day, no indicator at all
        masks[f"time:day{day}"] = k == day
    return {k: m for k, m in masks.items() if m.sum() >= 20}


def study_setup(data: dict, setup: str, cols: list[str], n_perm: int, progress) -> dict:
    book, gids = runner._book(data, setup)
    T, td = book.trades, book.td
    fwd = data["fwd"]
    O, Hh, L, C = (fwd[gids, j, :].astype(np.float64) for j in range(4))
    hz = O.shape[1]
    base = {"entry": T["entry"].to_numpy(), "stop0": T["stop0"].to_numpy(), "tgt": data["tgt"][gids],
            "ok": T["ok"].to_numpy(bool), "size": T["size"].to_numpy()}
    a = T["atr"].to_numpy()
    E, live = tc.action_tables(O, Hh, L, C, base, a, setup, data["slip"])
    live_ref = np.where(base["ok"], T["ret"].to_numpy() * T["size"].to_numpy(), 0.0)
    check = float(np.nanmax(np.abs(live - live_ref)))
    # sell at the next open must reproduce 7l's continuation value (-CV)
    kk = td["k"].to_numpy() - 1
    sell_d = E["sell"][book.sig, kk] - live[book.sig]
    okr = (kk < hz - 1) & base["ok"][book.sig]
    sell_check = float(np.nanmax(np.abs(sell_d[okr] + book.cv_eq[okr]))) if okr.any() else float("nan")
    masks = rule_masks(book, cols)
    names, B = tc.bitmasks(masks, book.sig, td["k"].to_numpy(), book.n)
    progress(f"{setup}: {book.n} trades, {len(names)} conditions x {len(E)} actions (live re-sim check {check:.2e}, sell vs -CV {sell_check:.2e})")
    risk_frac = np.where(base["ok"], (base["entry"] - base["stop0"]) / base["entry"] * base["size"], np.nan)

    def stats_of(x: np.ndarray, act: str) -> tuple[dict, np.ndarray]:
        d, fired = tc.deltas(x, E[act], live, hz)
        d_r = np.where(base["ok"] & (risk_frac > 0), d / np.where(risk_frac > 0, risk_frac, 1.0), 0.0)
        return book.stats(d, d_r, fired), d

    rules: dict = {}
    dl: dict = {}
    for j, nm in enumerate(names):
        for act in E:
            s_, d = stats_of(B[:, j], act)
            s_["family"] = "single"
            rules[f"{act} <- {nm}"] = s_
            dl[f"{act} <- {nm}"] = d
    progress(f"{setup}: {len(rules)} single rules")
    # ---------------- combinations: searched on the selection period only
    is_rows = np.flatnonzero(book.pmask["is"])
    fire_is = (B[is_rows] > 0).mean(axis=0)
    pool = [j for j, nm in enumerate(names)
            if 0.03 <= fire_is[j] <= 0.8 and "q05" not in nm and "q95" not in nm and "@" not in nm]
    t0 = time.time()
    best = tc.screen_pairs(B, pool, E, live, is_rows, hz)
    n_pairs = len(pool) * (len(pool) - 1) // 2
    progress(f"{setup}: screened {n_pairs:,} pairs x {len(E)} actions on 2020-24 ({time.time() - t0:.0f}s)")
    triples = {}
    for act, lst in best.items():
        triples[act] = tc.extend_triples(B, [(i, j) for _, i, j, _, _ in lst[:15]], pool, E[act], live, is_rows, hz)
    for act, lst in best.items():
        for t_is, i, j, _, _ in lst:
            nm = f"{act} <- {names[i]} & {names[j]}"
            s_, d = stats_of(B[:, i] & B[:, j], act)
            s_["family"] = "pair"
            s_["screen_t"] = t_is
            rules[nm], dl[nm] = s_, d
        for t_is, i, j, q, _, _ in triples[act]:
            nm = f"{act} <- {names[i]} & {names[j]} & {names[q]}"
            s_, d = stats_of(B[:, i] & B[:, j] & B[:, q], act)
            s_["family"] = "triple"
            s_["screen_t"] = t_is
            rules[nm], dl[nm] = s_, d
    g = hs.gate(rules)
    conf = book.pmask["hold"] | book.pmask["oos"]
    cand = [k for k, v in rules.items() if (v.get("is", {}).get("d_eq") or -1) > 0 and (v.get("is", {}).get("t") or 0) >= 1.65]
    if cand:
        pv, null95 = hs.maxt_permutation([book.date_means(dl[k], conf) for k in cand], n_perm=n_perm)
        for k, p_ in zip(cand, pv):
            rules[k]["maxt_p"] = float(p_)
    else:
        null95 = None
    e = {"n_trades": int(book.n), "n_conditions": len(names), "n_pool": len(pool), "n_pairs_screened": n_pairs,
         "actions": {k: tc.ACTION_TEXT[k] for k in E}, "gate": g, "maxt_null95": null95, "live_check": check, "sell_check": sell_check,
         "rules": rules, "n_candidates": len(cand)}
    progress(f"{setup}: IS candidates {g['n_is_candidates']}, bar t {g['bar_t']:.2f}, validated {g['validated'][:10]}")
    return e


def compact(res: dict, top: int = 15) -> dict:
    out = {k: v for k, v in res.items() if k != "setups"}
    out["setups"] = {}
    for s, e in res["setups"].items():
        rules = e["rules"]

        def rec(k: str) -> dict:
            v = rules[k]
            r_ = {"rule": k, "family": v.get("family"), "screen_t": _r(v.get("screen_t"), 2), "maxt_p": _r(v.get("maxt_p"), 3),
                  "pooled_t": _r(v.get("pooled", {}).get("t"), 2)}
            for p in ("is", "hold", "oos", "gfc"):
                x = v.get(p, {})
                r_[p] = [_r(x.get("d_eq")), _r(x.get("t"), 2), _r(x.get("fired"), 3), _r(x.get("d_r_fired"), 3)]
            return r_

        pt = lambda k: rules[k].get("pooled", {}).get("t") or -99  # noqa: E731
        ist = lambda k: rules[k].get("is", {}).get("t") or -99  # noqa: E731
        conf_both = [k for k in rules if (rules[k].get("hold", {}).get("d_eq") or -1) > 0 and (rules[k].get("oos", {}).get("d_eq") or -1) > 0]
        fam = {}
        for f_ in ("single", "pair", "triple"):
            ks = [k for k in rules if rules[k].get("family") == f_]
            ts = np.array([pt(k) for k in ks if pt(k) > -99])
            fam[f_] = {"n": len(ks), "pooled_t_le_-2": float((ts <= -2).mean()) if ts.size else None,
                       "pooled_t_ge_2": float((ts >= 2).mean()) if ts.size else None,
                       "is_candidates": sum(1 for k in ks if (rules[k].get("is", {}).get("d_eq") or -1) > 0 and ist(k) >= 1.65),
                       "both_confirmations_positive": sum(1 for k in ks if k in set(conf_both))}
        by_action = {}
        for act in e["actions"]:
            ks = [k for k in rules if k.startswith(act + " <- ")]
            ts = np.array([pt(k) for k in ks if pt(k) > -99])
            by_action[act] = {"n": len(ks), "median_pooled_t": float(np.median(ts)) if ts.size else None,
                              "share_t_ge_2": float((ts >= 2).mean()) if ts.size else None,
                              "share_t_le_-2": float((ts <= -2).mean()) if ts.size else None,
                              "best": rec(max(ks, key=pt)) if ks else None}
        out["setups"][s] = {k: e[k] for k in ("n_trades", "n_conditions", "n_pool", "n_pairs_screened", "actions", "gate", "maxt_null95", "n_candidates")}
        out["setups"][s].update({
            "families": fam, "by_action": by_action,
            "top_by_pooled_t": [rec(k) for k in sorted(rules, key=pt, reverse=True)[:top]],
            "top_is_candidates": [rec(k) for k in sorted([k for k in rules if (rules[k].get("is", {}).get("d_eq") or -1) > 0], key=ist, reverse=True)[:top]],
            "top_combos": [rec(k) for k in sorted([k for k in rules if rules[k].get("family") != "single"], key=pt, reverse=True)[:top]],
            "confirmed_in_both": [rec(k) for k in sorted(conf_both, key=pt, reverse=True)[:top]],
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--cache", required=True, help="holding_study.py cache directory (collect.pkl, tech.pkl)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--perm", type=int, default=500)
    ap.add_argument("--setups", default=",".join(hs.SETUPS))
    args = ap.parse_args(argv)
    t0 = time.time()
    settings = load_terminal_settings()
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    cache = Path(args.cache)
    parts, kept = pickle.loads((cache / "collect.pkl").read_bytes())
    tech = pickle.loads((cache / "tech.pkl").read_bytes()) if (cache / "tech.pkl").exists() else {}
    syms = [p["symbol"] for p in parts if "td" in p]
    print(f"{len(syms)} stocks; indicator library...", flush=True)
    extra = {}
    with ProcessPoolExecutor(args.workers, initializer=_lib_init, initargs=(args.prices,)) as ex:
        for sym, val in ex.map(_lib, syms, chunksize=4):
            extra[sym] = val
    print(f"library done ({time.time() - t0:.0f}s); assembling", flush=True)
    data = runner.assemble(parts, tech, extra=extra)
    del parts, extra
    data["slip"] = slip
    cols = runner.feature_columns(data["td"])
    print(f"assembled: {len(data['trades'])} trades, {len(data['td'])} trade-days, {len(cols)} indicator columns ({time.time() - t0:.0f}s)", flush=True)
    res = {"periods": hs.PERIODS, "n_indicator_columns": len(cols),
           "n_library": sum(c.startswith("L_") for c in cols), "library": il.columns(), "setups": {}}
    for setup in args.setups.split(","):
        res["setups"][setup] = study_setup(data, setup, cols, args.perm, progress=lambda m: print(m, flush=True))
    session = calendar.latest_completed_session()
    res.update({"generated_at": store.utc_now(), "session": session.isoformat(), "elapsed_seconds": round(time.time() - t0, 1),
                "n_stocks": len(kept), "survivorship_note": "Only stocks in today's universe are tested (survivorship bias); comparisons are paired."})
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / "trail_combo.json", json.dumps(store.jsonable(res), ensure_ascii=False))
    comp = compact(res)
    store.write_atomic(out / "trail_combo_compact.json", json.dumps(store.jsonable(comp), ensure_ascii=False, indent=1))
    print(f"Done in {time.time() - t0:.0f}s -> {out}/trail_combo.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
