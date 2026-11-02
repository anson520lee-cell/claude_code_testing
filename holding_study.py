"""Exit signals for HELD positions: candles, volume, 300+ indicators, trade state (research only).

See src/backtest/holding_study.py for the design and the pre-registered gate.

    python holding_study.py --prices prices.pkl [--hourly hourly.pkl] [--cache dir] [--workers 4] [--no-tech]

prices.pkl  = pickled {symbol: standard daily price frame} from Yahoo since 2003 (the database keeps 5 years only)
hourly.pkl  = pickled {symbol: 60-minute regular-hours bars, New York index, columns open/high/low/close/volume}
Writes reports/backtest/<session>/holding_study.json and config/holding_study.default.json (compact copy used by the
holdings page when no newer run exists).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import holding_study as hs
from src.daily import store
from src.daily.settings import load_terminal_settings, path
from src.market import calendar

_G: dict = {}


def _init(prices_file: str, hourly_file: str | None) -> None:
    _G["prices"] = pickle.loads(Path(prices_file).read_bytes())
    _G["hourly"] = pickle.loads(Path(hourly_file).read_bytes()) if hourly_file else {}
    _G["settings"] = load_terminal_settings()
    p = _G["prices"]
    _G["mkt"] = hs.market_frame(p["SPY"], p.get("^VIX"))


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
    sym, si, slip = args
    f, st = _features(sym)
    raw = _G["prices"][sym]
    hb = _G["hourly"].get(sym)
    return hs.collect_stock(sym, f, st, _G["settings"], _G["mkt"], si, slip, hourly=hb, raw=raw)


def _tech(args):
    from src.scanner.model_features import technique_features
    sym, ts = args
    f, _ = _features(sym)
    spy = _G["prices"]["SPY"]["adj_close"]
    spy = spy.reindex(f.index).ffill()
    rows = {}
    for t in ts:
        rows[int(t)] = technique_features(f, int(t), spy)
    return sym, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--hourly")
    ap.add_argument("--cache", help="directory for intermediate pickles (collection, technique features)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-tech", action="store_true", help="skip the analyst-technique features (49 techniques + volume profile)")
    ap.add_argument("--limit", type=int, help="first N stocks only (development)")
    ap.add_argument("--perm", type=int, default=500)
    args = ap.parse_args(argv)
    t0 = time.time()
    settings = load_terminal_settings()
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    cache = Path(args.cache) if args.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    prices = pickle.loads(Path(args.prices).read_bytes())
    syms = [s for s in settings["universe"] if s in prices and len(prices[s]) >= 300]
    init = (args.prices, args.hourly)
    coll_file = cache / "collect.pkl" if cache else None
    if coll_file and coll_file.exists():
        parts, kept = pickle.loads(coll_file.read_bytes())
    else:
        from src.scanner.universe_filter import apply_universe_filter
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as ex:
            last = dict(ex.map(_filter_row, syms, chunksize=4))
        kept = apply_universe_filter({s: f for s, f in last.items() if f is not None}, settings, {})["kept"]
        kept = [s for s in syms if s in kept][: args.limit or None]
        print(f"{len(kept)} stocks after the focus filter; collecting...", flush=True)
        parts = []
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as ex:
            for i, part in enumerate(ex.map(_collect, [(s, i, slip) for i, s in enumerate(kept)], chunksize=1)):
                parts.append(part)
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{len(kept)} stocks, {time.time() - t0:.0f}s", flush=True)
        if coll_file:
            coll_file.write_bytes(pickle.dumps((parts, kept), protocol=pickle.HIGHEST_PROTOCOL))
    # ---------------- technique features for setup trade-days k <= 5 (expensive: ~0.1 s per stock-day)
    tech: dict[str, dict[int, dict]] = {}
    if not args.no_tech:
        tech_file = cache / "tech.pkl" if cache else None
        if tech_file and tech_file.exists():
            tech = pickle.loads(tech_file.read_bytes())
        else:
            jobs = []
            for part in parts:
                if "td" not in part:
                    continue
                tr, td = part["trades"], part["td"]
                setup = tr["setup"].to_numpy()[td["sig"].to_numpy()]
                m = (setup != "random") & (td["k"].to_numpy() <= hs.TECH_MAX_DAY)
                jobs.append((part["symbol"], sorted(set(part["td_t"][m].tolist()))))
            print(f"technique features for {sum(len(j[1]) for j in jobs)} stock-days...", flush=True)
            with ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as ex:
                for i, (sym, rows) in enumerate(ex.map(_tech, jobs, chunksize=1)):
                    tech[sym] = rows
                    if (i + 1) % 20 == 0:
                        print(f"  tech {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
            if tech_file:
                tech_file.write_bytes(pickle.dumps(tech, protocol=pickle.HIGHEST_PROTOCOL))
    data = assemble(parts, tech)
    data["slip"] = slip
    print(f"assembled: {len(data['trades'])} trades, {len(data['td'])} trade-days, {data['td'].shape[1]} columns "
          f"({time.time() - t0:.0f}s)", flush=True)
    res = run(data, args.perm, progress=lambda m: print(m, flush=True))
    session = calendar.latest_completed_session()
    res.update({"generated_at": store.utc_now(), "session": session.isoformat(), "elapsed_seconds": round(time.time() - t0, 1),
                "n_stocks": len(kept),
                "survivorship_note": "Only stocks in today's universe are tested (survivorship bias); comparisons are paired."})
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / "holding_study.json", json.dumps(store.jsonable(res), ensure_ascii=False))
    comp = compact(res)
    store.write_atomic(out / "holding_study_compact.json", json.dumps(store.jsonable(comp), ensure_ascii=False, indent=1))
    print(f"Done in {time.time() - t0:.0f}s -> {out}/holding_study.json", flush=True)
    return 0


def assemble(parts: list[dict], tech: dict[str, dict[int, dict]]) -> dict:
    trades, tds, unis, firsts, fwds, tgts, unih = [], [], [], [], [], [], []
    off = 0
    for part in parts:
        if "uni" in part:
            unis.append(part["uni"])
        if part.get("uni_h"):
            unih.append(part["uni_h"])
        if "trades" not in part:
            continue
        tr = part["trades"].copy()
        tr["gid"] = np.arange(len(tr)) + off
        trades.append(tr)
        fwds.append(part["fwd_ohlc"])
        tgts.append(part["tgt"])
        if "td" in part:
            td = part["td"].copy()
            if tech:
                rows = tech.get(part["symbol"], {})
                keys = sorted({k for r in list(rows.values())[:50] for k in r})
                if keys:
                    T = np.full((len(td), len(keys)), np.nan, dtype=np.float32)
                    for j, t in enumerate(part["td_t"]):
                        r = rows.get(int(t))
                        if r:
                            T[j] = [r.get(k, np.nan) for k in keys]
                    td = pd.concat([td, pd.DataFrame(T, columns=keys)], axis=1)
            hrs = part.get("hourly") or {}
            if hrs:
                td = pd.concat([td, hrs["hd"], hrs["extra"]], axis=1)
                fr = pd.DataFrame(hrs["first_cv_eq"])
                fr["gid"] = tr["gid"].to_numpy()
                firsts.append(fr)
            td["sig"] = td["sig"].to_numpy() + off
            tds.append(td)
        off += len(tr)
    T = pd.concat(trades, ignore_index=True)
    TD = pd.concat(tds, ignore_index=True, sort=False)
    TD = TD.sort_values(["sig", "k"], kind="stable").reset_index(drop=True)
    U = {k: np.concatenate([u[k] for u in unis]) for k in unis[0]} if unis else {}
    F = pd.concat(firsts, ignore_index=True, sort=False) if firsts else pd.DataFrame()
    UH = {k: np.concatenate([u[k] for u in unih]) for k in unih[0]} if unih else {}
    return {"trades": T, "td": TD, "uni": U, "uni_h": UH, "hourly_first": F, "fwd": np.concatenate(fwds), "tgt": np.concatenate(tgts)}


def _book(data: dict, setup: str, periods=hs.PERIODS):
    T, TD = data["trades"], data["td"]
    m = (T["setup"] == setup).to_numpy()
    gids = T["gid"].to_numpy()[m]
    remap = np.full(len(T), -1)
    remap[gids] = np.arange(m.sum())
    sub = TD[np.isin(TD["sig"].to_numpy(), gids)].copy()
    sub["sig"] = remap[sub["sig"].to_numpy()]
    return hs.Book(T[m], sub, periods), gids


def feature_columns(td: pd.DataFrame) -> list[str]:
    skip = {"sig", "k", "cv_r", "cv_eq", "fwd1_atr", "fwd3_atr", "fwd5_atr"}
    return [c for c in td.columns if c not in skip and not c.startswith("hd_") and pd.api.types.is_numeric_dtype(td[c])]


def run(data: dict, n_perm: int = 500, progress=print) -> dict:
    res: dict = {"setups": {}, "periods": hs.PERIODS, "hourly_periods": hs.HPERIODS,
                 "counts": data["trades"]["setup"].value_counts().to_dict()}
    cols = feature_columns(data["td"])
    res["n_features"] = len(cols)
    res["feature_groups"] = pd.Series([c.split("_")[0] if not c.startswith(("w_", "wtd")) else "w" for c in cols]).value_counts().to_dict()
    res["universe_patterns"] = hs.universe_patterns(data["uni"]) if data["uni"] else {}
    res["universe_hourly_patterns"] = hs.universe_hourly_patterns(data["uni_h"]) if data.get("uni_h") else {}
    progress(f"universe patterns done; {len(cols)} features")
    rbook, _ = _book(data, "random")
    for setup in hs.SETUPS:
        book, gids = _book(data, setup)
        td = book.td
        e: dict = {"n_trades": int(book.n), "n_trade_days": int(len(td)),
                   "live": {p: {"avg_eq": float(np.nanmean((book.trades["ret"] * book.trades["size"]).to_numpy()[m]) * 100),
                                "avg_r": float(np.nanmean(book.trades["r"].to_numpy()[m])), "n": int(m.sum())} for p, m in book.pmask.items()}}
        masks = {}
        masks.update(hs.pattern_masks(td))
        masks.update(hs.hourly_masks(td))
        fam_pat = dict(masks)
        cl, stt = hs.classic_masks(td), hs.state_masks(td)
        ic, ind_masks, thr = hs.indicator_scan(book, cols)
        rules, deltas = {}, {}
        for fam, mm, ctx in (("pattern", fam_pat, True), ("classic", cl, False), ("state", stt, False), ("indicator", ind_masks, False)):
            st_, dl = hs.evaluate_masks(book, mm, contexts=ctx)
            for k, v in st_.items():
                v["family"] = fam
            rules.update(st_)
            deltas.update(dl)
        # stop -> signal bar's low (tighten instead of selling) for every candle variant
        rules_t = tighten_rules(data, book, gids, fam_pat)
        rules.update(rules_t)
        progress(f"{setup}: {len(rules)} rules evaluated")
        # gate: daily rules use is / hold / oos; hourly_eod rules only have is_h / oos_h data -> separate gate
        daily = {k: v for k, v in rules.items() if not k.startswith("hourly_eod:")}
        e["gate"] = hs.gate(daily)
        hd_rules = {k: v for k, v in rules.items() if k.startswith("hourly_eod:")}
        # max-t on the pooled confirmation sample
        conf = book.pmask["hold"] | book.pmask["oos"]
        names = [k for k in daily if k in deltas]
        pvals, null95 = hs.maxt_permutation([book.date_means(deltas[k], conf) for k in names], n_perm=n_perm)
        for k, p_ in zip(names, pvals):
            rules[k]["maxt_p"] = float(p_)
        e["maxt_null95"] = null95
        # hourly: end-of-day mode and intraday mode, gate on is_h -> oos_h
        hb = hs.Book(book.trades, td, hs.HPERIODS)
        hrules = {}
        for k, m in {k: v for k, v in hs.hourly_masks(td).items()}.items():
            for tag, cm in (("", None), ("|in_profit", td["s_open_r"].to_numpy(float) >= hs.PROFIT_R)):
                mm = m if cm is None else m & cm
                if mm.sum() < 20:
                    continue
                d_eq, d_r, fired = hb.delta_exit(mm)
                s_ = hb.stats(d_eq, d_r, fired)
                s_["pooled"] = s_.get("oos_h", {})
                hrules[k + tag] = s_
        F = data["hourly_first"]
        if not F.empty:
            sub = F[F["gid"].isin(gids)].set_index("gid").reindex(gids)
            hrules.update(hs.hourly_rules(hb, {c: sub[c].to_numpy(float) for c in sub.columns}))
        e["hourly_rules"] = hrules
        e["hourly_gate"] = hs.gate(hrules, is_key="is_h", conf_keys=("oos_h",), pooled_key="oos_h")
        e["hourly_n_trades"] = {p: int(m.sum()) for p, m in hb.pmask.items()}
        e["rules"] = rules
        e["ic"] = ic
        e["thresholds"] = thr
        e["conditional_cv"] = hs.conditional_cv(book, {**{k: v for k, v in fam_pat.items() if "@" not in k and not k.startswith("hourly")},
                                                       **{k: v for k, v in hs.hourly_masks(td).items() if "@" not in k}, **cl})
        e["day_profile"] = hs.day_profile(book)
        rp = hs.ridge_policy(book, cols)
        e["model"] = {k: v for k, v in rp.items() if k != "deltas"}
        # same rules on random entries (market property vs setup property): pattern + classic + state, no contexts
        rmasks = {**hs.pattern_masks(rbook.td), **hs.classic_masks(rbook.td), **hs.state_masks(rbook.td)}
        e["random_rules"] = hs.evaluate_masks(rbook, {k: v for k, v in rmasks.items() if k in rules}, contexts=False)[0] if setup == "crash_rebound" else None
        res["setups"][setup] = e
        progress(f"{setup}: IS candidates {e['gate']['n_is_candidates']}, bar t {e['gate']['bar_t']:.2f}, validated {e['gate']['validated'][:10]}"
                 f"; hourly validated {e['hourly_gate']['validated'][:10]}; model pooled {rp['stats'].get('pooled')}")
    return res


def tighten_rules(data: dict, book: hs.Book, gids: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """Instead of selling at the first firing, raise the stop to that bar's low - 0.1 ATR (daily candles, any volume)."""
    T = book.trades
    setup = T["setup"].iloc[0]
    # forward OHLC per trade is kept per stock in the collection; rebuild from the trade-day table is not possible, so
    # the runner stores it in data["fwd"] when available
    fwd = data.get("fwd")
    if fwd is None:
        return {}
    O, Hh, L, C = (fwd[gids, j, :].astype(np.float64) for j in range(4))
    base = {"entry": T["entry"].to_numpy(), "stop0": T["stop0"].to_numpy(), "tgt": data["tgt"][gids],
            "ok": T["ok"].to_numpy(bool), "size": T["size"].to_numpy()}
    live_eq = T["ret"].to_numpy() * T["size"].to_numpy()
    a = T["atr"].to_numpy()
    out = {}
    sig, k = book.sig, book.td["k"].to_numpy() - 1
    for name, m in masks.items():
        if "@" in name or name.startswith("hourly"):
            continue
        ff = hs.first_fire(m, sig, book.n)
        fired = ff >= 0
        if fired.sum() < 20:
            continue
        kf = np.where(fired, k[np.maximum(ff, 0)], -1)
        lowf = np.where(fired, L[np.arange(book.n), np.maximum(kf, 0)] - 0.1 * a, np.nan)
        eq = hs.resim_from(O, Hh, L, C, base, a, setup, kf, lowf, slip=data.get("slip", 0.001))
        d = np.where(base["ok"], eq - live_eq, 0.0)
        risk_frac = np.where(base["ok"], (T["entry"].to_numpy() - T["stop0"].to_numpy()) / T["entry"].to_numpy() * base["size"], np.nan)
        d_r = np.where(base["ok"] & (risk_frac > 0), d / np.where(risk_frac > 0, risk_frac, 1.0), 0.0)   # account change -> R
        s_ = book.stats(d, d_r, fired)
        s_["family"] = "tighten"
        out[name.replace("candle:", "tighten_to_low:").replace("mtf:", "tighten_to_low:mtf:")] = s_
    return out


def compact(res: dict, top: int = 12) -> dict:
    """Small copy for config/holding_study.default.json: everything the holdings page needs + the headline tables."""
    nz = lambda x: -9.0 if x is None or x != x else x  # noqa: E731
    out = {k: res.get(k) for k in ("session", "generated_at", "counts", "periods", "hourly_periods", "n_features",
                                   "feature_groups", "n_stocks", "survivorship_note")}
    out["universe_hourly_patterns"] = res.get("universe_hourly_patterns")
    up = res.get("universe_patterns") or {}
    out["universe_patterns"] = {k: {p: {kk: v[p][kk] for kk in ("n", "x1", "t1", "x5", "t5", "x10", "t10") if kk in v[p]}
                                    for p in ("hold", "is", "oos") if p in v} | {"freq": v.get("freq")}
                                for k, v in up.items()}
    out["setups"] = {}
    for setup, e in res["setups"].items():
        rules = e["rules"]

        def brief(nm: str, rr=rules) -> dict:
            v = rr[nm]
            return {"rule": nm, **{p: v.get(p) for p in ("is", "hold", "oos", "pooled", "is_h", "oos_h") if p in v},
                    "maxt_p": v.get("maxt_p"), "family": v.get("family")}
        fam = {}
        for nm, v in rules.items():
            f_ = fam.setdefault(v.get("family", "?"), {"n": 0, "n_pooled_pos": 0, "mean_pooled_d_eq": 0.0})
            f_["n"] += 1
            pd_ = (v.get("pooled") or {}).get("d_eq")
            if pd_ is not None and pd_ == pd_:
                f_["mean_pooled_d_eq"] += pd_
                f_["n_pooled_pos"] += int(pd_ > 0)
        for f_ in fam.values():
            f_["mean_pooled_d_eq"] /= max(1, f_["n"])
        hr = e.get("hourly_rules") or {}
        daily = [k for k in rules if not k.startswith("hourly_eod:")]   # hourly rules only have 2024+ data: own table

        def rec(v: dict, periods: tuple[str, ...]) -> dict:
            """Compact per-rule record for the holdings page: pooled d (account %) / t + each period's d, t, fired."""
            r_ = {"pd": _r((v.get("pooled") or {}).get("d_eq")), "pt": _r((v.get("pooled") or {}).get("t"), 2)}
            zero_r = all(not (v.get(p) or {}).get("d_r_fired") for p in periods)   # older runs: tighten rules had no R
            for p in periods:
                x = v.get(p) or {}
                if "d_eq" in x:
                    r_[p] = [_r(x.get("d_eq")), _r(x.get("t"), 2), _r(x.get("fired"), 3), None if zero_r else _r(x.get("d_r_fired"), 3)]
            if v.get("maxt_p") is not None:
                r_["maxt_p"] = _r(v["maxt_p"], 3)
            return r_
        lookup = {k: rec(v, ("is", "hold", "oos")) for k, v in rules.items()
                  if v.get("family") in ("pattern", "classic", "state", "tighten") and not k.startswith("hourly_eod:")}
        hourly_lookup = {k: rec(v, ("is_h", "oos_h")) for k, v in hr.items()}
        out["setups"][setup] = {
            "lookup": lookup, "hourly_lookup": hourly_lookup,
            "n_trades": e["n_trades"], "n_trade_days": e["n_trade_days"], "live": e["live"], "gate": e["gate"],
            "maxt_null95": e.get("maxt_null95"), "hourly_gate": e.get("hourly_gate"), "hourly_n_trades": e.get("hourly_n_trades"),
            "families": fam,
            "top_by_pooled_t": [brief(k) for k in sorted(daily, key=lambda k: -nz((rules[k].get("pooled") or {}).get("t")))[:top]],
            "top_by_is_t": [brief(k) for k in sorted(daily, key=lambda k: -nz((rules[k].get("is") or {}).get("t")))[:top]],
            "worst_by_pooled_t": [brief(k) for k in sorted(daily, key=lambda k: nz((rules[k].get("pooled") or {}).get("t")))[:5]],
            "hourly_top": [brief(k, hr) for k in sorted(hr, key=lambda k: -nz((hr[k].get("oos_h") or {}).get("t")))[:top]],
            "day_profile": e["day_profile"], "model": e["model"],
            "conditional_cv": e["conditional_cv"],
            "ic_top": sorted(({"feature": k, **v} for k, v in e["ic"].items() if "is" in v and "hold" in v and "oos" in v),
                             key=lambda r: -abs(r["is"]))[:25],
            "ic_consistent": [{"feature": k, **v} for k, v in e["ic"].items()
                              if all(p in v for p in ("is", "hold", "oos")) and np.sign(v["is"]) == np.sign(v["hold"]) == np.sign(v["oos"])
                              and min(abs(v["is"]), abs(v["hold"]), abs(v["oos"])) >= 0.03],
        }
    return out


def _r(x, nd: int = 4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if x == x and abs(x) != float("inf") else None


if __name__ == "__main__":
    sys.exit(main())
