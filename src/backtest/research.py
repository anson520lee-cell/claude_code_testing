"""SL / TP validation research: how accurate are the terminal's stops and targets, and what works better?

Everything is walk-forward safe: every signal, stop and target is computed from data up to the signal
day; the trade starts at the NEXT session's open. Results are split by time into an in-sample period
(first 60% of signal dates, used to choose) and an out-of-sample period (last 40%, used to judge).

1. Signals: every setup episode (first day a setup is active) across the focus universe.
2. Stop methods (long; short mirrored):
     structural  the live plan (structure + 0.25 ATR buffer, clamped to 1-3 ATR)
     atr_k       entry - k x ATR, k = 1.0 / 1.5 / 2.0 / 2.5 / 3.0
     swing       last confirmed swing low - 0.1 ATR (clamped 0.75-4 ATR)
     chip        nearest high-volume node (dense chips) or value-area low below entry - 0.25 ATR
                 (120-session volume profile at the signal date; clamped 0.75-4 ATR)
   Target methods:
     resistance  the live plan's Target 1 (nearest 20 / 60 / 252-day high >= 0.75 ATR, else 2 ATR)
     atr_m       entry + m x ATR, m = 2 / 3 / 4
     r2, r3      2R / 3R of the chosen stop
     chip        nearest high-volume node above, or the top of a vacuum zone directly above (else 2 ATR)
     trail       no fixed target: chandelier trailing stop (highest high - 3 ATR) until the time stop
   Time stops: 5 / 7 / 10 sessions.
3. Execution: stop and target checked on daily bars; a bar touching both counts as a STOP; gaps fill
   at the open; slippage on entry and exit; trades that open beyond the stop are skipped.
4. Accuracy diagnostics of the live plan: outcome rates, "stopped then target reached anyway" rate,
   MAE of winners (how much heat good trades took), MFE in R (how far trades ran), planned-R:R
   calibration against the break-even win rate, score calibration (do higher scores do better?),
   per setup / direction / regime / quarter.
Options-based stops cannot be back-tested (no free historical implied-volatility data); the volatility
proxy (HV-based 1-sigma band) is reported instead.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.analyst.techniques import volume_profile
from src.scanner.plan import plan_for_setup
from src.scanner.scoring import build_candidate
from src.scanner.setups import LONG_SETUPS, SETUP_DIRECTION, SETUP_NAMES, SHORT_SETUPS

H_MAX = 10
STOPS = ["structural", "atr1.0", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "swing", "chip"]
TARGETS = ["resistance", "atr2", "atr3", "atr4", "r2", "r3", "chip", "trail"]
HORIZONS = [5, 7, 10]
STOP_NAMES = {"structural": "Live structural stop", "atr1.0": "1.0 ATR", "atr1.5": "1.5 ATR", "atr2.0": "2.0 ATR", "atr2.5": "2.5 ATR",
              "atr3.0": "3.0 ATR", "swing": "Last swing low", "chip": "Volume node / value-area low"}
TARGET_NAMES = {"resistance": "Live resistance target", "atr2": "2 ATR", "atr3": "3 ATR", "atr4": "4 ATR", "r2": "2R", "r3": "3R",
                "chip": "Volume node / vacuum top above", "trail": "3-ATR trailing stop (no target)"}


# ====================================================================== signal collection
def _cross_ranks(features: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rs = pd.DataFrame({s: 0.5 * f["rel_20d"] + 0.3 * f["rel_60d"] + 0.2 * f["rel_5d"] for s, f in features.items()})
    sec = pd.DataFrame({s: f["rel_sector_20d"] for s, f in features.items()})
    return rs.rank(axis=1, pct=True) * 100, sec.rank(axis=1, pct=True) * 100


def _chip_levels(f: pd.DataFrame, t: int, entry: float, atr: float, long: bool) -> tuple[float, float]:
    """(chip stop level, chip target level) from the 120-session profile ending at t (NaN if none)."""
    if t < 120:
        return float("nan"), float("nan")
    vp = volume_profile(f.iloc[t - 119: t + 1])
    if not vp.get("available"):
        return float("nan"), float("nan")
    hvn = vp["hvn"]
    if long:
        below = [x for x in hvn + [vp["val"]] if x < entry - 0.5 * atr]
        above = [x for x in hvn if x > entry + 0.75 * atr]
        vac = [z["high"] for z in vp["vacuum"] if z["low"] >= entry and z["low"] - entry <= 1.5 * atr]
        stop = (max(below) - 0.25 * atr) if below else float("nan")
        target = min(above + vac) if (above or vac) else float("nan")
    else:
        above = [x for x in hvn + [vp["vah"]] if x > entry + 0.5 * atr]
        below = [x for x in hvn if x < entry - 0.75 * atr]
        vac = [z["low"] for z in vp["vacuum"] if z["high"] <= entry and entry - z["high"] <= 1.5 * atr]
        stop = (min(above) + 0.25 * atr) if above else float("nan")
        target = max(below + vac) if (below or vac) else float("nan")
    return stop, target


RAW_FACTORS = ["rsi14", "rsi6", "extension_atr", "volume_ratio", "rel_20d", "rel_60d", "rel_sector_20d", "adx",
               "bb_width_percentile", "atr_pct_percentile", "hv20", "beta120", "from_52w_high", "updown_vol_ratio", "clv",
               "gap_atr", "return_5d", "return_20d", "macd_hist_3d", "obv_slope20", "vol3_vs_mavol"]
FACTOR_NAMES = {"c_trend": "Trend sub-score", "c_momentum": "Momentum sub-score", "c_relative_strength": "Relative-strength sub-score", "c_volume": "Volume sub-score",
                "c_setup_quality": "Setup-quality sub-score", "c_reward_risk": "Reward/risk sub-score", "c_liquidity": "Liquidity sub-score",
                "c_regime_alignment": "Regime-alignment sub-score", "x_rsi14": "RSI14", "x_rsi6": "RSI6", "x_extension_atr": "Distance to EMA21 (ATR)",
                "x_volume_ratio": "Volume ratio", "x_rel_20d": "vs SPY 20d", "x_rel_60d": "vs SPY 60d", "x_rel_sector_20d": "vs sector 20d",
                "x_adx": "ADX", "x_bb_width_percentile": "Bollinger width pctile", "x_atr_pct_percentile": "ATR pctile", "x_hv20": "20d volatility",
                "x_beta120": "Beta", "x_from_52w_high": "From 52w high", "x_updown_vol_ratio": "Up/down volume", "x_clv": "Close location (CLV)",
                "x_gap_atr": "Gap (ATR)", "x_return_5d": "5d return", "x_return_20d": "20d return", "x_macd_hist_3d": "MACD hist 3d change",
                "x_obv_slope20": "OBV slope", "x_vol3_vs_mavol": "3d volume ratio", "x_regime_composite": "Regime score"}


def collect_signals(features: dict[str, pd.DataFrame], setups: dict[str, pd.DataFrame], settings: dict[str, Any],
                    regime_labels: pd.Series | None = None, progress: Callable[[str], None] | None = None,
                    regime_scores: pd.Series | None = None) -> dict[str, Any]:
    cfg = settings["scanner"]
    rs_pct, sec_pct = _cross_ranks(features)
    rows, O, Hh, L, C = [], [], [], [], []
    for n_sym, (sym, f) in enumerate(features.items()):
        st = setups[sym]
        o, h, lo, c = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        n = len(f)
        for setup in (*LONG_SETUPS, *SHORT_SETUPS):
            mask = st[f"setup_{setup}"].to_numpy(dtype=bool)
            starts = np.flatnonzero(mask & ~np.concatenate([[False], mask[:-1]]))
            starts = starts[starts + H_MAX < n]
            if starts.size == 0:
                continue
            plan = plan_for_setup(f, setup, cfg)
            long = SETUP_DIRECTION[setup] == "long"
            for t in starts:
                p = plan.iloc[t]
                atr = float(f["atr"].iat[t])
                if not (np.isfinite(atr) and atr > 0 and np.isfinite(p["stop"]) and np.isfinite(p["target1"])):
                    continue
                row = f.iloc[t]
                date = f.index[t]
                ref = float(p["entry"])
                chip_stop, chip_target = _chip_levels(f, t, ref, atr, long)
                swing = float(row["swing_low_last"] if long else row["swing_high_last"])
                plan_d = {"entry": ref, "stop": float(p["stop"]), "target1": float(p["target1"]), "target2": float(p["target2"]),
                          "reward_risk": float(p["reward_risk"]), "risk_pct": float(p["risk_pct"]), "wide_stop": bool(p["wide_stop"]),
                          "structural_stop_atr": float(p["structural_stop_atr"]), "risk_atr": float(p["risk_atr"])}
                ranks = pd.Series({"rs_pct": rs_pct.at[date, sym] if date in rs_pct.index else np.nan,
                                   "sector_rs_pct": sec_pct.at[date, sym] if date in sec_pct.index else np.nan})
                regime = regime_labels.get(date, "n/a") if regime_labels is not None else "n/a"
                try:
                    cand = build_candidate(sym, row, setup, float(st[f"quality_{setup}"].iat[t]), plan_d, ranks,
                                           {"regime": regime, "sector_quadrant": None}, settings)
                    score, tier, comps = cand["score"], cand["tier"], cand["components"]
                except Exception:  # noqa: BLE001
                    score, tier, comps = float("nan"), "n/a", {}
                factors = {f"c_{k}": v for k, v in comps.items()}
                factors.update({f"x_{k}": float(row.get(k, np.nan)) for k in RAW_FACTORS})
                factors["x_regime_composite"] = float(regime_scores.get(date, np.nan)) if regime_scores is not None else np.nan
                rows.append({"symbol": sym, "setup": setup, "long": long, "date": date, "ref": ref, "atr": atr,
                             "structural": float(p["stop"]), "resistance": float(p["target1"]), "planned_rr": float(p["reward_risk"]),
                             "swing": swing, "chip_stop": chip_stop, "chip_target": chip_target, "regime": regime,
                             "score": score, "tier": tier, "conditional": setup == "squeeze",
                             "hv20": float(row.get("hv20", np.nan)), **factors})
                O.append(o[t + 1: t + 1 + H_MAX])
                Hh.append(h[t + 1: t + 1 + H_MAX])
                L.append(lo[t + 1: t + 1 + H_MAX])
                C.append(c[t + 1: t + 1 + H_MAX])
        if progress and (n_sym + 1) % 50 == 0:
            progress(f"Collected historical signals for {n_sym + 1}/{len(features)} stocks")
    sig = pd.DataFrame(rows)
    return {"signals": sig, "O": np.array(O), "H": np.array(Hh), "L": np.array(L), "C": np.array(C)}


def collect_baseline(features: dict[str, pd.DataFrame], regime_labels: pd.Series | None = None, step: int = 5) -> dict[str, Any]:
    """Benchmark: enter EVERY ``step``-th session of every focus stock, long and short, with no setup at all.
    If a stop / target method makes money on these random entries too, the edge comes from the market drift
    (or the exit rule) - not from the setups."""
    rows, O, Hh, L, C = [], [], [], [], []
    for k, (sym, f) in enumerate(features.items()):
        o, h, lo, c, a = (f[col].to_numpy(dtype=float) for col in ("open", "high", "low", "close", "atr"))
        n = len(f)
        idx = np.arange(60 + k % step, n - H_MAX, step)
        idx = idx[np.isfinite(a[idx]) & (a[idx] > 0) & np.isfinite(c[idx])]
        sw_l, sw_h = f["swing_low_last"].to_numpy(dtype=float), f["swing_high_last"].to_numpy(dtype=float)
        hv = f["hv20"].to_numpy(dtype=float)
        for t in idx:
            date = f.index[t]
            regime = regime_labels.get(date, "n/a") if regime_labels is not None else "n/a"
            for long in (True, False):
                sgn = 1 if long else -1
                rows.append({"symbol": sym, "setup": "baseline", "long": long, "date": date, "ref": c[t], "atr": a[t],
                             "structural": c[t] - sgn * 1.5 * a[t], "resistance": c[t] + sgn * 2 * a[t], "planned_rr": np.nan,
                             "swing": sw_l[t] if long else sw_h[t], "chip_stop": np.nan, "chip_target": np.nan, "regime": regime,
                             "score": np.nan, "tier": "n/a", "conditional": False, "hv20": hv[t]})
                O.append(o[t + 1: t + 1 + H_MAX])
                Hh.append(h[t + 1: t + 1 + H_MAX])
                L.append(lo[t + 1: t + 1 + H_MAX])
                C.append(c[t + 1: t + 1 + H_MAX])
    return {"signals": pd.DataFrame(rows), "O": np.array(O), "H": np.array(Hh), "L": np.array(L), "C": np.array(C)}


def baseline_compare(res: dict[str, Any], base: dict[str, Any], slip: float, cutoff: pd.Timestamp) -> dict[str, Any]:
    """Same exit rules on random entries: the setup's real edge = setup result - baseline result."""
    out: dict[str, Any] = {}
    methods = {("atr1.5", "trail", 10), ("atr1.5", "r3", 10), ("atr2.0", "r3", 7), ("atr1.5", "r2", 7)}
    for side in ("long", "short"):
        for key in ("best_is",):
            for g in res[key][side][:3]:
                if g["stop"] not in ("structural", "chip") and g["target"] not in ("resistance", "chip"):
                    methods.add((g["stop"], g["target"], g["horizon"]))
    for sm, tm, hz in sorted(methods):
        tr = simulate(base, sm, tm, hz, slip)
        for side, flag in (("long", True), ("short", False)):
            part = tr[tr["long"] == flag]
            sig_row = next((g for g in res["grid"] if (g["stop"], g["target"], g["horizon"], g["side"]) == (sm, tm, hz, side)), None)
            b_is, b_oos = summary(part[part["date"] <= cutoff]), summary(part[part["date"] > cutoff])
            out[f"{side}|{sm}|{tm}|{hz}"] = {
                "side": side, "stop": sm, "target": tm, "horizon": hz, "baseline_is": b_is, "baseline_oos": b_oos,
                "signal_is": sig_row["is"] if sig_row else None, "signal_oos": sig_row["oos"] if sig_row else None,
                "edge_is": (sig_row["is"]["avg_r"] - b_is["avg_r"]) if sig_row and b_is.get("n") else None,
                "edge_oos": (sig_row["oos"]["avg_r"] - b_oos["avg_r"]) if sig_row and b_oos.get("n") else None}
    live_base = simulate(base, "atr1.5", "atr2", 7, slip)
    out["_note"] = "Baseline = enter the same stocks every 5th session regardless of setup, with the same stop / target rules."
    out["_baseline_n"] = int(len(live_base))
    return out


# ====================================================================== vectorised simulation
def stop_levels(sig: pd.DataFrame, entry: np.ndarray, method: str) -> np.ndarray:
    a, long = sig["atr"].to_numpy(), sig["long"].to_numpy()
    s = np.where(long, 1.0, -1.0)
    if method == "structural":
        lvl = sig["structural"].to_numpy()
    elif method.startswith("atr"):
        lvl = entry - s * float(method[3:]) * a
    else:
        raw = sig["swing"].to_numpy() - s * 0.1 * a if method == "swing" else sig["chip_stop"].to_numpy()
        dist = s * (entry - raw) / a
        lvl = np.where(np.isfinite(dist), entry - s * np.clip(dist, 0.75, 4.0) * a, sig["structural"].to_numpy())
    return lvl


def target_levels(sig: pd.DataFrame, entry: np.ndarray, stop: np.ndarray, method: str) -> np.ndarray:
    a, long = sig["atr"].to_numpy(), sig["long"].to_numpy()
    s = np.where(long, 1.0, -1.0)
    risk = s * (entry - stop)
    if method == "resistance":
        return sig["resistance"].to_numpy()
    if method.startswith("atr"):
        return entry + s * float(method[3:]) * a
    if method in ("r2", "r3"):
        return entry + s * float(method[1:]) * risk
    if method == "chip":
        ct = sig["chip_target"].to_numpy()
        return np.where(np.isfinite(ct), ct, entry + s * 2 * a)
    return np.full(len(sig), np.nan)  # trail


def simulate(data: dict[str, Any], stop_m: str, target_m: str, horizon: int, slip: float) -> pd.DataFrame:
    sig = data["signals"]
    O, H, L, C = (data[k][:, :horizon] for k in ("O", "H", "L", "C"))
    long = sig["long"].to_numpy()
    s = np.where(long, 1.0, -1.0)
    ref = sig["ref"].to_numpy()
    cond = sig["conditional"].to_numpy()
    entry_raw = np.where(cond, np.where(long, np.maximum(O[:, 0], ref), np.minimum(O[:, 0], ref)), O[:, 0])
    filled = ~cond | np.where(long, H[:, 0] >= ref, L[:, 0] <= ref)
    entry = entry_raw * (1 + s * slip)
    stop = stop_levels(sig, ref, stop_m)
    risk = s * (entry - stop)
    ok = filled & (risk > 0) & np.isfinite(risk) & np.isfinite(entry)
    a = sig["atr"].to_numpy()
    big = horizon + 1
    if target_m == "trail":
        hi_run = np.maximum.accumulate(np.where(long[:, None], H, -L), axis=1)
        prev = np.concatenate([np.where(long, entry, -entry)[:, None], hi_run[:, :-1]], axis=1)
        trail = np.where(long[:, None], np.maximum(stop[:, None], prev - 3 * a[:, None]),
                         np.minimum(stop[:, None], -prev + 3 * a[:, None]))
        hit_s = np.where(long[:, None], L <= trail, H >= trail)
        ks = np.where(hit_s.any(1), hit_s.argmax(1), big)
        kt = np.full(len(sig), big)
        stop_px = trail[np.arange(len(sig)), np.minimum(ks, horizon - 1)]
        target = np.full(len(sig), np.nan)
    else:
        target = target_levels(sig, ref, stop, target_m)
        hit_s = np.where(long[:, None], L <= stop[:, None], H >= stop[:, None])
        hit_t = np.where(long[:, None], H >= target[:, None], L <= target[:, None])
        ks = np.where(hit_s.any(1), hit_s.argmax(1), big)
        kt = np.where(hit_t.any(1), hit_t.argmax(1), big)
        stop_px = stop
    idx = np.arange(len(sig))
    stop_first = (ks <= kt) & (ks < big)
    tgt_first = (kt < ks) & (kt < big)
    ksafe, ktsafe = np.minimum(ks, horizon - 1), np.minimum(kt, horizon - 1)
    o_s, o_t = O[idx, ksafe], O[idx, ktsafe]
    ex_stop = np.where(ksafe > 0, np.where(long, np.minimum(o_s, stop_px), np.maximum(o_s, stop_px)), stop_px)
    ex_tgt = np.where(ktsafe > 0, np.where(long, np.maximum(o_t, target), np.minimum(o_t, target)), target)
    ex_time = C[:, horizon - 1]
    exit_px = np.where(stop_first, ex_stop, np.where(tgt_first, ex_tgt, ex_time))
    exit_px = exit_px * (1 - s * slip)
    pnl = s * (exit_px - entry)
    days = np.where(stop_first, ks + 1, np.where(tgt_first, kt + 1, horizon))
    inside = np.arange(horizon)[None, :] < days[:, None]
    window_hi = np.where(inside, H, -np.inf).max(1)
    window_lo = np.where(inside, L, np.inf).min(1)
    mfe = np.where(long, H.max(1) - entry, entry - L.min(1)) / np.where(risk > 0, risk, np.nan)
    mae_atr = np.where(long, window_lo - entry, entry - window_hi) / a
    after_stop_target = np.zeros(len(sig), dtype=bool)
    if target_m != "trail":
        for i in np.flatnonzero(stop_first & ok):
            k = ks[i]
            seg = H[i, k:] if long[i] else L[i, k:]
            after_stop_target[i] = bool((seg >= target[i]).any()) if long[i] else bool((seg <= target[i]).any())
    out = pd.DataFrame({
        "symbol": sig["symbol"], "setup": sig["setup"], "long": long, "date": sig["date"], "regime": sig["regime"],
        "score": sig["score"], "tier": sig["tier"], "planned_rr": sig["planned_rr"],
        "outcome": np.where(stop_first, "stop", np.where(tgt_first, "target", "time")),
        "r": pnl / risk, "ret": pnl / entry, "days": days, "mfe_r": mfe, "mae_atr": mae_atr,
        "risk_atr": risk / a, "reward_atr": s * (target - entry) / a, "stopped_then_target": after_stop_target,
        **{c: sig[c] for c in sig.columns if c.startswith(("c_", "x_"))},
    })
    return out[ok].reset_index(drop=True)


# ====================================================================== statistics
def summary(t: pd.DataFrame) -> dict[str, Any]:
    if t.empty:
        return {"n": 0}
    r = t["r"].replace([np.inf, -np.inf], np.nan).dropna()
    wins, losses = r[r > 0], r[r <= 0]
    eq = r.groupby(t.loc[r.index, "date"]).mean().sort_index().cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else float("nan")
    return {"n": int(len(t)), "win_rate": float((r > 0).mean()), "avg_r": float(r.mean()), "median_r": float(r.median()),
            "std_r": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
            "t_stat": float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))) if len(r) > 2 and r.std(ddof=1) > 0 else float("nan"),
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
            "avg_ret": float(t["ret"].mean()), "target_rate": float((t["outcome"] == "target").mean()),
            "stop_rate": float((t["outcome"] == "stop").mean()), "time_rate": float((t["outcome"] == "time").mean()),
            "avg_days": float(t["days"].mean()), "max_dd_r": dd}


def run_research(data: dict[str, Any], settings: dict[str, Any], is_fraction: float = 0.6,
                 baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    sig = data["signals"]
    if sig.empty:
        return {"available": False, "reason": "No historical signals."}
    cutoff = sig["date"].quantile(is_fraction)
    res: dict[str, Any] = {"available": True, "n_signals": int(len(sig)), "n_symbols": int(sig["symbol"].nunique()),
                           "first": sig["date"].min().strftime("%Y-%m-%d"), "last": sig["date"].max().strftime("%Y-%m-%d"),
                           "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"), "slippage_bps": slip * 10_000,
                           "stop_names": STOP_NAMES, "target_names": TARGET_NAMES, "method": __doc__}
    grid = []
    cache: dict[tuple, pd.DataFrame] = {}
    for sm in STOPS:
        for tm in TARGETS:
            for hz in HORIZONS:
                tr = simulate(data, sm, tm, hz, slip)
                cache[(sm, tm, hz)] = tr
                ins, oos = tr[tr["date"] <= cutoff], tr[tr["date"] > cutoff]
                for side, part in (("all", None), ("long", True), ("short", False)):
                    a_ = ins if part is None else ins[ins["long"] == part]
                    b_ = oos if part is None else oos[oos["long"] == part]
                    grid.append({"stop": sm, "target": tm, "horizon": hz, "side": side, "is": summary(a_), "oos": summary(b_)})
    res["grid"] = grid

    def best(side: str, k: int = 8) -> list[dict]:
        rows = [g for g in grid if g["side"] == side and g["is"].get("n", 0) >= 150]
        return sorted(rows, key=lambda g: -g["is"]["avg_r"])[:k]
    res["best_is"] = {side: best(side) for side in ("all", "long", "short")}
    live = cache[("structural", "resistance", int(settings["scanner"].get("plan_horizon_days", 7)))]
    res["live"] = {"all": summary(live), "is": summary(live[live["date"] <= cutoff]), "oos": summary(live[live["date"] > cutoff]),
                   "long": summary(live[live["long"]]), "short": summary(live[~live["long"]])}
    # selection honesty: the IS winner's out-of-sample result vs the live plan's
    top = res["best_is"]["all"][0] if res["best_is"]["all"] else None
    res["selected"] = top
    res["diagnostics"] = diagnostics(live)
    res["by_setup"] = {s: {"name": SETUP_NAMES[s], **summary(live[live["setup"] == s])} for s in live["setup"].unique()}
    res["best_by_setup"] = {}
    for s in live["setup"].unique():
        cands = []
        for (sm, tm, hz), tr in cache.items():
            part = tr[tr["setup"] == s]
            ins, oos = part[part["date"] <= cutoff], part[part["date"] > cutoff]
            if len(ins) >= 60:
                cands.append({"stop": sm, "target": tm, "horizon": hz, "is": summary(ins), "oos": summary(oos)})
        cands.sort(key=lambda g: -g["is"]["avg_r"])
        res["best_by_setup"][s] = {"name": SETUP_NAMES[s], "best": cands[:3]}
    res["by_regime"] = {k: summary(g) for k, g in live.groupby("regime")}
    res["by_regime_side"] = {f"{k[0]}|{'long' if k[1] else 'short'}": summary(g) for k, g in live.groupby(["regime", "long"])}
    res["selected_side"] = {}
    for side, flag in (("long", True), ("short", False)):
        top = res["best_is"][side][0] if res["best_is"][side] else None
        if not top:
            continue
        tr = cache[(top["stop"], top["target"], top["horizon"])]
        tr = tr[tr["long"] == flag]
        q = {str(k): summary(g)["avg_r"] for k, g in tr.groupby(tr["date"].dt.to_period("Q"))}
        fixed = [g for g in res["best_is"][side] if g["target"] != "trail"]
        res["selected_side"][side] = {**top, "quarters": q, "positive_quarters": float(np.mean([v > 0 for v in q.values()])) if q else None,
                                      "best_fixed_target": fixed[0] if fixed else None,
                                      "by_regime": {k: summary(g) for k, g in tr.groupby("regime")}}
    res["factors"] = factor_study(live, cutoff)
    if baseline is not None and len(baseline["signals"]):
        res["baseline"] = baseline_compare(res, baseline, slip, cutoff)
    res["by_quarter"] = {str(k): summary(g) for k, g in live.groupby(live["date"].dt.to_period("Q"))}
    res["score_calibration"] = score_calibration(live)
    res["verdict"] = verdict(res)
    return res


def diagnostics(t: pd.DataFrame) -> dict[str, Any]:
    """How accurate the live stop / target were."""
    if t.empty:
        return {}
    winners = t[t["outcome"] == "target"]
    stopped = t[t["outcome"] == "stop"]
    mae_w = (-winners["mae_atr"]).clip(lower=0).dropna()   # heat taken by eventual winners, in ATR (positive)
    mfe = t["mfe_r"].replace([np.inf, -np.inf], np.nan).dropna()
    rr_buckets = pd.cut(t["planned_rr"], [0, 1, 1.5, 2, 3, np.inf], labels=["<1", "1–1.5", "1.5–2", "2–3", "≥3"])
    calib = []
    for b, g in t.groupby(rr_buckets, observed=True):
        mid = {"<1": 0.75, "1–1.5": 1.25, "1.5–2": 1.75, "2–3": 2.5, "≥3": 3.5}[str(b)]
        calib.append({"bucket": str(b), "n": int(len(g)), "target_rate": float((g["outcome"] == "target").mean()),
                      "breakeven": 1 / (1 + mid), "avg_r": float(g["r"].mean())})
    return {
        "stopped_then_target": float(stopped["stopped_then_target"].mean()) if len(stopped) else None,
        "winner_mae_atr": {f"p{q}": float(mae_w.quantile(q / 100)) for q in (50, 75, 90)} if len(mae_w) else {},
        "mfe_r": {"p50": float(mfe.median()), "ge1": float((mfe >= 1).mean()), "ge2": float((mfe >= 2).mean()), "ge3": float((mfe >= 3).mean())},
        "avg_risk_atr": float(t["risk_atr"].mean()), "avg_reward_atr": float(t["reward_atr"].mean()),
        "rr_calibration": calib,
        "time_exits_positive": float((t.loc[t["outcome"] == "time", "r"] > 0).mean()) if (t["outcome"] == "time").any() else None,
    }


def score_calibration(t: pd.DataFrame) -> dict[str, Any]:
    s = t.dropna(subset=["score"])
    if len(s) < 50:
        return {}
    q = pd.qcut(s["score"].rank(method="first"), 5, labels=["Bottom 20%", "20–40%", "40–60%", "60–80%", "Top 20%"])
    by_q = [{"bucket": str(k), "score_range": f"{g['score'].min():.0f}–{g['score'].max():.0f}", **summary(g)} for k, g in s.groupby(q, observed=True)]
    by_tier = [{"tier": k, **summary(g)} for k, g in s.groupby("tier")]
    rho = float(s["score"].rank().corr(s["r"].rank()))
    return {"quintiles": by_q, "tiers": by_tier, "spearman": rho}


def verdict(res: dict[str, Any]) -> list[str]:
    out = []
    live = res["live"]
    d = res["diagnostics"]
    o = live["oos"]
    out.append(f"Live SL/TP (structural stop + resistance target + 7 days): {live['all']['n']} simulated trades, win rate {live['all']['win_rate'] * 100:.0f}%, "
               f"average {live['all']['avg_r']:+.2f}R; out of sample {o.get('n', 0)} trades averaging {o.get('avg_r', float('nan')):+.2f}R "
               f"(t {o.get('t_stat', float('nan')):.1f}; |t| < 2 means not distinguishable from zero).")
    if d.get("stopped_then_target") is not None:
        out.append(f"{d['stopped_then_target'] * 100:.0f}% of stopped-out trades still reached the target within the holding period (the higher, the tighter the stop).")
    if d.get("winner_mae_atr"):
        out.append(f"Trades that hit the target took a median heat of {d['winner_mae_atr']['p50']:.2f} ATR on the way (90% under {d['winner_mae_atr']['p90']:.2f} ATR); "
                   f"the live stop averages {d['avg_risk_atr']:.2f} ATR.")
    if d.get("mfe_r"):
        out.append(f"Median best open profit (MFE) {d['mfe_r']['p50']:.2f}R; {d['mfe_r']['ge2'] * 100:.0f}% ever reached 2R, "
                   f"{d['mfe_r']['ge3'] * 100:.0f}% reached 3R.")
    sel = res.get("selected")
    if sel:
        out.append(f"Best in-sample combination: {STOP_NAMES[sel['stop']]} + {TARGET_NAMES[sel['target']]} + {sel['horizon']} days: "
                   f"in sample {sel['is']['avg_r']:+.2f}R, out of sample {sel['oos'].get('avg_r', float('nan')):+.2f}R "
                   f"({sel['oos'].get('n', 0)} trades). " + ("Still better than the live plan out of sample." if sel["oos"].get("avg_r", -9) > o.get("avg_r", -9)
                                                        else "Not better than the live plan out of sample - overfitted, not recommended."))
    for side, zh in (("long", "Long"), ("short", "Short")):
        ss = (res.get("selected_side") or {}).get(side)
        if not ss:
            continue
        line = (f"{zh} best: {STOP_NAMES[ss['stop']]} + {TARGET_NAMES[ss['target']]} + {ss['horizon']} days: in sample {ss['is']['avg_r']:+.2f}R, "
                f"out of sample {ss['oos'].get('avg_r', float('nan')):+.2f}R (t {ss['oos'].get('t_stat', float('nan')):.1f}), "
                f"{(ss.get('positive_quarters') or 0) * 100:.0f}% of quarters positive. ")
        b_ = (res.get("baseline") or {}).get(f"{side}|{ss['stop']}|{ss['target']}|{ss['horizon']}")
        if b_ and b_.get("edge_oos") is not None:
            line += (f"Random entries with the same exits: {b_['baseline_oos']['avg_r']:+.2f}R out of sample, so the setups add only {b_['edge_oos']:+.2f}R"
                     + ("." if b_["edge_oos"] > 0.03 else " - the result comes from market drift and the exit rule, not the setups."))
        out.append(line)
    sc = res.get("score_calibration") or {}
    if sc.get("quintiles"):
        lo, hi = sc["quintiles"][0], sc["quintiles"][-1]
        out.append(f"Score calibration: bottom-20% signals average {lo['avg_r']:+.2f}R, top 20% {hi['avg_r']:+.2f}R, "
                   f"rank correlation {sc['spearman']:+.2f}. " + ("Higher scores do better - the score has predictive power." if hi["avg_r"] > lo["avg_r"] + 0.1 and sc["spearman"] > 0.03
                                                       else "Scores barely relate to outcomes - the live score has little predictive power."))
    return out


def factor_study(t: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    """Information coefficient (Spearman rank correlation with the trade's R) of every score component and raw
    indicator, in-sample and out-of-sample, per direction; plus a data-calibrated score fitted on the
    in-sample period only and judged out-of-sample by quintile."""
    cols = [c for c in t.columns if c.startswith(("c_", "x_"))]
    out: dict[str, Any] = {}
    for side, flag in (("long", True), ("short", False)):
        part = t[t["long"] == flag]
        ins, oos = part[part["date"] <= cutoff], part[part["date"] > cutoff]
        rows = []
        for c in cols:
            a_, b_ = ins[[c, "r"]].dropna(), oos[[c, "r"]].dropna()
            if len(a_) < 200 or len(b_) < 100 or a_[c].nunique() < 5:
                continue
            ic_is = float(a_[c].rank().corr(a_["r"].rank()))
            ic_oos = float(b_[c].rank().corr(b_["r"].rank()))
            rows.append({"factor": c, "name": FACTOR_NAMES.get(c, c), "ic_is": ic_is, "ic_oos": ic_oos,
                         "stable": bool(np.sign(ic_is) == np.sign(ic_oos) and abs(ic_is) >= 0.02)})
        rows.sort(key=lambda r: -abs(r["ic_is"]))
        # calibrated score: z-scored factors weighted by in-sample IC (|IC| >= 0.02), fitted on IS only
        chosen = [r for r in rows if abs(r["ic_is"]) >= 0.02][:10]
        calib = {}
        if chosen:
            def zscore(df: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({r["factor"]: ((df[r["factor"]] - ins[r["factor"]].mean()) / (ins[r["factor"]].std() or 1)).clip(-3, 3).fillna(0)
                                     for r in chosen})
            w = pd.Series({r["factor"]: r["ic_is"] for r in chosen})
            o_score = (zscore(oos) * w).sum(axis=1)
            i_score = (zscore(ins) * w).sum(axis=1)
            q = pd.qcut(o_score.rank(method="first"), 5, labels=False)
            calib = {"weights": {r["factor"]: r["ic_is"] for r in chosen},
                     "oos_quintiles": [summary(oos[q == k]) for k in range(5)],
                     "is_quintiles": [summary(ins[pd.qcut(i_score.rank(method="first"), 5, labels=False) == k]) for k in range(5)],
                     "oos_spearman": float(o_score.rank().corr(oos["r"].rank())),
                     "means": {r["factor"]: float(ins[r["factor"]].mean()) for r in chosen},
                     "stds": {r["factor"]: float(ins[r["factor"]].std() or 1) for r in chosen}}
        out[side] = {"factors": rows, "calibrated": calib}
    return out
