"""Scaling-in study (research only): does entering in tranches beat a single entry?

For the three setups the live terminal keeps (crash_rebound, squeeze, breakout) every setup episode and a
random-entry sample (every 10th session of every stock) is simulated with the exit the strategy lab chose
for that setup, once with a single entry and once for every scaling scheme below.

Risk is fixed per trade: the whole trade has a budget of 1R (= 1% of the account). Tranche i gets the
fraction w_i of that budget and is sized on its own fill price and the stop in force when it fills:
    shares_i = w_i / (fill_i - stop_at_fill)
A tranche that never triggers leaves its budget unused (less risk taken), so every scheme is compared on
R per 1R budget, the same unit as a single entry. All tranches share one stop (the setup's stop, or the
3-ATR trailing stop anchored on the first fill), one target and one time stop counted from the first fill.

Triggers (the first tranche is always the normal entry at the next open; squeeze = buy-stop):
    time      buy at the open of session d after the first fill (only if still in the trade)
    dip       buy limit x ATR below the first fill, valid for `window` sessions (pullback add)
    breakout  buy stop x ATR above the first fill, valid for `window` sessions (intraday confirmation)
    closeup   buy at the next open after a close >= first fill + x ATR (end-of-day confirmation)
A tranche is cancelled if its fill would sit less than 0.5 ATR above the stop (avoids huge positions).
Conservative intraday order: a tranche that fills on a bar whose low also hits the stop is stopped out.

Judgement: per setup the scheme is chosen on the in-sample dates (same 60/40 split as the strategy lab)
by the paired difference (scheme R - single R on the same trade). It is adopted only if out of sample the
paired difference is > 0 with t >= 2.5 (t on per-date means, so trades on the same day are not counted
as independent) and it also beats single entry in sample. Random entries run the same schemes: if a
scheme helps random entries as much, it is a property of the exit/market, not of the setup.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.scanner.plan import plan_for_setup

H = 20
SETUPS = ("crash_rebound", "squeeze", "breakout")
BASE_STEP = 10
MIN_DIST_ATR = 0.5


# ====================================================================== collection
def collect(features: dict[str, pd.DataFrame], setups: dict[str, pd.DataFrame], settings: dict[str, Any]) -> dict:
    rows, fwd = [], []
    cfg = settings["scanner"]
    for k, (sym, f) in enumerate(features.items()):
        st = setups[sym]
        o, h, lo, c, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "atr"))
        n = len(f)

        def add(t: int, setup: str, ref: float) -> None:
            rows.append({"symbol": sym, "setup": setup, "date": f.index[t], "ref": ref, "atr": a[t],
                         "conditional": setup == "squeeze"})
            fwd.append(np.stack([o[t + 1:t + 1 + H], h[t + 1:t + 1 + H], lo[t + 1:t + 1 + H], c[t + 1:t + 1 + H]]))
        for setup in SETUPS:
            p = plan_for_setup(f, setup, cfg)
            mask = st[f"setup_{setup}"].to_numpy(dtype=bool)
            starts = np.flatnonzero(mask & ~np.concatenate([[False], mask[:-1]]))
            for t in starts[(starts >= 260) & (starts + H < n)]:
                if np.isfinite(a[t]) and a[t] > 0 and np.isfinite(p["entry"].iat[t]):
                    add(t, setup, float(p["entry"].iat[t]))
        for t in range(260 + k % BASE_STEP, n - H, BASE_STEP):
            if np.isfinite(a[t]) and a[t] > 0:
                add(t, "baseline", c[t])
    sig = pd.DataFrame(rows)
    sig["date"] = pd.to_datetime(sig["date"])
    arr = np.array(fwd)
    return {"signals": sig, "O": arr[:, 0], "H": arr[:, 1], "L": arr[:, 2], "C": arr[:, 3]}


# ====================================================================== schemes
def schemes() -> dict[str, list[tuple]]:
    """name -> tranches [(weight, kind, x, window_or_day)], the first tranche is ('first')."""
    out: dict[str, list[tuple]] = {"single": [(1.0, "first", 0, 0)]}
    for w in (0.25, 0.33, 0.5, 0.67):
        r = round(1 - w, 2)
        for d in (1, 2, 3, 5):
            out[f"time {int(w*100)}/{int(r*100)} day+{d}"] = [(w, "first", 0, 0), (r, "time", 0, d)]
        for x in (0.5, 1.0, 1.5):
            for win in (3, 10):
                out[f"dip {int(w*100)}/{int(r*100)} -{x}ATR {win}d"] = [(w, "first", 0, 0), (r, "dip", x, win)]
                out[f"breakout {int(w*100)}/{int(r*100)} +{x}ATR {win}d"] = [(w, "first", 0, 0), (r, "breakout", x, win)]
                out[f"closeup {int(w*100)}/{int(r*100)} +{x}ATR {win}d"] = [(w, "first", 0, 0), (r, "closeup", x, win)]
    for ws in ((0.34, 0.33, 0.33), (0.5, 0.3, 0.2), (0.2, 0.3, 0.5)):
        lab = "/".join(str(int(round(v * 100))) for v in ws)
        out[f"time {lab} day+2,+4"] = [(ws[0], "first", 0, 0), (ws[1], "time", 0, 2), (ws[2], "time", 0, 4)]
        out[f"dip {lab} -0.5,-1.0ATR 10d"] = [(ws[0], "first", 0, 0), (ws[1], "dip", 0.5, 10), (ws[2], "dip", 1.0, 10)]
        out[f"breakout {lab} +0.5,+1.0ATR 10d"] = [(ws[0], "first", 0, 0), (ws[1], "breakout", 0.5, 10),
                                                   (ws[2], "breakout", 1.0, 10)]
        out[f"closeup {lab} +0.5,+1.5ATR 10d"] = [(ws[0], "first", 0, 0), (ws[1], "closeup", 0.5, 10),
                                                  (ws[2], "closeup", 1.5, 10)]
        out[f"mixed {lab} dip-1.0/brk+1.0 10d"] = [(ws[0], "first", 0, 0), (ws[1], "dip", 1.0, 10),
                                                   (ws[2], "breakout", 1.0, 10)]
    return out


# ====================================================================== simulation
def simulate_scaled(data: dict, stop_atr: float, target: str, horizon: int, slip: float,
                    tranches: list[tuple], own_stop: bool = False) -> pd.DataFrame:
    """R per 1R budget of every trade. own_stop=False: one shared stop (the setup's stop); True: each tranche
    gets its own stop `stop_atr` ATR below its fill. A trailing stop (hi since first fill - m ATR) applies to all."""
    sig = data["signals"]
    O, Hh, L, C = (data[k][:, :horizon] for k in ("O", "H", "L", "C"))
    n = len(sig)
    ref, a = sig["ref"].to_numpy(), sig["atr"].to_numpy()
    cond = sig["conditional"].to_numpy()
    first_px = np.where(cond, np.maximum(O[:, 0], ref), O[:, 0])
    filled0 = ~cond | (Hh[:, 0] >= ref)
    stop0 = ref - stop_atr * a
    entry0 = first_px * (1 + slip)
    ok = filled0 & (entry0 - stop0 > 0) & np.isfinite(entry0)
    trail_m = float(target[5:] or 3) if target.startswith("trail") else None
    tgt = np.full(n, np.inf)
    if trail_m is None:
        # same levels as research.simulate: measured from the planned entry (ref), not the fill
        tgt = ref + (float(target[1:]) * (ref - stop0) if target.startswith("r") else float(target[3:]) * a)
    nt = len(tranches)
    shares, cost = np.zeros((n, nt)), np.zeros((n, nt))
    base_stop = np.tile(stop0[:, None], (1, nt))
    exit_px = np.full((n, nt), np.nan)
    done = np.zeros((n, nt), dtype=bool)
    live = np.zeros((n, nt), dtype=bool)          # tranche holds shares now
    shares[:, 0] = np.where(ok, tranches[0][0] / np.where(ok, entry0 - stop0, 1), 0)
    cost[:, 0], done[:, 0], live[:, 0] = entry0, True, ok
    days = np.full(n, horizon)
    hi_prev = entry0.copy()              # highest high before today (trail anchor), starts at the first fill
    for k in range(horizon):
        trail = hi_prev - trail_m * a if trail_m is not None else np.full(n, -np.inf)
        on = live.any(1)
        # ---- fills of later tranches (before the stop check: conservative)
        if k > 0:
            for j in range(1, nt):
                w, kind, x, win = tranches[j]
                pend = on & ~done[:, j]
                if kind == "time":
                    px = np.where(pend & (k == win), O[:, k], np.nan)
                elif kind == "dip":
                    lvl = first_px - x * a
                    px = np.where(O[:, k] <= lvl, O[:, k], np.where(L[:, k] <= lvl, lvl, np.nan))
                    px = np.where(pend & (k <= win), px, np.nan)
                elif kind == "breakout":
                    lvl = first_px + x * a
                    px = np.where(O[:, k] >= lvl, O[:, k], np.where(Hh[:, k] >= lvl, lvl, np.nan))
                    px = np.where(pend & (k <= win), px, np.nan)
                else:                                     # closeup: yesterday's close confirmed
                    trig = C[:, k - 1] >= first_px + x * a
                    px = np.where(pend & trig & (k <= win), O[:, k], np.nan)
                fill = px * (1 + slip)
                bs = fill - stop_atr * a if own_stop else stop0
                dist = fill - np.maximum(bs, trail)
                good = np.isfinite(px) & (dist >= MIN_DIST_ATR * a)
                # a trigger that is not taken (e.g. too close to the stop) is not retried later
                done[:, j] |= np.isfinite(px) | (pend & (k >= win))
                shares[:, j] = np.where(good, w / np.where(good, dist, 1), shares[:, j])
                cost[:, j] = np.where(good, fill, cost[:, j])
                base_stop[:, j] = np.where(good, bs, base_stop[:, j])
                live[:, j] |= good
        # ---- exits (per tranche; with a shared stop they all leave together)
        stop_k = np.maximum(base_stop, trail[:, None])
        hit_s = live & (L[:, k][:, None] <= stop_k)
        px_s = np.minimum(O[:, k][:, None], stop_k) if k > 0 else stop_k
        hit_t = live & ~hit_s & (Hh[:, k] >= tgt)[:, None]
        px_t = np.maximum(O[:, k], tgt)[:, None] if k > 0 else np.tile(tgt[:, None], (1, nt))
        exit_px = np.where(hit_s, px_s, np.where(hit_t, px_t, exit_px))
        live &= ~(hit_s | hit_t)
        days = np.where(on & ~live.any(1), k + 1, days)
        hi_prev = np.maximum(hi_prev, Hh[:, k])
    exit_px = np.where(live, C[:, horizon - 1][:, None], exit_px) * (1 - slip)
    held = shares > 0
    pnl = np.where(held, shares * (exit_px - cost), 0).sum(1)
    w = np.array([t[0] for t in tranches])
    used = (held * w[None, :]).sum(1)                       # fraction of the 1R budget deployed
    out = pd.DataFrame({"setup": sig["setup"], "date": sig["date"], "symbol": sig["symbol"], "r": pnl,
                        "tranches": held.sum(1), "risk_used": used, "days": days})
    return out[ok].reset_index(drop=True)


# ====================================================================== statistics
def _t_by_date(diff: pd.Series, dates: pd.Series) -> float:
    m = diff.groupby(dates.to_numpy()).mean()
    if len(m) < 3 or m.std(ddof=1) == 0:
        return float("nan")
    return float(m.mean() / (m.std(ddof=1) / math.sqrt(len(m))))


def stats(tr: pd.DataFrame, single: pd.DataFrame) -> dict:
    if tr.empty:
        return {"n": 0}
    d = tr["r"] - single["r"]
    r = tr["r"]
    return {"n": int(len(tr)), "avg_r": float(r.mean()), "win_rate": float((r > 0).mean()),
            "t_r": _t_by_date(r, tr["date"]), "diff": float(d.mean()), "t_diff": _t_by_date(d, tr["date"]),
            "risk_used": float(tr["risk_used"].mean()), "avg_tranches": float(tr["tranches"].mean()),
            "r_per_risk": float(r.sum() / tr["risk_used"].sum()) if tr["risk_used"].sum() > 0 else None}


def run(data: dict, chosen: dict[str, dict], slip: float, cutoff: pd.Timestamp,
        progress: Callable[[str], None] | None = None) -> dict:
    sig = data["signals"]
    sch = schemes()
    res: dict[str, Any] = {"cutoff": cutoff.strftime("%Y-%m-%d"), "n_schemes": 2 * len(sch) - 1, "setups": {}}
    for setup in SETUPS:
        ex = chosen[setup]
        stop_atr = float(ex["stop"][3:])
        rows = {}
        for grp in (setup, "baseline"):
            m = (sig["setup"] == grp).to_numpy()
            sub = {"signals": sig[m].reset_index(drop=True), **{k: data[k][m] for k in ("O", "H", "L", "C")}}
            base = simulate_scaled(sub, stop_atr, ex["target"], int(ex["horizon"]), slip, sch["single"])
            for (name, tr), own in [(x, False) for x in sch.items()] + \
                    [((n + " | own stops", tr), True) for n, tr in sch.items() if n != "single"]:
                t = simulate_scaled(sub, stop_atr, ex["target"], int(ex["horizon"]), slip, tr, own_stop=own)
                assert len(t) == len(base)
                isx, oos = t["date"] <= cutoff, t["date"] > cutoff
                rows.setdefault(name, {})[grp] = {"is": stats(t[isx], base[isx]), "oos": stats(t[oos], base[oos]),
                                                  "all": stats(t, base)}
        for name, v in rows.items():
            v["edge_is"] = v[setup]["is"]["avg_r"] - v["baseline"]["is"]["avg_r"]
            v["edge_oos"] = v[setup]["oos"]["avg_r"] - v["baseline"]["oos"]["avg_r"]
        best = max((k for k in rows if k != "single"), key=lambda k: rows[k][setup]["is"]["diff"])
        b = rows[best][setup]
        passed = b["is"]["diff"] > 0 and b["oos"]["diff"] > 0 and (b["oos"]["t_diff"] or 0) >= 2.5
        oos_best = max((k for k in rows if k != "single"), key=lambda k: rows[k][setup]["oos"]["diff"])
        res["setups"][setup] = {"exit": ex["name"], "schemes": rows, "best_is": best, "passed": bool(passed),
                                "best_oos_hindsight": oos_best}
        if progress:
            progress(f"{setup}: best in-sample scheme '{best}' -> OOS diff {b['oos']['diff']:+.3f}R "
                     f"t {b['oos']['t_diff']:.2f} ({'PASS' if passed else 'fail'})")
    return res


def compact(res: dict, top: int = 10) -> dict:
    """Small copy for config/scale_in.default.json: single entry, the best schemes in sample, family averages."""
    out = {k: res[k] for k in ("cutoff", "n_schemes", "session", "generated_at", "counts", "survivorship_note") if k in res}
    out["setups"] = {}
    for setup, v in res["setups"].items():
        rows = v["schemes"]
        names = [k for k in rows if k != "single"]

        def brief(k: str) -> dict:
            r, b = rows[k][setup], rows[k]["baseline"]
            return {"name": k, "is": r["is"], "oos": r["oos"], "random_is_diff": b["is"].get("diff"),
                    "random_oos_diff": b["oos"].get("diff"), "edge_oos": rows[k]["edge_oos"]}
        fam: dict[str, dict] = {}
        for k in names:
            key = k.split()[0] + (" (own stops)" if k.endswith("own stops") else "")
            f = fam.setdefault(key, {"n": 0, "is_diff": 0.0, "oos_diff": 0.0, "best_oos_diff": -9.0})
            f["n"] += 1
            f["is_diff"] += rows[k][setup]["is"]["diff"]
            f["oos_diff"] += rows[k][setup]["oos"]["diff"]
            f["best_oos_diff"] = max(f["best_oos_diff"], rows[k][setup]["oos"]["diff"])
        for f in fam.values():
            f["is_diff"] /= f["n"]
            f["oos_diff"] /= f["n"]
        out["setups"][setup] = {
            "exit": v["exit"], "single": brief("single"), "passed": v["passed"], "best_is": brief(v["best_is"]),
            "best_oos_hindsight": brief(v["best_oos_hindsight"]),
            "better_than_single": {"is": sum(rows[k][setup]["is"]["diff"] > 0 for k in names),
                                   "oos": sum(rows[k][setup]["oos"]["diff"] > 0 for k in names),
                                   "both": sum(rows[k][setup]["is"]["diff"] > 0 and rows[k][setup]["oos"]["diff"] > 0
                                               for k in names)},
            "top_in_sample": [brief(k) for k in sorted(names, key=lambda k: -rows[k][setup]["is"]["diff"])[:top]],
            "families": fam}
    return out
