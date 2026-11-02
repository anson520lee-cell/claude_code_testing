"""Crash-rebound (buy the plunge) and spike-fade (short the surge) event study.

Walk-forward safe: every event is detected with data up to its own close; the trade is
entered at the NEXT session's open after the entry signal; volatility normalisation uses only
prior days. Results are split by date into in-sample (first 60% of events) and out-of-sample.

Events (long = crash, short = spike; spike rules are the mirror image):
    z3        one-day return <= -3 x the prior 60-day daily standard deviation
    pct8      one-day return <= -8%
    d3_15     3-day return <= -15%
    d5_20     5-day return <= -20%
    ext3      close at least 3 ATR below EMA21
A stock's events of one type closer than 10 sessions are merged (the first one counts).

Entry timing after the event day t (signal at a close, fill at the next open):
    d0..d10   signal at the close of day t + d
    green     first up close within 10 sessions (reversal confirmation)
    hold_hi   first close above the previous session's high within 10 sessions
Outcomes per entry: forward returns over 3 / 5 / 10 / 20 sessions (to the close), maximum
adverse / favourable excursion over 10 sessions, and a simple trade: stop 2 ATR below the entry,
target m ATR above (m = 1 / 1.5 / 2 / 3), or EMA21 (mean reversion), 10-session time stop.
Every number is compared with a random-entry baseline over the same stocks and dates.
Per stock: the best entry delay is estimated in-sample and checked out-of-sample against the
pooled best delay (does a stock-specific pattern beat the universe-wide one?).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EVENTS = {"z3": "1-day drop beyond 3σ", "pct8": "1-day drop ≥ 8%", "d3_15": "3-day drop ≥ 15%",
          "d5_20": "5-day drop ≥ 20%", "ext3": "≥ 3 ATR below EMA21"}
SPIKES = {"z3": "1-day surge beyond 3σ", "pct8": "1-day surge ≥ 8%", "d3_15": "3-day surge ≥ 15%",
          "d5_20": "5-day surge ≥ 20%", "ext3": "≥ 3 ATR above EMA21"}
DELAYS = [0, 1, 2, 3, 4, 5, 7, 10]
ENTRIES = [f"d{d}" for d in DELAYS] + ["green", "hold_hi"]
HORIZONS = [3, 5, 10, 20]
TARGETS = ["atr1", "atr1.5", "atr2", "atr3", "ema21"]
GAP = 10
LOOK = 34   # bars needed after the event: 10 (entry search) + 1 + 20 (horizon) + slack


def event_masks(f: pd.DataFrame, side: int) -> dict[str, np.ndarray]:
    """Boolean event masks; side = +1 crash (long), -1 spike (short)."""
    c = f["close"].astype(float)
    r = c.pct_change()
    sd = r.rolling(60).std().shift(1)
    s = -side   # crash: look for negative moves
    raw = {
        "z3": (s * r / sd) >= 3,
        "pct8": (s * r) >= 0.08,
        "d3_15": (s * (c / c.shift(3) - 1)) >= 0.15,
        "d5_20": (s * (c / c.shift(5) - 1)) >= 0.20,
        "ext3": (s * f["extension_atr"]) >= 3,
    }
    out = {}
    for k, m in raw.items():
        m = m.fillna(False).to_numpy()
        keep = np.zeros_like(m)
        last = -10 ** 9
        for i in np.flatnonzero(m):
            if i - last > GAP:
                keep[i] = True
            last = i if i - last > GAP else last
        out[k] = keep
    return out


def _entry_index(t: int, kind: str, c: np.ndarray, h: np.ndarray, side: int) -> int | None:
    """Index of the SIGNAL bar for an entry rule (the fill is at the next open)."""
    if kind.startswith("d"):
        return t + int(kind[1:])
    for j in range(t + 1, t + 11):
        if j >= len(c):
            return None
        if kind == "green" and side * (c[j] - c[j - 1]) > 0:
            return j
        if kind == "hold_hi" and ((side > 0 and c[j] > h[j - 1]) or (side < 0 and c[j] < h[j - 1])):
            return j
    return None


def collect_events(features: dict[str, pd.DataFrame], spy: pd.Series | None, side: int = 1) -> pd.DataFrame:
    rows = []
    spy_r = spy.pct_change() if spy is not None else None
    spy_sd = spy_r.rolling(60).std().shift(1) if spy is not None else None
    for sym, f in features.items():
        n = len(f)
        o, hi, lo, c = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        atr, ema21 = f["atr"].to_numpy(dtype=float), f["ema21"].to_numpy(dtype=float)
        ema200 = f["ema200"].to_numpy(dtype=float)
        vr = f["volume_ratio"].to_numpy(dtype=float)
        rsi = f["rsi14"].to_numpy(dtype=float)
        gap = f["gap_atr"].to_numpy(dtype=float)
        hh = hi if side > 0 else lo   # "previous high" for longs, previous low for shorts
        for ev, mask in event_masks(f, side).items():
            for t in np.flatnonzero(mask):
                if t + LOOK >= n or not np.isfinite(atr[t]):
                    continue
                date = f.index[t]
                mkt = None
                if spy_r is not None and date in spy_r.index:
                    zs = spy_r.at[date] / spy_sd.at[date] if spy_sd.at[date] else np.nan
                    mkt = bool(np.isfinite(zs) and (-side * zs) >= 2)   # the market itself crashed / spiked
                base = {"symbol": sym, "event": ev, "date": date, "atr_t": atr[t], "close_t": c[t],
                        "above_ema200": bool(c[t] > ema200[t]) if np.isfinite(ema200[t]) else None,
                        "volume_ratio": vr[t], "rsi14": rsi[t], "gap_atr": gap[t], "market_move": mkt,
                        "drop": c[t] / c[t - 5] - 1 if t >= 5 else np.nan}
                for kind in ENTRIES:
                    s_i = _entry_index(t, kind, c, hh, side)
                    if s_i is None or s_i + 21 >= n:
                        continue
                    e_i = s_i + 1
                    entry = o[e_i]
                    if not (np.isfinite(entry) and entry > 0):
                        continue
                    a = atr[s_i]
                    rec = {**base, "entry": kind, "wait": s_i - t, "entry_px": entry}
                    for hz in HORIZONS:
                        rec[f"ret{hz}"] = side * (c[e_i + hz - 1] / entry - 1)
                    win = slice(e_i, e_i + 10)
                    if side > 0:
                        rec["mae10_atr"], rec["mfe10_atr"] = (lo[win].min() - entry) / a, (hi[win].max() - entry) / a
                    else:
                        rec["mae10_atr"], rec["mfe10_atr"] = (entry - hi[win].max()) / a, (entry - lo[win].min()) / a
                    for tg in TARGETS:
                        rec[f"r_{tg}"] = _trade(o, hi, lo, c, e_i, entry, a, ema21[s_i], tg, side)
                    rows.append(rec)
    return pd.DataFrame(rows)


def _trade(o, hi, lo, c, e_i, entry, a, ema21, tg, side, k_stop=2.0, horizon=10) -> float:
    """R multiple of a stop-2-ATR / target / 10-day trade (a bar touching both counts as a stop)."""
    if tg == "ema21":
        dist = side * (ema21 - entry)
        dist = dist if dist > 0.5 * a else 1.0 * a
        target = entry + side * min(dist, 3 * a)
    else:
        target = entry + side * float(tg[3:]) * a
    stop = entry - side * k_stop * a
    risk = k_stop * a
    for j in range(e_i, e_i + horizon):
        first = j == e_i
        if side > 0:
            if lo[j] <= stop:
                return ((stop if first else min(o[j], stop)) - entry) / risk
            if hi[j] >= target:
                return ((target if first else max(o[j], target)) - entry) / risk
        else:
            if hi[j] >= stop:
                return (entry - (stop if first else max(o[j], stop))) / risk
            if lo[j] <= target:
                return (entry - (target if first else min(o[j], target))) / risk
    return side * (c[e_i + horizon - 1] - entry) / risk


def crash_history(f: pd.DataFrame, max_events: int = 40) -> dict[str, Any]:
    """This stock's past crashes (the crash_rebound setup's trigger) and what buying 0-2 days later did."""
    c = f["close"].astype(float)
    o, lo = f["open"].to_numpy(dtype=float), f["low"].to_numpy(dtype=float)
    ret3, ret5 = c / c.shift(3) - 1, c / c.shift(5) - 1
    ev = ((ret5 <= -0.20) | (ret3 <= -0.15) | (f["extension_atr"] <= -3)).fillna(False).to_numpy()
    cv, n, rows, last = c.to_numpy(), len(f), [], -10 ** 9
    for t in np.flatnonzero(ev):
        if t - last <= GAP:
            continue
        last = t
        rec = {"date": f.index[t].strftime("%Y-%m-%d"), "close": float(cv[t]),
               "drop5": float(ret5.iat[t]) if np.isfinite(ret5.iat[t]) else None,
               "below_ema21_atr": float(-f["extension_atr"].iat[t])}
        for d in (0, 1, 2):
            e_i = t + d + 1
            if e_i + 9 < n and o[e_i] > 0:
                rec[f"d{d}_ret5"] = float(cv[e_i + 4] / o[e_i] - 1)
                rec[f"d{d}_ret10"] = float(cv[e_i + 9] / o[e_i] - 1)
                rec[f"d{d}_mae10"] = float(lo[e_i:e_i + 10].min() / o[e_i] - 1)
        rows.append(rec)
    done = [r for r in rows if "d0_ret5" in r]
    summary: dict[str, Any] = {"n": len(rows), "n_complete": len(done), "since": f.index[0].strftime("%Y-%m-%d")}
    for d in (0, 1, 2):
        v = [r[f"d{d}_ret5"] for r in done if f"d{d}_ret5" in r]
        if v:
            summary[f"d{d}"] = {"mean5": float(np.mean(v)), "win5": float(np.mean(np.array(v) > 0)),
                                "mean10": float(np.mean([r[f"d{d}_ret10"] for r in done if f"d{d}_ret10" in r]))}
    return {"events": rows[::-1][:max_events], "summary": summary}


def baseline(features: dict[str, pd.DataFrame], side: int = 1, step: int = 5) -> pd.DataFrame:
    rows = []
    for sym, f in features.items():
        o, c = f["open"].to_numpy(dtype=float), f["close"].to_numpy(dtype=float)
        for t in range(60, len(f) - 22, step):
            e_i = t + 1
            if not (np.isfinite(o[e_i]) and o[e_i] > 0):
                continue
            rows.append({"date": f.index[t], **{f"ret{hz}": side * (c[e_i + hz - 1] / o[e_i] - 1) for hz in HORIZONS}})
    return pd.DataFrame(rows)


def _stats(x: pd.Series) -> dict[str, Any]:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return {"n": int(len(x))}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "win": float((x > 0).mean()),
            "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if x.std(ddof=1) > 0 else None}


def analyse(ev: pd.DataFrame, base: pd.DataFrame, is_fraction: float = 0.6) -> dict[str, Any]:
    if ev.empty:
        return {"available": False}
    cutoff = ev["date"].quantile(is_fraction)
    b_is, b_oos = base[base["date"] <= cutoff], base[base["date"] > cutoff]
    out: dict[str, Any] = {"available": True, "cutoff": str(pd.Timestamp(cutoff).date()), "n_events": int(ev.drop_duplicates(["symbol", "event", "date"]).shape[0]),
                           "baseline": {f"ret{h}": {"is": _stats(b_is[f"ret{h}"]), "oos": _stats(b_oos[f"ret{h}"])} for h in HORIZONS}}
    grid = []
    for (e, k), g in ev.groupby(["event", "entry"]):
        gi, go = g[g["date"] <= cutoff], g[g["date"] > cutoff]
        row = {"event": e, "entry": k, "n_is": len(gi), "n_oos": len(go)}
        for h in HORIZONS:
            row[f"ret{h}_is"], row[f"ret{h}_oos"] = _stats(gi[f"ret{h}"]), _stats(go[f"ret{h}"])
        for tg in TARGETS:
            row[f"r_{tg}_is"], row[f"r_{tg}_oos"] = _stats(gi[f"r_{tg}"]), _stats(go[f"r_{tg}"])
        grid.append(row)
    out["grid"] = grid
    # best in sample by 5-day excess return, judged out of sample
    def excess(row, h, part):
        st = row[f"ret{h}_{part}"]
        b = out["baseline"][f"ret{h}"][part].get("mean", 0)
        return (st.get("mean", np.nan) - b) if st.get("n", 0) >= 30 else np.nan
    for h in (5, 10):
        ranked = sorted([g for g in grid if np.isfinite(excess(g, h, "is"))], key=lambda g: -excess(g, h, "is"))
        out[f"best_ret{h}"] = [{"event": g["event"], "entry": g["entry"], "excess_is": excess(g, h, "is"), "excess_oos": excess(g, h, "oos"),
                                "is": g[f"ret{h}_is"], "oos": g[f"ret{h}_oos"]} for g in ranked[:10]]
    ranked = sorted([(g, tg) for g in grid for tg in TARGETS if g[f"r_{tg}_is"].get("n", 0) >= 30],
                    key=lambda x: -x[0][f"r_{x[1]}_is"].get("mean", -9))
    out["best_trade"] = [{"event": g["event"], "entry": g["entry"], "target": tg, "is": g[f"r_{tg}_is"], "oos": g[f"r_{tg}_oos"]} for g, tg in ranked[:10]]
    # conditions on the best event (by 5-day excess, in sample)
    top = out["best_ret5"][0] if out["best_ret5"] else None
    if top:
        sub = ev[(ev["event"] == top["event"]) & (ev["entry"] == top["entry"])]
        cond = {}
        for name, mask in (("above EMA200", sub["above_ema200"] == True), ("below EMA200", sub["above_ema200"] == False),  # noqa: E712
                           ("market also crashed", sub["market_move"] == True), ("stock-specific", sub["market_move"] == False),  # noqa: E712
                           ("capitulation volume (≥ 2.5x)", sub["volume_ratio"] >= 2.5), ("normal volume", sub["volume_ratio"] < 2.5),
                           ("gap ≥ 2 ATR", sub["gap_atr"].abs() >= 2), ("RSI14 < 25", sub["rsi14"] < 25)):
            part = sub[mask]
            cond[name] = {"is": _stats(part[part["date"] <= cutoff]["ret5"]), "oos": _stats(part[part["date"] > cutoff]["ret5"])}
        out["conditions"] = {"event": top["event"], "entry": top["entry"], "by": cond}
    # delay curve per event (pooled, all dates)
    out["delay_curve"] = {e: {k: _stats(g[g["entry"] == k]["ret5"]) for k in ENTRIES} for e, g in ev.groupby("event")}
    out["per_stock"] = per_stock(ev, cutoff)
    return out


def per_stock(ev: pd.DataFrame, cutoff: pd.Timestamp, event: str = "z3", h: int = 5, min_n: int = 3) -> dict[str, Any]:
    """Does each stock's own best entry delay (in sample) beat the pooled best delay out of sample?"""
    sub = ev[(ev["event"] == event) & ev["entry"].str.startswith("d")]
    ins, oos = sub[sub["date"] <= cutoff], sub[sub["date"] > cutoff]
    pooled = ins.groupby("entry")[f"ret{h}"].mean()
    if pooled.empty:
        return {}
    pooled_best = pooled.idxmax()
    own, pool = [], []
    table = {}
    for sym, g in ins.groupby("symbol"):
        m = g.groupby("entry")[f"ret{h}"].agg(["mean", "count"])
        m = m[m["count"] >= min_n]
        if m.empty:
            continue
        best = m["mean"].idxmax()
        go = oos[oos["symbol"] == sym]
        table[sym] = {"best_delay": best, "n_events_is": int(g.drop_duplicates("date").shape[0]), "mean_is": float(m.loc[best, "mean"])}
        if go.empty:
            continue
        a, b = go[go["entry"] == best][f"ret{h}"], go[go["entry"] == pooled_best][f"ret{h}"]
        if len(a) and len(b):
            own.append(float(a.mean()))
            pool.append(float(b.mean()))
    return {"event": event, "pooled_best": pooled_best, "stocks": table,
            "oos_own_best_mean": float(np.mean(own)) if own else None, "oos_pooled_best_mean": float(np.mean(pool)) if pool else None,
            "n_compared": len(own)}
