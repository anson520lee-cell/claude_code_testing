"""Strategy discovery: trading styles the terminal does not use yet, tested the same honest way.

Long only, daily bars, the focus universe, 5 years. Every style below comes from published research or
common professional practice; none was designed on this data. For each style:

* signals are found walk-forward (data up to the signal day), entry at the next open;
* the exit is chosen among 4 stops x 7 exits x 5 time stops on the first 60% of dates (in sample) and judged
  on the last 40% (out of sample), next to random entries with the SAME exit;
* t-statistics are clustered by date (one observation per signal day), because market-wide styles enter
  many stocks on the same day and would otherwise look far more certain than they are;
* with ~15 styles tested, a few will look good by luck, so a style "passes" only if it beats random entries
  by >= 0.05R both in AND out of sample, and its out-of-sample date-clustered t >= 2.5.

Styles
    rsi2          Connors RSI(2) pullback: RSI(2) < 5 while above EMA200 (short-term oversold in an uptrend)
    ibs           Internal bar strength: close in the bottom 15% of the day's range, down day, above EMA200
    down3         Three consecutive lower closes while above EMA50
    sector_lag    Idiosyncratic dip: 5-day return vs the stock's sector ETF more than 2 standard deviations
                  below normal, while above EMA200 (the stock fell, its sector did not)
    vix_spike     Fear spike: VIX closes 25% above its 20-day average (first day in 10) - buy the universe
    vix_invert    VIX term structure inverts (VIX > VIX3M, first day in 10) - buy the universe
    breadth_low   Breadth washout: fewer than 15% of the universe above EMA50 (first day in 10)
    breadth_thrust  Breadth thrust: share above EMA50 jumps from < 30% to > 60% within 10 sessions
    turn_month    Turn of the month: enter at the close of the month's last session
    pre_holiday   Enter at the close of the session before a market holiday
    event_drift   Post-event drift: gap up >= 2 ATR on >= 3x volume that holds (close in the upper half);
                  usually earnings or news - buy and let it drift
    gap_reversal  Gap-down reversal: gap down >= 2 ATR that closes in the top 30% of the range, above EMA200
    high52        52-week-high breakout: first close above the prior 252-session high in 20 sessions, volume >= 1.5x
    nr7           NR7 inside day: narrowest range of 7 sessions and inside the prior bar, above EMA50
Overnight drift is measured separately (close -> next open vs open -> close), as a fact about these stocks.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtest.research import simulate, summary
from src.market import calendar

H = 20
STOPS = ["atr1.0", "atr1.5", "atr2.0", "atr3.0"]
TARGETS = ["atr1", "atr1.5", "atr2", "atr3", "r2", "trail2", "trail3"]
HORIZONS = [2, 3, 5, 10, 20]
STYLE_NAMES = {
    "rsi2": "RSI(2) pullback in uptrend (Connors)", "ibs": "Close at the day's low (internal bar strength)",
    "down3": "Three lower closes in uptrend", "sector_lag": "Idiosyncratic dip vs sector",
    "vix_spike": "VIX fear spike - buy the universe", "vix_invert": "VIX term-structure inversion",
    "breadth_low": "Breadth washout (<15% above EMA50)", "breadth_thrust": "Breadth thrust (30% -> 60% in 10 days)",
    "turn_month": "Turn of the month", "pre_holiday": "Pre-holiday", "event_drift": "Post-event drift (gap up on volume)",
    "gap_reversal": "Gap-down reversal (closes strong)", "high52": "52-week-high breakout", "nr7": "NR7 inside day",
}


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _first(mask: pd.Series, cooldown: int = 1) -> pd.Series:
    """True on the first day of each episode (no signal in the previous ``cooldown`` sessions)."""
    m = mask.fillna(False).astype(bool)
    prior = m.shift(1, fill_value=False).rolling(cooldown, min_periods=1).max().astype(bool) if cooldown > 1 \
        else m.shift(1, fill_value=False)
    return m & ~prior


def market_signals(features: dict[str, pd.DataFrame], prices: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    vix = prices.get("^VIX")
    vix3 = prices.get("^VIX3M")
    if vix is not None:
        v = vix["close"].reindex(index).ffill()
        out["vix_spike"] = _first(v > 1.25 * v.rolling(20).mean(), 10)
        if vix3 is not None:
            v3 = vix3["close"].reindex(index).ffill()
            out["vix_invert"] = _first(v > v3, 10)
    if features:
        above = pd.DataFrame({s: (f["close"] > f["ema50"]).reindex(index) for s, f in features.items()}).mean(axis=1)
        out["breadth_low"] = _first(above < 0.15, 10)
        out["breadth_thrust"] = _first((above > 0.60) & (above.rolling(10).min() < 0.30), 10)
    dates = pd.Series(index, index=index)
    nxt = dates.shift(-1)
    if len(index):   # the latest session's next session is not in the data yet: take it from the calendar
        nxt.iloc[-1] = pd.Timestamp(calendar.next_trading_day(index[-1].date()))
    out["turn_month"] = (nxt.dt.month != dates.dt.month) & nxt.notna()
    def pre_holiday(d: pd.Timestamp, n: pd.Timestamp) -> bool:
        if pd.isna(n):
            return False
        gap = [d.date() + timedelta(days=k) for k in range(1, (n.date() - d.date()).days)]
        return any(x.weekday() < 5 and not calendar.is_trading_day(x) for x in gap)
    out["pre_holiday"] = pd.Series([pre_holiday(d, n) for d, n in zip(dates, nxt)], index=index)
    return out


def stock_signals(f: pd.DataFrame) -> dict[str, pd.Series]:
    c, h, lo = f["close"], f["high"], f["low"]
    rng = (h - lo).replace(0, np.nan)
    ibs = (c - lo) / rng
    up200, up50 = c > f["ema200"], c > f["ema50"]
    rs = f.get("rel_sector_5d")
    out = {
        "rsi2": _first((_rsi(c, 2) < 5) & up200, 5),
        "ibs": _first((ibs < 0.15) & (c < c.shift(1)) & up200, 3),
        "down3": _first((c < c.shift(1)) & (c.shift(1) < c.shift(2)) & (c.shift(2) < c.shift(3)) & up50, 5),
        "event_drift": _first((f["gap_atr"] >= 2) & (f["volume_ratio"] >= 3) & (ibs >= 0.5), 10),
        "gap_reversal": _first((f["gap_atr"] <= -2) & (ibs >= 0.7) & up200, 10),
        "high52": _first((c > h.shift(1).rolling(252, min_periods=250).max()) & (f["volume_ratio"] >= 1.5), 20),
        "nr7": _first((rng <= rng.rolling(7).min()) & (h < h.shift(1)) & (lo > lo.shift(1)) & up50, 5),
    }
    if rs is not None:
        z = rs / rs.rolling(120, min_periods=60).std()
        out["sector_lag"] = _first((z < -2) & up200, 10)
    return out


def collect(features: dict[str, pd.DataFrame], prices: dict[str, pd.DataFrame], regime: pd.Series,
            base_step: int = 10) -> dict[str, Any]:
    index = prices["SPY"].index
    mkt = market_signals(features, prices, index)
    rows, fwd = [], []
    for k, (sym, f) in enumerate(features.items()):
        o, h, lo, c, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "atr"))
        sw = f["swing_low_last"].to_numpy(dtype=float)
        n = len(f)
        sig = stock_signals(f)
        for name, s in mkt.items():
            sig[name] = s.reindex(f.index).fillna(False).astype(bool)
        base = np.zeros(n, dtype=bool)
        base[np.arange(260 + k % base_step, n, base_step)] = True
        sig["baseline"] = pd.Series(base, index=f.index)
        for name, s in sig.items():
            idx = np.flatnonzero(s.to_numpy(dtype=bool))
            idx = idx[(idx >= 260) & (idx + H < n)]
            for t in idx:
                if not (np.isfinite(a[t]) and a[t] > 0 and np.isfinite(c[t])):
                    continue
                d = f.index[t]
                rows.append({"symbol": sym, "setup": name, "date": d, "ref": c[t], "atr": a[t], "structural": c[t] - 1.5 * a[t],
                             "swing": sw[t], "conditional": False, "long": True, "regime": regime.get(d, "n/a")})
                fwd.append(np.stack([o[t + 1: t + 1 + H], h[t + 1: t + 1 + H], lo[t + 1: t + 1 + H], c[t + 1: t + 1 + H]]))
    sigdf = pd.DataFrame(rows)
    for col in ("score", "tier", "planned_rr", "chip_stop", "chip_target", "resistance"):
        sigdf[col] = np.nan
    arr = np.array(fwd)
    return {"signals": sigdf, "O": arr[:, 0], "H": arr[:, 1], "L": arr[:, 2], "C": arr[:, 3]}


def _subset(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {"signals": data["signals"][mask].reset_index(drop=True), **{k: data[k][mask] for k in ("O", "H", "L", "C")}}


def clustered(t: pd.DataFrame) -> dict[str, Any]:
    """Summary plus a t-statistic with one observation per signal date."""
    s = summary(t)
    if not s.get("n"):
        return s
    daily = t.groupby("date")["r"].mean()
    s["n_dates"] = int(len(daily))
    sd = daily.std(ddof=1)
    s["t_clustered"] = float(daily.mean() / (sd / math.sqrt(len(daily)))) if len(daily) > 2 and sd > 0 else float("nan")
    return s


def evaluate_styles(data: dict[str, Any], cutoff: pd.Timestamp, slip: float, names: list[str],
                    say: Callable[[str], None], min_is: int = 60) -> dict[str, Any]:
    """Choose each style's exit in sample, judge it out of sample against random entries with the same exit."""
    sig = data["signals"]
    base = _subset(data, (sig["setup"] == "baseline").to_numpy())
    base_cache: dict[tuple, pd.DataFrame] = {}

    def base_run(key):
        if key not in base_cache:
            base_cache[key] = simulate(base, *key, slip)
        return base_cache[key]
    styles: dict[str, Any] = {}
    for name in names:
        m = (sig["setup"] == name).to_numpy()
        if m.sum() < min_is:
            styles[name] = {"name": STYLE_NAMES[name], "n": int(m.sum()), "skipped": "too few signals"}
            continue
        d = _subset(data, m)
        best = None
        for sm in STOPS:
            for tm in TARGETS:
                for hz in HORIZONS:
                    tr = simulate(d, sm, tm, hz, slip)
                    ins = tr[tr["date"] <= cutoff]
                    if len(ins) < min_is:
                        continue
                    r_is = ins["r"].mean()
                    if best is None or r_is > best[0]:
                        best = (r_is, sm, tm, hz, tr)
        if best is None:
            styles[name] = {"name": STYLE_NAMES[name], "n": int(m.sum()), "skipped": "too few in-sample trades"}
            continue
        _, sm, tm, hz, tr = best
        br = base_run((sm, tm, hz))
        ins, oos = tr[tr["date"] <= cutoff], tr[tr["date"] > cutoff]
        b_is, b_oos = br[br["date"] <= cutoff], br[br["date"] > cutoff]
        s_is, s_oos = clustered(ins), clustered(oos)
        bi, bo = summary(b_is), summary(b_oos)
        edge_is = s_is.get("avg_r", np.nan) - bi.get("avg_r", np.nan)
        edge_oos = s_oos.get("avg_r", np.nan) - bo.get("avg_r", np.nan) if s_oos.get("n") else np.nan
        passes = bool(np.isfinite(edge_oos) and edge_is >= 0.05 and edge_oos >= 0.05 and s_oos.get("t_clustered", 0) >= 2.5)
        styles[name] = {
            "name": STYLE_NAMES[name], "n": int(len(tr)), "stop": sm, "target": tm, "horizon": hz,
            "is": s_is, "oos": s_oos, "baseline_is": bi, "baseline_oos": bo, "edge_is": edge_is, "edge_oos": edge_oos,
            "passes": passes, "by_regime": {k: summary(g) for k, g in tr.groupby("regime")},
            "per_year": {str(k): summary(g).get("avg_r") for k, g in tr.groupby(tr["date"].dt.year)}}
        say(f"{STYLE_NAMES[name]}: {len(tr)} trades, best exit {sm}/{tm}/{hz}d, in sample {s_is.get('avg_r', 0):+.2f}R "
            f"(edge {edge_is:+.2f}), out of sample {s_oos.get('avg_r', float('nan')):+.2f}R (edge {edge_oos:+.2f}, "
            f"t {s_oos.get('t_clustered', float('nan')):.1f}){'  PASS' if passes else ''}")
    return styles


INDEX_STYLES = ["rsi2", "ibs", "down3", "nr7", "high52", "vix_spike", "vix_invert", "turn_month", "pre_holiday"]


def index_study(frames: dict[str, pd.DataFrame], slip: float, split: str = "2013-01-01",
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The same styles on SPY and QQQ over their whole history (1993 / 1999 -), where market-wide signals have
    dozens of occurrences instead of a handful. Random entries = every session. In sample before ``split``."""
    from src.scanner.features import build_features
    say = progress or (lambda m: None)
    out: dict[str, Any] = {"split": split, "instruments": {}}
    for sym in ("SPY", "QQQ"):
        df = frames.get(sym)
        if df is None or len(df) < 1500:
            continue
        f = build_features(df, df["adj_close"], None)
        sig = stock_signals(f)
        sig.update(market_signals({}, {k: frames[k] for k in ("^VIX", "^VIX3M") if k in frames}, f.index))
        sig["baseline"] = pd.Series(True, index=f.index)
        o, h, lo, c, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "atr"))
        sw = f["swing_low_last"].to_numpy(dtype=float)
        rows, fwd = [], []
        for name, s in sig.items():
            if name != "baseline" and name not in INDEX_STYLES:
                continue
            idx = np.flatnonzero(s.reindex(f.index).fillna(False).to_numpy(dtype=bool))
            idx = idx[(idx >= 260) & (idx + H < len(f))]
            for t in idx:
                if not (np.isfinite(a[t]) and a[t] > 0):
                    continue
                rows.append({"symbol": sym, "setup": name, "date": f.index[t], "ref": c[t], "atr": a[t],
                             "structural": c[t] - 1.5 * a[t], "swing": sw[t], "conditional": False, "long": True, "regime": "n/a"})
                fwd.append(np.stack([o[t + 1: t + 1 + H], h[t + 1: t + 1 + H], lo[t + 1: t + 1 + H], c[t + 1: t + 1 + H]]))
        sigdf = pd.DataFrame(rows)
        for col in ("score", "tier", "planned_rr", "chip_stop", "chip_target", "resistance"):
            sigdf[col] = np.nan
        arr = np.array(fwd)
        data = {"signals": sigdf, "O": arr[:, 0], "H": arr[:, 1], "L": arr[:, 2], "C": arr[:, 3]}
        say(f"{sym} {f.index[0]:%Y}-{f.index[-1]:%Y}:")
        st = evaluate_styles(data, pd.Timestamp(split), slip, INDEX_STYLES, say, min_is=25)
        out["instruments"][sym] = {"first": f.index[0].strftime("%Y-%m-%d"), "last": f.index[-1].strftime("%Y-%m-%d"), "styles": st}
    return out


def index_timing_today(prices: dict[str, pd.DataFrame], study: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Live: which index styles fire at today's close on SPY / QQQ, with the exit and the long-history result
    from the discovery study. Only styles that passed, or beat random entries by >= 0.05R in AND out of sample
    with an out-of-sample t >= 2 ("near pass")."""
    from src.scanner.features import build_features
    out: list[dict[str, Any]] = []
    inst = ((study or {}).get("index") or {}).get("instruments") or {}
    for sym, res in inst.items():
        df = prices.get(sym)
        if df is None or len(df) < 260:
            continue
        f = build_features(df, df["adj_close"], None)
        sig = stock_signals(f)
        sig.update(market_signals({}, {}, f.index))
        a, c = float(f["atr"].iloc[-1]), float(f["close"].iloc[-1])
        for name, s in (res.get("styles") or {}).items():
            near = (s.get("edge_is", 0) >= 0.05 and (s.get("edge_oos") or 0) >= 0.05
                    and (s.get("oos") or {}).get("t_clustered", 0) >= 2)
            if s.get("skipped") or not (s.get("passes") or near):
                continue
            active = bool(sig.get(name, pd.Series(False, index=f.index)).iloc[-1])
            k = float(s["stop"][3:])
            stop = c - k * a
            tm = s["target"]
            target = None if tm.startswith("trail") else c + (float(tm[3:]) * a if tm.startswith("atr") else float(tm[1:]) * k * a)
            out.append({"symbol": sym, "style": name, "name": s["name"], "active": active, "close": c, "atr": a,
                        "stop": stop, "target": target, "trail_atr": float(tm[5:] or 3) if tm.startswith("trail") else None,
                        "horizon": s["horizon"], "passes": bool(s.get("passes")), "is": s["is"], "oos": s["oos"],
                        "edge_is": s["edge_is"], "edge_oos": s["edge_oos"],
                        "years_positive": sum(1 for v in s["per_year"].values() if v and v > 0), "years": len(s["per_year"]),
                        "period": f"{res.get('first', '')[:4]}-{res.get('last', '')[:4]}"})
    out.sort(key=lambda x: (not x["active"], not x["passes"], -(x["edge_oos"] or 0)))
    return out


def index_verdict(study: dict[str, Any]) -> list[str]:
    out = []
    for sym, res in (study.get("instruments") or {}).items():
        for s in sorted((x for x in res["styles"].values() if not x.get("skipped")), key=lambda x: -(x["edge_oos"] or -9)):
            pos = sum(1 for v in s["per_year"].values() if v and v > 0)
            both = s["edge_is"] > 0 and s["edge_oos"] > 0
            out.append(f"{'PASS' if s['passes'] else ('consistent' if both else 'fail')} · {sym} {s['name']} ({res['first'][:4]}-"
                       f"{res['last'][:4]}): edge {s['edge_is']:+.2f}R in sample, {s['edge_oos']:+.2f}R out of sample "
                       f"({s['oos'].get('n', 0)} trades, t {s['oos'].get('t_clustered', float('nan')):.1f}), positive in {pos}/{len(s['per_year'])} years.")
    return out


def overnight(features: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict[str, Any]:
    on, intra = [], []
    for f in features.values():
        c, o = f["close"], f["open"]
        on.append(np.log(o / c.shift(1)).rename("on"))
        intra.append(np.log(c / o).rename("in"))
    a = pd.concat(on, axis=1).mean(axis=1)
    b = pd.concat(intra, axis=1).mean(axis=1)
    out = {}
    for name, part in (("in_sample", a.index <= cutoff), ("out_of_sample", a.index > cutoff)):
        x, y = a[part].dropna(), b[part].dropna()
        out[name] = {"overnight_daily": float(x.mean()), "intraday_daily": float(y.mean()),
                     "overnight_t": float(x.mean() / (x.std() / math.sqrt(len(x)))), "intraday_t": float(y.mean() / (y.std() / math.sqrt(len(y)))),
                     "overnight_annual": float(np.expm1(x.mean() * 252)), "intraday_annual": float(np.expm1(y.mean() * 252)), "days": int(len(x))}
    return out


def run(data: dict[str, Any], features: dict[str, pd.DataFrame], settings: dict[str, Any],
        progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda m: None)
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    sig = data["signals"]
    cutoff = sig.loc[sig["setup"] != "baseline", "date"].quantile(0.6)
    res: dict[str, Any] = {"available": True, "method": __doc__, "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
                           "style_names": STYLE_NAMES, "styles": {}, "first": sig["date"].min().strftime("%Y-%m-%d"),
                           "last": sig["date"].max().strftime("%Y-%m-%d")}
    res["styles"] = evaluate_styles(data, cutoff, slip, list(STYLE_NAMES), say)
    res["overnight"] = overnight(features, pd.Timestamp(cutoff))
    res["verdict"] = verdict(res)
    return res


def verdict(res: dict[str, Any]) -> list[str]:
    st = [s for s in res["styles"].values() if not s.get("skipped")]
    passed = [s for s in st if s["passes"]]
    out = [f"{len(st)} styles tested; {len(passed)} passed (edge over random >= 0.05R in AND out of sample, "
           f"out-of-sample date-clustered t >= 2.5)."]
    for s in sorted(st, key=lambda x: -(x["edge_oos"] if np.isfinite(x["edge_oos"]) else -9)):
        out.append(f"{'PASS' if s['passes'] else 'fail'} · {s['name']}: out of sample {s['oos'].get('avg_r', float('nan')):+.2f}R "
                   f"over {s['oos'].get('n', 0)} trades on {s['oos'].get('n_dates', 0)} days, edge {s['edge_oos']:+.2f}R "
                   f"(in sample {s['edge_is']:+.2f}R), t {s['oos'].get('t_clustered', float('nan')):.1f}.")
    on = res.get("overnight", {}).get("out_of_sample")
    if on:
        out.append(f"Overnight vs intraday (average stock, out of sample): close-to-open {on['overnight_daily'] * 100:+.3f}%/day "
                   f"(t {on['overnight_t']:.1f}), open-to-close {on['intraday_daily'] * 100:+.3f}%/day (t {on['intraday_t']:.1f}).")
    return out
