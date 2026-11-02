"""Professional analysis toolkit: ~55 techniques run on one stock's daily bars.

Every technique returns a finding:
    {"id", "group", "name", "signal": -1 | 0 | +1, "strength": 0..1, "text": English, "stat": {...}}
signal is the directional read for the next 1-10 sessions (+1 constructive, -1 adverse, 0 neutral /
informational). Stock-specific statistics always report their sample size; fewer than 5 cases are
reported as "insufficient sample" and never produce a signal.

Groups
    Stock behaviour   how THIS stock has behaved at EMA9 / EMA21 / EMA50 / EMA200, band touches, RSI extremes,
               gaps, streaks, breakouts, big-move days, weekdays; autocorrelation and Hurst exponent
    Volume profile   volume profile (POC, value area, high-volume nodes = dense chips, low-volume nodes =
               vacuum zones), holder-profit ratio, anchored VWAPs, 20/60-day average holder cost
    Patterns   VCP, bull / bear flag, double top / bottom, pocket pivot, climax volume, key reversal,
               undercut-and-rally, failed breakout / breakdown, NR7, inside day, 3-weeks-tight,
               regression-channel position, support / resistance touch counts
    Indicator consensus   Stochastic, Williams %R, CCI, MFI, Chaikin money flow, A/D line, TTM squeeze, Keltner,
               Supertrend, Ichimoku, Parabolic SAR, Donchian, Heikin-Ashi, Elder ray, Aroon, RSI(2), z-score
    Relative performance   correlation / beta to SPY, up / down capture, idiosyncratic move today, RS line leadership
All calculations use data up to the analysis date only. Statistics are in-sample descriptions of the
past, not forecasts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.technical.indicators import atr as atr_fn, ema, rsi, sma
from src.technical.structure import swing_points

MIN_CASES = 5


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _pc(x: float, d: int = 1) -> str:
    return "n/a" if not np.isfinite(x) else f"{round(x * 100, d) + 0:+.{d}f}%"


def _n(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}" if np.isfinite(_f(x)) else "n/a"


def F(id_: str, group: str, name: str, signal: float, strength: float, text: str, **stat: Any) -> dict[str, Any]:
    return {"id": id_, "group": group, "name": name, "signal": int(np.sign(signal)) if np.isfinite(signal) else 0,
            "strength": float(np.clip(strength, 0, 1)) if np.isfinite(strength) else 0.0, "text": text, "stat": stat}


# ====================================================================== stock-specific behaviour
def level_reaction(f: pd.DataFrame, level: pd.Series, name: str, horizon: int = 3) -> dict[str, Any]:
    """Tests of a moving level from below (rejection vs break) and from above (bounce vs break)."""
    c, h, lo = f["close"], f["high"], f["low"]
    L = level
    prev_below = c.shift(1) < L.shift(1)
    prev_above = c.shift(1) > L.shift(1)
    test_below = prev_below & (h >= L * 0.998) & (c < L)          # poked up into the level, closed under it
    test_above = prev_above & (lo <= L * 1.002) & (c >= L)          # dipped into the level, held above it
    fut_close, fut_level = c.shift(-horizon), L.shift(-horizon)
    fwd = fut_close / c - 1
    valid = fut_close.notna()
    rb = test_below & valid
    ra = test_above & valid
    rej = (fut_close < fut_level)[rb]
    hold = (fut_close > fut_level)[ra]
    out = {"level": name, "n_below": int(rb.sum()), "rejected": float(rej.mean()) if rb.sum() else float("nan"),
           "fwd_below": float(fwd[rb].mean()) if rb.sum() else float("nan"),
           "n_above": int(ra.sum()), "held": float(hold.mean()) if ra.sum() else float("nan"),
           "fwd_above": float(fwd[ra].mean()) if ra.sum() else float("nan")}
    return out


def behaviour(f: pd.DataFrame, findings: list) -> None:
    c = f["close"]
    atr = _f(f["atr"].iloc[-1])
    last = f.iloc[-1]
    for span in (9, 21, 50, 200):
        col = f"ema{span}"
        if f[col].notna().sum() < 120:
            continue
        r = level_reaction(f.tail(500), f[col].tail(500), f"EMA{span}")
        dist = (last.close - last[col]) / atr if atr > 0 else float("nan")
        near = np.isfinite(dist) and abs(dist) <= 1.0
        parts, signal, strength = [], 0, 0.0
        if r["n_below"] >= MIN_CASES:
            parts.append(f"tested from below {r['n_below']} times, rejected within 3 days {r['rejected'] * 100:.0f}% of the time (avg 3-day return {_pc(r['fwd_below'])})")
        if r["n_above"] >= MIN_CASES:
            parts.append(f"tested from above {r['n_above']} times, held within 3 days {r['held'] * 100:.0f}% of the time (avg 3-day return {_pc(r['fwd_above'])})")
        if not parts:
            findings.append(F(f"ema{span}_reaction", "Stock behaviour", f"EMA{span} reaction stats", 0, 0, f"EMA{span}: too few tests in the last 2 years."))
            continue
        if near and dist < 0 and r["n_below"] >= MIN_CASES and r["rejected"] >= 0.65:
            signal, strength = -1, r["rejected"]
            parts.append(f"now {abs(dist):.1f} ATR below EMA{span}, where it has usually been rejected")
        elif near and dist >= 0 and r["n_above"] >= MIN_CASES and r["held"] >= 0.65:
            signal, strength = 1, r["held"]
            parts.append(f"now {dist:.1f} ATR above EMA{span}, which has usually held")
        findings.append(F(f"ema{span}_reaction", "Stock behaviour", f"EMA{span} reaction stats", signal, strength,
                          f"EMA{span}: " + "; ".join(parts) + ".", **r, distance_atr=dist))

    # Bollinger band touches: walk or revert
    up_touch = (f["close"] >= f["bb_upper"]) & f["bb_upper"].notna()
    fwd5 = c.shift(-5) / c - 1
    ut = up_touch & fwd5.notna()
    dt = (f["close"] <= f["bb_lower"]) & f["bb_lower"].notna() & fwd5.notna()
    if ut.sum() >= MIN_CASES or dt.sum() >= MIN_CASES:
        u_m, d_m = float(fwd5[ut].mean()) if ut.sum() else float("nan"), float(fwd5[dt].mean()) if dt.sum() else float("nan")
        state = "upper band" if last.close >= last.bb_upper else "lower band" if last.close <= last.bb_lower else None
        sig = 0
        if state == "upper band" and ut.sum() >= MIN_CASES:
            sig = 1 if u_m > 0.01 else -1 if u_m < -0.005 else 0
        if state == "lower band" and dt.sum() >= MIN_CASES:
            sig = 1 if d_m > 0.01 else -1 if d_m < -0.005 else 0
        findings.append(F("bb_touch", "Stock behaviour", "After Bollinger band touches", sig, 0.5,
                          f"Closed at the upper band {int(ut.sum())} times, next 5 days avg {_pc(u_m)}; at the lower band {int(dt.sum())} times, next 5 days avg {_pc(d_m)}. "
                          + (f"Closed at the {state} today; " + ("this stock tends to ride the band." if (u_m > 0.01 if state == 'upper band' else d_m < -0.005) else "this stock tends to revert to the middle band.") if state else ""),
                          up_n=int(ut.sum()), up_fwd5=u_m, down_n=int(dt.sum()), down_fwd5=d_m))

    # RSI14 extremes
    ob = (f["rsi14"] >= 70) & fwd5.notna()
    os_ = (f["rsi14"] <= 30) & fwd5.notna()
    if ob.sum() >= MIN_CASES or os_.sum() >= MIN_CASES:
        ob_m = float(fwd5[ob].mean()) if ob.sum() else float("nan")
        os_m = float(fwd5[os_].mean()) if os_.sum() else float("nan")
        cur = _f(last.rsi14)
        sig = (1 if ob_m > 0 else -1) if cur >= 70 and ob.sum() >= MIN_CASES else (1 if os_m > 0 else -1) if cur <= 30 and os_.sum() >= MIN_CASES else 0
        findings.append(F("rsi_extreme", "Stock behaviour", "After RSI extremes", sig, 0.5,
                          f"RSI14 ≥ 70 on {int(ob.sum())} days, next 5 days avg {_pc(ob_m)} ({'strength begets strength' if ob_m > 0 else 'overbought tends to fade'}); "
                          f"RSI14 ≤ 30 on {int(os_.sum())} days, next 5 days avg {_pc(os_m)}.", ob_n=int(ob.sum()), ob_fwd5=ob_m, os_n=int(os_.sum()), os_fwd5=os_m))

    # Gap fill tendency
    prev_c = c.shift(1)
    gap = f["open"] / prev_c - 1
    g_up = (gap > 0.01).to_numpy()
    g_dn = (gap < -0.01).to_numpy()
    lows, highs, pc = f["low"].to_numpy(), f["high"].to_numpy(), prev_c.to_numpy()
    filled_up, filled_dn, n_up, n_dn = 0, 0, 0, 0
    for i in range(1, len(f) - 5):
        if g_up[i]:
            n_up += 1
            filled_up += bool(np.nanmin(lows[i:i + 5]) <= pc[i])
        if g_dn[i]:
            n_dn += 1
            filled_dn += bool(np.nanmax(highs[i:i + 5]) >= pc[i])
    if n_up + n_dn >= MIN_CASES:
        today_gap = _f(gap.iloc[-1])
        rate_up = filled_up / n_up if n_up else float("nan")
        rate_dn = filled_dn / n_dn if n_dn else float("nan")
        sig = 0
        if today_gap > 0.01 and n_up >= MIN_CASES:
            sig = -1 if rate_up >= 0.65 else 1 if rate_up <= 0.35 else 0
        if today_gap < -0.01 and n_dn >= MIN_CASES:
            sig = 1 if rate_dn >= 0.65 else -1 if rate_dn <= 0.35 else 0
        findings.append(F("gap_fill", "Stock behaviour", "Gap-fill tendency", sig, 0.5,
                          f"{n_up} gaps up > 1%, {_n(rate_up * 100, 0)}% filled within 5 days; {n_dn} gaps down > 1%, "
                          f"{_n(rate_dn * 100, 0)}% filled within 5 days." + (f" Today's gap {_pc(today_gap)}." if abs(today_gap) > 0.01 else ""),
                          up_n=n_up, up_fill=rate_up, down_n=n_dn, down_fill=rate_dn))

    # streaks
    up = (c.diff() > 0).astype(int)
    run = up.groupby((up != up.shift()).cumsum()).cumsum() * up
    dn = (c.diff() < 0).astype(int)
    run_dn = dn.groupby((dn != dn.shift()).cumsum()).cumsum() * dn
    nxt = (c.shift(-1) > c)
    a3 = (run >= 3) & c.shift(-1).notna()
    d3 = (run_dn >= 3) & c.shift(-1).notna()
    if a3.sum() >= MIN_CASES and d3.sum() >= MIN_CASES:
        p_up_after_up = float(nxt[a3].mean())
        p_up_after_dn = float(nxt[d3].mean())
        cur_up, cur_dn = int(run.iloc[-1]), int(run_dn.iloc[-1])
        sig = (1 if p_up_after_up > 0.58 else -1 if p_up_after_up < 0.42 else 0) if cur_up >= 3 else \
              (1 if p_up_after_dn > 0.58 else -1 if p_up_after_dn < 0.42 else 0) if cur_dn >= 3 else 0
        findings.append(F("streaks", "Stock behaviour", "After winning / losing streaks", sig, 0.4,
                          f"After 3+ up days the next day rose {p_up_after_up * 100:.0f}% of the time ({int(a3.sum())} cases); after 3+ down days it bounced "
                          f"{p_up_after_dn * 100:.0f}% ({int(d3.sum())} cases). Now " + (f"up {cur_up} days in a row." if cur_up else f"down {cur_dn} days in a row." if cur_dn else "no streak."),
                          p_up_after_3up=p_up_after_up, p_up_after_3down=p_up_after_dn))

    # 20D breakout follow-through for THIS stock
    brk = (c > f["prior_high20"]) & ~(c.shift(1) > f["prior_high20"].shift(1)) & fwd5.notna()
    if brk.sum() >= MIN_CASES:
        ft = float((fwd5[brk] > 0).mean())
        findings.append(F("breakout_ft", "Stock behaviour", "Follow-through after 20-day-high breakouts",
                          1 if (c.iloc[-1] > f["prior_high20"].iloc[-1] and ft >= 0.6) else -1 if (c.iloc[-1] > f["prior_high20"].iloc[-1] and ft <= 0.4) else 0,
                          0.5, f"{int(brk.sum())} breakouts to 20-day highs; {ft * 100:.0f}% were still higher 5 days later, avg 5-day return {_pc(float(fwd5[brk].mean()))}.",
                          n=int(brk.sum()), follow_through=ft))

    # after big moves (> 2 sigma)
    ret = c.pct_change()
    sig_d = ret.rolling(60, min_periods=40).std().shift(1)
    big_up = (ret > 2 * sig_d) & fwd5.notna()
    big_dn = (ret < -2 * sig_d) & fwd5.notna()
    if big_up.sum() >= MIN_CASES or big_dn.sum() >= MIN_CASES:
        bu, bd = float(fwd5[big_up].mean()) if big_up.sum() else float("nan"), float(fwd5[big_dn].mean()) if big_dn.sum() else float("nan")
        today = _f(ret.iloc[-1]) / _f(sig_d.iloc[-1]) if _f(sig_d.iloc[-1]) > 0 else float("nan")
        sig = (1 if bu > 0.01 else -1 if bu < -0.01 else 0) if today > 2 else (1 if bd > 0.01 else -1 if bd < -0.01 else 0) if today < -2 else 0
        findings.append(F("big_moves", "Stock behaviour", "After big up / down days", sig, 0.45,
                          f"{int(big_up.sum())} up days beyond 2σ, next 5 days avg {_pc(bu)}; {int(big_dn.sum())} down days beyond 2σ, next 5 days avg {_pc(bd)}."
                          + (f" Today's move is {today:+.1f}σ." if np.isfinite(today) and abs(today) > 2 else ""), up_n=int(big_up.sum()), up_fwd5=bu, down_n=int(big_dn.sum()), down_fwd5=bd))

    # weekday seasonality (informational)
    wd = ret.groupby(ret.index.dayofweek).mean().tail(5)
    if len(wd) == 5:
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        best, worst = int(wd.idxmax()), int(wd.idxmin())
        findings.append(F("weekday", "Stock behaviour", "Day-of-week effect", 0, 0, f"Historically {names[best]} has been the best day ({_pc(wd[best], 2)}), "
                          f"{names[worst]} the worst ({_pc(wd[worst], 2)}). Differences are usually not significant."))

    # autocorrelation & Hurst: trending vs mean-reverting character
    r250 = ret.dropna().tail(250)
    if len(r250) >= 200:
        ac1 = float(r250.autocorr(1))
        hurst = _hurst(np.log(c.dropna().tail(250).to_numpy()))
        character = "trend-persistent" if hurst > 0.55 else "mean-reverting" if hurst < 0.45 else "random-walk-like"
        findings.append(F("character", "Stock behaviour", "Price character (Hurst / autocorrelation)", 0, 0,
                          f"Hurst exponent {hurst:.2f}, lag-1 autocorrelation of daily returns {ac1:+.2f}: {character}. "
                          + ("Breakout and trend-following suit it." if hurst > 0.55 else "Buying dips and trimming strength suit it." if hurst < 0.45 else "No clear statistical edge."),
                          hurst=hurst, autocorr1=ac1))


def _hurst(logp: np.ndarray) -> float:
    """Rescaled-range Hurst exponent estimate on log prices."""
    ret = np.diff(logp)
    sizes = [8, 16, 32, 64, 128]
    rs = []
    for n in sizes:
        if len(ret) < n * 2:
            continue
        vals = []
        for i in range(0, len(ret) - n + 1, n):
            seg = ret[i:i + n]
            dev = np.cumsum(seg - seg.mean())
            s = seg.std(ddof=1)
            if s > 0:
                vals.append((dev.max() - dev.min()) / s)
        if vals:
            rs.append((n, np.mean(vals)))
    if len(rs) < 3:
        return float("nan")
    x, y = np.log([a for a, _ in rs]), np.log([b for _, b in rs])
    return float(np.polyfit(x, y, 1)[0])


# ====================================================================== volume profile / cost
def volume_profile(f: pd.DataFrame, n: int = 120, bins: int = 48, decay: float = 0.985) -> dict[str, Any]:
    v = f.tail(n)
    lo, hi = float(v["low"].min()), float(v["high"].max())
    if not (hi > lo):
        return {"available": False}
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    chips = np.zeros(bins)
    ages = np.arange(len(v))[::-1]
    for (h, l, vv), age in zip(v[["high", "low", "volume"]].to_numpy(), ages):
        if not (np.isfinite(h) and np.isfinite(l) and np.isfinite(vv)) or vv <= 0:
            continue
        span = max(h - l, 1e-9)
        overlap = np.clip(np.minimum(edges[1:], h) - np.maximum(edges[:-1], l), 0, None) / span
        if overlap.sum() == 0:
            overlap[np.clip(np.searchsorted(edges, (h + l) / 2) - 1, 0, bins - 1)] = 1
        vol += overlap * vv
        chips += overlap * vv * decay ** age
    mids = (edges[:-1] + edges[1:]) / 2
    poc_i = int(vol.argmax())
    # value area: expand from POC until 70% of volume
    total, acc = vol.sum(), vol[poc_i]
    lo_i, hi_i = poc_i, poc_i
    while acc < 0.7 * total and (lo_i > 0 or hi_i < bins - 1):
        down = vol[lo_i - 1] if lo_i > 0 else -1
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    mean = vol.mean()
    hvn = [i for i in range(1, bins - 1) if vol[i] >= 1.3 * mean and vol[i] >= vol[i - 1] and vol[i] >= vol[i + 1]]
    lvn = [i for i in range(1, bins - 1) if vol[i] <= 0.45 * mean]
    close = float(f["close"].iloc[-1])
    profit_ratio = float(chips[mids <= close].sum() / chips.sum()) if chips.sum() > 0 else float("nan")
    # contiguous low-volume zones (vacuum)
    zones, cur = [], []
    for i in lvn:
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            if cur:
                zones.append(cur)
            cur = [i]
    if cur:
        zones.append(cur)
    vac = [{"low": float(edges[z[0]]), "high": float(edges[z[-1] + 1])} for z in zones if len(z) >= 2]
    return {"available": True, "poc": float(mids[poc_i]), "val": float(edges[lo_i]), "vah": float(edges[hi_i + 1]),
            "hvn": [float(mids[i]) for i in hvn], "vacuum": vac, "profit_ratio": profit_ratio,
            "bins": [{"price": float(m), "volume": float(x)} for m, x in zip(mids, vol)], "window": n}


def anchored_vwap(f: pd.DataFrame, anchor: pd.Timestamp) -> float:
    v = f.loc[anchor:]
    tp = (v["high"] + v["low"] + v["close"]) / 3
    w = v["volume"].fillna(0)
    return float((tp * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def chips(f: pd.DataFrame, findings: list) -> dict[str, Any]:
    vp = volume_profile(f)
    close, atr = float(f["close"].iloc[-1]), _f(f["atr"].iloc[-1])
    if vp.get("available"):
        pos = "above the value area" if close > vp["vah"] else "below the value area" if close < vp["val"] else "inside the value area"
        sig = 1 if close > vp["vah"] else -1 if close < vp["val"] else 0
        findings.append(F("vp_value_area", "Volume profile", "Volume profile: POC and value area", sig, 0.6,
                          f"Over {vp['window']} days the most traded price (POC) is {_n(vp['poc'])}; 70% of volume traded {_n(vp['val'])}–{_n(vp['vah'])} (value area). "
                          f"Price {_n(close)} is {pos}" + (": higher prices are being accepted - strength." if sig > 0 else ": higher prices are being rejected - weakness." if sig < 0 else ": in balance."),
                          poc=vp["poc"], val=vp["val"], vah=vp["vah"]))
        above = sorted([x for x in vp["hvn"] if x > close * 1.005])
        below = sorted([x for x in vp["hvn"] if x < close * 0.995], reverse=True)
        txt = "High-volume nodes (dense supply / demand): " + (f"above {', '.join(_n(x) for x in above[:2])} (likely resistance)" if above else "none above")
        txt += "; " + (f"below {', '.join(_n(x) for x in below[:2])} (likely support)" if below else "none below") + "."
        near_res = above and np.isfinite(atr) and (above[0] - close) <= 1.0 * atr
        findings.append(F("vp_hvn", "Volume profile", "High-volume nodes", -1 if near_res else 0, 0.5 if near_res else 0, txt, above=above[:3], below=below[:3]))
        vac_up = [z for z in vp["vacuum"] if z["low"] >= close * 0.995]
        vac_dn = [z for z in vp["vacuum"] if z["high"] <= close * 1.005]
        vtxt = "Volume vacuums (thin volume, price moves through quickly): " + (
            "above " + ", ".join(f"{_n(z['low'])}–{_n(z['high'])}" for z in vac_up[:2]) if vac_up else "none above") + "; " + (
            "below " + ", ".join(f"{_n(z['low'])}–{_n(z['high'])}" for z in vac_dn[-2:]) if vac_dn else "none below") + ". "
        vsig = 0
        if vac_up and np.isfinite(atr) and vac_up[0]["low"] - close <= 1.0 * atr:
            vsig = 1
            vtxt += "A vacuum sits just above - a breakout could accelerate."
        elif vac_dn and np.isfinite(atr) and close - vac_dn[-1]["high"] <= 1.0 * atr:
            vsig = -1
            vtxt += "A vacuum sits just below - a breakdown could accelerate."
        findings.append(F("vp_vacuum", "Volume profile", "Volume vacuums", vsig, 0.5, vtxt, up=vac_up[:2], down=vac_dn[-2:]))
        pr = vp["profit_ratio"]
        findings.append(F("profit_ratio", "Volume profile", "Share of holders in profit", 1 if pr >= 0.85 else -1 if pr <= 0.15 else 0, 0.4,
                          f"From 120 days of volume (recent days weighted more), about {pr * 100:.0f}% of shares were bought below the current price (in profit). "
                          + ("Nearly every holder is in profit - little trapped supply overhead." if pr >= 0.85 else "Most holders are under water - rallies may meet break-even selling." if pr <= 0.15 else ""),
                          profit_ratio=pr))
    # anchored VWAPs
    window = f.tail(250)
    anchors = {"52-week high": window["high"].idxmax(), "52-week low": window["low"].idxmin()}
    gaps = f.tail(60)
    big_gap = gaps[gaps["gap_atr"].abs() >= 1]
    if not big_gap.empty:
        anchors["last big gap"] = big_gap.index[-1]
    av = {k: anchored_vwap(f, d) for k, d in anchors.items()}
    above = [k for k, v in av.items() if np.isfinite(v) and close > v]
    findings.append(F("avwap", "Volume profile", "Anchored VWAP", 1 if len(above) == len(av) else -1 if not above else 0, 0.5,
                      "Anchored VWAP (average cost of all volume since the anchor): " + "; ".join(f"from the {k} ({d:%Y-%m-%d}) {_n(av[k])}, price {'above' if close > av[k] else 'below'}"
                                                                    for k, d in anchors.items()) + ". "
                      + ("Every major holder cohort is in profit." if len(above) == len(av) else "Every major holder cohort is under water." if not above else ""),
                      **{k: v for k, v in av.items()}))
    cost20, cost60 = _f(f["vwap20"].iloc[-1]), anchored_vwap(f, f.index[-60]) if len(f) >= 60 else float("nan")
    findings.append(F("holder_cost", "Volume profile", "Short / medium-term holder cost", 1 if close > cost20 and close > cost60 else -1 if close < cost20 and close < cost60 else 0, 0.4,
                      f"Volume-weighted average cost 20d {_n(cost20)}, 60d {_n(cost60)}; price vs 20-day cost {_pc(close / cost20 - 1)}.", cost20=cost20, cost60=cost60))
    return vp


# ====================================================================== patterns
def patterns(f: pd.DataFrame, findings: list) -> None:
    c, h, lo, v = f["close"], f["high"], f["low"], f["volume"]
    t, y = f.iloc[-1], f.iloc[-2]
    atr = _f(t.atr)
    is_high, is_low = swing_points(h, lo, 3)
    ph = [(d, float(h[d])) for d in h.index[is_high.fillna(False)]][-8:]
    pl = [(d, float(lo[d])) for d in lo.index[is_low.fillna(False)]][-8:]

    # VCP: successive pullbacks getting shallower
    depths = []
    for d, p in ph[-5:]:
        after = lo[d:].iloc[1:12]
        if not after.empty:
            depths.append(1 - float(after.min()) / p)
    vcp = len(depths) >= 3 and all(depths[i] > depths[i + 1] for i in range(len(depths) - 3, len(depths) - 1)) and depths[-1] < 0.12 \
        and t.close > _f(t.ema50)
    findings.append(F("vcp", "Patterns", "Volatility contraction pattern (VCP)", 1 if vcp else 0, 0.7 if vcp else 0,
                      ("Each recent pullback is shallower (" + " → ".join(f"{x * 100:.0f}%" for x in depths[-3:]) + "): supply is being absorbed, "
                       "a classic pre-breakout pattern." if vcp else "No volatility contraction pattern."), depths=depths[-3:]))

    # flags
    ret10 = c / c.shift(10) - 1
    pole_end = ret10.tail(25).idxmax() if ret10.tail(25).notna().any() else None
    flag = None
    if pole_end is not None and _f(ret10[pole_end]) >= 0.15:
        cons = f.loc[pole_end:].iloc[1:]
        if 4 <= len(cons) <= 15:
            pole = _f(ret10[pole_end]) * float(c.shift(10)[pole_end])
            depth = float(h[pole_end] - cons["low"].min())
            if depth <= 0.5 * pole and cons["volume"].mean() < v.loc[:pole_end].tail(10).mean():
                flag = "bull"
    pole_end_d = ret10.tail(25).idxmin() if ret10.tail(25).notna().any() else None
    if flag is None and pole_end_d is not None and _f(ret10[pole_end_d]) <= -0.15:
        cons = f.loc[pole_end_d:].iloc[1:]
        if 4 <= len(cons) <= 15:
            pole = -_f(ret10[pole_end_d]) * float(c.shift(10)[pole_end_d])
            bounce = float(cons["high"].max() - lo[pole_end_d])
            if bounce <= 0.5 * pole:
                flag = "bear"
    findings.append(F("flag", "Patterns", "Flag", 1 if flag == "bull" else -1 if flag == "bear" else 0, 0.6 if flag else 0,
                      "Bull flag: a sharp rise (pole) then a tight, quiet consolidation; a break above it may resume the advance." if flag == "bull" else
                      "Bear flag: a sharp drop then a weak bounce; a break below it may resume the decline." if flag == "bear" else "No flag."))

    # double top / bottom
    dt_txt, dt_sig = "No double top or bottom.", 0
    if len(ph) >= 2:
        (d1, p1), (d2, p2) = ph[-2], ph[-1]
        gap_bars = f.index.get_loc(d2) - f.index.get_loc(d1)
        trough = float(lo[d1:d2].min())
        if abs(p2 / p1 - 1) <= 0.02 and gap_bars >= 8 and trough <= min(p1, p2) * 0.95 and f.index.get_loc(f.index[-1]) - f.index.get_loc(d2) <= 30:
            dt_sig = -1 if t.close < trough else 0
            dt_txt = (f"Double top: highs {_n(p1)} ({d1:%m-%d}) and {_n(p2)} ({d2:%m-%d}), neckline {_n(trough)}. "
                      + ("Neckline broken - confirmed." if t.close < trough else "Neckline intact - not confirmed."))
    if dt_sig == 0 and len(pl) >= 2:
        (d1, p1), (d2, p2) = pl[-2], pl[-1]
        gap_bars = f.index.get_loc(d2) - f.index.get_loc(d1)
        peak = float(h[d1:d2].max())
        if abs(p2 / p1 - 1) <= 0.02 and gap_bars >= 8 and peak >= max(p1, p2) * 1.05 and f.index.get_loc(f.index[-1]) - f.index.get_loc(d2) <= 30:
            dt_sig = 1 if t.close > peak else 0
            dt_txt = (f"Double bottom: lows {_n(p1)} ({d1:%m-%d}) and {_n(p2)} ({d2:%m-%d}), neckline {_n(peak)}. "
                      + ("Neckline cleared - confirmed." if t.close > peak else "Neckline not cleared - not confirmed."))
    findings.append(F("double", "Patterns", "Double top / bottom", dt_sig, 0.6, dt_txt))

    # pocket pivot (O'Neil): up day volume > max down-day volume of prior 10 sessions
    prev10 = f.iloc[-11:-1]
    max_down_vol = float(prev10.loc[prev10["close"] < prev10["close"].shift(1), "volume"].max()) if len(prev10) else float("nan")
    pp = t.close > y.close and np.isfinite(max_down_vol) and t.volume > max_down_vol and t.close > _f(t.ema21)
    findings.append(F("pocket_pivot", "Patterns", "Pocket pivot", 1 if pp else 0, 0.6 if pp else 0,
                      "Up-day volume above any down day of the last 10, above EMA21: a footprint of institutional buying." if pp else "No pocket pivot today."))

    # climax volume / exhaustion
    vmax60 = float(v.tail(60).max())
    climax = t.volume >= vmax60 and (t.high - t.low) > 1.5 * atr
    cl_sig = -1 if climax and t.close > _f(t.ema21) and _f(t.clv) < 0.4 else 1 if climax and t.close < _f(t.ema21) and _f(t.clv) > 0.6 else 0
    findings.append(F("climax", "Patterns", "Climax volume", cl_sig, 0.6 if cl_sig else 0,
                      ("60-day-high volume on a huge range, " + ("closing weak in an uptrend: possible buying climax / top." if cl_sig < 0 else
                                                     "closing strong in a downtrend: possible capitulation low." if cl_sig > 0 else "needs follow-through to confirm."))
                      if climax else "No climax volume."))

    # key reversal / undercut & rally / failed breakout / failed breakdown
    kr = t.high > _f(f["high"].iloc[-21:-1].max()) and t.close < y.close
    findings.append(F("key_reversal", "Patterns", "Key reversal day", -1 if kr else 0, 0.6 if kr else 0,
                      "New 20-day high, then a close below yesterday's close (bearish key reversal)." if kr else "No key reversal."))
    low10 = float(f["low"].iloc[-11:-1].min())
    uc = t.low < low10 and t.close > low10
    findings.append(F("undercut_rally", "Patterns", "Undercut & rally", 1 if uc else 0, 0.6 if uc else 0,
                      f"Undercut the 10-day low {_n(low10)} intraday and closed back above it - stops shaken out, then a recovery." if uc else "No undercut & rally."))
    recent = f.tail(6)
    fb = bool(((recent["close"] > recent["prior_high20"]).iloc[:-1]).any()) and t.close < _f(f["prior_high20"].iloc[-6])
    fd = bool(((recent["close"] < recent["prior_low20"]).iloc[:-1]).any()) and t.close > _f(f["prior_low20"].iloc[-6])
    findings.append(F("failed_break", "Patterns", "Failed breakout / breakdown", -1 if fb else 1 if fd else 0, 0.7 if (fb or fd) else 0,
                      "Broke the 20-day high within 5 days but is back below it (failed breakout, bull trap)." if fb else
                      "Broke the 20-day low within 5 days but is back above it (failed breakdown, bear trap)." if fd else "No failed breakout or breakdown."))

    # NR7 / inside day
    rng = h - lo
    nr7 = rng.iloc[-1] <= rng.tail(7).min()
    inside = t.high < y.high and t.low > y.low
    findings.append(F("nr7", "Patterns", "Narrow range (NR7 / inside day)", 0, 0,
                      ("Narrowest range in 7 days (NR7)" + (" and an inside day" if inside else "") + ": volatility compression often precedes a big move.")
                      if nr7 or inside else "Not a narrow-range day.", nr7=bool(nr7), inside=bool(inside)))

    # 3 weeks tight (weekly closes within 1.5%)
    wc = c.resample("W-FRI").last().dropna().tail(3)
    tight = len(wc) == 3 and (wc.max() / wc.min() - 1) <= 0.015
    findings.append(F("three_weeks_tight", "Patterns", "Three weeks tight", 1 if tight and t.close > _f(t.ema50) else 0, 0.5 if tight else 0,
                      "The last 3 weekly closes are within 1.5%: stable holders, common before breakouts in leaders." if tight else "Not three weeks tight."))

    # regression channel position
    y60 = c.tail(60).to_numpy()
    if len(y60) == 60:
        x = np.arange(60)
        slope, icpt = np.polyfit(x, y60, 1)
        resid = y60 - (slope * x + icpt)
        z = resid[-1] / resid.std(ddof=1)
        findings.append(F("reg_channel", "Patterns", "60-day regression channel", -1 if z > 1.8 else 1 if z < -1.8 else 0, 0.4,
                          f"60-day regression slope {slope / y60.mean() * 100:+.2f}% per day; price sits {z:+.1f}σ within the channel"
                          + (" (near the top - stretched)" if z > 1.8 else " (near the bottom - depressed)" if z < -1.8 else "") + ".", z=float(z)))

    # support / resistance touches
    lvl_txt = []
    for name, series, cmp in (("resistance", h, 1), ("support", lo, -1)):
        ref = float(f["high"].tail(60).max()) if cmp > 0 else float(f["low"].tail(60).min())
        touches = int(((series.tail(120) - ref).abs() <= 0.5 * atr).sum()) if np.isfinite(atr) else 0
        lvl_txt.append(f"60-day {'high' if cmp > 0 else 'low'} {_n(ref)} as {name}, tested {touches} times in 120 days")
    findings.append(F("sr_touches", "Patterns", "Support / resistance tests", 0, 0, "; ".join(lvl_txt) + ". The more a level is tested, the more it matters - a break often triggers a bigger move."))


# ====================================================================== indicator consensus
def indicators(f: pd.DataFrame, findings: list) -> None:
    h, lo, c, v = f["high"], f["low"], f["close"], f["volume"]
    t = f.iloc[-1]
    # Stochastic 14,3
    ll, hh = lo.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    kd = k.rolling(3).mean()
    dd = kd.rolling(3).mean()
    st_sig = 1 if kd.iloc[-1] > dd.iloc[-1] and kd.iloc[-2] <= dd.iloc[-2] and kd.iloc[-1] < 40 else \
        -1 if kd.iloc[-1] < dd.iloc[-1] and kd.iloc[-2] >= dd.iloc[-2] and kd.iloc[-1] > 60 else 0
    findings.append(F("stoch", "Indicator consensus", "Stochastic (14,3,3)", st_sig, 0.4, f"%K {_n(kd.iloc[-1], 0)}, %D {_n(dd.iloc[-1], 0)}"
                      + (": bullish cross from oversold." if st_sig > 0 else ": bearish cross from overbought." if st_sig < 0 else ".")))
    # Williams %R
    wr = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    findings.append(F("williams", "Indicator consensus", "Williams %R (14)", 1 if wr.iloc[-1] < -80 and wr.iloc[-1] > wr.iloc[-2] else -1 if wr.iloc[-1] > -20 and wr.iloc[-1] < wr.iloc[-2] else 0,
                      0.3, f"%R {_n(wr.iloc[-1], 0)} (below -80 oversold, above -20 overbought)."))
    # CCI 20
    tp = (h + lo + c) / 3
    md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - tp.rolling(20).mean()) / (0.015 * md)
    findings.append(F("cci", "Indicator consensus", "CCI (20)", 1 if cci.iloc[-1] > 100 else -1 if cci.iloc[-1] < -100 else 0, 0.3,
                      f"CCI {_n(cci.iloc[-1], 0)} (> 100 strong, < -100 weak)."))
    # MFI 14
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    findings.append(F("mfi", "Indicator consensus", "Money Flow Index (14)", -1 if mfi.iloc[-1] > 80 else 1 if mfi.iloc[-1] < 20 else (1 if mfi.iloc[-1] > 50 else -1), 0.3,
                      f"MFI {_n(mfi.iloc[-1], 0)} (volume-weighted RSI; > 80 overheated, < 20 exhausted)."))
    # Chaikin money flow 20 & A/D line
    clv = ((c - lo) - (h - c)) / (h - lo).replace(0, np.nan)
    cmf = (clv * v).rolling(20).sum() / v.rolling(20).sum()
    ad = (clv.fillna(0) * v).cumsum()
    ad_up = ad.iloc[-1] > ad.iloc[-21]
    findings.append(F("cmf", "Indicator consensus", "Chaikin Money Flow (20) / A/D line", 1 if cmf.iloc[-1] > 0.05 and ad_up else -1 if cmf.iloc[-1] < -0.05 and not ad_up else 0, 0.4,
                      f"CMF {_n(cmf.iloc[-1])}, accumulation/distribution line {'rising' if ad_up else 'falling'} over 20 days (close-location-weighted money flow)."))
    # TTM squeeze & Keltner
    mid = sma(c, 20)
    sd = c.rolling(20).std(ddof=0)
    atr20 = atr_fn(h, lo, c, 20)
    kc_up, kc_dn = ema(c, 20) + 1.5 * atr20, ema(c, 20) - 1.5 * atr20
    squeeze_on = (mid + 2 * sd < kc_up) & (mid - 2 * sd > kc_dn)
    mom = c - ((hh + ll) / 2 + mid) / 2
    fired = bool(squeeze_on.iloc[-2]) and not bool(squeeze_on.iloc[-1])
    findings.append(F("ttm_squeeze", "Indicator consensus", "TTM squeeze", (1 if mom.iloc[-1] > 0 else -1) if fired else 0, 0.6 if fired else 0,
                      ("Squeeze just fired (Bollinger bands expanding out of the Keltner channel), momentum " + ("up." if mom.iloc[-1] > 0 else "down.")) if fired else
                      f"Squeeze {'on (Bollinger bands inside the Keltner channel, energy building)' if squeeze_on.iloc[-1] else 'off'}, momentum {'up' if mom.iloc[-1] > 0 else 'down'}."))
    findings.append(F("keltner", "Indicator consensus", "Keltner channel", 1 if c.iloc[-1] > kc_up.iloc[-1] else -1 if c.iloc[-1] < kc_dn.iloc[-1] else 0, 0.3,
                      f"Keltner channel (EMA20 ± 1.5 ATR): {_n(kc_dn.iloc[-1])}–{_n(kc_up.iloc[-1])}, price "
                      + ("above the upper band (strong)." if c.iloc[-1] > kc_up.iloc[-1] else "below the lower band (weak)." if c.iloc[-1] < kc_dn.iloc[-1] else "inside the channel.")))
    # Supertrend (10, 3)
    st_dir = _supertrend(h, lo, c, 10, 3.0)
    findings.append(F("supertrend", "Indicator consensus", "Supertrend (10,3)", st_dir["dir"], 0.5,
                      f"Supertrend {'up' if st_dir['dir'] > 0 else 'down'}, line {_n(st_dir['line'])}, for {st_dir['age']} days"
                      + (" (just flipped)." if st_dir["age"] <= 2 else "."), line=st_dir["line"]))
    # Ichimoku
    tenkan = (h.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    top, bot = max(_f(span_a.iloc[-1]), _f(span_b.iloc[-1])), min(_f(span_a.iloc[-1]), _f(span_b.iloc[-1]))
    ich = 1 if c.iloc[-1] > top and tenkan.iloc[-1] > kijun.iloc[-1] else -1 if c.iloc[-1] < bot and tenkan.iloc[-1] < kijun.iloc[-1] else 0
    findings.append(F("ichimoku", "Indicator consensus", "Ichimoku", ich, 0.5,
                      f"Cloud {_n(bot)}–{_n(top)}, price " + ("above the cloud" if c.iloc[-1] > top else "below the cloud" if c.iloc[-1] < bot else "inside the cloud")
                      + f"; Tenkan {_n(tenkan.iloc[-1])} {'above' if tenkan.iloc[-1] > kijun.iloc[-1] else 'below'} Kijun {_n(kijun.iloc[-1])}."))
    # Parabolic SAR
    sar_dir, sar = _psar(h, lo)
    findings.append(F("psar", "Indicator consensus", "Parabolic SAR", sar_dir, 0.3, f"SAR {_n(sar)}, {'below price (uptrend)' if sar_dir > 0 else 'above price (downtrend)'}."))
    # Donchian
    d20h, d20l = h.iloc[-21:-1].max(), lo.iloc[-21:-1].min()
    d55h, d55l = h.iloc[-56:-1].max(), lo.iloc[-56:-1].min()
    don = 1 if c.iloc[-1] > d55h else -1 if c.iloc[-1] < d55l else (1 if c.iloc[-1] > d20h else -1 if c.iloc[-1] < d20l else 0)
    findings.append(F("donchian", "Indicator consensus", "Donchian channels (Turtle 20/55)", don, 0.4,
                      f"20-day channel {_n(d20l)}–{_n(d20h)}, 55-day {_n(d55l)}–{_n(d55h)}"
                      + ("; above the 55-day high (Turtle long-term entry)." if c.iloc[-1] > d55h else "; below the 55-day low." if c.iloc[-1] < d55l else ".")))
    # Heikin-Ashi
    ha_c = (f["open"] + h + lo + c) / 4
    ha_o = ha_c.copy()
    ha_o.iloc[0] = (f["open"].iloc[0] + c.iloc[0]) / 2
    hc, ho = ha_c.to_numpy(copy=True), ha_o.to_numpy(copy=True)
    for i in range(1, len(ho)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    color = np.sign(hc - ho)
    streak = 1
    while streak < len(color) and color[-1 - streak] == color[-1]:
        streak += 1
    findings.append(F("heikin_ashi", "Indicator consensus", "Heikin-Ashi", int(color[-1]), 0.3,
                      f"{streak} consecutive {'up' if color[-1] > 0 else 'down'} Heikin-Ashi candles (noise-filtered trend direction)."))
    # Elder ray
    e13 = ema(c, 13)
    bull, bear = h - e13, lo - e13
    er = 1 if e13.iloc[-1] > e13.iloc[-2] and bear.iloc[-1] < 0 and bear.iloc[-1] > bear.iloc[-2] else \
        -1 if e13.iloc[-1] < e13.iloc[-2] and bull.iloc[-1] > 0 and bull.iloc[-1] < bull.iloc[-2] else 0
    findings.append(F("elder_ray", "Indicator consensus", "Elder Ray", er, 0.4, f"Bull power {_n(bull.iloc[-1])}, bear power {_n(bear.iloc[-1])}, EMA13 {'rising' if e13.iloc[-1] > e13.iloc[-2] else 'falling'}"
                      + (": bear power fading in an uptrend - Elder's buy condition." if er > 0 else ": bull power fading in a downtrend - Elder's sell condition." if er < 0 else ".")))
    # Aroon 25
    au = 100 * (25 - (h.tail(26).to_numpy()[::-1].argmax())) / 25
    ad_ = 100 * (25 - (lo.tail(26).to_numpy()[::-1].argmin())) / 25
    findings.append(F("aroon", "Indicator consensus", "Aroon (25)", 1 if au > 70 and ad_ < 30 else -1 if ad_ > 70 and au < 30 else 0, 0.3,
                      f"Aroon up {au:.0f}, down {ad_:.0f} (time since the latest high / low)."))
    # RSI(2) short-term mean reversion (Connors)
    r2 = rsi(c, 2)
    r2s = 1 if r2.iloc[-1] < 10 and c.iloc[-1] > _f(t.ema200) else -1 if r2.iloc[-1] > 90 and c.iloc[-1] < _f(t.ema200) else 0
    findings.append(F("rsi2", "Indicator consensus", "RSI(2) short-term mean reversion", r2s, 0.4, f"RSI(2) {_n(r2.iloc[-1], 0)}"
                      + (": deeply oversold within a long-term uptrend (Connors buy condition)." if r2s > 0 else ": deeply overbought within a long-term downtrend (Connors sell condition)." if r2s < 0 else ".")))
    # z-score vs 20-day mean
    z = (c.iloc[-1] - mid.iloc[-1]) / sd.iloc[-1] if sd.iloc[-1] > 0 else float("nan")
    findings.append(F("zscore", "Indicator consensus", "20-day z-score", -1 if z > 2.2 else 1 if z < -2.2 else 0, 0.3, f"Price is {z:+.1f}σ from its 20-day mean."))


def _supertrend(h: pd.Series, lo: pd.Series, c: pd.Series, n: int, mult: float) -> dict[str, Any]:
    a = atr_fn(h, lo, c, n).to_numpy()
    hl2 = ((h + lo) / 2).to_numpy()
    up, dn = hl2 - mult * a, hl2 + mult * a
    cc = c.to_numpy()
    direction = np.ones(len(cc))
    fu, fd = up.copy(), dn.copy()
    for i in range(1, len(cc)):
        if np.isnan(a[i]):
            continue
        fu[i] = max(up[i], fu[i - 1]) if cc[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if cc[i - 1] < fd[i - 1] else dn[i]
        direction[i] = 1 if cc[i] > fd[i - 1] else -1 if cc[i] < fu[i - 1] else direction[i - 1]
    age = 1
    while age < len(direction) and direction[-1 - age] == direction[-1]:
        age += 1
    return {"dir": int(direction[-1]), "line": float(fu[-1] if direction[-1] > 0 else fd[-1]), "age": age}


def _psar(h: pd.Series, lo: pd.Series, step: float = 0.02, cap: float = 0.2) -> tuple[int, float]:
    hh, ll = h.to_numpy(), lo.to_numpy()
    up, af, ep, sar = True, step, hh[0], ll[0]
    for i in range(1, len(hh)):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, ll[i - 1], ll[i - 2] if i > 1 else ll[i - 1])
            if ll[i] < sar:
                up, sar, ep, af = False, ep, ll[i], step
            elif hh[i] > ep:
                ep, af = hh[i], min(af + step, cap)
        else:
            sar = max(sar, hh[i - 1], hh[i - 2] if i > 1 else hh[i - 1])
            if hh[i] > sar:
                up, sar, ep, af = True, ep, hh[i], step
            elif ll[i] < ep:
                ep, af = ll[i], min(af + step, cap)
    return (1 if up else -1), float(sar)


# ====================================================================== relative behaviour
def relative(f: pd.DataFrame, spy: pd.Series | None, findings: list) -> None:
    if spy is None:
        return
    s = spy.reindex(f.index).ffill()
    r, m = f["close"].pct_change(), s.pct_change()
    j = pd.concat([r, m], axis=1).dropna().tail(120)
    if len(j) < 60:
        return
    corr = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
    beta = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
    up_days, dn_days = j[j.iloc[:, 1] > 0], j[j.iloc[:, 1] < 0]
    upcap = float(up_days.iloc[:, 0].mean() / up_days.iloc[:, 1].mean()) if len(up_days) else float("nan")
    dncap = float(dn_days.iloc[:, 0].mean() / dn_days.iloc[:, 1].mean()) if len(dn_days) else float("nan")
    meaningful = beta > 0.3 and upcap > 0 and dncap > 0
    cap_sig = (1 if upcap > dncap * 1.15 else -1 if dncap > upcap * 1.15 else 0) if meaningful else 0
    findings.append(F("capture", "Relative performance", "Up / down capture", cap_sig, 0.5,
                      f"120-day correlation with SPY {corr:.2f}, beta {beta:.2f}; "
                      + (f"on up days it moves {upcap:.1f}x the market, on down days {dncap:.1f}x"
                         + (" (stronger up, more resilient down - a leader trait)." if cap_sig > 0 else " (falls more on down days - weak high beta)." if cap_sig < 0 else ".")
                         if meaningful else "weak or inverse link to the market, so capture ratios mean little."),
                      corr=corr, beta=beta, up_capture=upcap, down_capture=dncap))
    resid = j.iloc[:, 0] - beta * j.iloc[:, 1]
    today = float(r.iloc[-1] - beta * m.iloc[-1]) if np.isfinite(r.iloc[-1]) and np.isfinite(m.iloc[-1]) else float("nan")
    zt = today / float(resid.std(ddof=1)) if resid.std(ddof=1) > 0 else float("nan")
    findings.append(F("idio", "Relative performance", "Idiosyncratic move today", 1 if zt > 2 else -1 if zt < -2 else 0, 0.5 if abs(_f(zt)) > 2 else 0,
                      f"After removing the market (beta × SPY), today's stock-specific return is {_pc(today, 2)}, about {zt:+.1f}σ"
                      + (" - possibly news or flow driven." if abs(_f(zt)) > 2 else ".")))
    rs = (f["close"] / s).dropna()
    lead = len(rs) >= 60 and rs.iloc[-1] >= rs.tail(60).max() * 0.995 and f["close"].iloc[-1] < f["close"].tail(60).max() * 0.98
    findings.append(F("rs_lead", "Relative performance", "RS line leading", 1 if lead else 0, 0.6 if lead else 0,
                      "The RS line made a 60-day high before price did: a classic sign of money flowing in early." if lead else "The RS line is not leading price to new highs."))


# ====================================================================== entry point
def run_all(f: pd.DataFrame, spy: pd.Series | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    vp: dict[str, Any] = {}
    for step in (lambda: behaviour(f, findings), lambda: patterns(f, findings), lambda: indicators(f, findings),
                 lambda: relative(f, spy, findings)):
        try:
            step()
        except Exception as exc:  # noqa: BLE001 - one broken technique must not kill the note
            findings.append(F("error", "System", "Calculation error", 0, 0, f"Some technical calculations failed: {type(exc).__name__}"))
    try:
        vp = chips(f, findings)
    except Exception as exc:  # noqa: BLE001
        findings.append(F("error_vp", "System", "Calculation error", 0, 0, f"Volume-profile calculation failed: {type(exc).__name__}"))
    groups: dict[str, dict[str, Any]] = {}
    for x in findings:
        g = groups.setdefault(x["group"], {"group": x["group"], "bull": 0, "bear": 0, "score": 0.0, "n": 0})
        g["n"] += 1
        g["bull"] += x["signal"] > 0
        g["bear"] += x["signal"] < 0
        g["score"] += x["signal"] * x["strength"]
    return {"findings": findings, "groups": list(groups.values()), "volume_profile": vp, "count": len(findings)}
