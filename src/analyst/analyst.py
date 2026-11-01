"""Analyst note: one synthesised, plain-language view of a stock (Traditional Chinese).

It combines everything the terminal knows about a stock and writes it up the way a desk analyst would:
bottom line first, then the evidence by theme, key levels, scenarios, what to watch and the risks.

Scorecard (each -2 ... +2, weights sum to 100; bias = sum(weight x score) / 2 -> -100 ... +100):
    weekly trend 11 · daily trend 11 · momentum 9 · indicator consensus 6 · relative strength 9 ·
    volume / money flow 8 · chip distribution 8 · patterns & stock behaviour 8 · position / extension 5 ·
    fundamentals 9 · options sentiment 4 · positioning 4 · catalysts 4 · market environment 4
The ~50 techniques in techniques.py feed the consensus, chips and patterns components and are all listed.
Label: >= 45 Strongly bullish | >= 20 Bullish | >= 7 Leaning bullish | > -7 Neutral | > -20 Leaning bearish | > -45 Bearish | else Strongly bearish
Confidence: share of non-zero components agreeing with the bias, |bias|, data completeness and imminent
earnings (a binary event lowers confidence).

Every statement is derived from data with an explicit rule below; nothing is invented. Descriptive
research, not investment advice.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.technical.indicators import ema, rsi
from src.analyst.techniques import run_all
from src.technical.structure import support_resistance, swing_points

WEIGHTS = {"weekly": 11, "daily": 11, "momentum": 9, "consensus": 6, "relative": 9, "volume": 8, "chips": 8,
           "patterns": 8, "position": 5, "fundamentals": 9, "options": 4, "positioning": 4, "catalysts": 4, "market": 4}
NAMES = {"weekly": "Weekly trend", "daily": "Daily trend", "momentum": "Momentum", "consensus": "Indicator consensus", "relative": "Relative strength",
         "volume": "Volume / money flow", "chips": "Volume profile", "patterns": "Patterns & behaviour", "position": "Position / extension",
         "fundamentals": "Fundamentals", "options": "Options sentiment", "positioning": "Positioning / short", "catalysts": "Catalysts", "market": "Market backdrop"}
TREND_SCORE = {"Strong short-term uptrend": 2, "Short-term uptrend": 1.5, "Improving": 0.5, "Mixed": 0,
               "Deteriorating": -0.5, "Short-term downtrend": -1.5, "Strong short-term downtrend": -2}
REGIME_SCORE = {"Strong Bullish": 2, "Bullish": 1, "Neutral": 0, "Bearish": -1, "Strong Bearish": -2}
QUADRANT_SCORE = {"Leading": 1, "Improving": 0.5, "Weakening": -0.5, "Lagging": -1}


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _ok(x: Any) -> bool:
    return np.isfinite(_f(x))


def _p(x: float, d: int = 1, sign: bool = True) -> str:
    if not _ok(x):
        return "n/a"
    v = round(x * 100, d) + 0
    return f"{v:+.{d}f}%" if sign else f"{v:.{d}f}%"


def _n(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}" if _ok(x) else "n/a"


def _clip(x: float, lo: float = -2, hi: float = 2) -> float:
    return float(np.clip(x, lo, hi)) if np.isfinite(x) else 0.0


# ------------------------------------------------------------------ analytics
def weekly_view(f: pd.DataFrame) -> dict[str, Any]:
    """Weekly bars (Friday close): EMA10 / EMA30, weekly RSI14 and trend."""
    w = pd.DataFrame({"close": f["close"].resample("W-FRI").last(), "high": f["high"].resample("W-FRI").max(),
                      "low": f["low"].resample("W-FRI").min(), "volume": f["volume"].resample("W-FRI").sum()}).dropna()
    if len(w) < 35:
        return {"available": False}
    e10, e30 = ema(w["close"], 10), ema(w["close"], 30)
    r = rsi(w["close"], 14)
    c, a, b = w["close"].iloc[-1], e10.iloc[-1], e30.iloc[-1]
    slope30 = e30.iloc[-1] / e30.iloc[-4] - 1 if len(e30.dropna()) > 4 else float("nan")
    if c > a > b and slope30 > 0:
        label, score = "uptrend", 2
    elif c > b and slope30 > 0:
        label, score = "pullback in an uptrend", 1
    elif c < a < b and slope30 < 0:
        label, score = "downtrend", -2
    elif c < b and slope30 < 0:
        label, score = "bounce in a downtrend", -1
    else:
        label, score = "range-bound", 0
    return {"available": True, "label": label, "score": score, "close": float(c), "ema10": float(a), "ema30": float(b),
            "rsi14": float(r.iloc[-1]), "ema30_slope": float(slope30), "weeks_up": int((w["close"].diff().tail(8) > 0).sum())}


def trend_quality(close: pd.Series, n: int = 50) -> dict[str, float]:
    """R² and annualised slope of a log-price regression over n sessions (how clean the trend is)."""
    y = np.log(close.dropna().tail(n).to_numpy())
    if len(y) < n * 0.8:
        return {"r2": float("nan"), "slope_ann": float("nan")}
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    ss_res, ss_tot = float(((y - fit) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {"r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "slope_ann": float(math.expm1(slope * 252))}


def divergences(f: pd.DataFrame, lookback: int = 70) -> list[dict[str, Any]]:
    """Price vs RSI14 / MACD-histogram divergence between the last two CONFIRMED swing points."""
    view = f.tail(lookback)
    is_high, is_low = swing_points(view["high"], view["low"], 3)
    out = []
    for kind, mask, col in (("bearish divergence", is_high, "high"), ("bullish divergence", is_low, "low")):
        idx = [i for i, flag in enumerate(mask.to_numpy()) if flag]
        if len(idx) < 2:
            continue
        i1, i2 = idx[-2], idx[-1]
        p1, p2 = view[col].iloc[i1], view[col].iloc[i2]
        r1, r2 = view["rsi14"].iloc[i1], view["rsi14"].iloc[i2]
        h1, h2 = view["macd_hist"].iloc[i1], view["macd_hist"].iloc[i2]
        d1, d2 = view.index[i1].strftime("%m-%d"), view.index[i2].strftime("%m-%d")
        if kind == "bearish divergence" and p2 > p1 and _ok(r1) and r2 < r1 - 2:
            out.append({"type": kind, "indicator": "RSI14", "text": f"Price high rose from {_n(p1)} ({d1}) to {_n(p2)} ({d2}) "
                        f"but RSI14 fell from {r1:.0f} to {r2:.0f}: upside momentum is fading.", "bias": -1, "age": len(view) - 1 - i2})
        if kind == "bullish divergence" and p2 < p1 and _ok(r1) and r2 > r1 + 2:
            out.append({"type": kind, "indicator": "RSI14", "text": f"Price low fell from {_n(p1)} ({d1}) to {_n(p2)} ({d2}) "
                        f"but RSI14 rose from {r1:.0f} to {r2:.0f}: downside momentum is fading.", "bias": 1, "age": len(view) - 1 - i2})
        if kind == "bearish divergence" and p2 > p1 and _ok(h1) and h1 > 0 and h2 < h1 * 0.7:
            out.append({"type": kind, "indicator": "MACD", "text": "MACD histogram shrank clearly at the new price high (MACD bearish divergence).",
                        "bias": -1, "age": len(view) - 1 - i2})
        if kind == "bullish divergence" and p2 < p1 and _ok(h1) and h1 < 0 and h2 > h1 * 0.7:
            out.append({"type": kind, "indicator": "MACD", "text": "MACD histogram shrank clearly at the new price low (MACD bullish divergence).",
                        "bias": 1, "age": len(view) - 1 - i2})
    return [d for d in out if d["age"] <= 25]


def money_flow(f: pd.DataFrame, n: int = 25) -> dict[str, Any]:
    """Accumulation / distribution days (IBD-style): close up / down > 0.2% on higher volume than the day before."""
    v = f.tail(n + 1)
    ret, vol = v["close"].pct_change(), v["volume"]
    up = (ret > 0.002) & (vol > vol.shift(1))
    down = (ret < -0.002) & (vol > vol.shift(1))
    obv = f["obv"].tail(20)
    price_chg = f["close"].iloc[-1] / f["close"].iloc[-21] - 1 if len(f) > 21 else float("nan")
    obv_slope = _f(f["obv_slope20"].iloc[-1])
    return {"accumulation": int(up.sum()), "distribution": int(down.sum()), "updown": _f(f["updown_vol_ratio"].iloc[-1]),
            "obv_divergence": ("price_up_obv_down" if price_chg > 0.03 and obv_slope < 0 else
                               "price_down_obv_up" if price_chg < -0.03 and obv_slope > 0 else None),
            "obv_up": bool(obv.iloc[-1] > obv.iloc[0]) if len(obv) == 20 else None}


def candles(f: pd.DataFrame) -> list[str]:
    """Price action of the latest session(s)."""
    out = []
    t, y = f.iloc[-1], f.iloc[-2]
    body, rng = abs(t.close - t.open), t.high - t.low
    upper, lower = t.high - max(t.close, t.open), min(t.close, t.open) - t.low
    atr = _f(t.atr)
    if y.close < y.open and t.close > t.open and t.open <= y.close and t.close >= y.open:
        out.append("Bullish engulfing: today's up candle engulfs yesterday's down candle - buyers taking over.")
    if y.close > y.open and t.close < t.open and t.open >= y.close and t.close <= y.open:
        out.append("Bearish engulfing: today's down candle engulfs yesterday's up candle - sellers taking over.")
    if rng > 0 and lower >= 2 * max(body, 1e-9) and upper <= 0.35 * rng and t.low <= f["low"].tail(10).min() * 1.005:
        out.append("Hammer: long lower wick at a recent low - buyers defended it.")
    if rng > 0 and upper >= 2 * max(body, 1e-9) and lower <= 0.35 * rng and t.high >= f["high"].tail(10).max() * 0.995:
        out.append("Shooting star: long upper wick at a recent high - supply overhead.")
    if t.high < y.high and t.low > y.low:
        out.append("Inside day: today's range sits inside yesterday's - watch the breakout direction.")
    if np.isfinite(atr) and rng > 1.8 * atr:
        out.append(f"Wide-range bar: today's range is {rng / atr:.1f}x ATR, closing {'near the high' if _f(t.clv) > 0.7 else 'near the low' if _f(t.clv) < 0.3 else 'mid-range'}.")
    g = _f(t.gap_atr)
    if np.isfinite(g) and abs(g) >= 1:
        out.append(f"Gap {'up' if g > 0 else 'down'} of {abs(g):.1f} ATR today, {'unfilled' if (g > 0 and t.low > y.high) or (g < 0 and t.high < y.low) else 'partly filled'}.")
    return out


def fib_levels(f: pd.DataFrame, n: int = 60) -> dict[str, Any]:
    v = f.tail(n)
    hi, lo = float(v["high"].max()), float(v["low"].min())
    up_leg = v["low"].idxmin() < v["high"].idxmax()  # low came first -> retracement of an up-move
    levels = {}
    for r in (0.382, 0.5, 0.618):
        levels[r] = hi - (hi - lo) * r if up_leg else lo + (hi - lo) * r
    return {"high": hi, "low": lo, "direction": "up-leg" if up_leg else "down-leg", "levels": levels}


def rs_line_high(f: pd.DataFrame, spy: pd.Series | None) -> bool | None:
    if spy is None:
        return None
    rs = (f["close"] / spy.reindex(f.index).ffill()).dropna()
    return bool(rs.iloc[-1] >= rs.tail(60).max() * 0.995) if len(rs) >= 60 else None


# ------------------------------------------------------------------ the note
def analyze(symbol: str, f: pd.DataFrame, candidate: dict | None = None, deep: dict | None = None,
            regime: dict | None = None, sector: dict | None = None, spy_close: pd.Series | None = None,
            rs_percentile: float | None = None, setup_names: list[str] | None = None) -> dict[str, Any]:
    deep = deep or {}
    t = f.iloc[-1]
    close, atr = float(t.close), _f(t.atr)
    wk = weekly_view(f)
    tq = trend_quality(f["close"])
    divs = divergences(f)
    mf = money_flow(f)
    cds = candles(f)
    fib = fib_levels(f)
    rsh = rs_line_high(f, spy_close)
    lv = support_resistance(f, 60, 3, count=3)
    tk = run_all(f, spy_close)
    by_group: dict[str, list] = {}
    for x in tk["findings"]:
        by_group.setdefault(x["group"], []).append(x)
    sc: dict[str, dict[str, Any]] = {}

    def put(key: str, score: float, text: str) -> None:
        sc[key] = {"key": key, "name": NAMES[key], "weight": WEIGHTS[key], "score": round(_clip(score), 2), "text": text}

    # 1 weekly
    if wk.get("available"):
        put("weekly", wk["score"], f"Weekly {wk['label']}: weekly close {_n(wk['close'])}, weekly EMA10 {_n(wk['ema10'])}, weekly EMA30 {_n(wk['ema30'])}"
            f" ({'rising' if wk['ema30_slope'] > 0 else 'falling'}), weekly RSI14 {wk['rsi14']:.0f}.")
    else:
        put("weekly", 0, "Not enough history for the weekly trend.")
    # 2 daily
    ds = TREND_SCORE.get(str(t.trend), 0)
    if str(t.swing_structure) == "HH/HL":
        ds += 0.5
    elif str(t.swing_structure) == "LH/LL":
        ds -= 0.5
    q = "a smooth trend" if _f(tq["r2"]) >= 0.7 else "a trend with some noise" if _f(tq["r2"]) >= 0.4 else "choppy, directionless"
    put("daily", ds, f"Daily {str(t.trend).lower()}, swing structure {t.swing_structure}; ADX {_n(t.adx, 0)}"
        f" ({'trending' if _f(t.adx) >= 25 else 'weak trend' if _f(t.adx) < 20 else 'moderate trend'}); 50-day regression R² {_n(tq['r2'], 2)}, {q}.")
    # 3 momentum
    r14, r6 = _f(t.rsi14), _f(t.rsi6)
    ms = (1 if r14 >= 55 else -1 if r14 <= 45 else 0) + (0.5 if _f(t.macd_hist) > 0 else -0.5) \
        + (0.5 if _f(t.macd_hist_3d) > 0 else -0.5) + sum(d["bias"] for d in divs) * 0.75
    div_txt = " ".join(d["text"] for d in divs)
    put("momentum", ms, f"RSI14 {r14:.0f}, RSI6 {r6:.0f}; MACD histogram {'positive' if _f(t.macd_hist) > 0 else 'negative'} and "
        f"{'widening' if _f(t.macd_hist_3d) > 0 else 'narrowing'} over 3 days; 20-day rate of change {_p(_f(t.return_20d))}." + (f" {div_txt}" if div_txt else ""))
    # 3b indicator consensus (Stochastic, CCI, MFI, CMF, TTM squeeze, Supertrend, Ichimoku, SAR, Donchian ...)
    cons_list = by_group.get("Indicator consensus", [])
    nb, ns = sum(x["signal"] > 0 for x in cons_list), sum(x["signal"] < 0 for x in cons_list)
    bull_names = [x["name"].split(" (")[0] for x in cons_list if x["signal"] > 0][:4]
    bear_names = [x["name"].split(" (")[0] for x in cons_list if x["signal"] < 0][:4]
    put("consensus", (nb - ns) / max(len(cons_list), 1) * 3,
        f"Of {len(cons_list)} common indicators, {nb} bullish, {ns} bearish, {len(cons_list) - nb - ns} neutral"
        + (f" (bullish: {', '.join(bull_names)})" if bull_names else "")
        + (f" (bearish: {', '.join(bear_names)})" if bear_names else "") + ".")
    # 4 relative strength
    rel20, rel60, rels = _f(t.rel_20d), _f(t.rel_60d), _f(t.rel_sector_20d)
    rs_score = _clip((rel20 / 0.05) * 0.6 + (rel60 / 0.10) * 0.6) + (0.5 if rsh else 0)
    put("relative", rs_score, f"vs SPY: 20d {_p(rel20)}, 60d {_p(rel60)}; vs sector ETF 20d {_p(rels)}"
        + (f"; universe RS percentile {rs_percentile:.0f}" if rs_percentile is not None else "")
        + ("; RS line at a 60-day high" if rsh else "") + ".")
    # 5 volume / money flow
    net = mf["accumulation"] - mf["distribution"]
    vs = _clip(net / 3) + (0.5 if mf["updown"] >= 1.3 else -0.5 if mf["updown"] <= 0.75 else 0)
    obv_txt = {"price_up_obv_down": "but OBV is falling while price rises (volume divergence - the advance lacks volume)",
               "price_down_obv_up": "OBV is rising while price falls (accumulation on weakness)"}.get(mf["obv_divergence"], "")
    put("volume", vs - (0.5 if mf["obv_divergence"] == "price_up_obv_down" else 0) + (0.5 if mf["obv_divergence"] == "price_down_obv_up" else 0),
        f"{mf['accumulation']} accumulation and {mf['distribution']} distribution days in 25; 20-day up/down volume {_n(mf['updown'])}; "
        f"today's volume ratio {_n(t.volume_ratio)}x" + (f"; {obv_txt}" if obv_txt else "") + ".")
    # 5b chip distribution (volume profile) and 5c patterns / stock-specific behaviour
    chip_list = by_group.get("Volume profile", [])
    put("chips", sum(x["signal"] * x["strength"] for x in chip_list) * 1.2,
        " ".join(x["text"] for x in chip_list if x["id"] in ("vp_value_area", "profit_ratio"))[:220] if chip_list else "No volume-profile data.")
    pat_list = by_group.get("Patterns", []) + by_group.get("Stock behaviour", [])
    active = [x for x in pat_list if x["signal"] != 0]
    put("patterns", sum(x["signal"] * x["strength"] for x in pat_list) * 1.2,
        ("Active: " + "; ".join(x["name"] + (" (bullish)" if x["signal"] > 0 else " (bearish)") for x in active[:5]) + ".") if active
        else "No notable pattern or behaviour signal.")
    # 6 position / extension
    ext, from_hi = _f(t.extension_atr), _f(t.from_52w_high)
    ps = 0.0
    pos_txt = f"Close {ext:+.1f} ATR from EMA21, {_p(from_hi)} from the 52-week high. "
    if ext > 3 or r6 > 85:
        ps = -1.5
        pos_txt += "Clearly over-extended short term - poor reward/risk for chasing, prone to a pullback."
    elif ext < -3 or r6 < 15:
        ps = 1
        pos_txt += "Oversold short term - room for a technical bounce (not a trend change)."
    elif from_hi > -0.05 and ext >= 0:
        ps = 0.5
        pos_txt += "Near the 52-week high - little overhead supply."
    put("position", ps, pos_txt)
    # 7 fundamentals
    fund = deep.get("fundamentals") or {}
    if fund.get("available"):
        flags = fund.get("flags") or []
        fs = sum(0.6 for x in flags if x["kind"] == "strength") - sum(1.0 if x["severity"] == "high" else 0.6 for x in flags if x["kind"] == "risk")
        m = fund["metrics"]
        put("fundamentals", fs, f"Revenue YoY {_p(m.get('revenue_yoy'), 0)}, EPS YoY {_p(m.get('eps_yoy'), 0)}, operating margin {_p(m.get('operating_margin_ttm'), 0, False)}, "
            f"free cash flow {'positive' if _f(m.get('fcf_ttm')) > 0 else 'negative' if _ok(m.get('fcf_ttm')) else 'n/a'}, share count YoY {_p(m.get('shares_yoy'), 1)}.")
    else:
        put("fundamentals", 0, "No fundamentals (leveraged ETF or no deep dive).")
    # 8 options
    opt = deep.get("options") or {}
    if opt.get("available"):
        pcr, ivhv = _f(opt.get("pcr_volume")), _f(opt.get("iv_hv_ratio"))
        os_ = (0.75 if pcr < 0.6 else -0.75 if pcr > 1.3 else 0)
        calls = sum(u["premium"] for u in opt.get("unusual", []) if u["side"] == "call")
        puts = sum(u["premium"] for u in opt.get("unusual", []) if u["side"] == "put")
        os_ += 0.5 if calls > 2 * puts and calls > 0 else -0.5 if puts > 2 * calls and puts > 0 else 0
        put("options", os_, f"Put/call volume {_n(pcr)}; ATM implied volatility {_p(opt.get('atm_iv'), 0, False)}, {_n(ivhv)}x realised; "
            f"expected move to {opt.get('expected_move_expiry')} ±{_p(opt.get('expected_move_pct'), 1, False)}"
            + ("; unusual activity is call-heavy" if calls > 2 * puts and calls > 0 else "; unusual activity is put-heavy" if puts > 2 * calls and puts > 0 else "")
            + ". (Options data cannot separate speculation from hedging - supporting evidence only.)")
    else:
        put("options", 0, "No options data.")
    # 9 positioning
    pos = deep.get("positioning") or {}
    si, ins = _f(pos.get("short_percent_float")), pos.get("insider_90d") or {}
    pp = 0.0
    pos_parts = []
    if np.isfinite(si):
        pos_parts.append(f"short interest {si * 100:.1f}% of float")
        if si >= 0.15:
            pp += 0.75 if ds > 0 else -0.25
    if ins.get("available"):
        pos_parts.append(f"insiders: {ins['n_buys']} buys, {ins['n_sales']} sales in 90 days")
        pp += 0.75 if ins["n_buys"] >= 2 and ins["net_value"] > 0 else 0
    put("positioning", pp, ("; ".join(pos_parts).capitalize() + ".") if pos_parts else "No positioning data.")
    # 10 catalysts
    cat = deep.get("catalysts") or {}
    ne = cat.get("next_earnings")
    cs = 0.0
    cat_parts = []
    if ne:
        cat_parts.append(f"next earnings {ne['date']} (in {ne['sessions_until']} sessions, {str(ne['status']).lower()})")
    if cat.get("beats_last4") is not None:
        cat_parts.append(f"EPS beat in {cat['beats_last4']} of the last 4 reports")
        cs += 0.5 if cat["beats_last4"] >= 3 else -0.5 if cat["beats_last4"] <= 1 else 0
    neg = [x for x in cat.get("filings", []) if x["sentiment"] == "Negative" and x.get("age_days", 99) <= 30]
    if neg:
        cs -= 1
        cat_parts.append(f"{len(neg)} negative-type SEC filings in 30 days ({', '.join(sorted({x['form'] for x in neg}))})")
    ups = [a for a in cat.get("analyst", []) if a.get("age_days", 99) <= 14]
    if ups:
        cs += 0.5 * sum(1 if a["sentiment"] == "Positive" else -1 if a["sentiment"] == "Negative" else 0 for a in ups)
        cat_parts.append(f"{len(ups)} analyst rating changes in 14 days")
    put("catalysts", cs, ("; ".join(cat_parts).capitalize() + ".") if cat_parts else "No known near-term catalysts.")
    # 11 market
    rl = (regime or {}).get("label")
    quad = (sector or {}).get("quadrant")
    put("market", REGIME_SCORE.get(rl, 0) + QUADRANT_SCORE.get(quad, 0),
        f"Market regime {rl or 'n/a'}; sector {(sector or {}).get('symbol', 'n/a')} ({(sector or {}).get('name') or ''}) is {quad or 'n/a'} on the rotation map.")

    # ---------------------------------------------------------- synthesis
    bias = sum(WEIGHTS[k] * sc[k]["score"] for k in WEIGHTS) / 2
    label = ("Strongly bullish" if bias >= 45 else "Bullish" if bias >= 20 else "Leaning bullish" if bias >= 7 else "Neutral" if bias > -7
             else "Leaning bearish" if bias > -20 else "Bearish" if bias > -45 else "Strongly bearish")
    nz = [s["score"] for s in sc.values() if abs(s["score"]) >= 0.25]
    agree = (sum(1 for s in nz if np.sign(s) == np.sign(bias)) / len(nz)) if nz and bias != 0 else 0.0
    complete = bool(fund.get("available") or opt.get("available"))
    earnings_soon = bool(ne and ne.get("sessions_until") is not None and ne["sessions_until"] <= 5)
    conf_score = agree + (0.15 if abs(bias) >= 25 else 0) + (0.1 if complete else -0.1) - (0.25 if earnings_soon else 0)
    confidence = "high" if conf_score >= 0.85 else "medium" if conf_score >= 0.6 else "low"

    res, sup = lv["resistance"], lv["support"]
    r1 = res[0].price if res else close + 2 * atr
    r2 = res[1].price if len(res) > 1 else r1 + 1.5 * atr
    s1 = sup[0].price if sup else close - 2 * atr
    s2 = sup[1].price if len(sup) > 1 else s1 - 1.5 * atr
    hv20 = _f(t.hv20)
    move7 = close * hv20 * math.sqrt(7 / 252) if np.isfinite(hv20) else float("nan")
    opt_move = _f(opt.get("expected_move_pct")) * close if opt.get("available") else float("nan")

    ranked = sorted(sc.values(), key=lambda s: -s["score"] * s["weight"])
    pros = [s for s in ranked if s["score"] >= 0.5][:3]
    cons = [s for s in reversed(ranked) if s["score"] <= -0.5][:3]
    wk_txt = f"weekly {wk['label']}" if wk.get("available") else "unclear weekly trend"
    day_txt = f"daily {str(t.trend).lower()}"
    headline = f"{label}: {wk_txt[:1].upper() + wk_txt[1:]}, {day_txt}" + (", over-extended short term" if ps <= -1.5 else ", oversold short term" if ps >= 1 else "")
    stance = _stance(label, ps, close, atr, r1, s1, float(t.ema21), earnings_soon, ne)
    summary = (f"{symbol} at {_n(close)}: {wk_txt}, {day_txt} ({t.swing_structure}). "
               + (f"Main positives: {'; '.join(p['name'] + ' - ' + _short(p['text']) for p in pros)}. " if pros else "")
               + (f"Main negatives: {'; '.join(c['name'] + ' - ' + _short(c['text']) for c in cons)}. " if cons else "")
               + f"Across {len(WEIGHTS)} dimensions the overall call is {label} ({bias:+.0f}/100, {confidence} confidence). {stance}")

    sections = [
        {"title": "Trend structure (weekly + daily)", "body": [sc["weekly"]["text"], sc["daily"]["text"],
                                          f"Moving averages: close {'above' if close > _f(t.ema21) else 'below'} EMA21 ({_n(t.ema21)}), "
                                          f"{'above' if close > _f(t.ema50) else 'below'} EMA50 ({_n(t.ema50)}), "
                                          f"{'above' if close > _f(t.ema200) else 'below'} EMA200 ({_n(t.ema200)})."]},
        {"title": "Momentum & divergence", "body": [sc["momentum"]["text"]] + ([] if divs else ["No clear price/indicator divergence recently."])},
        {"title": "Volume & money flow", "body": [sc["volume"]["text"]]},
        {"title": "Relative strength", "body": [sc["relative"]["text"]]},
        {"title": "Volatility & price action", "body": [
            f"ATR {_n(atr)} ({_n(t.atr_pct, 1)}% of price), {_n(t.atr_pct_percentile, 0)}th percentile of the year; "
            f"Bollinger width in the {_n(t.bb_width_percentile, 0)}th percentile of 6 months"
            + (" - volatility is compressing, which often precedes a bigger directional move" if _f(t.bb_width_percentile) <= 20 else
               " - volatility is expanding" if _f(t.bb_width_percentile) >= 80 else "") + ".",
            f"Realised volatility 10d {_p(t.hv10, 0, False)}, 60d {_p(t.hv60, 0, False)}; beta (120d) {_n(t.beta120)}.",
            sc["position"]["text"], *cds]},
    ]
    if fund.get("available"):
        sections.append({"title": "Fundamentals", "body": [sc["fundamentals"]["text"], *[x["text"] for x in (fund.get("flags") or [])[:4]]]})
    if opt.get("available") or pos:
        sections.append({"title": "Options & positioning", "body": [x for x in (sc["options"]["text"] if opt.get("available") else None,
                                                                  sc["positioning"]["text"] if pos else None) if x]
                         + list((opt.get("interpretation") or [])[:2])})
    sections.append({"title": "Catalysts & market", "body": [sc["catalysts"]["text"], sc["market"]["text"]]})
    for group, title in (("Volume profile", "Volume profile (nodes, vacuum zones, holder cost)"), ("Stock behaviour", "Stock behaviour (this stock, last 2 years)"),
                         ("Patterns", "Pattern detection"), ("Indicator consensus", "Indicator consensus"), ("Relative performance", "Relative performance")):
        items = by_group.get(group, [])
        if items:
            sections.append({"title": title, "body": [x["text"] for x in items],
                             "signals": [{"name": x["name"], "signal": x["signal"]} for x in items]})

    scenarios = [
        {"name": "Bull case", "trigger": f"Close above {_n(r1)} ({_lvl(res[0].label()) if res else '2 ATR above'}) on rising volume",
         "target": f"{_n(r2)}" + (f"; options-implied upper bound ~{_n(close + opt_move)}" if np.isfinite(opt_move) else ""),
         "invalidation": f"Back below {_n(r1 - 0.5 * atr)}"},
        {"name": "Base case", "trigger": f"Range between {_n(s1)} and {_n(r1)}",
         "target": f"7-day 1σ move ~±{_n(move7)} (±{_p(move7 / close, 1, False)})" if np.isfinite(move7) else "n/a",
         "invalidation": "A close outside the range"},
        {"name": "Bear case", "trigger": f"Close below {_n(s1)} ({_lvl(sup[0].label()) if sup else '2 ATR below'})",
         "target": f"{_n(s2)}" + (f"; options-implied lower bound ~{_n(close - opt_move)}" if np.isfinite(opt_move) else ""),
         "invalidation": f"Back above {_n(s1 + 0.5 * atr)}"},
    ]
    watch = [f"Hold {_n(s1)} support ({_n((s1 / close - 1) * 100, 1)}%)?", f"Break {_n(r1)} resistance ({_n((r1 / close - 1) * 100, 1)}%)?",
             f"EMA21 ({_n(t.ema21)}): {'holding above' if close > _f(t.ema21) else 'below it'}"]
    if divs:
        watch.append("Is the divergence confirmed by price (e.g. a break of the last swing low)?")
    if ne and ne.get("sessions_until") is not None and ne["sessions_until"] <= 15:
        watch.append(f"Gap risk around earnings ({ne['date']})")
    if _f(t.bb_width_percentile) <= 20:
        watch.append("Breakout direction after the squeeze, and breakout-day volume")
    risks = list((candidate or {}).get("contradicting", []))[:6] or [c["name"] + ": " + c["text"] for c in cons]
    return {
        "symbol": symbol, "as_of": f.index[-1].strftime("%Y-%m-%d"), "close": close,
        "bias": label, "bias_score": round(bias, 1), "confidence": confidence, "agreement": round(agree, 2),
        "headline": headline, "summary": summary, "stance": stance,
        "scorecard": [sc[k] for k in WEIGHTS], "sections": sections, "scenarios": scenarios, "watch": watch, "risks": risks,
        "levels": {"resistance": [{"price": x.price, "label": _lvl(x.label()), "atr": x.distance_atr, "pct": x.distance_pct} for x in res],
                   "support": [{"price": x.price, "label": _lvl(x.label()), "atr": x.distance_atr, "pct": x.distance_pct} for x in sup],
                   "fib": {"direction": fib["direction"], "high": fib["high"], "low": fib["low"],
                           "levels": {str(k): v for k, v in fib["levels"].items()}},
                   "ema21": float(t.ema21), "ema50": float(t.ema50), "ema200": _f(t.ema200),
                   "high52": _f(t.high252), "low52": _f(t.low252)},
        "signals": {"divergences": divs, "candles": cds, "money_flow": mf, "weekly": wk, "trend_quality": tq,
                    "rs_line_high": rsh, "setups": setup_names or []},
        "techniques": {"count": tk["count"], "groups": tk["groups"],
                       "findings": [{k: x[k] for k in ("id", "group", "name", "signal", "strength", "text")} for x in tk["findings"]]},
        "volume_profile": {k: tk["volume_profile"].get(k) for k in ("poc", "val", "vah", "hvn", "vacuum", "profit_ratio", "bins", "window")}
                          if tk["volume_profile"].get("available") else None,
        "data_complete": complete,
        "method": __doc__,
    }


def _lvl(label: str) -> str:
    import re
    t = str(label or "")
    return re.sub(r"(\d+)D (high|low)", lambda m: f"{m.group(1)}-day {m.group(2)}", t)


def _short(text: str, n: int = 46) -> str:
    first = text.split("; ")[0].split(". ")[0].rstrip(".")
    return first if len(first) <= n else first[: n - 1] + "…"


def _stance(label: str, ps: float, close: float, atr: float, r1: float, s1: float, ema21: float,
            earnings_soon: bool, ne: dict | None) -> str:
    bull, bear = "ullish" in label, "earish" in label
    if bull and ps <= -1.5:
        txt = (f"The trend is constructive but over-extended; a better setup is a pullback toward EMA21 (~{_n(ema21)}) or a quiet consolidation, "
               "rather than chasing here.")
    elif bull:
        txt = (f"The bullish picture holds while {_n(s1)} holds; a close below {_n(s1)} calls for a rethink. "
               f"First resistance is {_n(r1)}.")
    elif bear and ps >= 1:
        txt = f"Still weak but oversold - watch for a technical bounce; rejection near {_n(r1)} would extend the weakness."
    elif bear:
        txt = (f"Rejection near {_n(r1)} would extend the weakness; only a sustained reclaim of EMA21 (~{_n(ema21)}) would signal strength.")
    else:
        txt = f"No clear direction - wait for a break above {_n(r1)} or below {_n(s1)}."
    if earnings_soon and ne:
        txt += f" Earnings are due in {ne['sessions_until']} sessions ({ne['date']}), so any position carries gap risk."
    return txt
