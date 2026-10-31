"""Analyst note: one synthesised, plain-language view of a stock (Traditional Chinese).

It combines everything the terminal knows about a stock and writes it up the way a desk analyst would:
bottom line first, then the evidence by theme, key levels, scenarios, what to watch and the risks.

Scorecard (each -2 ... +2, weights sum to 100; bias = sum(weight x score) / 2 -> -100 ... +100):
    weekly trend 11 · daily trend 11 · momentum 9 · indicator consensus 6 · relative strength 9 ·
    volume / money flow 8 · chip distribution 8 · patterns & stock behaviour 8 · position / extension 5 ·
    fundamentals 9 · options sentiment 4 · positioning 4 · catalysts 4 · market environment 4
The ~50 techniques in techniques.py feed the consensus, chips and patterns components and are all listed.
Label: >= 45 強烈偏多 | >= 20 偏多 | >= 7 中性偏多 | > -7 中性 | > -20 中性偏空 | > -45 偏空 | else 強烈偏空
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

from src.i18n.zh import tr
from src.technical.indicators import ema, rsi
from src.analyst.techniques import run_all
from src.technical.structure import support_resistance, swing_points

WEIGHTS = {"weekly": 11, "daily": 11, "momentum": 9, "consensus": 6, "relative": 9, "volume": 8, "chips": 8,
           "patterns": 8, "position": 5, "fundamentals": 9, "options": 4, "positioning": 4, "catalysts": 4, "market": 4}
NAMES = {"weekly": "週線趨勢", "daily": "日線趨勢", "momentum": "動量", "consensus": "技術指標共識", "relative": "相對強弱",
         "volume": "量能／資金流", "chips": "籌碼分佈", "patterns": "形態與個股習性", "position": "位置／延伸",
         "fundamentals": "基本面", "options": "期權情緒", "positioning": "持倉／做空", "catalysts": "催化劑", "market": "市場環境"}
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
        label, score = "上升趨勢", 2
    elif c > b and slope30 > 0:
        label, score = "上升趨勢中回調", 1
    elif c < a < b and slope30 < 0:
        label, score = "下跌趨勢", -2
    elif c < b and slope30 < 0:
        label, score = "下跌趨勢中反彈", -1
    else:
        label, score = "橫行整理", 0
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
    for kind, mask, col in (("頂背離", is_high, "high"), ("底背離", is_low, "low")):
        idx = [i for i, flag in enumerate(mask.to_numpy()) if flag]
        if len(idx) < 2:
            continue
        i1, i2 = idx[-2], idx[-1]
        p1, p2 = view[col].iloc[i1], view[col].iloc[i2]
        r1, r2 = view["rsi14"].iloc[i1], view["rsi14"].iloc[i2]
        h1, h2 = view["macd_hist"].iloc[i1], view["macd_hist"].iloc[i2]
        d1, d2 = view.index[i1].strftime("%m-%d"), view.index[i2].strftime("%m-%d")
        if kind == "頂背離" and p2 > p1 and _ok(r1) and r2 < r1 - 2:
            out.append({"type": kind, "indicator": "RSI14", "text": f"價格高點由 {_n(p1)}（{d1}）升至 {_n(p2)}（{d2}），"
                        f"但 RSI14 由 {r1:.0f} 降至 {r2:.0f}，上升動力減弱。", "bias": -1, "age": len(view) - 1 - i2})
        if kind == "底背離" and p2 < p1 and _ok(r1) and r2 > r1 + 2:
            out.append({"type": kind, "indicator": "RSI14", "text": f"價格低點由 {_n(p1)}（{d1}）跌至 {_n(p2)}（{d2}），"
                        f"但 RSI14 由 {r1:.0f} 升至 {r2:.0f}，下跌動力減弱。", "bias": 1, "age": len(view) - 1 - i2})
        if kind == "頂背離" and p2 > p1 and _ok(h1) and h1 > 0 and h2 < h1 * 0.7:
            out.append({"type": kind, "indicator": "MACD", "text": "價格創新高時 MACD 柱狀明顯縮短（MACD 頂背離）。",
                        "bias": -1, "age": len(view) - 1 - i2})
        if kind == "底背離" and p2 < p1 and _ok(h1) and h1 < 0 and h2 > h1 * 0.7:
            out.append({"type": kind, "indicator": "MACD", "text": "價格創新低時 MACD 柱狀明顯縮短（MACD 底背離）。",
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
        out.append("看漲吞沒：今日陽燭完全包住昨日陰燭，買盤轉強。")
    if y.close > y.open and t.close < t.open and t.open >= y.close and t.close <= y.open:
        out.append("看跌吞沒：今日陰燭完全包住昨日陽燭，賣壓轉強。")
    if rng > 0 and lower >= 2 * max(body, 1e-9) and upper <= 0.35 * rng and t.low <= f["low"].tail(10).min() * 1.005:
        out.append("錘頭線：在近期低位出現長下影線，低位有承接。")
    if rng > 0 and upper >= 2 * max(body, 1e-9) and lower <= 0.35 * rng and t.high >= f["high"].tail(10).max() * 0.995:
        out.append("射擊之星：在近期高位出現長上影線，高位有沽壓。")
    if t.high < y.high and t.low > y.low:
        out.append("內包日：今日波幅在昨日之內，市場在觀望，留意之後突破的方向。")
    if np.isfinite(atr) and rng > 1.8 * atr:
        out.append(f"寬幅 K 線：今日波幅是 ATR 的 {rng / atr:.1f} 倍，{'收在高位' if _f(t.clv) > 0.7 else '收在低位' if _f(t.clv) < 0.3 else '收在中間'}。")
    g = _f(t.gap_atr)
    if np.isfinite(g) and abs(g) >= 1:
        out.append(f"今日{'跳空高開' if g > 0 else '跳空低開'} {abs(g):.1f} 個 ATR，{'缺口未補' if (g > 0 and t.low > y.high) or (g < 0 and t.high < y.low) else '缺口已部分回補'}。")
    return out


def fib_levels(f: pd.DataFrame, n: int = 60) -> dict[str, Any]:
    v = f.tail(n)
    hi, lo = float(v["high"].max()), float(v["low"].min())
    up_leg = v["low"].idxmin() < v["high"].idxmax()  # low came first -> retracement of an up-move
    levels = {}
    for r in (0.382, 0.5, 0.618):
        levels[r] = hi - (hi - lo) * r if up_leg else lo + (hi - lo) * r
    return {"high": hi, "low": lo, "direction": "上升波段" if up_leg else "下跌波段", "levels": levels}


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
        put("weekly", wk["score"], f"週線{wk['label']}：週收市 {_n(wk['close'])}，週 EMA10 {_n(wk['ema10'])}，週 EMA30 {_n(wk['ema30'])}"
            f"（{'向上' if wk['ema30_slope'] > 0 else '向下'}），週 RSI14 {wk['rsi14']:.0f}。")
    else:
        put("weekly", 0, "歷史不足，無法判斷週線趨勢。")
    # 2 daily
    ds = TREND_SCORE.get(str(t.trend), 0)
    if str(t.swing_structure) == "HH/HL":
        ds += 0.5
    elif str(t.swing_structure) == "LH/LL":
        ds -= 0.5
    q = "趨勢相當順暢" if _f(tq["r2"]) >= 0.7 else "趨勢有一定雜音" if _f(tq["r2"]) >= 0.4 else "走勢反覆、缺乏方向"
    put("daily", ds, f"日線{tr(t.trend)}，波段結構{tr(t.swing_structure)}；ADX {_n(t.adx, 0)}"
        f"（{'有趨勢' if _f(t.adx) >= 25 else '趨勢偏弱' if _f(t.adx) < 20 else '趨勢一般'}）；50 日回歸 R² {_n(tq['r2'], 2)}，{q}。")
    # 3 momentum
    r14, r6 = _f(t.rsi14), _f(t.rsi6)
    ms = (1 if r14 >= 55 else -1 if r14 <= 45 else 0) + (0.5 if _f(t.macd_hist) > 0 else -0.5) \
        + (0.5 if _f(t.macd_hist_3d) > 0 else -0.5) + sum(d["bias"] for d in divs) * 0.75
    div_txt = "；".join(d["text"] for d in divs)
    put("momentum", ms, f"RSI14 {r14:.0f}、RSI6 {r6:.0f}；MACD 柱 {'正' if _f(t.macd_hist) > 0 else '負'}值且近 3 日"
        f"{'擴大' if _f(t.macd_hist_3d) > 0 else '收窄'}；20 日變動率 {_p(_f(t.return_20d))}。" + (f"{div_txt}" if div_txt else ""))
    # 3b indicator consensus (Stochastic, CCI, MFI, CMF, TTM squeeze, Supertrend, Ichimoku, SAR, Donchian ...)
    cons_list = by_group.get("指標共識", [])
    nb, ns = sum(x["signal"] > 0 for x in cons_list), sum(x["signal"] < 0 for x in cons_list)
    bull_names = [x["name"].split("（")[0] for x in cons_list if x["signal"] > 0][:4]
    bear_names = [x["name"].split("（")[0] for x in cons_list if x["signal"] < 0][:4]
    put("consensus", (nb - ns) / max(len(cons_list), 1) * 3,
        f"{len(cons_list)} 個常用技術指標中，{nb} 個偏多、{ns} 個偏空、{len(cons_list) - nb - ns} 個中性"
        + (f"（偏多：{'、'.join(bull_names)}）" if bull_names else "")
        + (f"（偏空：{'、'.join(bear_names)}）" if bear_names else "") + "。")
    # 4 relative strength
    rel20, rel60, rels = _f(t.rel_20d), _f(t.rel_60d), _f(t.rel_sector_20d)
    rs_score = _clip((rel20 / 0.05) * 0.6 + (rel60 / 0.10) * 0.6) + (0.5 if rsh else 0)
    put("relative", rs_score, f"相對 SPY：20 日 {_p(rel20)}、60 日 {_p(rel60)}；相對板塊 ETF 20 日 {_p(rels)}"
        + (f"；股票池強弱排名第 {rs_percentile:.0f} 百分位" if rs_percentile is not None else "")
        + ("；相對強弱線創 60 日新高" if rsh else "") + "。")
    # 5 volume / money flow
    net = mf["accumulation"] - mf["distribution"]
    vs = _clip(net / 3) + (0.5 if mf["updown"] >= 1.3 else -0.5 if mf["updown"] <= 0.75 else 0)
    obv_txt = {"price_up_obv_down": "但價格上升時 OBV 下降（量價背離，升勢欠缺成交支持）",
               "price_down_obv_up": "價格下跌時 OBV 反而上升（有資金在低位吸納）"}.get(mf["obv_divergence"], "")
    put("volume", vs - (0.5 if mf["obv_divergence"] == "price_up_obv_down" else 0) + (0.5 if mf["obv_divergence"] == "price_down_obv_up" else 0),
        f"近 25 日有 {mf['accumulation']} 個吸納日、{mf['distribution']} 個派發日；20 日上漲／下跌成交量比 {_n(mf['updown'])}；"
        f"今日量比 {_n(t.volume_ratio)} 倍" + (f"；{obv_txt}" if obv_txt else "") + "。")
    # 5b chip distribution (volume profile) and 5c patterns / stock-specific behaviour
    chip_list = by_group.get("籌碼分佈", [])
    put("chips", sum(x["signal"] * x["strength"] for x in chip_list) * 1.2,
        "；".join(x["text"].rstrip("。") for x in chip_list if x["id"] in ("vp_value_area", "profit_ratio"))[:160] + "。" if chip_list else "沒有籌碼數據。")
    pat_list = by_group.get("形態", []) + by_group.get("個股習性", [])
    active = [x for x in pat_list if x["signal"] != 0]
    put("patterns", sum(x["signal"] * x["strength"] for x in pat_list) * 1.2,
        ("目前觸發：" + "；".join(x["name"] + ("（偏多）" if x["signal"] > 0 else "（偏空）") for x in active[:5]) + "。") if active
        else "沒有觸發明顯的形態或個股習性訊號。")
    # 6 position / extension
    ext, from_hi = _f(t.extension_atr), _f(t.from_52w_high)
    ps = 0.0
    pos_txt = f"收市距 EMA21 {ext:+.1f} 個 ATR，距 52 週高位 {_p(from_hi)}。"
    if ext > 3 or r6 > 85:
        ps = -1.5
        pos_txt += "短線明顯過度延伸，追高的風險報酬差，容易出現回吐。"
    elif ext < -3 or r6 < 15:
        ps = 1
        pos_txt += "短線超賣，有技術性反彈的條件（但不代表趨勢轉向）。"
    elif from_hi > -0.05 and ext >= 0:
        ps = 0.5
        pos_txt += "接近 52 週高位，上方阻力較少。"
    put("position", ps, pos_txt)
    # 7 fundamentals
    fund = deep.get("fundamentals") or {}
    if fund.get("available"):
        flags = fund.get("flags") or []
        fs = sum(0.6 for x in flags if x["kind"] == "strength") - sum(1.0 if x["severity"] == "high" else 0.6 for x in flags if x["kind"] == "risk")
        m = fund["metrics"]
        put("fundamentals", fs, f"營收按年 {_p(m.get('revenue_yoy'), 0)}、EPS 按年 {_p(m.get('eps_yoy'), 0)}、經營利潤率 {_p(m.get('operating_margin_ttm'), 0, False)}、"
            f"自由現金流 {'為正' if _f(m.get('fcf_ttm')) > 0 else '為負' if _ok(m.get('fcf_ttm')) else 'n/a'}、股數按年 {_p(m.get('shares_yoy'), 1)}。")
    else:
        put("fundamentals", 0, "沒有基本面數據（槓桿 ETF 或未做深入分析）。")
    # 8 options
    opt = deep.get("options") or {}
    if opt.get("available"):
        pcr, ivhv = _f(opt.get("pcr_volume")), _f(opt.get("iv_hv_ratio"))
        os_ = (0.75 if pcr < 0.6 else -0.75 if pcr > 1.3 else 0)
        calls = sum(u["premium"] for u in opt.get("unusual", []) if u["side"] == "call")
        puts = sum(u["premium"] for u in opt.get("unusual", []) if u["side"] == "put")
        os_ += 0.5 if calls > 2 * puts and calls > 0 else -0.5 if puts > 2 * calls and puts > 0 else 0
        put("options", os_, f"認沽／認購成交比 {_n(pcr)}；平價隱含波動率 {_p(opt.get('atm_iv'), 0, False)}，是實際波動的 {_n(ivhv)} 倍；"
            f"期權預期到 {opt.get('expected_move_expiry')} 的波幅 ±{_p(opt.get('expected_move_pct'), 1, False)}"
            + ("；異常成交以認購為主" if calls > 2 * puts and calls > 0 else "；異常成交以認沽為主" if puts > 2 * calls and puts > 0 else "")
            + "。（期權數據無法區分投機與對沖，只作輔助。）")
    else:
        put("options", 0, "沒有期權數據。")
    # 9 positioning
    pos = deep.get("positioning") or {}
    si, ins = _f(pos.get("short_percent_float")), pos.get("insider_90d") or {}
    pp = 0.0
    pos_parts = []
    if np.isfinite(si):
        pos_parts.append(f"做空佔流通股 {si * 100:.1f}%")
        if si >= 0.15:
            pp += 0.75 if ds > 0 else -0.25
    if ins.get("available"):
        pos_parts.append(f"90 日內部人士買入 {ins['n_buys']} 筆、賣出 {ins['n_sales']} 筆")
        pp += 0.75 if ins["n_buys"] >= 2 and ins["net_value"] > 0 else 0
    put("positioning", pp, ("；".join(pos_parts) + "。") if pos_parts else "沒有持倉數據。")
    # 10 catalysts
    cat = deep.get("catalysts") or {}
    ne = cat.get("next_earnings")
    cs = 0.0
    cat_parts = []
    if ne:
        cat_parts.append(f"下次財報 {ne['date']}（{ne['sessions_until']} 個交易日後，{tr(ne['status'])}）")
    if cat.get("beats_last4") is not None:
        cat_parts.append(f"最近 4 次財報有 {cat['beats_last4']} 次 EPS 超預期")
        cs += 0.5 if cat["beats_last4"] >= 3 else -0.5 if cat["beats_last4"] <= 1 else 0
    neg = [x for x in cat.get("filings", []) if x["sentiment"] == "Negative" and x.get("age_days", 99) <= 30]
    if neg:
        cs -= 1
        cat_parts.append(f"近 30 日有 {len(neg)} 份負面類型的 SEC 文件（{', '.join(sorted({x['form'] for x in neg}))}）")
    ups = [a for a in cat.get("analyst", []) if a.get("age_days", 99) <= 14]
    if ups:
        cs += 0.5 * sum(1 if a["sentiment"] == "Positive" else -1 if a["sentiment"] == "Negative" else 0 for a in ups)
        cat_parts.append(f"近 14 日分析師評級變動 {len(ups)} 次")
    put("catalysts", cs, ("；".join(cat_parts) + "。") if cat_parts else "沒有已知的近期催化劑。")
    # 11 market
    rl = (regime or {}).get("label")
    quad = (sector or {}).get("quadrant")
    put("market", REGIME_SCORE.get(rl, 0) + QUADRANT_SCORE.get(quad, 0),
        f"大市狀態「{tr(rl) if rl else 'n/a'}」；所屬板塊 {(sector or {}).get('symbol', 'n/a')}（{tr((sector or {}).get('name'))}）處於「{tr(quad) if quad else 'n/a'}」象限。")

    # ---------------------------------------------------------- synthesis
    bias = sum(WEIGHTS[k] * sc[k]["score"] for k in WEIGHTS) / 2
    label = ("強烈偏多" if bias >= 45 else "偏多" if bias >= 20 else "中性偏多" if bias >= 7 else "中性" if bias > -7
             else "中性偏空" if bias > -20 else "偏空" if bias > -45 else "強烈偏空")
    nz = [s["score"] for s in sc.values() if abs(s["score"]) >= 0.25]
    agree = (sum(1 for s in nz if np.sign(s) == np.sign(bias)) / len(nz)) if nz and bias != 0 else 0.0
    complete = bool(fund.get("available") or opt.get("available"))
    earnings_soon = bool(ne and ne.get("sessions_until") is not None and ne["sessions_until"] <= 5)
    conf_score = agree + (0.15 if abs(bias) >= 25 else 0) + (0.1 if complete else -0.1) - (0.25 if earnings_soon else 0)
    confidence = "高" if conf_score >= 0.85 else "中" if conf_score >= 0.6 else "低"

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
    wk_txt = f"週線{wk['label']}" if wk.get("available") else "週線未明"
    day_txt = f"日線{tr(t.trend)}"
    headline = f"{label}：{wk_txt}，{day_txt}" + ("，短線過度延伸" if ps <= -1.5 else "，短線超賣" if ps >= 1 else "")
    stance = _stance(label, ps, close, atr, r1, s1, float(t.ema21), earnings_soon, ne)
    summary = (f"{symbol} 現價 {_n(close)}，{wk_txt}，{day_txt}（{tr(t.swing_structure)}）。"
               + (f"主要支持因素：{'；'.join(p['name'] + '－' + _short(p['text']) for p in pros)}。" if pros else "")
               + (f"主要不利因素：{'；'.join(c['name'] + '－' + _short(c['text']) for c in cons)}。" if cons else "")
               + f"綜合 {len(WEIGHTS)} 項評估，整體判斷為「{label}」（{bias:+.0f}/100，信心{confidence}）。{stance}")

    sections = [
        {"title": "趨勢結構（週線＋日線）", "body": [sc["weekly"]["text"], sc["daily"]["text"],
                                          f"均線位置：收市{'高於' if close > _f(t.ema21) else '低於'} EMA21（{_n(t.ema21)}），"
                                          f"{'高於' if close > _f(t.ema50) else '低於'} EMA50（{_n(t.ema50)}），"
                                          f"{'高於' if close > _f(t.ema200) else '低於'} EMA200（{_n(t.ema200)}）。"]},
        {"title": "動量與背離", "body": [sc["momentum"]["text"]] + ([] if divs else ["近期沒有偵測到明顯的價格與指標背離。"])},
        {"title": "量能與資金流", "body": [sc["volume"]["text"]]},
        {"title": "相對強弱", "body": [sc["relative"]["text"]]},
        {"title": "波動與價格行為", "body": [
            f"ATR {_n(atr)}（股價的 {_n(t.atr_pct, 1)}%），處於一年內第 {_n(t.atr_pct_percentile, 0)} 百分位；"
            f"布林帶寬處於半年第 {_n(t.bb_width_percentile, 0)} 百分位"
            + ("，波動正在收窄，通常預示之後會有較大的方向性走勢" if _f(t.bb_width_percentile) <= 20 else
               "，波動正處於擴張階段" if _f(t.bb_width_percentile) >= 80 else "") + "。",
            f"10 日實際波動率 {_p(t.hv10, 0, False)}，60 日 {_p(t.hv60, 0, False)}；Beta（120 日）{_n(t.beta120)}。",
            sc["position"]["text"], *cds]},
    ]
    if fund.get("available"):
        sections.append({"title": "基本面", "body": [sc["fundamentals"]["text"], *[x["text"] for x in (fund.get("flags") or [])[:4]]]})
    if opt.get("available") or pos:
        sections.append({"title": "期權與持倉", "body": [x for x in (sc["options"]["text"] if opt.get("available") else None,
                                                                  sc["positioning"]["text"] if pos else None) if x]
                         + list((opt.get("interpretation") or [])[:2])})
    sections.append({"title": "催化劑與市場環境", "body": [sc["catalysts"]["text"], sc["market"]["text"]]})
    for group, title in (("籌碼分佈", "籌碼分佈（成交量分佈、真空區、持有人成本）"), ("個股習性", "個股習性（此股過去 2 年的統計）"),
                         ("形態", "形態偵測"), ("指標共識", "技術指標共識"), ("相對表現", "相對大市表現")):
        items = by_group.get(group, [])
        if items:
            sections.append({"title": title, "body": [x["text"] for x in items],
                             "signals": [{"name": x["name"], "signal": x["signal"]} for x in items]})

    scenarios = [
        {"name": "多頭情景", "trigger": f"收市升穿 {_n(r1)}（{_lvl(res[0].label()) if res else '2 ATR 上方'}）並放量確認",
         "target": f"{_n(r2)}" + (f"；期權預期上限約 {_n(close + opt_move)}" if np.isfinite(opt_move) else ""),
         "invalidation": f"重新跌回 {_n(r1 - 0.5 * atr)} 下方"},
        {"name": "基本情景", "trigger": f"在 {_n(s1)} 至 {_n(r1)} 區間內整理",
         "target": f"7 日 1σ 波幅約 ±{_n(move7)}（±{_p(move7 / close, 1, False)}）" if np.isfinite(move7) else "n/a",
         "invalidation": "收市突破區間任何一邊"},
        {"name": "空頭情景", "trigger": f"收市跌穿 {_n(s1)}（{_lvl(sup[0].label()) if sup else '2 ATR 下方'}）",
         "target": f"{_n(s2)}" + (f"；期權預期下限約 {_n(close - opt_move)}" if np.isfinite(opt_move) else ""),
         "invalidation": f"重新收復 {_n(s1 + 0.5 * atr)}"},
    ]
    watch = [f"能否守住 {_n(s1)} 支持（{_n((s1 / close - 1) * 100, 1)}%）", f"能否突破 {_n(r1)} 阻力（{_n((r1 / close - 1) * 100, 1)}%）",
             f"EMA21（{_n(t.ema21)}）的得失：{'目前在其上方' if close > _f(t.ema21) else '目前在其下方'}"]
    if divs:
        watch.append("背離是否被價格確認（例如跌穿最近的波段低點）")
    if ne and ne.get("sessions_until") is not None and ne["sessions_until"] <= 15:
        watch.append(f"財報（{ne['date']}）前後的跳空風險")
    if _f(t.bb_width_percentile) <= 20:
        watch.append("波動收窄後的突破方向，以及突破當日成交量")
    risks = list((candidate or {}).get("contradicting", []))[:6] or [c["name"] + "：" + c["text"] for c in cons]
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
    t = str(label or "").replace("swing high", "波段高點").replace("swing low", "波段低點")
    return re.sub(r"(\d+)D (high|low)", lambda m: f"{m.group(1)} 日{'高位' if m.group(2) == 'high' else '低位'}", t)


def _short(text: str, n: int = 46) -> str:
    first = text.split("；")[0].split("。")[0]
    return first if len(first) <= n else first[: n - 1] + "…"


def _stance(label: str, ps: float, close: float, atr: float, r1: float, s1: float, ema21: float,
            earnings_soon: bool, ne: dict | None) -> str:
    bull, bear = "偏多" in label, "偏空" in label
    if bull and ps <= -1.5:
        txt = (f"趨勢雖然向好，但短線過度延伸；較合理的研究方向是等回調至 EMA21（約 {_n(ema21)}）附近或出現縮量整理，"
               "而不是在現價追入。")
    elif bull:
        txt = (f"若價格守住 {_n(s1)} 一帶，偏多格局維持；收市跌穿 {_n(s1)} 代表看法需要重新評估。"
               f"上方首個阻力在 {_n(r1)}。")
    elif bear and ps >= 1:
        txt = f"弱勢格局未改，但短線已超賣，留意技術反彈；反彈到 {_n(r1)} 附近受阻的話，弱勢延續。"
    elif bear:
        txt = (f"反彈至 {_n(r1)} 附近受阻的話弱勢延續；要重新收復 EMA21（約 {_n(ema21)}）並站穩，才代表轉強。")
    else:
        txt = f"方向未明，較穩妥的做法是等待價格突破 {_n(r1)} 或跌穿 {_n(s1)} 再判斷。"
    if earnings_soon and ne:
        txt += f"另外，財報將於 {ne['sessions_until']} 個交易日後（{ne['date']}）公布，任何部署都要面對跳空風險。"
    return txt
