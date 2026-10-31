"""Professional analysis toolkit: ~55 techniques run on one stock's daily bars.

Every technique returns a finding:
    {"id", "group", "name", "signal": -1 | 0 | +1, "strength": 0..1, "text": Traditional Chinese, "stat": {...}}
signal is the directional read for the next 1-10 sessions (+1 constructive, -1 adverse, 0 neutral /
informational). Stock-specific statistics always report their sample size; fewer than 5 cases are
reported as "樣本不足" and never produce a signal.

Groups
    個股習性   how THIS stock has behaved at EMA9 / EMA21 / EMA50 / EMA200, band touches, RSI extremes,
               gaps, streaks, breakouts, big-move days, weekdays; autocorrelation and Hurst exponent
    籌碼分佈   volume profile (POC, value area, high-volume nodes = dense chips, low-volume nodes =
               vacuum zones), holder-profit ratio, anchored VWAPs, 20/60-day average holder cost
    形態       VCP, bull / bear flag, double top / bottom, pocket pivot, climax volume, key reversal,
               undercut-and-rally, failed breakout / breakdown, NR7, inside day, 3-weeks-tight,
               regression-channel position, support / resistance touch counts
    指標共識   Stochastic, Williams %R, CCI, MFI, Chaikin money flow, A/D line, TTM squeeze, Keltner,
               Supertrend, Ichimoku, Parabolic SAR, Donchian, Heikin-Ashi, Elder ray, Aroon, RSI(2), z-score
    相對表現   correlation / beta to SPY, up / down capture, idiosyncratic move today, RS line leadership
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


# ====================================================================== 個股習性 (stock-specific behaviour)
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
            parts.append(f"由下方觸及 {r['n_below']} 次，{r['rejected'] * 100:.0f}% 在 3 日內仍被壓在其下（平均 3 日回報 {_pc(r['fwd_below'])}）")
        if r["n_above"] >= MIN_CASES:
            parts.append(f"由上方回踩 {r['n_above']} 次，{r['held'] * 100:.0f}% 在 3 日內守住（平均 3 日回報 {_pc(r['fwd_above'])}）")
        if not parts:
            findings.append(F(f"ema{span}_reaction", "個股習性", f"EMA{span} 反應統計", 0, 0, f"EMA{span}：過去 2 年測試次數樣本不足。"))
            continue
        if near and dist < 0 and r["n_below"] >= MIN_CASES and r["rejected"] >= 0.65:
            signal, strength = -1, r["rejected"]
            parts.append(f"目前正位於 EMA{span} 下方 {abs(dist):.1f} ATR，歷史上多數會受壓")
        elif near and dist >= 0 and r["n_above"] >= MIN_CASES and r["held"] >= 0.65:
            signal, strength = 1, r["held"]
            parts.append(f"目前在 EMA{span} 上方 {dist:.1f} ATR 附近，歷史上多數守得住")
        findings.append(F(f"ema{span}_reaction", "個股習性", f"EMA{span} 反應統計", signal, strength,
                          f"EMA{span}：" + "；".join(parts) + "。", **r, distance_atr=dist))

    # Bollinger band touches: walk or revert
    up_touch = (f["close"] >= f["bb_upper"]) & f["bb_upper"].notna()
    fwd5 = c.shift(-5) / c - 1
    ut = up_touch & fwd5.notna()
    dt = (f["close"] <= f["bb_lower"]) & f["bb_lower"].notna() & fwd5.notna()
    if ut.sum() >= MIN_CASES or dt.sum() >= MIN_CASES:
        u_m, d_m = float(fwd5[ut].mean()) if ut.sum() else float("nan"), float(fwd5[dt].mean()) if dt.sum() else float("nan")
        state = "上軌" if last.close >= last.bb_upper else "下軌" if last.close <= last.bb_lower else None
        sig = 0
        if state == "上軌" and ut.sum() >= MIN_CASES:
            sig = 1 if u_m > 0.01 else -1 if u_m < -0.005 else 0
        if state == "下軌" and dt.sum() >= MIN_CASES:
            sig = 1 if d_m > 0.01 else -1 if d_m < -0.005 else 0
        findings.append(F("bb_touch", "個股習性", "布林帶觸及後表現", sig, 0.5,
                          f"收市觸及上軌 {int(ut.sum())} 次，其後 5 日平均 {_pc(u_m)}；觸及下軌 {int(dt.sum())} 次，其後 5 日平均 {_pc(d_m)}。"
                          + (f"今日收在{state}，" + ("此股傾向沿軌道延續。" if (u_m > 0.01 if state == '上軌' else d_m < -0.005) else "此股傾向回歸中軌。") if state else ""),
                          up_n=int(ut.sum()), up_fwd5=u_m, down_n=int(dt.sum()), down_fwd5=d_m))

    # RSI14 extremes
    ob = (f["rsi14"] >= 70) & fwd5.notna()
    os_ = (f["rsi14"] <= 30) & fwd5.notna()
    if ob.sum() >= MIN_CASES or os_.sum() >= MIN_CASES:
        ob_m = float(fwd5[ob].mean()) if ob.sum() else float("nan")
        os_m = float(fwd5[os_].mean()) if os_.sum() else float("nan")
        cur = _f(last.rsi14)
        sig = (1 if ob_m > 0 else -1) if cur >= 70 and ob.sum() >= MIN_CASES else (1 if os_m > 0 else -1) if cur <= 30 and os_.sum() >= MIN_CASES else 0
        findings.append(F("rsi_extreme", "個股習性", "RSI 超買／超賣後表現", sig, 0.5,
                          f"RSI14 ≥ 70 共 {int(ob.sum())} 日，其後 5 日平均 {_pc(ob_m)}（{'強者恆強型' if ob_m > 0 else '超買易回吐型'}）；"
                          f"RSI14 ≤ 30 共 {int(os_.sum())} 日，其後 5 日平均 {_pc(os_m)}。", ob_n=int(ob.sum()), ob_fwd5=ob_m, os_n=int(os_.sum()), os_fwd5=os_m))

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
        findings.append(F("gap_fill", "個股習性", "缺口回補傾向", sig, 0.5,
                          f"跳空高開 > 1% 共 {n_up} 次，{_n(rate_up * 100, 0)}% 在 5 日內回補；跳空低開 > 1% 共 {n_dn} 次，"
                          f"{_n(rate_dn * 100, 0)}% 在 5 日內回補。" + (f"今日跳空 {_pc(today_gap)}。" if abs(today_gap) > 0.01 else ""),
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
        findings.append(F("streaks", "個股習性", "連升／連跌後表現", sig, 0.4,
                          f"連升 3 日以上後，翌日再升的機率 {p_up_after_up * 100:.0f}%（{int(a3.sum())} 次）；連跌 3 日以上後，翌日反彈機率 "
                          f"{p_up_after_dn * 100:.0f}%（{int(d3.sum())} 次）。目前" + (f"已連升 {cur_up} 日。" if cur_up else f"已連跌 {cur_dn} 日。" if cur_dn else "沒有連續走勢。"),
                          p_up_after_3up=p_up_after_up, p_up_after_3down=p_up_after_dn))

    # 20D breakout follow-through for THIS stock
    brk = (c > f["prior_high20"]) & ~(c.shift(1) > f["prior_high20"].shift(1)) & fwd5.notna()
    if brk.sum() >= MIN_CASES:
        ft = float((fwd5[brk] > 0).mean())
        findings.append(F("breakout_ft", "個股習性", "突破 20 日高位後的延續率",
                          1 if (c.iloc[-1] > f["prior_high20"].iloc[-1] and ft >= 0.6) else -1 if (c.iloc[-1] > f["prior_high20"].iloc[-1] and ft <= 0.4) else 0,
                          0.5, f"此股突破 20 日高位共 {int(brk.sum())} 次，5 日後仍上升的佔 {ft * 100:.0f}%，平均 5 日回報 {_pc(float(fwd5[brk].mean()))}。",
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
        findings.append(F("big_moves", "個股習性", "大升／大跌日後表現", sig, 0.45,
                          f"單日升幅超過 2 個標準差共 {int(big_up.sum())} 次，其後 5 日平均 {_pc(bu)}；跌幅超過 2 個標準差共 {int(big_dn.sum())} 次，其後 5 日平均 {_pc(bd)}。"
                          + (f"今日變動是 {today:+.1f} 個標準差。" if np.isfinite(today) and abs(today) > 2 else ""), up_n=int(big_up.sum()), up_fwd5=bu, down_n=int(big_dn.sum()), down_fwd5=bd))

    # weekday seasonality (informational)
    wd = ret.groupby(ret.index.dayofweek).mean().tail(5)
    if len(wd) == 5:
        names = ["一", "二", "三", "四", "五"]
        best, worst = int(wd.idxmax()), int(wd.idxmin())
        findings.append(F("weekday", "個股習性", "星期效應", 0, 0, f"過去數據中星期{names[best]}平均表現最好（{_pc(wd[best], 2)}），"
                          f"星期{names[worst]}最差（{_pc(wd[worst], 2)}）。差異通常不顯著，只作參考。"))

    # autocorrelation & Hurst: trending vs mean-reverting character
    r250 = ret.dropna().tail(250)
    if len(r250) >= 200:
        ac1 = float(r250.autocorr(1))
        hurst = _hurst(np.log(c.dropna().tail(250).to_numpy()))
        character = "趨勢延續型" if hurst > 0.55 else "均值回歸型" if hurst < 0.45 else "隨機漫步型"
        findings.append(F("character", "個股習性", "走勢性格（Hurst／自相關）", 0, 0,
                          f"Hurst 指數 {hurst:.2f}、日回報一階自相關 {ac1:+.2f}，屬於{character}："
                          + ("突破和順勢策略較適合。" if hurst > 0.55 else "回調買入、高位減持較適合。" if hurst < 0.45 else "沒有明顯的統計優勢。"),
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


# ====================================================================== 籌碼分佈 (volume profile / cost)
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
        pos = "價值區上方" if close > vp["vah"] else "價值區下方" if close < vp["val"] else "價值區之內"
        sig = 1 if close > vp["vah"] else -1 if close < vp["val"] else 0
        findings.append(F("vp_value_area", "籌碼分佈", "成交量分佈：控制點與價值區", sig, 0.6,
                          f"近 {vp['window']} 日成交最密集的價位（控制點 POC）在 {_n(vp['poc'])}，70% 成交集中在 {_n(vp['val'])}–{_n(vp['vah'])}（價值區）。"
                          f"現價 {_n(close)} 位於{pos}" + ("，市場接受較高價格，屬強勢。" if sig > 0 else "，市場拒絕較高價格，屬弱勢。" if sig < 0 else "，處於均衡區。"),
                          poc=vp["poc"], val=vp["val"], vah=vp["vah"]))
        above = sorted([x for x in vp["hvn"] if x > close * 1.005])
        below = sorted([x for x in vp["hvn"] if x < close * 0.995], reverse=True)
        txt = "籌碼密集區（高成交量節點）：" + (f"上方 {', '.join(_n(x) for x in above[:2])}（容易成為阻力）" if above else "上方沒有")
        txt += "；" + (f"下方 {', '.join(_n(x) for x in below[:2])}（容易成為支持）" if below else "下方沒有") + "。"
        near_res = above and np.isfinite(atr) and (above[0] - close) <= 1.0 * atr
        findings.append(F("vp_hvn", "籌碼分佈", "籌碼密集區", -1 if near_res else 0, 0.5 if near_res else 0, txt, above=above[:3], below=below[:3]))
        vac_up = [z for z in vp["vacuum"] if z["low"] >= close * 0.995]
        vac_dn = [z for z in vp["vacuum"] if z["high"] <= close * 1.005]
        vtxt = "籌碼真空區（低成交量，價格容易快速穿過）：" + (
            "上方 " + "、".join(f"{_n(z['low'])}–{_n(z['high'])}" for z in vac_up[:2]) if vac_up else "上方沒有") + "；" + (
            "下方 " + "、".join(f"{_n(z['low'])}–{_n(z['high'])}" for z in vac_dn[-2:]) if vac_dn else "下方沒有") + "。"
        vsig = 0
        if vac_up and np.isfinite(atr) and vac_up[0]["low"] - close <= 1.0 * atr:
            vsig = 1
            vtxt += "上方真空區就在附近，突破後可能加速上升。"
        elif vac_dn and np.isfinite(atr) and close - vac_dn[-1]["high"] <= 1.0 * atr:
            vsig = -1
            vtxt += "下方真空區就在附近，跌穿後可能加速下跌。"
        findings.append(F("vp_vacuum", "籌碼分佈", "籌碼真空區", vsig, 0.5, vtxt, up=vac_up[:2], down=vac_dn[-2:]))
        pr = vp["profit_ratio"]
        findings.append(F("profit_ratio", "籌碼分佈", "獲利盤比例", 1 if pr >= 0.85 else -1 if pr <= 0.15 else 0, 0.4,
                          f"按近 120 日成交量（較近期權重較高）估算，約 {pr * 100:.0f}% 的籌碼成本低於現價（處於獲利）。"
                          + ("幾乎所有持有人都在賺錢，上方套牢盤少。" if pr >= 0.85 else "大部分持有人處於虧損，反彈時容易遇到解套沽壓。" if pr <= 0.15 else ""),
                          profit_ratio=pr))
    # anchored VWAPs
    window = f.tail(250)
    anchors = {"52 週高位": window["high"].idxmax(), "52 週低位": window["low"].idxmin()}
    gaps = f.tail(60)
    big_gap = gaps[gaps["gap_atr"].abs() >= 1]
    if not big_gap.empty:
        anchors["最近大缺口"] = big_gap.index[-1]
    av = {k: anchored_vwap(f, d) for k, d in anchors.items()}
    above = [k for k, v in av.items() if np.isfinite(v) and close > v]
    findings.append(F("avwap", "籌碼分佈", "錨定 VWAP", 1 if len(above) == len(av) else -1 if not above else 0, 0.5,
                      "錨定 VWAP（由該日起所有成交的平均成本）：" + "；".join(f"{k}（{d:%Y-%m-%d}）起 {_n(av[k])}，現價{'高於' if close > av[k] else '低於'}"
                                                                    for k, d in anchors.items()) + "。"
                      + ("所有主要群組的持有人都處於獲利。" if len(above) == len(av) else "所有主要群組的持有人都處於虧損。" if not above else ""),
                      **{k: v for k, v in av.items()}))
    cost20, cost60 = _f(f["vwap20"].iloc[-1]), anchored_vwap(f, f.index[-60]) if len(f) >= 60 else float("nan")
    findings.append(F("holder_cost", "籌碼分佈", "短中期持有人平均成本", 1 if close > cost20 and close > cost60 else -1 if close < cost20 and close < cost60 else 0, 0.4,
                      f"20 日成交量加權平均成本 {_n(cost20)}、60 日 {_n(cost60)}；現價相對 20 日成本 {_pc(close / cost20 - 1)}。", cost20=cost20, cost60=cost60))
    return vp


# ====================================================================== 形態 (patterns)
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
    findings.append(F("vcp", "形態", "波動收縮形態（VCP）", 1 if vcp else 0, 0.7 if vcp else 0,
                      ("近期每次回調幅度遞減（" + " → ".join(f"{x * 100:.0f}%" for x in depths[-3:]) + "），賣壓逐步被吸收，"
                       "典型的突破前形態。" if vcp else "沒有偵測到波動收縮形態。"), depths=depths[-3:]))

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
    findings.append(F("flag", "形態", "旗形整理", 1 if flag == "bull" else -1 if flag == "bear" else 0, 0.6 if flag else 0,
                      "偵測到上升旗形：急升（旗桿）後縮量窄幅整理，突破整理區高位可能延續升勢。" if flag == "bull" else
                      "偵測到下跌旗形：急跌後弱勢反彈整理，跌穿整理區低位可能延續跌勢。" if flag == "bear" else "沒有偵測到旗形整理。"))

    # double top / bottom
    dt_txt, dt_sig = "沒有偵測到雙頂或雙底。", 0
    if len(ph) >= 2:
        (d1, p1), (d2, p2) = ph[-2], ph[-1]
        gap_bars = f.index.get_loc(d2) - f.index.get_loc(d1)
        trough = float(lo[d1:d2].min())
        if abs(p2 / p1 - 1) <= 0.02 and gap_bars >= 8 and trough <= min(p1, p2) * 0.95 and f.index.get_loc(f.index[-1]) - f.index.get_loc(d2) <= 30:
            dt_sig = -1 if t.close < trough else 0
            dt_txt = (f"雙頂：{d1:%m-%d} 高位 {_n(p1)} 與 {d2:%m-%d} 高位 {_n(p2)} 接近，頸線 {_n(trough)}。"
                      + ("已跌穿頸線，形態確認。" if t.close < trough else "未跌穿頸線，形態未確認。"))
    if dt_sig == 0 and len(pl) >= 2:
        (d1, p1), (d2, p2) = pl[-2], pl[-1]
        gap_bars = f.index.get_loc(d2) - f.index.get_loc(d1)
        peak = float(h[d1:d2].max())
        if abs(p2 / p1 - 1) <= 0.02 and gap_bars >= 8 and peak >= max(p1, p2) * 1.05 and f.index.get_loc(f.index[-1]) - f.index.get_loc(d2) <= 30:
            dt_sig = 1 if t.close > peak else 0
            dt_txt = (f"雙底：{d1:%m-%d} 低位 {_n(p1)} 與 {d2:%m-%d} 低位 {_n(p2)} 接近，頸線 {_n(peak)}。"
                      + ("已升穿頸線，形態確認。" if t.close > peak else "未升穿頸線，形態未確認。"))
    findings.append(F("double", "形態", "雙頂／雙底", dt_sig, 0.6, dt_txt))

    # pocket pivot (O'Neil): up day volume > max down-day volume of prior 10 sessions
    prev10 = f.iloc[-11:-1]
    max_down_vol = float(prev10.loc[prev10["close"] < prev10["close"].shift(1), "volume"].max()) if len(prev10) else float("nan")
    pp = t.close > y.close and np.isfinite(max_down_vol) and t.volume > max_down_vol and t.close > _f(t.ema21)
    findings.append(F("pocket_pivot", "形態", "口袋支點（Pocket Pivot）", 1 if pp else 0, 0.6 if pp else 0,
                      "今日上漲成交量大於過去 10 日任何下跌日的成交量，而且在 EMA21 之上：機構買入的痕跡。" if pp else "今日沒有口袋支點。"))

    # climax volume / exhaustion
    vmax60 = float(v.tail(60).max())
    climax = t.volume >= vmax60 and (t.high - t.low) > 1.5 * atr
    cl_sig = -1 if climax and t.close > _f(t.ema21) and _f(t.clv) < 0.4 else 1 if climax and t.close < _f(t.ema21) and _f(t.clv) > 0.6 else 0
    findings.append(F("climax", "形態", "高潮成交量", cl_sig, 0.6 if cl_sig else 0,
                      ("今日成交量為 60 日最高且波幅巨大，" + ("在升勢中收在低位：可能是買盤高潮／見頂。" if cl_sig < 0 else
                                                     "在跌勢中收在高位：可能是恐慌拋售見底。" if cl_sig > 0 else "需要觀察之後的走勢確認。"))
                      if climax else "沒有高潮成交量。"))

    # key reversal / undercut & rally / failed breakout / failed breakdown
    kr = t.high > _f(f["high"].iloc[-21:-1].max()) and t.close < y.close
    findings.append(F("key_reversal", "形態", "關鍵反轉日", -1 if kr else 0, 0.6 if kr else 0,
                      "今日創 20 日新高後收市低於昨日收市（頂部關鍵反轉）。" if kr else "沒有關鍵反轉日。"))
    low10 = float(f["low"].iloc[-11:-1].min())
    uc = t.low < low10 and t.close > low10
    findings.append(F("undercut_rally", "形態", "跌穿後收復（Undercut & Rally）", 1 if uc else 0, 0.6 if uc else 0,
                      f"今日盤中跌穿近 10 日低位 {_n(low10)} 後收回其上，洗走止損盤後回升。" if uc else "沒有跌穿後收復的訊號。"))
    recent = f.tail(6)
    fb = bool(((recent["close"] > recent["prior_high20"]).iloc[:-1]).any()) and t.close < _f(f["prior_high20"].iloc[-6])
    fd = bool(((recent["close"] < recent["prior_low20"]).iloc[:-1]).any()) and t.close > _f(f["prior_low20"].iloc[-6])
    findings.append(F("failed_break", "形態", "假突破／假跌破", -1 if fb else 1 if fd else 0, 0.7 if (fb or fd) else 0,
                      "過去 5 日曾突破 20 日高位，但已跌回突破位之下（假突破、多頭陷阱）。" if fb else
                      "過去 5 日曾跌穿 20 日低位，但已重回其上（假跌破、空頭陷阱）。" if fd else "沒有假突破或假跌破。"))

    # NR7 / inside day
    rng = h - lo
    nr7 = rng.iloc[-1] <= rng.tail(7).min()
    inside = t.high < y.high and t.low > y.low
    findings.append(F("nr7", "形態", "窄幅日（NR7／內包日）", 0, 0,
                      ("今日是 7 日內波幅最窄的一天（NR7）" + ("，同時是內包日" if inside else "") + "：波動壓縮，常在大波動之前出現。")
                      if nr7 or inside else "今日不是窄幅日。", nr7=bool(nr7), inside=bool(inside)))

    # 3 weeks tight (weekly closes within 1.5%)
    wc = c.resample("W-FRI").last().dropna().tail(3)
    tight = len(wc) == 3 and (wc.max() / wc.min() - 1) <= 0.015
    findings.append(F("three_weeks_tight", "形態", "三週緊密收市", 1 if tight and t.close > _f(t.ema50) else 0, 0.5 if tight else 0,
                      "最近 3 個週收市價相差不足 1.5%，籌碼穩定，常見於強勢股突破前。" if tight else "沒有三週緊密收市。"))

    # regression channel position
    y60 = c.tail(60).to_numpy()
    if len(y60) == 60:
        x = np.arange(60)
        slope, icpt = np.polyfit(x, y60, 1)
        resid = y60 - (slope * x + icpt)
        z = resid[-1] / resid.std(ddof=1)
        findings.append(F("reg_channel", "形態", "60 日回歸通道", -1 if z > 1.8 else 1 if z < -1.8 else 0, 0.4,
                          f"60 日回歸線斜率每日 {slope / y60.mean() * 100:+.2f}%；現價在通道中的位置 {z:+.1f} 個標準差"
                          + ("（貼近上緣，短線偏高）" if z > 1.8 else "（貼近下緣，短線偏低）" if z < -1.8 else "") + "。", z=float(z)))

    # support / resistance touches
    lvl_txt = []
    for name, series, cmp in (("阻力", h, 1), ("支持", lo, -1)):
        ref = float(f["high"].tail(60).max()) if cmp > 0 else float(f["low"].tail(60).min())
        touches = int(((series.tail(120) - ref).abs() <= 0.5 * atr).sum()) if np.isfinite(atr) else 0
        lvl_txt.append(f"60 日{'高位' if cmp > 0 else '低位'} {_n(ref)} 作為{name}，近 120 日被測試 {touches} 次")
    findings.append(F("sr_touches", "形態", "支持／阻力測試次數", 0, 0, "；".join(lvl_txt) + "。測試越多次的價位越重要，一旦失守往往觸發較大的走勢。"))


# ====================================================================== 指標共識 (indicator consensus)
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
    findings.append(F("stoch", "指標共識", "隨機指標 KD(14,3,3)", st_sig, 0.4, f"%K {_n(kd.iloc[-1], 0)}、%D {_n(dd.iloc[-1], 0)}"
                      + ("，低位金叉。" if st_sig > 0 else "，高位死叉。" if st_sig < 0 else "。")))
    # Williams %R
    wr = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    findings.append(F("williams", "指標共識", "威廉指標 %R(14)", 1 if wr.iloc[-1] < -80 and wr.iloc[-1] > wr.iloc[-2] else -1 if wr.iloc[-1] > -20 and wr.iloc[-1] < wr.iloc[-2] else 0,
                      0.3, f"%R {_n(wr.iloc[-1], 0)}（低於 -80 為超賣，高於 -20 為超買）。"))
    # CCI 20
    tp = (h + lo + c) / 3
    md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - tp.rolling(20).mean()) / (0.015 * md)
    findings.append(F("cci", "指標共識", "順勢指標 CCI(20)", 1 if cci.iloc[-1] > 100 else -1 if cci.iloc[-1] < -100 else 0, 0.3,
                      f"CCI {_n(cci.iloc[-1], 0)}（> 100 強勢，< -100 弱勢）。"))
    # MFI 14
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    findings.append(F("mfi", "指標共識", "資金流量指標 MFI(14)", -1 if mfi.iloc[-1] > 80 else 1 if mfi.iloc[-1] < 20 else (1 if mfi.iloc[-1] > 50 else -1), 0.3,
                      f"MFI {_n(mfi.iloc[-1], 0)}（成交量加權的 RSI；> 80 資金過熱，< 20 資金枯竭）。"))
    # Chaikin money flow 20 & A/D line
    clv = ((c - lo) - (h - c)) / (h - lo).replace(0, np.nan)
    cmf = (clv * v).rolling(20).sum() / v.rolling(20).sum()
    ad = (clv.fillna(0) * v).cumsum()
    ad_up = ad.iloc[-1] > ad.iloc[-21]
    findings.append(F("cmf", "指標共識", "蔡金資金流 CMF(20)／A/D 線", 1 if cmf.iloc[-1] > 0.05 and ad_up else -1 if cmf.iloc[-1] < -0.05 and not ad_up else 0, 0.4,
                      f"CMF {_n(cmf.iloc[-1])}，累積／派發線 20 日{'上升' if ad_up else '下降'}（收市位置加權的資金流）。"))
    # TTM squeeze & Keltner
    mid = sma(c, 20)
    sd = c.rolling(20).std(ddof=0)
    atr20 = atr_fn(h, lo, c, 20)
    kc_up, kc_dn = ema(c, 20) + 1.5 * atr20, ema(c, 20) - 1.5 * atr20
    squeeze_on = (mid + 2 * sd < kc_up) & (mid - 2 * sd > kc_dn)
    mom = c - ((hh + ll) / 2 + mid) / 2
    fired = bool(squeeze_on.iloc[-2]) and not bool(squeeze_on.iloc[-1])
    findings.append(F("ttm_squeeze", "指標共識", "TTM 擠壓", (1 if mom.iloc[-1] > 0 else -1) if fired else 0, 0.6 if fired else 0,
                      ("擠壓剛釋放（布林帶由 Keltner 通道內擴張出來），動量" + ("向上。" if mom.iloc[-1] > 0 else "向下。")) if fired else
                      f"擠壓{'進行中（布林帶在 Keltner 通道內，能量累積）' if squeeze_on.iloc[-1] else '未出現'}，動量{'向上' if mom.iloc[-1] > 0 else '向下'}。"))
    findings.append(F("keltner", "指標共識", "Keltner 通道", 1 if c.iloc[-1] > kc_up.iloc[-1] else -1 if c.iloc[-1] < kc_dn.iloc[-1] else 0, 0.3,
                      f"Keltner 通道（EMA20 ± 1.5 ATR）：{_n(kc_dn.iloc[-1])}–{_n(kc_up.iloc[-1])}，現價"
                      + ("在上軌之上（強勢）。" if c.iloc[-1] > kc_up.iloc[-1] else "在下軌之下（弱勢）。" if c.iloc[-1] < kc_dn.iloc[-1] else "在通道內。")))
    # Supertrend (10, 3)
    st_dir = _supertrend(h, lo, c, 10, 3.0)
    findings.append(F("supertrend", "指標共識", "超級趨勢 Supertrend(10,3)", st_dir["dir"], 0.5,
                      f"Supertrend 方向{'向上' if st_dir['dir'] > 0 else '向下'}，線位 {_n(st_dir['line'])}，已維持 {st_dir['age']} 日"
                      + ("（剛翻轉）。" if st_dir["age"] <= 2 else "。"), line=st_dir["line"]))
    # Ichimoku
    tenkan = (h.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    top, bot = max(_f(span_a.iloc[-1]), _f(span_b.iloc[-1])), min(_f(span_a.iloc[-1]), _f(span_b.iloc[-1]))
    ich = 1 if c.iloc[-1] > top and tenkan.iloc[-1] > kijun.iloc[-1] else -1 if c.iloc[-1] < bot and tenkan.iloc[-1] < kijun.iloc[-1] else 0
    findings.append(F("ichimoku", "指標共識", "一目均衡表", ich, 0.5,
                      f"雲帶 {_n(bot)}–{_n(top)}，現價" + ("在雲上" if c.iloc[-1] > top else "在雲下" if c.iloc[-1] < bot else "在雲中")
                      + f"；轉換線 {_n(tenkan.iloc[-1])} {'高於' if tenkan.iloc[-1] > kijun.iloc[-1] else '低於'}基準線 {_n(kijun.iloc[-1])}。"))
    # Parabolic SAR
    sar_dir, sar = _psar(h, lo)
    findings.append(F("psar", "指標共識", "拋物線 SAR", sar_dir, 0.3, f"SAR {_n(sar)}，在價格{'下方（升勢）' if sar_dir > 0 else '上方（跌勢）'}。"))
    # Donchian
    d20h, d20l = h.iloc[-21:-1].max(), lo.iloc[-21:-1].min()
    d55h, d55l = h.iloc[-56:-1].max(), lo.iloc[-56:-1].min()
    don = 1 if c.iloc[-1] > d55h else -1 if c.iloc[-1] < d55l else (1 if c.iloc[-1] > d20h else -1 if c.iloc[-1] < d20l else 0)
    findings.append(F("donchian", "指標共識", "唐奇安通道（海龜 20/55）", don, 0.4,
                      f"20 日通道 {_n(d20l)}–{_n(d20h)}，55 日通道 {_n(d55l)}–{_n(d55h)}"
                      + ("；突破 55 日高位（海龜長線入場訊號）。" if c.iloc[-1] > d55h else "；跌穿 55 日低位。" if c.iloc[-1] < d55l else "。")))
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
    findings.append(F("heikin_ashi", "指標共識", "平均 K 線（Heikin-Ashi）", int(color[-1]), 0.3,
                      f"平均 K 線連續 {streak} 根{'陽燭' if color[-1] > 0 else '陰燭'}（過濾雜訊後的趨勢方向）。"))
    # Elder ray
    e13 = ema(c, 13)
    bull, bear = h - e13, lo - e13
    er = 1 if e13.iloc[-1] > e13.iloc[-2] and bear.iloc[-1] < 0 and bear.iloc[-1] > bear.iloc[-2] else \
        -1 if e13.iloc[-1] < e13.iloc[-2] and bull.iloc[-1] > 0 and bull.iloc[-1] < bull.iloc[-2] else 0
    findings.append(F("elder_ray", "指標共識", "艾達透視（Elder Ray）", er, 0.4, f"多頭力量 {_n(bull.iloc[-1])}、空頭力量 {_n(bear.iloc[-1])}，EMA13 {'上升' if e13.iloc[-1] > e13.iloc[-2] else '下降'}"
                      + ("：升勢中空頭力量減弱，是 Elder 的買入條件。" if er > 0 else "：跌勢中多頭力量減弱，是 Elder 的沽出條件。" if er < 0 else "。")))
    # Aroon 25
    au = 100 * (25 - (h.tail(26).to_numpy()[::-1].argmax())) / 25
    ad_ = 100 * (25 - (lo.tail(26).to_numpy()[::-1].argmin())) / 25
    findings.append(F("aroon", "指標共識", "阿隆指標 Aroon(25)", 1 if au > 70 and ad_ < 30 else -1 if ad_ > 70 and au < 30 else 0, 0.3,
                      f"Aroon 上 {au:.0f}、下 {ad_:.0f}（衡量距離最近高／低位的時間）。"))
    # RSI(2) short-term mean reversion (Connors)
    r2 = rsi(c, 2)
    r2s = 1 if r2.iloc[-1] < 10 and c.iloc[-1] > _f(t.ema200) else -1 if r2.iloc[-1] > 90 and c.iloc[-1] < _f(t.ema200) else 0
    findings.append(F("rsi2", "指標共識", "RSI(2) 短線均值回歸", r2s, 0.4, f"RSI(2) {_n(r2.iloc[-1], 0)}"
                      + ("：長線升勢中的短線極度超賣（Connors 買入條件）。" if r2s > 0 else "：長線跌勢中的短線極度超買（Connors 沽出條件）。" if r2s < 0 else "。")))
    # z-score vs 20-day mean
    z = (c.iloc[-1] - mid.iloc[-1]) / sd.iloc[-1] if sd.iloc[-1] > 0 else float("nan")
    findings.append(F("zscore", "指標共識", "20 日 Z 分數", -1 if z > 2.2 else 1 if z < -2.2 else 0, 0.3, f"現價偏離 20 日均線 {z:+.1f} 個標準差。"))


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


# ====================================================================== 相對表現 (relative behaviour)
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
    findings.append(F("capture", "相對表現", "上漲／下跌捕捉率", cap_sig, 0.5,
                      f"近 120 日與 SPY 相關係數 {corr:.2f}、Beta {beta:.2f}；"
                      + (f"大市上升日的平均升幅是大市的 {upcap:.1f} 倍，大市下跌日的平均跌幅是大市的 {dncap:.1f} 倍"
                         + ("（升市時更強、跌市時相對抗跌，領導股特徵）。" if cap_sig > 0 else "（跌市時跌得更多，屬弱勢高 Beta）。" if cap_sig < 0 else "。")
                         if meaningful else "與大市走勢關係很弱或相反，捕捉率沒有參考價值。"),
                      corr=corr, beta=beta, up_capture=upcap, down_capture=dncap))
    resid = j.iloc[:, 0] - beta * j.iloc[:, 1]
    today = float(r.iloc[-1] - beta * m.iloc[-1]) if np.isfinite(r.iloc[-1]) and np.isfinite(m.iloc[-1]) else float("nan")
    zt = today / float(resid.std(ddof=1)) if resid.std(ddof=1) > 0 else float("nan")
    findings.append(F("idio", "相對表現", "今日個股獨立走勢", 1 if zt > 2 else -1 if zt < -2 else 0, 0.5 if abs(_f(zt)) > 2 else 0,
                      f"扣除大市影響（Beta × SPY）後，今日個股獨立回報 {_pc(today, 2)}，約 {zt:+.1f} 個標準差"
                      + ("，有個股消息或資金推動的可能。" if abs(_f(zt)) > 2 else "。")))
    rs = (f["close"] / s).dropna()
    lead = len(rs) >= 60 and rs.iloc[-1] >= rs.tail(60).max() * 0.995 and f["close"].iloc[-1] < f["close"].tail(60).max() * 0.98
    findings.append(F("rs_lead", "相對表現", "相對強弱線領先", 1 if lead else 0, 0.6 if lead else 0,
                      "相對強弱線已創 60 日新高，但股價仍未創新高：資金領先流入的典型訊號。" if lead else "相對強弱線沒有領先股價創新高。"))


# ====================================================================== entry point
def run_all(f: pd.DataFrame, spy: pd.Series | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    vp: dict[str, Any] = {}
    for step in (lambda: behaviour(f, findings), lambda: patterns(f, findings), lambda: indicators(f, findings),
                 lambda: relative(f, spy, findings)):
        try:
            step()
        except Exception as exc:  # noqa: BLE001 - one broken technique must not kill the note
            findings.append(F("error", "系統", "計算錯誤", 0, 0, f"部分技術計算失敗：{type(exc).__name__}"))
    try:
        vp = chips(f, findings)
    except Exception as exc:  # noqa: BLE001
        findings.append(F("error_vp", "系統", "計算錯誤", 0, 0, f"籌碼分佈計算失敗：{type(exc).__name__}"))
    groups: dict[str, dict[str, Any]] = {}
    for x in findings:
        g = groups.setdefault(x["group"], {"group": x["group"], "bull": 0, "bear": 0, "score": 0.0, "n": 0})
        g["n"] += 1
        g["bull"] += x["signal"] > 0
        g["bear"] += x["signal"] < 0
        g["score"] += x["signal"] * x["strength"]
    return {"findings": findings, "groups": list(groups.values()), "volume_profile": vp, "count": len(findings)}
