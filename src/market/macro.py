"""Cross-asset context, breadth and industry-group strength (context only - not part of the regime score).

Cross-asset (Yahoo daily bars, aligned to NYSE sessions by carrying the last value forward):
    US Dollar Index (DX-Y.NYB), WTI crude (CL=F), gold (GC=F), Bitcoin (BTC-USD), 10-year yield
    proxy is already in the regime. For each: last value, 1 / 5 / 20-session change, position vs EMA50,
    z-score of the 20-session change against its own 1-year history, and the 60-session correlation of
    daily returns with QQQ (how much the asset currently moves with growth stocks).

Advance / decline: computed on the terminal's own universe (~250 high-beta stocks). NOT the NYSE-wide
    A/D line - no free, reliable source exists for that. Advancers / decliners / unchanged per session,
    cumulative A/D line, McClellan-style oscillator (EMA19 - EMA39 of (adv - dec) / (adv + dec) x 1000),
    up-volume share, and 10-session net advances.

Industry groups: every universe stock is assigned its Yahoo industry (cached); groups with >= 2 members
    are ranked by the median of the stocks' relative-strength score (0.5 x rel_20d + 0.3 x rel_60d + 0.2 x
    rel_5d, the same mix as the stock RS percentile). The rank 5 sessions ago shows the direction of travel.
    Stocks whose industry is unknown are grouped by their sector ETF.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

MACRO_DEFAULT = {"DX-Y.NYB": "美元指數 DXY", "CL=F": "WTI 原油", "GC=F": "黃金", "BTC-USD": "比特幣"}


def _aligned_close(prices: dict[str, pd.DataFrame], symbol: str, index: pd.DatetimeIndex) -> pd.Series | None:
    f = prices.get(symbol)
    if f is None or f.empty:
        return None
    s = f["close"].astype(float)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def macro_table(prices: dict[str, pd.DataFrame], macro: dict[str, str], index: pd.DatetimeIndex) -> list[dict[str, Any]]:
    qqq = _aligned_close(prices, "QQQ", index)
    out = []
    for sym, name in macro.items():
        c = _aligned_close(prices, sym, index)
        if c is None or c.dropna().size < 60:
            out.append({"symbol": sym, "name": name, "available": False})
            continue
        c = c.dropna()
        ema50 = c.ewm(span=50, adjust=False).mean()
        ch20 = c.pct_change(20)
        hist = ch20.tail(252).dropna()
        z = float((ch20.iloc[-1] - hist.mean()) / hist.std()) if len(hist) > 30 and hist.std() > 0 else None
        corr = None
        if qqq is not None:
            j = pd.concat([c.pct_change(), qqq.reindex(c.index).pct_change()], axis=1).dropna().tail(60)
            corr = float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) >= 40 else None
        out.append({"symbol": sym, "name": name, "available": True, "close": float(c.iloc[-1]),
                    "ret_1d": float(c.pct_change().iloc[-1]), "ret_5d": float(c.pct_change(5).iloc[-1]),
                    "ret_20d": float(ch20.iloc[-1]), "z20": z, "above_ema50": bool(c.iloc[-1] > ema50.iloc[-1]),
                    "corr_qqq_60d": corr, "spark": [float(x) for x in c.tail(60)],
                    "comment": _macro_comment(sym, float(ch20.iloc[-1]), z, corr)})
    return out


def _macro_comment(sym: str, ch20: float, z: float | None, corr: float | None) -> str:
    strong = z is not None and abs(z) >= 1.5
    up = ch20 > 0
    if sym == "DX-Y.NYB":
        txt = "美元走強，通常收緊金融條件，對高估值成長股不利。" if up else "美元走弱，金融條件較寬鬆，對成長股和商品較有利。"
    elif sym == "CL=F":
        txt = "油價上升推高通脹預期，可能令減息預期降溫。" if up else "油價回落，通脹壓力減輕。"
    elif sym == "GC=F":
        txt = "金價上升，避險或實際利率下降的需求增加。" if up else "金價回落，避險需求減少。"
    elif sym == "BTC-USD":
        txt = "比特幣上升，投機風險胃納較強，常與高 Beta 股同步。" if up else "比特幣回落，投機風險胃納轉弱，留意高 Beta 股。"
    else:
        txt = ""
    if strong:
        txt = f"20 日變化屬一年內的極端（z = {z:+.1f}）：" + txt
    if corr is not None and abs(corr) >= 0.4:
        txt += f" 近 60 日與 QQQ 的相關係數 {corr:+.2f}，{'同向' if corr > 0 else '反向'}關係明顯。"
    return txt


def advance_decline(prices: dict[str, pd.DataFrame], symbols: list[str], index: pd.DatetimeIndex,
                    keep: int = 160) -> dict[str, Any]:
    closes = pd.DataFrame({s: prices[s]["close"] for s in symbols if s in prices}).reindex(index)
    vols = pd.DataFrame({s: prices[s]["volume"] for s in symbols if s in prices}).reindex(index)
    if closes.shape[1] < 10:
        return {"available": False}
    ch = closes.diff()
    adv, dec = (ch > 0).sum(axis=1), (ch < 0).sum(axis=1)
    unch = (ch == 0).sum(axis=1)
    valid = (adv + dec) > 0
    adv, dec, unch = adv[valid], dec[valid], unch[valid]
    net = adv - dec
    ratio = (net / (adv + dec)) * 1000
    mcc = ratio.ewm(span=19, adjust=False).mean() - ratio.ewm(span=39, adjust=False).mean()
    upv = vols[ch > 0].sum(axis=1)[valid]
    totv = vols[ch.notna() & (ch != 0)].sum(axis=1)[valid]
    ad_line = net.cumsum()
    tail = slice(-keep, None)
    last = {"date": adv.index[-1].strftime("%Y-%m-%d"), "advancers": int(adv.iloc[-1]), "decliners": int(dec.iloc[-1]),
            "unchanged": int(unch.iloc[-1]), "up_volume_share": float(upv.iloc[-1] / totv.iloc[-1]) if totv.iloc[-1] else None,
            "mcclellan": float(mcc.iloc[-1]), "net_10d": int(net.tail(10).sum()), "n": int(closes.shape[1])}
    hi, ad_tail = ad_line.tail(60).max(), ad_line.iloc[-1]
    spy = _aligned_close(prices, "SPY", adv.index)
    divergence = None
    if spy is not None:
        spy_high = spy.iloc[-1] >= spy.tail(60).max() * 0.99
        ad_high = ad_tail >= hi - 1e-9
        if spy_high and not ad_high:
            divergence = "SPY 接近 60 日高位，但股票池的 A/D 線沒有創新高：上升由少數股票帶動（負背離）。"
        elif not spy_high and ad_high:
            divergence = "股票池的 A/D 線創 60 日新高而 SPY 未有：高 Beta 股普遍走強（正背離）。"
    mcv = last["mcclellan"]
    last["comment"] = (("McClellan 振盪指標 " + f"{mcv:+.0f}：" + ("廣度動量強，升市參與面廣。" if mcv > 50 else "廣度動量偏弱，跌市參與面廣。"
                       if mcv < -50 else "廣度動量中性。")) + (" " + divergence if divergence else ""))
    return {"available": True, "scope": f"股票池 {closes.shape[1]} 檔（非全市場）", **last,
            "series": {"dates": [d.strftime("%Y-%m-%d") for d in adv.index[tail]], "ad_line": [int(x) for x in ad_line.iloc[tail]],
                       "net": [int(x) for x in net.iloc[tail]], "mcclellan": [float(x) for x in mcc.iloc[tail]]}}


def _rs_score(f: pd.DataFrame) -> pd.Series:
    return 0.5 * f["rel_20d"] + 0.3 * f["rel_60d"] + 0.2 * f["rel_5d"]


def industry_groups(features: dict[str, pd.DataFrame], industries: dict[str, str | None], fallback: dict[str, str],
                    lookback: int = 5) -> list[dict[str, Any]]:
    """Industry groups ranked by the median RS score of their members, now and ``lookback`` sessions ago."""
    rows = []
    for sym, f in features.items():
        g = industries.get(sym) or f"{fallback.get(sym, '其他')}（行業未知）"
        rs = _rs_score(f)
        if rs.dropna().size < lookback + 1:
            continue
        rows.append({"symbol": sym, "group": g, "rs": float(rs.iloc[-1]), "rs_prev": float(rs.iloc[-1 - lookback]),
                     "ret_5d": float(f["return_5d"].iloc[-1]), "ret_20d": float(f["return_20d"].iloc[-1]),
                     "above_ema50": bool(f["close"].iloc[-1] > f["ema50"].iloc[-1]) if "ema50" in f else None})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    agg = df.groupby("group").agg(n=("symbol", "size"), rs=("rs", "median"), rs_prev=("rs_prev", "median"),
                                  ret_5d=("ret_5d", "median"), ret_20d=("ret_20d", "median"),
                                  breadth=("above_ema50", "mean"), members=("symbol", list))
    agg = agg[agg["n"] >= 2].copy()
    if agg.empty:
        return []
    agg["rank"] = agg["rs"].rank(ascending=False, method="min").astype(int)
    agg["rank_prev"] = agg["rs_prev"].rank(ascending=False, method="min").astype(int)
    agg["pct"] = agg["rs"].rank(pct=True) * 100
    leaders = df.sort_values("rs", ascending=False).groupby("group")["symbol"].apply(lambda s: list(s)[:3])
    out = []
    for g, r in agg.sort_values("rank").iterrows():
        out.append({"group": g, "n": int(r["n"]), "rank": int(r["rank"]), "rank_prev": int(r["rank_prev"]),
                    "rank_change": int(r["rank_prev"] - r["rank"]), "rs": float(r["rs"]), "pct": float(r["pct"]),
                    "ret_5d": float(r["ret_5d"]), "ret_20d": float(r["ret_20d"]),
                    "breadth": float(r["breadth"]) if pd.notna(r["breadth"]) else None,
                    "leaders": leaders.get(g, []), "members": sorted(r["members"])})
    return out


def group_lookup(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """symbol -> {group, rank, of, rank_change, pct}"""
    total = len(groups)
    out = {}
    for g in groups:
        for s in g["members"]:
            out[s] = {"group": g["group"], "rank": g["rank"], "of": total, "rank_change": g["rank_change"], "pct": g["pct"]}
    return out


__all__ = ["MACRO_DEFAULT", "macro_table", "advance_decline", "industry_groups", "group_lookup"]
