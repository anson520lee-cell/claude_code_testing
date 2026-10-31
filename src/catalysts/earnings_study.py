"""Earnings reaction study: what the options market implies vs what the stock usually does.

For every past earnings date (Yahoo earnings calendar, up to ~16 quarters):
    reaction session  before the open -> that day; after the close -> the next session;
                      time unknown -> whichever of the two sessions moved more (gap vs prior close)
    gap               open(reaction) / close(prior session) - 1
    move              close(reaction) / close(prior session) - 1
    drift 5 / 10      close(reaction + 5 / 10 sessions) / close(reaction) - 1   (post-earnings drift)
Summary: median and mean absolute move, largest moves, share of up reactions, whether the drift
continues in the direction of the reaction (post-earnings announcement drift), and the options-implied
move (ATM straddle of the first expiry after the next earnings date) compared with the historical
median absolute move: implied > history -> options price a bigger move than usual (rich), and vice versa.
Descriptive statistics only - small samples (8-16 events).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.market import calendar


def _naive(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return idx


def reactions(dates: pd.DataFrame | None, prices: pd.DataFrame, session: date) -> list[dict[str, Any]]:
    if dates is None or len(dates) == 0 or prices.empty:
        return []
    idx = _naive(dates.index)
    close, open_ = prices["close"].astype(float), prices["open"].astype(float)
    days = prices.index
    out = []
    for ts, (_, r) in zip(idx, dates.iterrows()):
        if ts.normalize() > pd.Timestamp(session):
            continue
        minutes = ts.hour * 60 + ts.minute
        day = pd.Timestamp(ts.date())
        pos_day = days.searchsorted(day)              # first session on / after the date
        if pos_day >= len(days) or pos_day == 0:
            continue
        if 0 < minutes < 9 * 60 + 30:
            pos, timing = pos_day if days[pos_day] == day else pos_day, "before open"
        elif minutes >= 16 * 60:
            pos, timing = (pos_day + 1 if days[pos_day] == day else pos_day), "after close"
        else:  # unknown: pick the larger gap of the date's session and the next one
            cands = [p for p in (pos_day, pos_day + 1) if 0 < p < len(days)]
            pos = max(cands, key=lambda p: abs(open_.iat[p] / close.iat[p - 1] - 1))
            timing = "estimated"
        if pos >= len(days) or pos == 0:
            continue
        prev = close.iat[pos - 1]
        rec = {"date": ts.strftime("%Y-%m-%d"), "reaction_session": days[pos].strftime("%Y-%m-%d"), "timing": timing,
               "gap": open_.iat[pos] / prev - 1, "move": close.iat[pos] / prev - 1,
               "drift5": close.iat[pos + 5] / close.iat[pos] - 1 if pos + 5 < len(days) else None,
               "drift10": close.iat[pos + 10] / close.iat[pos] - 1 if pos + 10 < len(days) else None,
               "surprise_pct": float(r.get("Surprise(%)")) if pd.notna(r.get("Surprise(%)")) else None}
        out.append(rec)
    out.sort(key=lambda x: x["date"], reverse=True)
    seen, unique = set(), []
    for x in out:   # Yahoo sometimes lists the same quarter twice
        if x["reaction_session"] not in seen:
            seen.add(x["reaction_session"])
            unique.append(x)
    return unique


def implied_earnings_move(chains: list[dict[str, Any]] | None, spot: float, reaction: date | None) -> dict[str, Any] | None:
    """ATM straddle of the first expiry on / after the reaction session (the move the options price in)."""
    if not chains or reaction is None or not spot:
        return None
    for c in sorted(chains, key=lambda c: c["days"]):
        if date.fromisoformat(str(c["expiry"])) < reaction:
            continue
        calls, puts = c["calls"], c["puts"]
        if calls.empty or puts.empty:
            return None
        k = float(calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]["strike"].iloc[0])
        def mid(frame: pd.DataFrame) -> float | None:
            row = frame[frame["strike"] == k]
            if row.empty:
                return None
            v = row["mid"].iloc[0] if "mid" in row and pd.notna(row["mid"].iloc[0]) else row["lastPrice"].iloc[0]
            return float(v) if pd.notna(v) and v > 0 else None
        cm, pm = mid(calls), mid(puts)
        if cm is None or pm is None:
            return None
        out = {"expiry": str(c["expiry"]), "strike": k, "straddle": cm + pm, "straddle_pct": (cm + pm) / spot,
               "move_pct": (cm + pm) / spot, "days": int(c["days"]), "method": "straddle"}
        gap = calendar.trading_days_between(reaction, date.fromisoformat(str(c["expiry"])))
        before = [b for b in chains if date.fromisoformat(str(b["expiry"])) < reaction and b["days"] >= 2]
        if c["days"] > 3 and before:
            # isolate the event: total variance to the post-earnings expiry minus the "normal" variance
            # implied by the last pre-earnings expiry; expected absolute move = sqrt(2/pi) x event sigma
            try:
                from src.derivatives.options import _atm_iv
                iv_after, iv_before = _atm_iv(c, spot), _atm_iv(before[-1], spot)
            except Exception:  # noqa: BLE001 - chains without iv columns (tests, partial data)
                iv_after = iv_before = None
            if iv_after and iv_before:
                t = max(c["days"], 1) / 365
                ev = iv_after ** 2 * t - iv_before ** 2 * max(t - 1 / 365, 0)
                out.update({"iv_after": iv_after, "iv_before": iv_before, "expiry_before": str(before[-1]["expiry"])})
                if ev > 0:
                    out.update({"move_pct": float(np.sqrt(2 / np.pi) * np.sqrt(ev)), "method": "event_variance"})
                else:   # the post-earnings expiry is not priced richer than the pre-earnings one: no event premium
                    out.update({"move_pct": None, "method": "no_event_premium"})
        out["expiry_gap_sessions"] = gap
        return out
    return None


def earnings_study(dates: pd.DataFrame | None, prices: pd.DataFrame, session: date, next_earnings: dict | None,
                   chains: list[dict[str, Any]] | None) -> dict[str, Any]:
    hist = reactions(dates, prices, session)
    out: dict[str, Any] = {"available": bool(hist), "events": hist[:16], "n": len(hist)}
    if not hist:
        out["reason"] = "沒有足夠的歷史財報日期。"
        return out
    mv = np.array([h["move"] for h in hist], dtype=float)
    gp = np.array([h["gap"] for h in hist], dtype=float)
    out.update({"median_abs_move": float(np.median(np.abs(mv))), "mean_abs_move": float(np.mean(np.abs(mv))),
                "max_up": float(mv.max()), "max_down": float(mv.min()), "up_share": float((mv > 0).mean()),
                "median_abs_gap": float(np.median(np.abs(gp))),
                "gap_fade_share": float(np.mean([(h["move"] - h["gap"]) * np.sign(h["gap"]) < 0 for h in hist]))})
    d5 = [(h["move"], h["drift5"]) for h in hist if h["drift5"] is not None]
    d10 = [(h["move"], h["drift10"]) for h in hist if h["drift10"] is not None]
    if d5:
        out["drift5_same_dir"] = float(np.mean([np.sign(m) == np.sign(d) for m, d in d5]))
        out["drift5_signed_mean"] = float(np.mean([np.sign(m) * d for m, d in d5]))
    if d10:
        out["drift10_same_dir"] = float(np.mean([np.sign(m) == np.sign(d) for m, d in d10]))
        out["drift10_signed_mean"] = float(np.mean([np.sign(m) * d for m, d in d10]))
    spot = float(prices["close"].iloc[-1])
    reaction = date.fromisoformat(next_earnings["reaction_session"]) if next_earnings and next_earnings.get("reaction_session") else None
    imp = implied_earnings_move(chains, spot, reaction)
    out["next"] = next_earnings
    out["implied"] = imp
    if imp and imp["method"] == "no_event_premium":
        out["implied_vs_history"] = None
        out["note"] = (f"財報後到期日 {imp['expiry']} 的引伸波幅（{imp['iv_after'] * 100:.0f}%）不高於財報前到期日 {imp['expiry_before']}"
                       f"（{imp['iv_before'] * 100:.0f}%），期權暫時沒有為財報定出額外波幅；過去 {len(hist)} 次財報實際波幅中位數 ±{out['median_abs_move'] * 100:.1f}%。")
    elif imp:
        ratio = imp["move_pct"] / out["median_abs_move"] if out["median_abs_move"] else None
        out["implied_vs_history"] = ratio
        if imp["method"] == "event_variance":
            extra = (f"（{imp['expiry']} 到期的跨式為 ±{imp['straddle_pct'] * 100:.1f}%，覆蓋 {imp['days']} 個曆日；"
                     f"已扣除財報前到期日 {imp['expiry_before']} 引伸波幅 {imp['iv_before'] * 100:.0f}% 所代表的平日波動，只保留財報當日的部分）")
        elif imp.get("days", 0) > 5:
            extra = (f"（{imp['expiry']} 到期的跨式覆蓋 {imp['days']} 個曆日，又沒有財報前的到期日可作扣減，"
                     "包含財報以外的日子，隱含的財報波幅會被高估）")
        else:
            extra = ""
        out["note"] = (f"期權隱含財報波幅 ±{imp['move_pct'] * 100:.1f}%，過去 {len(hist)} 次財報實際波幅中位數 ±{out['median_abs_move'] * 100:.1f}%"
                       + (f"：期權定價比歷史{'大' if ratio > 1.15 else '小' if ratio < 0.85 else '接近'}（{ratio:.2f} 倍）" if ratio else "") + "。" + extra)
    else:
        out["note"] = f"過去 {len(hist)} 次財報實際波幅中位數 ±{out['median_abs_move'] * 100:.1f}%。" + (
            "沒有覆蓋財報日的期權到期日，無法比較期權隱含波幅。" if next_earnings else "未有下一次財報日期。")
    if "drift5_same_dir" in out:
        out["drift_note"] = (f"財報後 5 日延續首日方向的比例 {out['drift5_same_dir'] * 100:.0f}%，平均（按首日方向）{out['drift5_signed_mean'] * 100:+.1f}%："
                             + ("有財報後漂移（PEAD）傾向。" if out["drift5_same_dir"] >= 0.6 and out["drift5_signed_mean"] > 0
                                else "首日反應之後傾向回吐。" if out["drift5_same_dir"] <= 0.4 and out["drift5_signed_mean"] < 0 else "沒有明顯的延續或回吐傾向。"))
    return out
