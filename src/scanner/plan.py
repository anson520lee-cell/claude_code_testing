"""Risk plan for each setup: entry reference, stop, two targets, reward/risk - vectorised.

The SAME rules are used for today's candidates and for the historical simulation, so
the historical statistics describe exactly the plan shown in the report.

Long (short is the mirror image with lows <-> highs):

    Entry reference  = today's close (the report is read after the close; the simulation
                       fills at the NEXT session's open, so overnight gaps are included).
                       squeeze: conditional entry at the 20-session high + 0.1 ATR (buy-stop);
                       it is only "filled" if the next session trades there.
    Structural stop  breakout : min(today's Low, previous 20-session high) - 0.25 ATR
                     pullback : min(Low of last 5 sessions, EMA21) - 0.25 ATR
                     squeeze  : Low of last 5 sessions - 0.25 ATR
                     reclaim  : Low of last 3 sessions - 0.25 ATR
                     momentum : max(EMA21 - 0.5 ATR, Low of last 5 sessions - 0.25 ATR)
    Stop distance is clamped to [min_stop_atr, max_stop_atr] ATRs (default 1-3). A structural
    stop beyond the cap is flagged "wide stop" (the capped stop sits inside the structure).
    Target 1 = nearest overhead reference at least 0.75 ATR above entry among the 20-, 60- and
               252-session highs; if none (price at highs): entry + 2 ATR.
    Target 2 = max(Target 1 + 1 ATR, entry + 2 x risk).
    Reward/risk = (Target 1 - entry) / (entry - stop).
    Time stop = ``plan_horizon_days`` sessions (default 7): exit at that close.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.scanner.setups import SETUP_DIRECTION


def _nearest_above(entry: pd.Series, gap: pd.Series, candidates: list[pd.Series]) -> pd.Series:
    stack = np.vstack([c.to_numpy(dtype=float) for c in candidates])
    threshold = (entry + gap).to_numpy(dtype=float)
    valid = np.where(stack > threshold, stack, np.inf)
    best = valid.min(axis=0)
    return pd.Series(np.where(np.isfinite(best), best, np.nan), index=entry.index)


def _nearest_below(entry: pd.Series, gap: pd.Series, candidates: list[pd.Series]) -> pd.Series:
    stack = np.vstack([c.to_numpy(dtype=float) for c in candidates])
    threshold = (entry - gap).to_numpy(dtype=float)
    valid = np.where(stack < threshold, stack, -np.inf)
    best = valid.max(axis=0)
    return pd.Series(np.where(np.isfinite(best), best, np.nan), index=entry.index)


def plan_for_setup(df: pd.DataFrame, setup: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Plan columns for one setup on every day (values are meaningful where the setup is active)."""
    a, c = df["atr"], df["close"]
    direction = SETUP_DIRECTION[setup]
    min_atr, max_atr = float(cfg.get("min_stop_atr", 1.0)), float(cfg.get("max_stop_atr", 3.0))
    conditional = setup == "squeeze"
    if direction == "long":
        entry = (df["high20"] + 0.1 * a) if conditional else c
        structural = {
            "breakout": np.minimum(df["low"], df["prior_high20"]) - 0.25 * a,
            "pullback": np.minimum(df["low5"], df["ema21"]) - 0.25 * a,
            "squeeze": df["low5"] - 0.25 * a,
            "reclaim": df["low3"] - 0.25 * a,
            "momentum": np.maximum(df["ema21"] - 0.5 * a, df["low5"] - 0.25 * a),
        }[setup]
        dist = (entry - structural) / a
        risk_atr = dist.clip(lower=min_atr, upper=max_atr)
        stop = entry - risk_atr * a
        t1 = _nearest_above(entry, 0.75 * a, [df["high20"], df["high60"], df["high252"]])
        t1 = t1.fillna(entry + 2 * a)
        risk = entry - stop
        t2 = np.maximum(t1 + a, entry + 2 * risk)
        reward = t1 - entry
    else:
        entry = c
        structural = {
            "breakdown": np.maximum(df["high"], df["prior_low20"]) + 0.25 * a,
            "rally_fail": np.maximum(df["high5"], df["ema21"]) + 0.25 * a,
            "ema21_loss": df["high3"] + 0.25 * a,
        }[setup]
        dist = (structural - entry) / a
        risk_atr = dist.clip(lower=min_atr, upper=max_atr)
        stop = entry + risk_atr * a
        t1 = _nearest_below(entry, 0.75 * a, [df["low20"], df["low60"], df["low252"]])
        t1 = t1.fillna(entry - 2 * a)
        risk = stop - entry
        t2 = np.minimum(t1 - a, entry - 2 * risk)
        reward = entry - t1
    return pd.DataFrame({
        "entry": entry, "stop": stop, "target1": t1, "target2": t2,
        "risk_atr": risk_atr, "structural_stop_atr": dist, "wide_stop": dist > max_atr,
        "reward_risk": reward / risk.where(risk > 0), "risk_pct": risk / entry,
        "conditional_entry": conditional,
    }, index=df.index)


STOP_BASIS = {  # why the stop sits where it sits (shown next to every plan)
    "breakout": "突破當日低位與突破位（前 20 日高位）兩者較低者，再減 0.25 ATR：收市跌回突破位之下代表突破失敗",
    "pullback": "近 5 日最低位與 EMA21 兩者較低者，再減 0.25 ATR：跌穿回調低點和 EMA21 代表上升趨勢受損",
    "squeeze": "近 5 日最低位減 0.25 ATR：跌穿收窄區間的下沿，代表向上突破的前提消失",
    "reclaim": "近 3 日最低位減 0.25 ATR：跌回收復 EMA21 前的低點，代表收復失敗",
    "momentum": "EMA21 − 0.5 ATR 與近 5 日低位 − 0.25 ATR 兩者較高者：強勢股不應跌穿 EMA21",
    "breakdown": "跌破當日高位與跌破位（前 20 日低位）兩者較高者，再加 0.25 ATR：收市重回跌破位之上代表跌破失敗",
    "rally_fail": "近 5 日最高位與 EMA21 兩者較高者，再加 0.25 ATR：升穿反彈高點和 EMA21 代表下跌趨勢受損",
    "ema21_loss": "近 3 日最高位加 0.25 ATR：重回跌穿前的高點，代表跌穿失敗",
}


def plan_basis(df: pd.DataFrame, setup: str, plan: dict[str, Any], cfg: dict[str, Any]) -> dict[str, str]:
    """Plain-language reason for today's stop and targets (structure, ATR noise floor, resistance)."""
    last = df.iloc[-1]
    long = SETUP_DIRECTION[setup] == "long"
    stop = STOP_BASIS[setup] + "。"
    sdist = plan["structural_stop_atr"]
    if plan.get("wide_stop"):
        stop += f"結構止損距離 {sdist:.1f} ATR 超過上限，已收窄至 {cfg.get('max_stop_atr', 3)} ATR（較易被掃）。"
    elif sdist < float(cfg.get("min_stop_atr", 1.0)):
        stop += f"結構止損只有 {sdist:.1f} ATR，小於日常波動，已放寬至 {cfg.get('min_stop_atr', 1.0)} ATR 以免被雜訊掃走。"
    else:
        stop += f"距離 {plan['risk_atr']:.1f} ATR（每股風險 {plan['risk_pct'] * 100:.1f}%）。"
    t1 = plan["target1"]
    names = {"high20": "20 日高位", "high60": "60 日高位", "high252": "52 週高位"} if long else \
        {"low20": "20 日低位", "low60": "60 日低位", "low252": "52 週低位"}
    src = next((n for col, n in names.items() if pd.notna(last.get(col)) and abs(float(last[col]) - t1) < 1e-6), None)
    target = (f"目標一 = 上方最近而且至少 0.75 ATR 以外的{src}（阻力）" if long else f"目標一 = 下方最近而且至少 0.75 ATR 以外的{src}（支持）") \
        if src else f"目標一 = 進場價 {'+' if long else '−'} 2 ATR（{'上方' if long else '下方'}沒有 20／60／252 日的阻力支持位）"
    target += "；目標二 = 目標一再加 1 ATR 與 2 倍風險兩者較遠者。時間止損 = 持倉 " + str(plan.get("horizon_days", 7)) + " 個交易日後收市出場。"
    return {"stop_basis": stop, "target_basis": target,
            "method": "結構位 + ATR 噪音下限 + 歷史阻力／支持（未使用籌碼和期權；期權預期波幅只作參考）"}


def plan_row(df: pd.DataFrame, setup: str, cfg: dict[str, Any], horizon: int) -> dict[str, Any]:
    """Today's plan for one setup as a plain dict."""
    p = plan_for_setup(df, setup, cfg).iloc[-1]
    a = float(df["atr"].iloc[-1])
    out = {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, (bool, np.bool_)) else bool(v))
           for k, v in p.items()}
    out["direction"] = SETUP_DIRECTION[setup]
    out["horizon_days"] = horizon
    out["atr"] = a
    # 1-sigma move over the horizon from realised volatility (hv20): close x hv20 x sqrt(h / 252)
    hv = float(df["hv20"].iloc[-1]) if pd.notna(df["hv20"].iloc[-1]) else float("nan")
    out["expected_move_hv"] = float(df["close"].iloc[-1] * hv * np.sqrt(horizon / 252)) if np.isfinite(hv) else None
    out.update(plan_basis(df, setup, out, cfg))
    return out
