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
               252-session highs, capped at entry + 2 ATR (``max_target1_atr``: the SL/TP backtest showed
               far targets are rarely reached); if none (price at highs): entry + 2 ATR.
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
            "crash_rebound": entry - 2.0 * a,
        }[setup]
        dist = (entry - structural) / a
        risk_atr = dist.clip(lower=min_atr, upper=max_atr)
        stop = entry - risk_atr * a
        t1 = _nearest_above(entry, 0.75 * a, [df["high20"], df["high60"], df["high252"]])
        t1 = np.minimum(t1.fillna(entry + 2 * a), entry + float(cfg.get("max_target1_atr", 2.0)) * a)
        risk = entry - stop
        t2 = np.maximum(t1 + a, entry + 2 * risk)
        if setup == "crash_rebound":   # mean-reversion bounce: close targets (crash study: 1.5 ATR, EMA21)
            t1 = entry + 1.5 * a
            t2 = np.maximum(t1 + 0.5 * a, np.minimum(df["ema21"], entry + 3 * a))
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
    "breakout": "Lower of the breakout-day low and the breakout level (prior 20-day high), minus 0.25 ATR: a close back below the breakout level means the breakout failed",
    "pullback": "Lower of the 5-day low and EMA21, minus 0.25 ATR: losing the pullback low and EMA21 damages the uptrend",
    "squeeze": "5-day low minus 0.25 ATR: breaking the bottom of the squeeze range removes the upside-breakout premise",
    "reclaim": "3-day low minus 0.25 ATR: falling back below the pre-reclaim low means the reclaim failed",
    "momentum": "Higher of EMA21 − 0.5 ATR and the 5-day low − 0.25 ATR: a momentum leader should not lose EMA21",
    "breakdown": "Higher of the breakdown-day high and the breakdown level (prior 20-day low), plus 0.25 ATR: a close back above it means the breakdown failed",
    "rally_fail": "Higher of the 5-day high and EMA21, plus 0.25 ATR: clearing the rally high and EMA21 damages the downtrend",
    "ema21_loss": "3-day high plus 0.25 ATR: regaining the pre-loss high means the EMA21 loss failed",
    "crash_rebound": "Entry − 2 ATR: crash volatility is wide, and the crash study found 2 ATR gives the rebound room to work",
}


def plan_basis(df: pd.DataFrame, setup: str, plan: dict[str, Any], cfg: dict[str, Any]) -> dict[str, str]:
    """Plain-language reason for today's stop and targets (structure, ATR noise floor, resistance)."""
    last = df.iloc[-1]
    long = SETUP_DIRECTION[setup] == "long"
    stop = STOP_BASIS[setup] + ". "
    sdist = plan["structural_stop_atr"]
    if plan.get("wide_stop"):
        stop += f"The structural stop is {sdist:.1f} ATR away, above the cap, so it is tightened to {cfg.get('max_stop_atr', 3)} ATR (easier to get stopped out)."
    elif sdist < float(cfg.get("min_stop_atr", 1.0)):
        stop += f"The structural stop is only {sdist:.1f} ATR away, inside normal daily noise, so it is widened to {cfg.get('min_stop_atr', 1.0)} ATR."
    else:
        stop += f"Distance {plan['risk_atr']:.1f} ATR ({plan['risk_pct'] * 100:.1f}% risk per share)."
    if setup == "crash_rebound":
        return {"stop_basis": stop, "target_basis": "T1 = entry + 1.5 ATR (a quick bounce: in the crash study 1.5–2 ATR targets "
                "had the best out-of-sample results, win rate ~65%); T2 = back to EMA21, capped at 3 ATR. Time stop "
                f"{plan.get('horizon_days', 10)} sessions.", "method": "Crash event study (buy 0–2 days after the first crash in 10 sessions)"}
    t1 = plan["target1"]
    names = {"high20": "20-day high", "high60": "60-day high", "high252": "52-week high"} if long else \
        {"low20": "20-day low", "low60": "60-day low", "low252": "52-week low"}
    src = next((n for col, n in names.items() if pd.notna(last.get(col)) and abs(float(last[col]) - t1) < 1e-6), None)
    target = (f"T1 = the nearest {src} at least 0.75 ATR above (resistance), capped at entry + {cfg.get('max_target1_atr', 2.0)} ATR" if long else f"T1 = the nearest {src} at least 0.75 ATR below (support)") \
        if src else f"T1 = entry {'+' if long else '−'} 2 ATR (no 20 / 60 / 252-day {'resistance above' if long else 'support below'})"
    target += "; T2 = the further of T1 + 1 ATR and 2x risk. Time stop = exit at the close after " + str(plan.get("horizon_days", 7)) + " sessions."
    return {"stop_basis": stop, "target_basis": target,
            "method": "Price structure + ATR noise floor + historical resistance / support (no volume profile or options; the options expected move is context only)"}


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
