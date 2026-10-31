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
    return out
