"""One feature vector per (stock, day): every indicator the terminal computes, in a scale-free form.

Shared by the strategy lab (history) and the live run, so a model trained on the past sees exactly the
same inputs as today's candidates. Uses data up to and including day ``t`` only.

Groups (prefix):
    f_   stationary columns of build_features (oscillators, returns, relative strength, volatility,
         volume ratios, slopes, sub-scores, rule-based pattern flags as 0 / 1)
    d_   price-level distances in ATR: close vs EMA9/21/50/200/250, VWAP20, 20/60/252-day highs and lows,
         Bollinger bands; MACD in ATR; EMA stack alignment flags
    t_   the 49 analyst techniques (behaviour at moving averages, patterns, indicator consensus, volume
         profile, relative performance): signal (-1 / 0 / +1) x strength (0..1)
    v_   volume profile ("chip distribution", 120 sessions): close vs POC / VAH / VAL in ATR, inside the
         value area, nearest dense-chip node above / below, vacuum directly above, holder-profit ratio
    m_   market context passed in by the caller (regime score, cross-sectional RS percentile, ...)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analyst.techniques import run_all

WINDOW = 600            # sessions handed to the analyst techniques (the live deep dive uses the same depth)

LEVELS = {"open", "high", "low", "close", "volume", "ema9", "ema21", "ema50", "ema200", "ema250", "macd", "macd_signal",
          "macd_hist", "atr", "bb_mid", "bb_upper", "bb_lower", "mavol20", "avg_volume_5d", "obv", "vwap20", "high20",
          "low20", "high60", "low60", "prior_high20", "prior_low20", "high252", "low252", "prior_high55", "prior_low55",
          "dollar_volume20", "low3", "low5", "low10", "high3", "high5", "high10", "swing_high_last", "swing_high_prev",
          "swing_low_last", "swing_low_prev", "macd_hist_3d"}


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def base_features(f: pd.DataFrame, t: int) -> dict[str, float]:
    """f_ and d_ groups (cheap: one row of build_features)."""
    row = f.iloc[t]
    out: dict[str, float] = {}
    for col in f.columns:
        if col in LEVELS:
            continue
        val = row[col]
        if isinstance(val, (bool, np.bool_)):
            out[f"f_{col}"] = float(val)
        elif isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
            out[f"f_{col}"] = _num(val)
    c, a = _num(row["close"]), _num(row["atr"])
    if not (np.isfinite(c) and np.isfinite(a) and a > 0):
        return out
    for col in ("ema9", "ema21", "ema50", "ema200", "ema250", "vwap20", "bb_upper", "bb_lower", "high20", "low20",
                "high60", "low60", "high252", "low252"):
        if col in f.columns:
            out[f"d_{col}"] = (c - _num(row[col])) / a
    for col in ("macd", "macd_signal", "macd_hist", "macd_hist_3d"):
        if col in f.columns:
            out[f"d_{col}"] = _num(row[col]) / a
    e9, e21, e50, e200 = (_num(row.get(k)) for k in ("ema9", "ema21", "ema50", "ema200"))
    out["d_stack_bull"] = float(c > e9 > e21 > e50 > e200)
    out["d_stack_bear"] = float(c < e9 < e21 < e50 < e200)
    out["d_above_ema200"] = float(c > e200)
    out["d_ema50_above_200"] = float(e50 > e200)
    return out


def technique_features(f: pd.DataFrame, t: int, spy: pd.Series | None) -> dict[str, float]:
    """t_ and v_ groups (the 49 analyst techniques + volume profile on the last WINDOW sessions)."""
    window = f.iloc[max(0, t - WINDOW + 1): t + 1]
    spy_w = spy.loc[: window.index[-1]] if spy is not None else None
    try:
        res = run_all(window, spy_w)
    except Exception:  # noqa: BLE001 - a technique failing on odd data must not stop the study
        return {}
    out = {f"t_{x['id']}": float(x["signal"]) * float(x["strength"]) for x in res.get("findings", [])}
    vp = res.get("volume_profile") or {}
    c, a = float(window["close"].iat[-1]), float(window["atr"].iat[-1]) if "atr" in window else float("nan")
    if vp.get("available") and np.isfinite(a) and a > 0:
        poc, vah, val = vp["poc"], vp["vah"], vp["val"]
        out["v_poc"] = (c - poc) / a
        out["v_vah"] = (c - vah) / a
        out["v_val"] = (c - val) / a
        out["v_inside_va"] = float(val <= c <= vah)
        above = [x for x in vp.get("hvn", []) if x > c]
        below = [x for x in vp.get("hvn", []) if x < c]
        out["v_hvn_above"] = (min(above) - c) / a if above else 5.0
        out["v_hvn_below"] = (c - max(below)) / a if below else 5.0
        out["v_vacuum_above"] = float(any(z["low"] >= c and z["low"] - c <= 1.5 * a for z in vp.get("vacuum", [])))
        out["v_profit_ratio"] = _num(vp.get("profit_ratio"))
    return out


def feature_vector(f: pd.DataFrame, t: int, spy: pd.Series | None, context: dict[str, float] | None = None,
                   techniques: bool = True) -> dict[str, float]:
    out = base_features(f, t)
    if techniques:
        out.update(technique_features(f, t, spy))
    for k, v in (context or {}).items():
        out[f"m_{k}"] = _num(v)
    return out


GROUP_NAMES = {"f": "Indicators", "d": "Moving averages / levels", "t": "Analyst techniques", "v": "Volume profile (chips)",
               "m": "Market context", "s": "Setup type"}
