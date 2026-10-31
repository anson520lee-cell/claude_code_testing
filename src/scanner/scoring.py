"""Transparent composite score for swing candidates.

composite = sum_i(weight_i x subscore_i) / 100  -  sum(penalties)      (clipped to 0 ... 100)

Weights and penalties live in config/terminal.yaml. Every sub-score is 0-100 and is
computed for the candidate's DIRECTION (a short candidate scores high on "trend" when
its trend is down). Sub-scores:

trend (long)       C > EMA21 +20 · EMA21 > EMA50 +20 · EMA50 > EMA200 +15 · EMA21 rising +15 ·
                   EMA50 rising +10 · swing structure HH/HL +20 (Mixed +10).        Short: mirror.
momentum (long)    RSI14 in 50-70: 40, 45-50 or 70-78: 25, otherwise 10 (> 78: 5) · MACD histogram > 0 +20 ·
                   histogram rising over 3 sessions +20 · ROC10 > 0 +20.           Short: mirror (RSI 30-50 best).
relative_strength  70% x universe percentile of RS = 0.5 x rel20 + 0.3 x rel60 + 0.2 x rel5 (vs SPY)
                   + 30% x universe percentile of the 20d return vs the sector ETF. Short: 100 - that.
volume             40% accumulation (up/down volume ratio 20d: 0.8 -> 0, 1.6 -> 100) + 30% OBV slope20 sign
                   + 30% setup-specific: breakout / reclaim / breakdown / loss want volume_ratio high
                   (1 -> 0, 2.5 -> 100); pullback / rally_fail / squeeze want quiet volume (1.1 -> 0, 0.6 -> 100).
setup_quality      the setup's own 0-100 quality (see setups.py).
reward_risk        reward/risk to Target 1: 1.0 -> 20, 1.5 -> 50, 2.0 -> 70, >= 3.0 -> 100 (linear between).
liquidity          log-scale 20d dollar volume between the minimum (0) and "good" (100) thresholds.
regime_alignment   60% regime (long: Strong Bullish 100, Bullish 80, Neutral 50, Bearish 25, Strong Bearish 10;
                   short reversed) + 40% sector rotation quadrant of the stock's ETF
                   (long: Leading 100, Improving 70, Weakening 40, Lagging 15; short reversed).

Penalties (subtracted, configurable): earnings inside the plan window · extended (> N ATR from EMA21
in the trade direction) · RSI6 > 85 (long) / < 15 (short) · dilution filing in last 30 days (long only) ·
financial distress flags (long only) · reward/risk below minimum · wide structural stop ·
data-quality warning · options pricing a large event premium (IV/HV20 > 1.6).

Tiers: A >= 75, B >= 62, C >= 50, otherwise "Watch". These rank RESEARCH priority, not trade advice.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.scanner.setups import SETUP_DIRECTION, SETUP_NAMES

REGIME_LONG = {"Strong Bullish": 100, "Bullish": 80, "Neutral": 50, "Bearish": 25, "Strong Bearish": 10}
REGIME_SHORT = {"Strong Bullish": 10, "Bullish": 25, "Neutral": 50, "Bearish": 80, "Strong Bearish": 100}
QUADRANT_LONG = {"Leading": 100, "Improving": 70, "Weakening": 40, "Lagging": 15}
QUADRANT_SHORT = {"Leading": 15, "Improving": 40, "Weakening": 70, "Lagging": 100}
HIGH_VOLUME_SETUPS = ("breakout", "reclaim", "breakdown", "ema21_loss", "momentum")


def _f(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _lin(x: float, x0: float, x1: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip((x - x0) / (x1 - x0), 0, 1) * 100)


def cross_sectional_ranks(latest: pd.DataFrame) -> pd.DataFrame:
    """Universe percentiles (0-100) used by the relative-strength sub-score."""
    rs = 0.5 * latest["rel_20d"] + 0.3 * latest["rel_60d"] + 0.2 * latest["rel_5d"]
    return pd.DataFrame({
        "rs_raw": rs,
        "rs_pct": rs.rank(pct=True) * 100,
        "sector_rs_pct": latest["rel_sector_20d"].rank(pct=True) * 100,
    }, index=latest.index)


def trend_score(row: pd.Series, long: bool) -> float:
    s = 1 if long else -1
    c, e21, e50, e200 = (_f(row.get(k)) for k in ("close", "ema21", "ema50", "ema200"))
    pts = 0.0
    pts += 20 * (s * (c - e21) > 0)
    pts += 20 * (s * (e21 - e50) > 0)
    pts += 15 * (np.isfinite(e200) and s * (e50 - e200) > 0)
    pts += 15 * (s * _f(row.get("ema21_slope")) > 0)
    pts += 10 * (s * _f(row.get("ema50_slope")) > 0)
    structure = row.get("swing_structure")
    pts += 20 if structure == ("HH/HL" if long else "LH/LL") else 10 if structure == "Mixed" else 0
    return float(pts)


def momentum_score(row: pd.Series, long: bool) -> float:
    rsi = _f(row.get("rsi14"))
    r = rsi if long else 100 - rsi
    pts = 0.0
    if np.isfinite(r):
        pts += 40 if 50 <= r <= 70 else 25 if (45 <= r < 50 or 70 < r <= 78) else 5 if r > 78 else 10
    s = 1 if long else -1
    pts += 20 * (s * _f(row.get("macd_hist")) > 0)
    pts += 20 * (s * _f(row.get("macd_hist_3d")) > 0)
    pts += 20 * (s * _f(row.get("roc10")) > 0)
    return float(min(pts, 100))


def volume_score(row: pd.Series, setup: str, long: bool) -> float:
    ud = _f(row.get("updown_vol_ratio"))
    accumulation = _lin(ud, 0.8, 1.6) if long else _lin(-ud, -1.25, -0.6)  # short: ratio 1.25 -> 0, 0.6 -> 100
    obv = 100.0 if (1 if long else -1) * _f(row.get("obv_slope20")) > 0 else 0.0
    if setup in HIGH_VOLUME_SETUPS:
        today = _lin(_f(row.get("volume_ratio")), 1.0, 2.5)
    else:
        today = _lin(-_f(row.get("vol3_vs_mavol")), -1.1, -0.6)
    return 0.4 * accumulation + 0.3 * obv + 0.3 * today


def reward_risk_score(rr: float) -> float:
    if not np.isfinite(rr):
        return 0.0
    points = [(1.0, 20), (1.5, 50), (2.0, 70), (3.0, 100)]
    if rr <= 1.0:
        return max(0.0, 20 * rr)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if rr <= x1:
            return y0 + (y1 - y0) * (rr - x0) / (x1 - x0)
    return 100.0


def liquidity_score(dollar_volume: float, minimum: float, good: float) -> float:
    if not np.isfinite(dollar_volume) or dollar_volume <= 0:
        return 0.0
    return float(np.clip((math.log10(dollar_volume) - math.log10(minimum))
                         / (math.log10(good) - math.log10(minimum)), 0, 1) * 100)


def build_candidate(symbol: str, row: pd.Series, setup: str, quality: float, plan: dict[str, Any],
                    ranks: pd.Series, context: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Score one (symbol, setup) pair. ``context``: regime label, sector quadrant, sector ETF, quality status."""
    long = SETUP_DIRECTION[setup] == "long"
    liq = settings["liquidity"]
    rs = 0.7 * _f(ranks.get("rs_pct")) + 0.3 * np.nan_to_num(_f(ranks.get("sector_rs_pct")), nan=50.0)
    rs = rs if long else 100 - rs
    regime_map, quad_map = (REGIME_LONG, QUADRANT_LONG) if long else (REGIME_SHORT, QUADRANT_SHORT)
    components = {
        "trend": trend_score(row, long),
        "momentum": momentum_score(row, long),
        "relative_strength": float(np.nan_to_num(rs, nan=50.0)),
        "volume": volume_score(row, setup, long),
        "setup_quality": float(quality),
        "reward_risk": reward_risk_score(_f(plan.get("reward_risk"))),
        "liquidity": liquidity_score(_f(row.get("dollar_volume20")), liq["min_avg_dollar_volume"],
                                     liq["good_avg_dollar_volume"]),
        "regime_alignment": 0.6 * regime_map.get(context.get("regime"), 50)
                            + 0.4 * quad_map.get(context.get("sector_quadrant"), 50),
    }
    candidate = {
        "symbol": symbol, "setup": setup, "setup_name": SETUP_NAMES[setup], "direction": "long" if long else "short",
        "setup_quality": float(quality), "components": components, "plan": plan,
        "sector_etf": context.get("sector_etf"), "sector_quadrant": context.get("sector_quadrant"),
        "data_quality": context.get("data_quality", "ok"),
        "technical": technical_snapshot(row, ranks),
        "penalties": {}, "supporting": [], "contradicting": [],
    }
    technical_penalties(candidate, row, settings)
    finalize(candidate, settings)
    return candidate


def technical_snapshot(row: pd.Series, ranks: pd.Series) -> dict[str, Any]:
    keys = ["close", "return_1d", "return_5d", "return_20d", "return_60d", "ema9", "ema21", "ema50", "ema200",
            "ema250", "rsi6", "rsi14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_mid", "bb_lower",
            "bb_pct_b", "bb_width_percentile", "atr", "atr_pct", "hv10", "hv20", "hv60", "volume", "mavol20",
            "volume_ratio", "obv_slope20", "updown_vol_ratio", "roc10", "roc20", "rel_5d", "rel_20d", "rel_60d",
            "rel_sector_20d", "gap", "gap_atr", "high252", "low252", "from_52w_high", "from_52w_low",
            "extension_atr", "adx", "vwap20", "dollar_volume20", "clv", "high20", "low20"]
    out = {k: (None if pd.isna(row.get(k)) else float(row.get(k))) for k in keys}
    for k in ("trend", "momentum", "swing_structure", "structure", "swing_condition", "relative_strength",
              "volatility", "band_state", "macd_state", "rsi_zone"):
        out[k] = None if row.get(k) is None else str(row.get(k))
    out["swing_score"] = None if pd.isna(row.get("swing_score")) else float(row.get("swing_score"))
    out["rs_percentile"] = None if pd.isna(ranks.get("rs_pct")) else float(ranks.get("rs_pct"))
    return out


def technical_penalties(c: dict[str, Any], row: pd.Series, settings: dict[str, Any]) -> None:
    pen, sc = settings["penalties"], settings["scanner"]
    long = c["direction"] == "long"
    plan, t = c["plan"], c["technical"]
    ext = _f(row.get("extension_atr"))
    if np.isfinite(ext) and (ext if long else -ext) > sc["extension_atr"]:
        c["penalties"]["extended"] = pen["extended"]
        c["contradicting"].append(f"Extended: close is {abs(ext):.1f} ATR {'above' if long else 'below'} EMA21 "
                                  f"(> {sc['extension_atr']} ATR) - chasing risk, mean-reversion likely.")
    rsi6 = _f(row.get("rsi6"))
    if np.isfinite(rsi6) and ((long and rsi6 > 85) or (not long and rsi6 < 15)):
        c["penalties"]["overbought_extreme"] = pen["overbought_extreme"]
        c["contradicting"].append(f"RSI6 {rsi6:.0f} is at an extreme - short-term exhaustion risk.")
    rr = _f(plan.get("reward_risk"))
    if np.isfinite(rr) and rr < sc["min_reward_risk"]:
        c["penalties"]["poor_reward_risk"] = pen["poor_reward_risk"]
        c["contradicting"].append(f"Reward/risk to Target 1 is only {rr:.2f} (< {sc['min_reward_risk']}).")
    if plan.get("wide_stop"):
        c["penalties"]["wide_stop"] = pen["wide_stop"]
        c["contradicting"].append(f"Structural stop is {plan['structural_stop_atr']:.1f} ATR away; the plan caps it "
                                  f"at {sc['max_stop_atr']} ATR, inside the structure (higher stop-out risk).")
    if c["data_quality"] == "warning":
        c["penalties"]["data_quality"] = pen["data_quality"]
        c["contradicting"].append("Data-quality warning on this symbol (see Data Quality).")

    # evidence lists (both directions)
    s = 1 if long else -1
    side = "above" if long else "below"
    if t["rs_percentile"] is not None:
        pct = t["rs_percentile"] if long else 100 - t["rs_percentile"]
        (c["supporting"] if pct >= 70 else c["contradicting"] if pct < 40 else []).append(
            f"Relative strength vs SPY in the {t['rs_percentile']:.0f}th universe percentile.")
    if t["ema200"] is not None and t["close"] is not None:
        if s * (t["close"] - t["ema200"]) > 0:
            c["supporting"].append(f"Close {side} EMA200 (long-term trend agrees).")
        else:
            c["contradicting"].append(f"Close is {'below' if long else 'above'} EMA200 - trading against the "
                                      "long-term trend.")
    if t["swing_structure"] == ("HH/HL" if long else "LH/LL"):
        c["supporting"].append(f"Swing structure {t['swing_structure']} (trend structure confirmed).")
    elif t["swing_structure"] == ("LH/LL" if long else "HH/HL"):
        c["contradicting"].append(f"Swing structure is {t['swing_structure']} - opposite to the trade direction.")
    if t["macd_hist"] is not None and _f(row.get("macd_hist_3d")) * s < 0:
        c["contradicting"].append("MACD histogram has weakened over the last 3 sessions (momentum fading).")
    vr = t["volume_ratio"]
    if c["setup"] in ("breakout", "breakdown") and vr is not None and vr >= 2:
        c["supporting"].append(f"Volume {vr:.1f}x MAVOL20 on the {'breakout' if long else 'breakdown'} "
                               "(institution-sized participation likely).")
    if c["setup"] in ("pullback", "rally_fail") and _f(row.get("vol3_vs_mavol")) < 0.8:
        c["supporting"].append("Quiet volume during the counter-move (no heavy distribution / accumulation).")
    gap = _f(row.get("gap_atr"))
    if np.isfinite(gap) and s * gap > 2:
        c["contradicting"].append(f"Gapped {gap:+.1f} ATR today - gap-fill risk.")
    ud = _f(row.get("updown_vol_ratio"))
    if np.isfinite(ud):
        if (long and ud >= 1.3) or (not long and ud <= 0.75):
            c["supporting"].append(f"Up/down volume ratio (20d) {ud:.2f} - {'accumulation' if long else 'distribution'}.")
        elif (long and ud < 0.8) or (not long and ud > 1.25):
            c["contradicting"].append(f"Up/down volume ratio (20d) {ud:.2f} works against the trade.")
    if t["volatility"] in ("High", "Extreme"):
        c["contradicting"].append(f"ATR in the {t['volatility'].lower()} part of its 1-year range - wider swings, "
                                  "smaller size needed.")


def finalize(c: dict[str, Any], settings: dict[str, Any]) -> None:
    weights = settings["weights"]
    raw = sum(weights[k] * c["components"][k] for k in weights) / 100
    penalty = sum(c["penalties"].values())
    c["raw_score"] = round(float(raw), 2)
    c["penalty_total"] = float(penalty)
    c["score"] = round(float(np.clip(raw - penalty, 0, 100)), 2)
    tiers = settings["tiers"]
    c["tier"] = "A" if c["score"] >= tiers["A"] else "B" if c["score"] >= tiers["B"] else \
        "C" if c["score"] >= tiers["C"] else "Watch"
    c["contribution"] = {k: round(weights[k] * c["components"][k] / 100, 2) for k in weights}


def apply_enrichment(c: dict[str, Any], enrichment: dict[str, Any], settings: dict[str, Any]) -> None:
    """Deep-dive penalties / evidence from catalysts, fundamentals, options and positioning."""
    pen, sc = settings["penalties"], settings["scanner"]
    long = c["direction"] == "long"
    horizon = int(sc["event_window_days"])
    cat = enrichment.get("catalysts") or {}
    earn = cat.get("next_earnings")
    if earn and earn.get("sessions_until") is not None and earn["sessions_until"] <= horizon:
        c["penalties"]["earnings_in_window"] = pen["earnings_in_window"]
        c["contradicting"].append(f"Earnings {earn['date']} ({earn.get('timing') or 'time n/a'}) in "
                                  f"{earn['sessions_until']} session(s) - binary gap risk inside the holding window.")
    dilution = [f for f in cat.get("filings", []) if f.get("dilution_risk") and f.get("age_days", 99) <= 30]
    if dilution:
        forms = ", ".join(sorted({f["form"] for f in dilution}))
        if long:
            c["penalties"]["dilution_filing"] = pen["dilution_filing"]
            c["contradicting"].append(f"Recent dilution-related SEC filing(s): {forms} (offering / shelf).")
        else:
            c["supporting"].append(f"Recent dilution-related SEC filing(s): {forms} - supply overhang.")
    for item in cat.get("analyst", [])[:3]:
        if item.get("age_days", 99) <= 10:
            positive = item["sentiment"] == "Positive"
            target = c["supporting"] if positive == long and item["sentiment"] != "Uncertain" else \
                c["contradicting"] if item["sentiment"] != "Uncertain" else None
            if target is not None:
                target.append(f"Analyst {item['description']} ({item['date']}).")
    fund = enrichment.get("fundamentals") or {}
    flags = fund.get("flags") or []
    severe = [f for f in flags if f.get("severity") == "high"]
    if severe and long:
        c["penalties"]["financial_distress"] = pen["financial_distress"]
    for f in flags:
        bucket = c["contradicting"] if (f.get("kind") == "risk") == long else c["supporting"]
        bucket.append(f"Fundamentals: {f['text']}")
    opt = enrichment.get("options") or {}
    if opt.get("available"):
        ratio = opt.get("iv_hv_ratio")
        if ratio is not None and ratio > 1.6:
            c["penalties"]["options_event_premium"] = pen["options_event_premium"]
            c["contradicting"].append(f"Options price elevated risk: ATM IV is {ratio:.1f}x 20-day realised "
                                      "volatility (event premium / expected large move).")
        em = opt.get("expected_move_pct")
        if em is not None and c["plan"].get("risk_pct"):
            if em > 2.5 * c["plan"]["risk_pct"]:
                c["contradicting"].append(f"Options-implied move to {opt.get('expected_move_expiry')} is "
                                          f"+/-{em * 100:.1f}%, > 2.5x the planned stop distance.")
    pos = enrichment.get("positioning") or {}
    si = pos.get("short_percent_float")
    if si is not None and si >= 0.15:
        (c["supporting"] if long else c["contradicting"]).append(
            f"Short interest {si * 100:.0f}% of float (as of {pos.get('short_interest_date')}) - squeeze "
            f"{'potential' if long else 'risk'}.")
    insider = pos.get("insider_90d") or {}
    if insider.get("net_value") is not None and insider.get("n_buys", 0) >= 2 and insider["net_value"] > 0:
        (c["supporting"] if long else c["contradicting"]).append("Insider open-market buying in the last 90 days.")
    finalize(c, settings)


def stats_note(stats: dict[str, Any] | None) -> str:
    if not stats or not stats.get("n"):
        return "No historical occurrences."
    note = (f"N={stats['n']}, win {stats['win_rate'] * 100:.0f}%, avg {stats['avg_r']:+.2f}R, "
            f"target hit {stats['hit_target_rate'] * 100:.0f}%")
    return note + (" (small sample)" if stats.get("small_sample") else "")
