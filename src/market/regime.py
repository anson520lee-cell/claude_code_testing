"""Market regime classification - transparent, rule-based, computed for every date.

The regime is a weighted sum of six component scores. Each component is scored on a
-2 ... +2 scale from explicit formulas (below); the composite is

    composite = sum_i( weight_i * score_i / 2 )        -> range -100 ... +100

and is classified as

    >= +45 Strong Bullish | >= +15 Bullish | > -15 Neutral | > -45 Bearish | else Strong Bearish

with one consistency guard: "Strong Bullish" requires SPY above its EMA50 and
"Strong Bearish" requires SPY below it (otherwise the label is softened one step).

Components (weights in ``COMPONENT_WEIGHTS``):

1. Index trend (25). Per index: 0.5 x [sign(C - EMA21) + sign(C - EMA50) + sign(EMA50 - EMA200)
   + sign(EMA21 3-day slope)]. Weighted SPY 40%, QQQ 30%, IWM 15%, DIA 15%.
2. Index momentum (15). 20-day return of SPY and QQQ divided by its typical size
   (daily std over 60 days x sqrt(20)), clipped to +/-2, averaged.
3. Breadth (20). Within the screened universe (NOT the whole market - a proxy):
   0.5 x clip((%>EMA50 - 50) / 15) + 0.3 x clip((%>EMA200 - 50) / 15)
   + 0.2 x clip(net % at 20-day highs minus lows / 10).
4. Volatility (20). VIX level (<15 +2, <18 +1, <22 0, <28 -1, else -2); VIX/VIX3M > 1
   (inverted term structure = near-term stress) -1, < 0.90 +0.5; VIX up > 25% in 5 days -0.5.
5. Risk appetite (15). 0.4 x clip(HYG-IEF 20d return / 1%) (credit) + 0.4 x clip(offensive minus
   defensive sectors 20d return / 2%) + 0.2 x clip(IWM-SPY 20d return / 2%).
6. Rates (5). 10-year Treasury yield change over 20 sessions: > +30bp -2, > +15bp -1,
   < -15bp +1, else 0. (Falling yields are scored mildly positive; in a growth scare they are
   not - the evidence text says so.)

Missing inputs make a component "n/a"; the composite then re-weights the available ones
and the evidence lists what was missing. Nothing here is a forecast.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.technical.indicators import ema

COMPONENT_WEIGHTS = {"trend": 25, "momentum": 15, "breadth": 20, "volatility": 20, "risk_appetite": 15, "rates": 5}
COMPONENT_NAMES = {"trend": "Index trend", "momentum": "Index momentum", "breadth": "Breadth (universe)",
                   "volatility": "Volatility (VIX)", "risk_appetite": "Risk appetite", "rates": "Rates"}
LABELS = ["Strong Bearish", "Bearish", "Neutral", "Bullish", "Strong Bullish"]
INDEX_WEIGHTS = {"SPY": 0.40, "QQQ": 0.30, "IWM": 0.15, "DIA": 0.15}

GUIDANCE = {
    "Strong Bullish": ("Environment supports long setups; breakouts have the best odds of follow-through. "
                       "Short setups fight the tape - only on clear individual breakdowns.", 1.0, 0.25),
    "Bullish": ("Favours long setups, especially pullbacks in leading groups. Be selective with shorts.", 0.8, 0.4),
    "Neutral": ("Mixed tape: reduce size, demand higher-quality setups, prefer relative-strength leaders "
                "for longs and relative-weakness laggards for shorts; take profits faster.", 0.5, 0.5),
    "Bearish": ("Favours short setups and cash; long breakouts fail more often. Longs only in clear "
                "relative-strength leaders with tight risk.", 0.3, 0.8),
    "Strong Bearish": ("Risk-off: capital preservation first. Long setups have poor odds; short setups in "
                       "weak groups align with the tape, but volatility (and squeeze risk) is high.", 0.1, 1.0),
}


def _clip(x: pd.Series | float, lo: float = -2.0, hi: float = 2.0):
    return np.clip(x, lo, hi)


def _close(prices: dict[str, pd.DataFrame], symbol: str, index: pd.DatetimeIndex) -> pd.Series | None:
    frame = prices.get(symbol)
    if frame is None or frame.empty:
        return None
    column = "adj_close" if frame["adj_close"].notna().all() else "close"
    return frame[column].astype(float).reindex(index).ffill()


def _ret(series: pd.Series, n: int) -> pd.Series:
    return series / series.shift(n) - 1


def index_trend_score(close: pd.Series) -> pd.Series:
    e21, e50, e200 = ema(close, 21), ema(close, 50), ema(close, 200)
    parts = [np.sign(close - e21), np.sign(close - e50), np.sign(e50 - e200), np.sign(e21 / e21.shift(3) - 1)]
    return 0.5 * sum(parts)


def compute_regime_history(prices: dict[str, pd.DataFrame], universe: list[str], settings: dict[str, Any],
                           yields: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per trading date with every component score, the composite and the label."""
    spy = prices.get("SPY")
    if spy is None or spy.empty:
        raise ValueError("SPY prices are required for the market regime.")
    index = spy.index
    market = settings["market"]
    out = pd.DataFrame(index=index)

    # 1. trend
    parts, weights = [], []
    for symbol, weight in INDEX_WEIGHTS.items():
        close = _close(prices, symbol, index)
        if close is not None:
            out[f"trend_{symbol}"] = index_trend_score(close)
            parts.append(out[f"trend_{symbol}"] * weight)
            weights.append(weight)
    out["trend"] = sum(parts) / sum(weights) if parts else np.nan

    # 2. momentum
    moms = []
    for symbol in ("SPY", "QQQ"):
        close = _close(prices, symbol, index)
        if close is not None:
            sigma = close.pct_change().rolling(60, min_periods=40).std()
            out[f"ret20_{symbol}"] = _ret(close, 20)
            moms.append(_clip(out[f"ret20_{symbol}"] / (sigma * np.sqrt(20))))
    out["momentum"] = sum(moms) / len(moms) if moms else np.nan

    # 3. breadth (universe)
    above50, above200, highs, lows, valid = [], [], [], [], []
    for symbol in universe:
        frame = prices.get(symbol)
        if frame is None or len(frame) < 60:
            continue
        close = frame["close"].astype(float).reindex(index)
        e50, e200 = ema(close.ffill(), 50), ema(close.ffill(), 200)
        ok = close.notna()
        above50.append(((close > e50) & e50.notna()).where(ok))
        above200.append(((close > e200) & e200.notna()).where(ok & e200.notna()))
        hi20 = frame["high"].reindex(index).rolling(20, min_periods=20).max()
        lo20 = frame["low"].reindex(index).rolling(20, min_periods=20).min()
        highs.append((frame["high"].reindex(index) >= hi20).where(hi20.notna()))
        lows.append((frame["low"].reindex(index) <= lo20).where(lo20.notna()))
        valid.append(ok)
    if above50:
        def pct(items: list[pd.Series]) -> pd.Series:
            frame = pd.concat(items, axis=1).astype(float)
            return 100 * frame.mean(axis=1, skipna=True)
        out["pct_above_ema50"] = pct(above50)
        out["pct_above_ema200"] = pct(above200)
        out["pct_new_high20"] = pct(highs)
        out["pct_new_low20"] = pct(lows)
        out["breadth_n"] = pd.concat(valid, axis=1).sum(axis=1)
        net = out["pct_new_high20"] - out["pct_new_low20"]
        out["breadth"] = (0.5 * _clip((out["pct_above_ema50"] - 50) / 15)
                          + 0.3 * _clip((out["pct_above_ema200"] - 50) / 15) + 0.2 * _clip(net / 10))
    else:
        out["breadth"] = np.nan

    # 4. volatility
    vix = _close(prices, "^VIX", index)
    vix3m = _close(prices, "^VIX3M", index)
    if vix is not None:
        out["vix"] = vix
        level = pd.Series(np.select([vix < 15, vix < 18, vix < 22, vix < 28], [2, 1, 0, -1], -2), index=index)
        level = level.where(vix.notna())
        adj = pd.Series(0.0, index=index)
        if vix3m is not None:
            out["vix_term"] = vix / vix3m
            adj = adj + np.where(out["vix_term"] > 1.0, -1.0, np.where(out["vix_term"] < 0.90, 0.5, 0.0))
        out["vix_chg5"] = _ret(vix, 5)
        adj = adj + np.where(out["vix_chg5"] > 0.25, -0.5, 0.0)
        out["volatility"] = _clip(level + adj)
    else:
        out["volatility"] = np.nan

    # 5. risk appetite
    pieces, piece_w = [], []
    hyg, ief = _close(prices, "HYG", index), _close(prices, "IEF", index)
    if hyg is not None and ief is not None:
        out["credit_20d"] = _ret(hyg, 20) - _ret(ief, 20)
        pieces.append(0.4 * _clip(out["credit_20d"] / 0.01))
        piece_w.append(0.4)
    off = [c for c in (_close(prices, s, index) for s in market.get("offensive_sectors", [])) if c is not None]
    dfn = [c for c in (_close(prices, s, index) for s in market.get("defensive_sectors", [])) if c is not None]
    if off and dfn:
        out["off_def_20d"] = sum(_ret(c, 20) for c in off) / len(off) - sum(_ret(c, 20) for c in dfn) / len(dfn)
        pieces.append(0.4 * _clip(out["off_def_20d"] / 0.02))
        piece_w.append(0.4)
    iwm, spy_c = _close(prices, "IWM", index), _close(prices, "SPY", index)
    if iwm is not None:
        out["small_vs_large_20d"] = _ret(iwm, 20) - _ret(spy_c, 20)
        pieces.append(0.2 * _clip(out["small_vs_large_20d"] / 0.02))
        piece_w.append(0.2)
    out["risk_appetite"] = sum(pieces) / sum(piece_w) if pieces else np.nan

    # 6. rates
    if yields is not None and not yields.empty and "DGS10" in yields:
        y10 = yields["DGS10"].reindex(index.union(yields.index)).ffill().reindex(index)
        out["us10y"] = y10
        if "DGS2" in yields:
            out["us2y"] = yields["DGS2"].reindex(index.union(yields.index)).ffill().reindex(index)
            out["curve_2s10s"] = out["us10y"] - out["us2y"]
        change = (y10 - y10.shift(20)) * 100
        out["us10y_chg20_bp"] = change
        out["rates"] = pd.Series(np.select([change > 30, change > 15, change < -15], [-2, -1, 1], 0),
                                 index=index).where(change.notna())
    else:
        out["rates"] = np.nan

    # composite (re-weights available components)
    scores = out[list(COMPONENT_WEIGHTS)]
    w = pd.Series(COMPONENT_WEIGHTS, dtype=float)
    available = scores.notna()
    weighted = (scores.fillna(0) * w).sum(axis=1)
    total_w = (available * w).sum(axis=1)
    out["composite"] = (weighted / 2 * 100 / total_w).where(total_w >= 50)
    out["spy_above_ema50"] = spy_c > ema(spy_c, 50)
    out["label"] = [_label(c, a) for c, a in zip(out["composite"], out["spy_above_ema50"])]
    return out


def _label(composite: float, spy_above_ema50: bool) -> str:
    if not np.isfinite(composite):
        return "n/a"
    if composite >= 45:
        i = 4
    elif composite >= 15:
        i = 3
    elif composite > -15:
        i = 2
    elif composite > -45:
        i = 1
    else:
        i = 0
    if i == 4 and not spy_above_ema50:
        i = 3
    if i == 0 and spy_above_ema50:
        i = 1
    return LABELS[i]


def _fmt_pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def evidence(row: pd.Series) -> dict[str, list[str]]:
    """Plain-language evidence for each component on one date."""
    ev: dict[str, list[str]] = {k: [] for k in COMPONENT_WEIGHTS}
    for symbol in INDEX_WEIGHTS:
        value = row.get(f"trend_{symbol}")
        if pd.notna(value):
            word = "uptrend" if value >= 1 else "downtrend" if value <= -1 else "mixed trend"
            ev["trend"].append(f"{symbol} trend score {value:+.1f} of +/-2 ({word}).")
    for symbol in ("SPY", "QQQ"):
        if pd.notna(row.get(f"ret20_{symbol}")):
            ev["momentum"].append(f"{symbol} 20-day return {_fmt_pct(row[f'ret20_{symbol}'])}.")
    if pd.notna(row.get("pct_above_ema50")):
        ev["breadth"].append(f"{row['pct_above_ema50']:.0f}% of {int(row.get('breadth_n', 0))} universe stocks above "
                             f"EMA50, {row['pct_above_ema200']:.0f}% above EMA200.")
        ev["breadth"].append(f"{row['pct_new_high20']:.0f}% at 20-day highs vs {row['pct_new_low20']:.0f}% at "
                             "20-day lows. (Universe proxy, not NYSE-wide breadth.)")
    if pd.notna(row.get("vix")):
        text = f"VIX {row['vix']:.1f}"
        if pd.notna(row.get("vix_term")):
            shape = "inverted (stress)" if row["vix_term"] > 1 else "normal contango"
            text += f"; VIX/VIX3M {row['vix_term']:.2f} - term structure {shape}"
        if pd.notna(row.get("vix_chg5")):
            text += f"; 5-day change {round(row['vix_chg5'] * 100) + 0:+d}%"
        ev["volatility"].append(text + ".")
    if pd.notna(row.get("credit_20d")):
        ev["risk_appetite"].append(f"High yield vs Treasuries (HYG-IEF) 20d: {_fmt_pct(row['credit_20d'])}.")
    if pd.notna(row.get("off_def_20d")):
        ev["risk_appetite"].append(f"Offensive minus defensive sectors 20d: {_fmt_pct(row['off_def_20d'])}.")
    if pd.notna(row.get("small_vs_large_20d")):
        ev["risk_appetite"].append(f"Small caps vs S&P 500 (IWM-SPY) 20d: {_fmt_pct(row['small_vs_large_20d'])}.")
    if pd.notna(row.get("us10y")):
        text = f"10-year yield {row['us10y']:.2f}%"
        if pd.notna(row.get("us10y_chg20_bp")):
            text += f" ({row['us10y_chg20_bp']:+.0f}bp over 20 sessions)"
        if pd.notna(row.get("curve_2s10s")):
            text += f"; 2s10s curve {row['curve_2s10s'] * 100:+.0f}bp"
        ev["rates"].append(text + " (FRED H.15, published with ~1 day lag).")
    for key in COMPONENT_WEIGHTS:
        if pd.isna(row.get(key)):
            ev[key] = ["Input data unavailable - component excluded and the composite re-weighted."]
    return ev


def sector_rotation(prices: dict[str, pd.DataFrame], etfs: dict[str, str], session_index: pd.DatetimeIndex) -> list[dict]:
    """Relative-rotation table for sector / industry ETFs vs SPY.

    RS line = ETF / SPY.  RS-trend = RS / SMA50(RS) - 1;  RS-momentum = RS-trend - RS-trend 10 sessions ago.
    Quadrant: Leading (trend > 0, momentum > 0), Weakening (> 0, < 0), Lagging (< 0, < 0), Improving (< 0, > 0).
    (A transparent analogue of a relative rotation graph; not the proprietary JdK RS-Ratio.)
    """
    spy = _close(prices, "SPY", session_index)
    rows = []
    for symbol, name in etfs.items():
        close = _close(prices, symbol, session_index)
        if close is None or spy is None or close.dropna().empty:
            continue
        rs = close / spy
        rs_trend = rs / rs.rolling(50, min_periods=50).mean() - 1
        rs_mom = rs_trend - rs_trend.shift(10)
        t, m = rs_trend.iloc[-1], rs_mom.iloc[-1]
        quadrant = ("n/a" if not (np.isfinite(t) and np.isfinite(m)) else
                    "Leading" if t > 0 and m > 0 else "Weakening" if t > 0 else
                    "Lagging" if m < 0 else "Improving")
        e21, e50 = ema(close, 21), ema(close, 50)
        rows.append({
            "symbol": symbol, "name": name,
            "close": float(close.iloc[-1]),
            "ret_1d": float(_ret(close, 1).iloc[-1]), "ret_5d": float(_ret(close, 5).iloc[-1]),
            "ret_20d": float(_ret(close, 20).iloc[-1]), "ret_60d": float(_ret(close, 60).iloc[-1]),
            "rel_20d": float(_ret(close, 20).iloc[-1] - _ret(spy, 20).iloc[-1]),
            "rel_60d": float(_ret(close, 60).iloc[-1] - _ret(spy, 60).iloc[-1]),
            "rs_trend": float(t) if np.isfinite(t) else None, "rs_momentum": float(m) if np.isfinite(m) else None,
            "quadrant": quadrant,
            "above_ema21": bool(close.iloc[-1] > e21.iloc[-1]), "above_ema50": bool(close.iloc[-1] > e50.iloc[-1]),
            "rs_tail": [float(v) if np.isfinite(v) else None for v in rs_trend.tail(10)],
            "mom_tail": [float(v) if np.isfinite(v) else None for v in rs_mom.tail(10)],
        })
    rows.sort(key=lambda r: r["rel_20d"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank_20d"] = rank
    return rows


def index_table(prices: dict[str, pd.DataFrame], symbols: list[str], index: pd.DatetimeIndex) -> list[dict]:
    rows = []
    for symbol in symbols:
        close = _close(prices, symbol, index)
        if close is None or close.dropna().empty:
            continue
        frame = prices[symbol].reindex(index)
        e9, e21, e50, e200 = (ema(close, s) for s in (9, 21, 50, 200))
        hi52 = frame["high"].rolling(252, min_periods=200).max().iloc[-1]
        rows.append({
            "symbol": symbol, "close": float(frame["close"].iloc[-1]),
            "ret_1d": float(_ret(close, 1).iloc[-1]), "ret_5d": float(_ret(close, 5).iloc[-1]),
            "ret_20d": float(_ret(close, 20).iloc[-1]), "ret_ytd": _ytd(close),
            "vs_ema21": float(close.iloc[-1] / e21.iloc[-1] - 1), "vs_ema50": float(close.iloc[-1] / e50.iloc[-1] - 1),
            "vs_ema200": float(close.iloc[-1] / e200.iloc[-1] - 1) if np.isfinite(e200.iloc[-1]) else None,
            "from_52w_high": float(frame["close"].iloc[-1] / hi52 - 1) if np.isfinite(hi52) else None,
            "ema_stack": ("bullish" if close.iloc[-1] > e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1] else
                          "bearish" if close.iloc[-1] < e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1] else "mixed"),
        })
    return rows


def _ytd(close: pd.Series) -> float | None:
    year = close.index[-1].year
    prior = close[close.index.year < year]
    return float(close.iloc[-1] / prior.iloc[-1] - 1) if not prior.empty else None


def summarize(history: pd.DataFrame) -> dict[str, Any]:
    """Today's regime: label, composite, components with evidence, guidance and recent path."""
    row = history.iloc[-1]
    ev = evidence(row)
    components = []
    for key, weight in COMPONENT_WEIGHTS.items():
        score = row.get(key)
        components.append({
            "key": key, "name": COMPONENT_NAMES[key], "weight": weight,
            "score": float(score) if pd.notna(score) else None,
            "contribution": float(weight * score / 2) if pd.notna(score) else None,
            "evidence": ev[key],
        })
    label = row["label"]
    text, long_budget, short_budget = GUIDANCE.get(label, ("Insufficient data.", 0.0, 0.0))
    previous = history["label"].iloc[-6] if len(history) > 6 else None
    recent = history[["composite", "label"]].tail(130)
    return {
        "date": history.index[-1].strftime("%Y-%m-%d"),
        "label": label, "composite": float(row["composite"]) if pd.notna(row["composite"]) else None,
        "label_5d_ago": previous,
        "composite_5d_change": (float(row["composite"] - history["composite"].iloc[-6])
                                if len(history) > 6 and pd.notna(history["composite"].iloc[-6]) else None),
        "components": components,
        "guidance": text, "long_risk_budget": long_budget, "short_risk_budget": short_budget,
        "breadth": {k: (float(row[k]) if pd.notna(row.get(k)) else None)
                    for k in ("pct_above_ema50", "pct_above_ema200", "pct_new_high20", "pct_new_low20", "breadth_n")},
        "volatility": {k: (float(row[k]) if pd.notna(row.get(k)) else None) for k in ("vix", "vix_term", "vix_chg5")},
        "rates": {k: (float(row[k]) if pd.notna(row.get(k)) else None)
                  for k in ("us10y", "us2y", "curve_2s10s", "us10y_chg20_bp")},
        "history": [{"date": d.strftime("%Y-%m-%d"), "composite": (float(c) if pd.notna(c) else None), "label": lab}
                    for d, c, lab in zip(recent.index, recent["composite"], recent["label"])],
        "methodology": __doc__,
    }
