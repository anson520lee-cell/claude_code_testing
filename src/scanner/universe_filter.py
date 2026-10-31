"""Daily focus filter: keep only high-beta, high-volatility, large-enough, liquid names.

Rules (config/terminal.yaml -> universe_filter, liquidity):
    beta120   >= min_beta        120-session beta vs SPY
    hv60      >= min_hv60        60-session annualised volatility
    market cap >= min_market_cap  (leveraged ETFs exempt; unknown cap is not held against a stock)
    20-day average dollar volume >= liquidity.min_avg_dollar_volume
A stock failing any rule is excluded from candidates, the universe table and the setup statistics;
the reasons are kept so the dashboard can show why.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def apply_universe_filter(features: dict[str, pd.DataFrame], settings: dict[str, Any],
                          caps: dict[str, float | None]) -> dict[str, Any]:
    cfg = settings.get("universe_filter") or {}
    leveraged = set(settings.get("leveraged_etfs") or [])
    min_dv = float(settings["liquidity"]["min_avg_dollar_volume"])
    rows, kept, excluded = [], [], []
    for symbol, frame in features.items():
        last = frame.iloc[-1]
        item = {"symbol": symbol, "beta": _num(last.get("beta120")), "hv60": _num(last.get("hv60")),
                "market_cap": caps.get(symbol), "dollar_volume20": _num(last.get("dollar_volume20")),
                "leveraged": symbol in leveraged}
        reasons = []
        if cfg.get("enabled", True):
            if item["beta"] is None or item["beta"] < cfg.get("min_beta", 0):
                reasons.append(f"Beta {item['beta']:.2f} 低於 {cfg.get('min_beta')}" if item["beta"] is not None else "Beta 無法計算")
            if item["hv60"] is None or item["hv60"] < cfg.get("min_hv60", 0):
                reasons.append(f"60 日波動率 {item['hv60'] * 100:.0f}% 低於 {cfg.get('min_hv60', 0) * 100:.0f}%"
                               if item["hv60"] is not None else "波動率無法計算")
            cap = item["market_cap"]
            if not item["leveraged"] and cap is not None and cap < cfg.get("min_market_cap", 0):
                reasons.append(f"市值 ${cap / 1e9:.2f}B 低於 ${cfg.get('min_market_cap', 0) / 1e9:.0f}B")
            dv = item["dollar_volume20"]
            if dv is None or dv < min_dv:
                reasons.append(f"日均成交額 ${(dv or 0) / 1e6:.1f}M 低於 ${min_dv / 1e6:.0f}M")
        item["reason"] = "；".join(reasons)
        (excluded if reasons else kept).append(symbol)
        rows.append(item)
    return {"kept": kept, "excluded": [r for r in rows if r["reason"]], "kept_detail": [r for r in rows if not r["reason"]],
            "rules": {"enabled": cfg.get("enabled", True), "min_beta": cfg.get("min_beta"), "min_hv60": cfg.get("min_hv60"),
                      "min_market_cap": cfg.get("min_market_cap"), "min_avg_dollar_volume": min_dv,
                      "leveraged_exempt": sorted(leveraged)}}
