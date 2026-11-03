"""Display labels for the research terminal (web dashboard + daily report).

Internal keys (used in logic, the database and tests) are mapped to readable English;
``tr(x)`` returns the label or ``x`` unchanged.
"""

from __future__ import annotations

LABELS: dict[str, str] = {
    # internal keys -> readable English (everything else is shown as it is)
    "long": "Long", "short": "Short", "Watch": "Watch",
    "trend": "Trend", "momentum": "Momentum", "relative_strength": "Relative strength", "volume": "Volume",
    "setup_quality": "Setup quality", "reward_risk": "Reward/risk", "liquidity": "Liquidity", "regime_alignment": "Regime alignment",
    "earnings_in_window": "Earnings in window", "extended": "Extended", "overbought_extreme": "RSI extreme",
    "dilution_filing": "Dilution filing", "insider_exec_buy": "CEO/CFO buy", "finra_low_short_volume": "Low short volume", "high_short_volume": "Short volume not low", "financial_distress": "Financial distress", "poor_reward_risk": "Poor reward/risk",
    "wide_stop": "Wide stop", "data_quality": "Data quality", "options_event_premium": "Options event premium",
    "ALL_LONG": "All long setups", "ALL_SHORT": "All short setups",
    "bullish": "Bullish stack", "bearish": "Bearish stack", "mixed": "Mixed",
    "before open": "Before open", "after close": "After close", "during session": "During session",
    "ok": "OK", "warning": "Warning", "excluded": "Excluded",
    "success": "Success", "success_with_warnings": "Success (warnings)", "failed": "Failed", "running": "Running",
    "risk": "Risk", "strength": "Strength",
}


def tr(value: object) -> str:
    """Readable label for an internal key (unchanged when there is none)."""
    text = "" if value is None else str(value)
    return LABELS.get(text, text)


def methodology() -> dict[str, str]:
    """The documented rules, straight from the module docstrings (single source of truth)."""
    from src.market import regime
    from src.scanner import backtest, plan, scoring, setups
    return {"regime": regime.__doc__ or "", "setups": setups.__doc__ or "", "plan": plan.__doc__ or "",
            "scoring": scoring.__doc__ or "", "backtest": backtest.__doc__ or ""}
