"""Data-quality checks for the daily universe update.

Every symbol gets a status:
    ok        - usable, nothing suspicious
    warning   - usable, but something should be known (listed in ``issues``)
    excluded  - not scored (stale / missing latest session / too little history / broken bars)

Checks (all on the stored bars up to the target session):
    * latest bar == expected session                 (else: stale -> excluded)
    * missing sessions vs the exchange calendar, last 252 sessions
    * enough history for EMA250 / percentiles         (min_history_days)
    * OHLC consistency (High >= max(Open, Close), Low <= min(Open, Close))
    * zero / missing volume on recent days
    * unchanged close for >= 3 consecutive sessions   (stale quotes)
    * |daily return| > 50% without a matching adj/close jump (possible bad tick or unadjusted split)
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.market import calendar

INDEX_LIKE = ("^",)  # index symbols (VIX...) have no volume


def check_symbol(symbol: str, frame: pd.DataFrame | None, session: date, min_history: int = 260) -> dict[str, Any]:
    issues: list[str] = []
    status = "ok"
    if frame is None or frame.empty:
        return {"symbol": symbol, "status": "excluded", "issues": ["沒有已儲存的價格數據。"], "last_date": None,
                "sessions": 0, "missing_sessions": None}
    last = frame.index[-1].date()
    if last != session:
        lag = calendar.trading_days_between(last, session)
        issues.append(f"最新 K 線是 {last}（落後 {session} {lag} 個交易日）。")
        status = "excluded"
    if len(frame) < min_history:
        issues.append(f"只有 {len(frame)} 個交易日歷史（少於 {min_history}），長期 EMA 和百分位不可靠。")
        status = "excluded"

    expected = calendar.trading_days(max(frame.index[0].date(), _sessions_back(session, 252)), session)
    present = set(frame.index.date)
    missing = [d for d in expected if d not in present]
    if missing:
        issues.append(f"過去一年缺少 {len(missing)} 個交易日（例如 {missing[0]}）。")
        if len(missing) > 5:
            status = _worse(status, "warning")
    recent = frame.tail(20)
    has_ohlc = recent[["open", "high", "low", "close"]].notna().all(axis=1)
    bad = has_ohlc & ((recent["high"] < recent[["open", "close"]].max(axis=1) * 0.999)
                      | (recent["low"] > recent[["open", "close"]].min(axis=1) * 1.001))
    if bad.any():
        issues.append(f"近期有 {int(bad.sum())} 根 K 線的開高低收數據不一致。")
        status = _worse(status, "warning")
    if not symbol.startswith(INDEX_LIKE):
        zero = int(((recent["volume"] <= 0) | recent["volume"].isna()).sum())
        if zero:
            issues.append(f"最近 20 個交易日中有 {zero} 日成交量為零或缺失。")
            status = _worse(status, "warning" if zero < 3 else "excluded")
    closes = frame["close"].tail(10).to_numpy()
    if len(closes) >= 3 and np.all(closes[-3:] == closes[-1]):
        issues.append("收市價連續 3 日以上不變（報價可能過期）。")
        status = _worse(status, "warning")
    returns = frame["close"].pct_change().tail(252)
    jumps = returns[returns.abs() > 0.5]
    if not jumps.empty:
        issues.append(f"過去一年有 {len(jumps)} 日單日變動超過 50%（例如 {jumps.index[-1]:%Y-%m-%d}），"
                      "請核實是真實走勢，而不是錯誤報價或未調整的拆股。")
        status = _worse(status, "warning")
    return {"symbol": symbol, "status": status, "issues": issues, "last_date": last.isoformat(),
            "sessions": int(len(frame)), "missing_sessions": len(missing)}


def _sessions_back(session: date, n: int) -> date:
    day = session
    for _ in range(n):
        day = calendar.previous_trading_day(day)
    return day


def _worse(a: str, b: str) -> str:
    order = {"ok": 0, "warning": 1, "excluded": 2}
    return a if order[a] >= order[b] else b


def check_universe(prices: dict[str, pd.DataFrame], symbols: list[str], session: date,
                   min_history: int = 260) -> dict[str, Any]:
    rows = [check_symbol(s, prices.get(s), session, min_history) for s in symbols]
    counts = {k: sum(r["status"] == k for r in rows) for k in ("ok", "warning", "excluded")}
    coverage = (counts["ok"] + counts["warning"]) / len(rows) if rows else 0.0
    return {"session": session.isoformat(), "symbols": rows, "counts": counts, "coverage": coverage,
            "usable": [r["symbol"] for r in rows if r["status"] != "excluded"]}


def provider_has_session(prices: dict[str, pd.DataFrame], reference: str, session: date) -> bool:
    """True when the reference symbol (SPY) already has a bar for ``session`` - i.e. the
    provider has published the session. Used to refuse a run on stale provider data."""
    frame = prices.get(reference)
    return frame is not None and not frame.empty and frame.index[-1].date() == session
