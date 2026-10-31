"""Portfolio view of today's candidates: position sizing, correlation, total heat.

Sizing (fixed fractional, the standard swing-trading rule):
    dollar risk   = account x risk_per_trade x regime budget (the regime's long / short risk multiplier)
    shares        = dollar risk / |entry - stop|           (the stop distance already scales with ATR)
    capped so one position never exceeds max_position_pct of the account.
Correlation: 60-session daily returns, direction-adjusted (a short position's return is -return), so a
    long and a short in two highly correlated stocks HEDGE each other (negative position correlation).
Total heat = sum of the dollar risk of every position (worst case: every stop hit together).
    Correlation-adjusted risk = sqrt(sum_ij rho_ij) units of one position's risk, and the "effective number
    of independent bets" = n^2 / sum_ij rho_ij: 10 positions that all move together are ~1 bet.
Clusters: groups of positions whose position correlation is >= 0.7 (they will hit their stops together).
Research only: the terminal never places orders.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def size_position(entry: float, stop: float, account: float, risk_pct: float, budget: float = 1.0,
                  max_position_pct: float = 0.25) -> dict[str, Any]:
    per_share = abs(entry - stop)
    if not (per_share > 0 and entry > 0):
        return {"shares": 0, "dollar_risk": 0.0, "value": 0.0, "capped": False}
    risk = account * risk_pct * budget
    shares = np.floor(risk / per_share)
    cap = np.floor(account * max_position_pct / entry)
    capped = shares > cap
    shares = float(min(shares, cap))
    return {"shares": shares, "dollar_risk": shares * per_share, "value": shares * entry,
            "value_pct": shares * entry / account, "capped": bool(capped)}


def _clusters(rho: pd.DataFrame, threshold: float = 0.7) -> list[list[str]]:
    names, seen, out = list(rho.index), set(), []
    for n in names:
        if n in seen:
            continue
        stack, comp = [n], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(y for y in names if y not in seen and rho.at[x, y] >= threshold)
        if len(comp) > 1:
            out.append(sorted(comp))
    return out


def portfolio_report(features: dict[str, pd.DataFrame], candidates: dict[str, list[dict]], cfg: dict[str, Any],
                     budgets: dict[str, float], per_side: int = 10, window: int = 60) -> dict[str, Any]:
    account = float(cfg.get("account_size", 100_000))
    risk_pct = float(cfg.get("risk_per_trade", 0.005))
    max_pos = float(cfg.get("max_position_pct", 0.25))
    max_heat = float(cfg.get("max_heat", 0.04))
    picks = [c for side in ("long", "short") for c in candidates.get(side, [])[:per_side]]
    rows = []
    for c in picks:
        p = c["plan"]
        sz = size_position(p["entry"], p["stop"], account, risk_pct, budgets.get(c["direction"], 1.0), max_pos)
        rows.append({"symbol": c["symbol"], "direction": c["direction"], "setup": c["setup"], "score": c["score"],
                     "entry": p["entry"], "stop": p["stop"], "atr": p.get("atr"), **sz})
    out: dict[str, Any] = {"account_size": account, "risk_per_trade": risk_pct, "max_position_pct": max_pos,
                           "max_heat": max_heat, "budgets": budgets, "positions": rows}
    syms = [r["symbol"] for r in rows]
    rets = pd.DataFrame({r["symbol"] + ("" if r["direction"] == "long" else " (空)"):
                         features[r["symbol"]]["close"].pct_change().tail(window) * (1 if r["direction"] == "long" else -1)
                         for r in rows if r["symbol"] in features})
    if rets.shape[1] >= 2:
        rho = rets.corr(min_periods=int(window * 0.6)).fillna(0.0)
        n = rho.shape[0]
        total = float(rho.to_numpy().sum())
        risk_units = [r["dollar_risk"] for r in rows if r["symbol"] in features]
        w = np.array(risk_units, dtype=float)
        port_risk = float(np.sqrt(max(w @ rho.to_numpy() @ w, 0.0)))
        out.update({"labels": list(rho.index), "correlation": [[round(float(v), 3) for v in row] for row in rho.to_numpy()],
                    "avg_pair_corr": float((total - n) / (n * (n - 1))), "effective_bets": float(n * n / total) if total > 0 else None,
                    "clusters": _clusters(rho), "corr_adjusted_risk": port_risk})
    heat = float(sum(r["dollar_risk"] for r in rows))
    out.update({"total_heat": heat, "total_heat_pct": heat / account if account else None,
                "gross_exposure": float(sum(r["value"] for r in rows)),
                "net_exposure": float(sum(r["value"] * (1 if r["direction"] == "long" else -1) for r in rows))})
    notes = []
    if out["total_heat_pct"] and out["total_heat_pct"] > max_heat:
        notes.append(f"如果全部 {len(rows)} 個候選都建倉，總風險 {out['total_heat_pct'] * 100:.1f}% 超過上限 {max_heat * 100:.1f}%："
                     f"最多約 {int(max_heat / (risk_pct or 1))} 個標準倉位，請按分數和相關性挑選。")
    if out.get("effective_bets") is not None:
        notes.append(f"{len(rows)} 個倉位的平均兩兩相關 {out['avg_pair_corr']:+.2f}，相當於約 {out['effective_bets']:.1f} 個獨立的注碼。")
    for cl in out.get("clusters", []):
        notes.append(f"高度相關（≥ 0.7）會一起觸發止損：{'、'.join(cl)}。只揀其中一兩隻。")
    out["notes"] = notes
    out["symbols"] = syms
    return out
