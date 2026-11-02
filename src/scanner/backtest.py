"""Historical simulation of every setup's plan across the universe (walk-forward, no look-ahead).

For each setup episode (first day the setup is active after a day it was not):

1. Signal at the close of day t. All inputs (indicators, setups, plan levels) use data up to t.
2. Entry at the OPEN of day t+1, worsened by ``slippage_bps`` (conditional squeeze entries:
   only if t+1 trades through the trigger, filled at max(open, trigger)). If the open is
   already beyond the stop, no trade is taken.
3. Each following session up to the time stop (H sessions):
   - long: Low <= stop -> exit at the stop (or at the open if it gapped below the stop);
     else High >= Target 1 -> exit at Target 1 (or the open if it gapped above).
     If stop and target are both inside one bar, the STOP is assumed first (conservative).
   - short: mirror image.
4. Otherwise exit at the close of session H.  Exits are also worsened by the slippage.

Results are in R multiples (P&L / initial risk) and percent. Statistics are grouped by
setup and by the market regime label on the signal day.

Known biases (stated in every report): survivorship (today's universe), in-sample
(the same history that inspired the rules), overlapping signals across correlated
stocks, no commissions / borrow costs, daily bars only (intraday path unknown).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.scanner.plan import plan_for_setup
from src.scanner.setups import LONG_SETUPS, SETUP_DIRECTION, SHORT_SETUPS

MIN_SAMPLE = 30


def simulate_symbol(symbol: str, df: pd.DataFrame, setups: pd.DataFrame, cfg: dict[str, Any],
                    regime_labels: pd.Series | None = None) -> list[dict[str, Any]]:
    horizon = int(cfg.get("plan_horizon_days", 7))
    slip = float(cfg.get("slippage_bps", 10)) / 10_000
    o, h, lo, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(df)
    trades: list[dict[str, Any]] = []
    for setup in (*LONG_SETUPS, *SHORT_SETUPS):
        mask = setups[f"setup_{setup}"].to_numpy(dtype=bool)
        starts = np.flatnonzero(mask & ~np.concatenate([[False], mask[:-1]]))
        starts = starts[starts + horizon < n]
        if starts.size == 0:
            continue
        plan = plan_for_setup(df, setup, cfg)
        long = SETUP_DIRECTION[setup] == "long"
        for t in starts:
            entry_ref, stop, target = plan["entry"].iat[t], plan["stop"].iat[t], plan["target1"].iat[t]
            if not (np.isfinite(entry_ref) and np.isfinite(stop) and np.isfinite(target)):
                continue
            first = t + 1
            if setup == "squeeze":
                if h[first] < entry_ref:
                    continue  # trigger never reached on the next session -> no trade
                entry = max(o[first], entry_ref)
            else:
                entry = o[first]
            if not np.isfinite(entry):
                continue
            entry = entry * (1 + slip) if long else entry * (1 - slip)
            risk = (entry - stop) if long else (stop - entry)
            if risk <= 0:
                continue  # opened beyond the stop: a disciplined trader does not enter
            exit_price, outcome, days = None, "time", horizon
            for k in range(first, first + horizon):
                if long:
                    if lo[k] <= stop:
                        exit_price, outcome = (min(o[k], stop) if k > first else stop), "stop"
                    elif h[k] >= target:
                        exit_price, outcome = (max(o[k], target) if k > first else target), "target"
                else:
                    if h[k] >= stop:
                        exit_price, outcome = (max(o[k], stop) if k > first else stop), "stop"
                    elif lo[k] <= target:
                        exit_price, outcome = (min(o[k], target) if k > first else target), "target"
                if exit_price is not None:
                    days = k - first + 1
                    break
            if exit_price is None:
                exit_price = c[first + horizon - 1]
            exit_price = exit_price * (1 - slip) if long else exit_price * (1 + slip)
            pnl = (exit_price - entry) if long else (entry - exit_price)
            window = slice(first, first + horizon)
            mfe = (np.nanmax(h[window]) / entry - 1) if long else (1 - np.nanmin(lo[window]) / entry)
            mae = (np.nanmin(lo[window]) / entry - 1) if long else (1 - np.nanmax(h[window]) / entry)
            trades.append({
                "symbol": symbol, "setup": setup, "direction": "long" if long else "short",
                "signal_date": df.index[t], "entry": entry, "exit": exit_price, "outcome": outcome, "days": days,
                "r": pnl / risk if risk > 0 else np.nan, "ret": pnl / entry, "mfe": mfe, "mae": mae,
                "regime": regime_labels.get(df.index[t], "n/a") if regime_labels is not None else "n/a",
                **{f"fwd_{k}d": ((c[t + k] / c[t] - 1) * (1 if long else -1)) if t + k < n else np.nan
                   for k in (3, 5, 7, 10)},
            })
    return trades


def summarize(trades: pd.DataFrame) -> dict[str, Any]:
    """Aggregate statistics; ``small_sample`` when N < 30."""
    if trades.empty:
        return {"n": 0, "small_sample": True}
    r = trades["r"].dropna()
    wins, losses = r[r > 0], r[r <= 0]
    out = {
        "n": int(len(trades)),
        "win_rate": float((r > 0).mean()) if len(r) else None,
        "avg_r": float(r.mean()) if len(r) else None,
        "median_r": float(r.median()) if len(r) else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
        "hit_target_rate": float((trades["outcome"] == "target").mean()),
        "stop_rate": float((trades["outcome"] == "stop").mean()),
        "time_exit_rate": float((trades["outcome"] == "time").mean()),
        "avg_days": float(trades["days"].mean()),
        "avg_ret": float(trades["ret"].mean()),
        "median_ret": float(trades["ret"].median()),
        "avg_mfe": float(trades["mfe"].mean()),
        "avg_mae": float(trades["mae"].mean()),
        "small_sample": len(trades) < MIN_SAMPLE,
    }
    for k in (3, 5, 7, 10):
        col = trades[f"fwd_{k}d"].dropna()
        out[f"fwd_{k}d_mean"] = float(col.mean()) if len(col) else None
        out[f"fwd_{k}d_positive"] = float((col > 0).mean()) if len(col) else None
    return out


def run_backtest(features: dict[str, pd.DataFrame], setups: dict[str, pd.DataFrame], cfg: dict[str, Any],
                 regime_labels: pd.Series | None = None, lookback_sessions: int | None = None) -> dict[str, Any]:
    all_trades: list[dict[str, Any]] = []
    for symbol, df in features.items():
        st = setups[symbol]
        if lookback_sessions:
            df, st = df.tail(lookback_sessions), st.tail(lookback_sessions)
        all_trades.extend(simulate_symbol(symbol, df, st, cfg, regime_labels))
    trades = pd.DataFrame(all_trades)
    result: dict[str, Any] = {"by_setup": {}, "by_setup_regime": {}, "baseline": {}, "n_trades": int(len(trades)),
                              "horizon_days": int(cfg.get("plan_horizon_days", 7)),
                              "slippage_bps": float(cfg.get("slippage_bps", 10))}
    if trades.empty:
        return result
    result["first_signal"] = trades["signal_date"].min().strftime("%Y-%m-%d")
    result["last_signal"] = trades["signal_date"].max().strftime("%Y-%m-%d")
    for setup, group in trades.groupby("setup"):
        result["by_setup"][setup] = summarize(group)
        result["by_setup_regime"][setup] = {regime: summarize(g) for regime, g in group.groupby("regime")}
    for direction, group in trades.groupby("direction"):
        result["by_setup"][f"ALL_{direction.upper()}"] = summarize(group)
    # Baseline: unconditional forward returns of every stock-day (long view), for comparison.
    base = []
    for df in features.values():
        c = df["close"]
        base.append(pd.DataFrame({f"fwd_{k}d": c.shift(-k) / c - 1 for k in (3, 5, 7, 10)}).dropna())
    if base:
        b = pd.concat(base)
        result["baseline"] = {f"fwd_{k}d_mean": float(b[f"fwd_{k}d"].mean()) for k in (3, 5, 7, 10)}
        result["baseline"].update({f"fwd_{k}d_positive": float((b[f"fwd_{k}d"] > 0).mean()) for k in (3, 5, 7, 10)})
        result["baseline"]["n"] = int(len(b))
    result["recent_trades"] = (trades.sort_values("signal_date").tail(200)
                               .assign(signal_date=lambda t: t["signal_date"].dt.strftime("%Y-%m-%d"))
                               .replace({np.nan: None}).to_dict("records"))
    result["trades_frame"] = trades
    return result
