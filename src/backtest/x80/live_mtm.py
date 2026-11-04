"""python -m src.backtest.x80.live_mtm SIGNALS FLAGS PRICES OUT_PARQUET : the live swing account, marked to market daily.

Same rules and sizing as src.backtest.edge_hunt.account (10 slots, 1% risk, 25% cap, crash > squeeze priority,
random tie-break), but an open trade is valued every session at the stock's adjusted close against the signal-day
close, and the trade's real return is booked on its exit day. The edge-hunt curve books P&L only at exit, which hides
intra-trade swings (smaller drawdowns, lower correlation with the market than reality).
Variant: crash rebound + squeeze, live-like stocks, FINRA low short-volume filter (the live set-up since 2026-11-04).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest.edge_hunt.account import PRIORITY
from src.backtest.edge_hunt.account_run import END, START


def simulate_mtm(trades: pd.DataFrame, cal: pd.DatetimeIndex, adj: pd.DataFrame, risk: float = 0.01, slots: int = 10,
                 cap: float = 0.25, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = trades.copy()
    t["risk_pct"] = (t["ret"] / t["r"]).where(t["r"].abs() > 1e-6)
    t = t.dropna(subset=["risk_pct"])
    t = t[t["risk_pct"] > 0]
    pos = np.searchsorted(cal.values, t["date"].values.astype("datetime64[ns]"))
    t["entry_i"] = pos + 1
    t["exit_i"] = t["entry_i"] + t["days"].astype(int) - 1
    t["prio"] = t["setup"].map(PRIORITY) + rng.random(len(t)) * 0.5
    t = t.sort_values(["entry_i", "prio"])
    a = adj.reindex(index=cal).ffill()
    cols = {c: j for j, c in enumerate(a.columns)}
    A = a.to_numpy()
    eq = 1.0
    open_pos: list[tuple[int, float, float, int, int]] = []       # (exit_i, size, ret, col, entry_i)
    curve = np.full(len(cal), np.nan)
    by_day = {i: g for i, g in t.groupby("entry_i")}
    for i in range(len(cal)):
        keep = []
        for ex, size, ret, j, ei in open_pos:
            if ex <= i:
                eq += size * ret
            else:
                keep.append((ex, size, ret, j, ei))
        open_pos = keep
        g = by_day.get(i)
        if g is not None:
            for _, row in g.iterrows():
                if len(open_pos) >= slots:
                    break
                j = cols.get(row["symbol"])
                if j is None:
                    continue
                size = min(risk / row["risk_pct"], cap) * eq
                open_pos.append((int(row["exit_i"]), size, float(row["ret"]), j, i))
        unreal = 0.0
        for ex, size, ret, j, ei in open_pos:
            ref = A[ei - 1, j] if ei >= 1 else np.nan
            now = A[i, j]
            if np.isfinite(ref) and np.isfinite(now) and ref > 0:
                unreal += size * (now / ref - 1)
        curve[i] = eq + unreal
    return pd.Series(curve, index=cal)


def main() -> None:
    sig, flags, prices, out = sys.argv[1:5]
    s = pd.read_parquet(sig, columns=["symbol", "date", "setup", "member", "live_like", "r", "ret", "days"])
    s = s[s["member"] & s["setup"].isin(["crash_rebound", "squeeze"]) & (s["date"] >= START) & (s["date"] <= END)]
    s = s.dropna(subset=["r", "ret", "days"])
    s = s.merge(pd.read_parquet(flags)[["symbol", "date", "svr5_good"]], on=["symbol", "date"], how="left")
    t = s[s["live_like"] & (s["svr5_good"] == 1)]
    p = pd.read_parquet(prices, columns=["symbol", "date", "adj_close"])
    p["date"] = pd.to_datetime(p["date"])
    adj = p.pivot_table(index="date", columns="symbol", values="adj_close").sort_index()
    spy = adj["SPY"] if "SPY" in adj else None
    cal = adj.index[(adj.index >= START) & (adj.index <= END)]
    curves = {f"seed{k}": simulate_mtm(t, cal, adj, seed=k) for k in range(5)}
    c = pd.DataFrame(curves)
    c.to_parquet(out)
    for k, x in c.items():
        r = x.pct_change().dropna()
        yrs = len(r) / 252
        dd = (x / x.cummax() - 1).min()
        cg = (x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1
        sh = r.mean() / r.std() * np.sqrt(252)
        corr = r.corr(spy.pct_change().reindex(r.index)) if spy is not None else np.nan
        print(k, f"CAGR {cg:.3f} maxDD {dd:.3f} Sharpe(daily, raw) {sh:.2f} corr SPY {corr:.2f}")


if __name__ == "__main__":
    main()
