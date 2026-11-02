"""Portfolio-level check for the strategies that pass: a realistic account can only hold so many positions.

simulate(): signals in date order fill at most `max_pos` equal slots (1/max_pos of the account each, no leverage);
a slot is busy from the entry session to the exit session; when more signals arrive than free slots, the ones with
the best `rank` (default: random, fixed seed) are taken. Monthly returns = realised trade P&L x slot weight, booked
evenly over each trade's holding sessions (approximate mark-to-market), plus cash earning nothing.
combine(): blends the monthly streams of several strategies with weights chosen IN-SAMPLE (inverse variance),
judged out of sample against equal weights and SPY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import IS_END, START


def simulate(p, sig, out, held, max_pos=10, rank=None, seed=0):
    rng = np.random.default_rng(seed)
    ok = sig & p.elig & np.isfinite(out) & np.isfinite(held)
    ti, tj = np.where(ok)
    if len(ti) == 0:
        return pd.Series(dtype=float), {}
    pr = rank[ti, tj] if rank is not None else rng.random(len(ti))
    order = np.lexsort((-pr, ti))
    ti, tj = ti[order], tj[order]
    busy_until = []  # exit rows of open slots
    pnl = np.zeros(p.D + 2)
    taken = 0
    open_names = {}
    for d, s in zip(ti, tj):
        busy_until = [x for x in busy_until if x > d]  # slot frees after its exit session
        if len(busy_until) >= max_pos or open_names.get(s, -1) > d:
            continue
        ex = int(d + held[d, s])
        busy_until.append(ex)
        open_names[s] = ex
        # spread the trade's P&L evenly over its holding sessions (approximate mark-to-market)
        e0, e1 = d + 1, min(ex, p.D - 1)
        per = out[d, s] / max_pos / max(e1 - e0 + 1, 1)
        pnl[e0] += per
        pnl[e1 + 1] -= per
        taken += 1
    daily = pd.Series(np.cumsum(pnl)[:p.D], index=p.dates)
    m = daily.groupby(daily.index.to_period("M")).sum()
    m.index = m.index.to_timestamp("M")
    m = m[m.index >= START]
    return m, {"taken": taken, "signals": int(ok.sum())}


def perf(m: pd.Series) -> dict:
    res = {}
    for name, mask in (("IS", m.index <= IS_END), ("OOS", m.index > IS_END), ("ALL", m.index >= START)):
        x = m[mask]
        if len(x) < 6:
            continue
        eq = (1 + x).cumprod()
        yrs = len(x) / 12
        res[name] = {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(x.std() * np.sqrt(12)),
                     "sharpe": float(x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else None,
                     "maxdd": float((eq / eq.cummax() - 1).min()), "months": int(len(x))}
    return res


def combine(streams: dict[str, pd.Series]) -> dict:
    df = pd.DataFrame(streams).fillna(0.0)
    ins = df[df.index <= IS_END]
    var = ins.var().replace(0, np.nan)
    w = (1 / var).fillna(0)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1 / len(df.columns), index=df.columns)
    ivw = (df * w).sum(1)
    ew = df.mean(1)
    return {"weights": {k: float(v) for k, v in w.items()}, "inv_var": perf(ivw), "equal": perf(ew),
            "corr_oos": df[df.index > IS_END].corr().round(2).to_dict()}
