from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from reality.engine import stats, by_period, month_ends, run_weights, leveraged, PERIODS, TRADING_DAYS
from reality.panel import load

close, vol, tab = load()
pit = pd.read_pickle("/tmp/rc/pit.pkl")
ret = close.pct_change(fill_method=None)
dates = close.index
cur_members = set(pit.iloc[-1][pit.iloc[-1]].index)          # today's S&P 500 = the "survivor" universe
survivor = pd.DataFrame(False, index=dates, columns=close.columns); survivor[list(cur_members)] = True
dvol = (close * vol).rolling(60, min_periods=40).mean()
mom = close.shift(21) / close.shift(252) - 1
mom6 = close.shift(21) / close.shift(126) - 1
START = pd.Timestamp("2005-01-01")
ME = [d for d in month_ends(dates) if d >= START]

def eligible(universe: pd.DataFrame, d, top_liq: int | None = None):
    ok = universe.loc[d] & close.loc[d].notna() & mom.loc[d].notna()
    if top_liq:
        liq = dvol.loc[d][ok].nlargest(top_liq).index
        ok = ok & close.columns.isin(liq)
    return ok

def weights(universe, score: pd.DataFrame, n: int, top_liq=None, rng=None, ascending=False):
    rows = {}
    for d in ME:
        ok = eligible(universe, d, top_liq)
        if ok.sum() < max(n * 2, 30):
            continue
        if rng is None:
            s = score.loc[d][ok].dropna()
            pick = s.nsmallest(n).index if ascending else s.nlargest(n).index
        else:
            pick = rng.choice(ok[ok].index, size=n, replace=False)
        w = pd.Series(0.0, index=close.columns); w[list(pick)] = 1.0 / n
        rows[d] = w
    return pd.DataFrame(rows).T

def row(name, r, extra=None):
    d = {"name": name, "periods": by_period(r)}
    if extra: d.update(extra)
    return d

def run_momentum(universe, label, n, bps, top_liq=None, seeds=60):
    W = weights(universe, mom, n, top_liq)
    r, t = run_weights(ret, W, bps)
    rnd = []
    for s in range(seeds):
        Wr = weights(universe, mom, n, top_liq, rng=np.random.default_rng(s))
        rr, _ = run_weights(ret, Wr, bps)
        rnd.append(by_period(rr))
    return r, t, rnd
