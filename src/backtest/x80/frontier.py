"""python -m src.backtest.x80.frontier OUT_DIR : combine the sleeves, lever them, and trace CAGR against max drawdown.

Each sleeve is the version chosen on its own early period (none is picked on 2018-26 results):
BTC trend (Donchian 20/10, vol-targeted), ETH trend (7-day momentum), crypto cross-sectional momentum (in-sample
best), BTC funding carry, short VIX futures with the term-structure filter, QQQ volatility target,
multi-asset trend (18 ETFs), S&P 500 market-neutral composite, Cboe PutWrite, and our live swing set
(crash rebound + squeeze, low-short-volume filter, S&P 500 point-in-time account marked to market daily).
Weekly excess returns over T-bills. Walk-forward combination: inverse-vol weights from the trailing 52 weeks,
the whole book scaled to a target vol from its trailing 26-week vol (cap 10x), financed at T-bill + 1% (crypto
sleeves T-bill + 4%). The funding-carry books are reported on their own, not levered here (see EXCLUDE).
Also the hindsight max-Sharpe mix (weights fitted on the very period it is scored on) as an upper bound.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from src.backtest.x80 import crypto_trend as ct
from src.backtest.x80 import vrp
from src.backtest.x80.common import cash_daily, load
from src.backtest.x80.equity_lev import book, total_return
from src.backtest.x80.misc import cboe
from src.backtest.research_paths import research_path

OUT = research_path("x80", "out")


def weekly_excess(r: pd.Series, cash: pd.Series) -> pd.Series:
    ex = (r - cash.reindex(r.index).fillna(0.0)).dropna()
    return (1 + ex).resample("W-FRI").prod() - 1


def sleeves(px: dict) -> pd.DataFrame:
    s = {}
    for asset, rule in (("BTC-USD", "don20_10"), ("ETH-USD", "mom7")):
        close = px[asset]["close"]
        r = close.pct_change().dropna()
        cash = cash_daily(px, r.index, 365)
        sig = ct.signals(close)[rule]
        if asset == "BTC-USD":
            rv = r.rolling(30).std() * np.sqrt(365)
            sig = sig * (0.8 / rv).clip(upper=3.0).reindex(sig.index).fillna(0.0)
        s[f"{asset[:3]} trend"] = weekly_excess(ct.run_exposure(r, sig.reindex(r.index).fillna(0.0), cash), cash)
    xs = pd.read_parquet(f"{OUT}/crypto_xs_series.parquet")
    xst = pd.read_csv(f"{OUT}/crypto_xs.csv")
    best = xst[(xst["period"] == "IS 2018-19") & xst["rule"].str.startswith("mom")].sort_values("sharpe").iloc[-1]["rule"]
    c365 = cash_daily(px, xs.index, 365)
    s["crypto XS momentum"] = weekly_excess(xs[best].dropna(), c365)
    if os.path.exists(f"{OUT}/crypto_carry_series.parquet"):
        cc = pd.read_parquet(f"{OUT}/crypto_carry_series.parquet")
        for col, name in (("BTC always on", "BTC funding carry"), ("top-5 funding, weekly", "funding rotation")):
            x = cc[col].dropna()                                           # capital is in the coin: excess = carry - T-bill
            s[name] = weekly_excess(x, cash_daily(px, x.index, 365))
    if os.path.exists(f"{OUT}/crypto_ls_series.parquet"):
        cl = pd.read_parquet(f"{OUT}/crypto_ls_series.parquet")["long BTC / short top50 alts"].dropna()
        s["BTC vs alts"] = weekly_excess(cl, cash_daily(px, cl.index, 365))
    sv, cash = vrp.short_vol_series(px)
    vix, vix3m = px["^VIX"]["close"], px["^VIX3M"]["close"]
    ts = (vix / vix3m).reindex(sv.index)
    s["short VIX + term filter"] = weekly_excess(vrp.account(sv, cash, (ts < 1.0).astype(float), 1), cash)
    r = total_return(px, "QQQ", "^NDX", 0.005)
    cash = cash_daily(px, r.index)
    rv = r.rolling(20).std() * np.sqrt(252)
    s["QQQ vol target"] = weekly_excess(book(r, (0.2 / rv).clip(upper=2.0).shift(1), cash), cash)
    tsm = pd.read_parquet(f"{OUT}/misc_tsmom.parquet")["tsmom"]
    s["multi-asset trend"] = weekly_excess(tsm, cash_daily(px, tsm.index))
    ls = pd.read_parquet(f"{OUT}/ls_neutral_series.parquet")["composite_all|beta"].dropna()
    s["S&P 500 long/short"] = weekly_excess(ls, cash_daily(px, ls.index))
    put = cboe(research_path("x80", "raw", "cboe", "PUT.csv"))
    s["PutWrite"] = weekly_excess(put, cash_daily(px, put.index))
    acct = pd.read_parquet(f"{OUT}/live_mtm.parquet")["seed0"].dropna()      # marked to market daily (live_mtm.py)
    ar = acct.pct_change().dropna()
    s["live swing (crash+squeeze)"] = weekly_excess(ar, cash_daily(px, ar.index))
    return pd.DataFrame(s)


CRYPTO = {"BTC trend", "ETH trend", "crypto XS momentum", "BTC funding carry", "funding rotation", "BTC vs alts"}
# the two funding-carry books are left out of the levered mix: their weekly vol is tiny (so inverse-vol weighting would
# pile into them and lever them 20x+), their excess over T-bills since 2022 is 2-3%/yr while crypto borrowing costs
# T-bill + 4%, and their real risk (an exchange failing, a liquidation) never shows in the price history.
EXCLUDE = {"BTC funding carry", "funding rotation"}


def spread_of(name: str) -> float:
    return 0.04 if name in CRYPTO else 0.01


def combine(w_ex: pd.DataFrame, cash_w: pd.Series, target: float, cap: float = 10.0) -> pd.Series:
    w_ex = w_ex.drop(columns=[c for c in EXCLUDE if c in w_ex])
    vol = w_ex.rolling(52, min_periods=26).std().shift(1)
    inv = (1 / vol).where(w_ex.notna())
    wts = inv.div(inv.sum(axis=1), axis=0)
    base = (wts * w_ex.fillna(0.0)).sum(axis=1).where(wts.notna().any(axis=1))
    spread = (wts.fillna(0.0) * pd.Series({c: spread_of(c) for c in w_ex.columns})).sum(axis=1)
    bvol = base.rolling(26, min_periods=13).std().shift(1) * np.sqrt(52)
    lev = (target / bvol).clip(upper=cap)
    ret = lev * base + cash_w - (lev - 1).clip(lower=0) * spread / 52
    return ret.dropna().clip(lower=-1.0)


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 26:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 52
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0, "max_dd": float((eq / eq.cummax() - 1).min()),
            "vol": float(r.std() * np.sqrt(52))}


def main() -> None:
    out_dir = sys.argv[1]
    px = load()
    w_ex = sleeves(px)
    w_ex = w_ex[w_ex.index >= "2004-01-01"]
    cash_w = (1 + cash_daily(px, pd.date_range("2004-01-01", "2026-11-04"), 365)).resample("W-FRI").prod().reindex(w_ex.index).fillna(0) - 1
    w_ex.to_parquet(os.path.join(out_dir, "sleeves_weekly.parquet"))
    stat = {}
    for c in w_ex:
        x = w_ex[c].dropna()
        for lab, a in (("2018-26", "2018-01-01"), ("2022-26", "2022-01-01")):
            y = x[x.index >= a]
            stat[(c, lab)] = y.mean() / y.std() * np.sqrt(52) if len(y) > 26 else np.nan
    sh = pd.Series(stat).unstack()
    print("sleeve Sharpe (weekly, excess of T-bills):\n", sh.round(2).to_string())
    corr = w_ex[w_ex.index >= "2018-01-01"].corr()
    print("correlation 2018-26:\n", corr.round(2).to_string())
    corr.to_csv(os.path.join(out_dir, "sleeve_corr.csv"))
    rows = []
    for tv in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00):
        r = combine(w_ex, cash_w, tv)
        for lab, a in (("2018-26", "2018-01-01"), ("2022-26", "2022-01-01")):
            p = perf(r[r.index >= a])
            rows.append({"mix": "walk-forward inverse vol", "target_vol": tv, "period": lab, **p})
    # hindsight upper bound: long-only max-Sharpe weights fitted on the period itself
    for lab, a in (("2018-26", "2018-01-01"), ("2022-26", "2022-01-01")):
        x = w_ex.drop(columns=[c for c in EXCLUDE if c in w_ex])
        x = x[x.index >= a].dropna(axis=1, thresh=int(0.9 * len(x[x.index >= a]))).fillna(0.0)
        mu, cov = x.mean().values, x.cov().values
        w = np.clip(np.linalg.pinv(cov) @ mu, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(mu)) / len(mu)
        for _ in range(200):                                               # projected-gradient refinement, long only
            g = mu / np.sqrt(w @ cov @ w) - (w @ mu) * (cov @ w) / (w @ cov @ w) ** 1.5
            w = np.clip(w + 0.05 * g / (np.abs(g).max() + 1e-12), 0, None)
            w = w / w.sum()
        base = x.values @ w
        sr = base.mean() / base.std() * np.sqrt(52)
        print(f"hindsight max-Sharpe {lab}: Sharpe {sr:.2f}", dict(zip(x.columns, np.round(w, 2))))
        sp = float(w @ np.array([spread_of(c) for c in x.columns]))
        for tv in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00):
            lev = tv / (base.std() * np.sqrt(52))
            r = pd.Series(lev * base, index=x.index) + cash_w.reindex(x.index) - max(lev - 1, 0) * sp / 52
            rows.append({"mix": "hindsight max-Sharpe", "target_vol": tv, "period": lab, "sharpe": sr, **perf(r.clip(lower=-1))})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, "frontier.csv"), index=False)
    print(t.round(3).to_string())
    json.dump({"sleeve_sharpe": sh.round(3).to_dict()}, open(os.path.join(out_dir, "frontier_sleeves.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
