"""python -m src.backtest.optimize.final POOL_DIR CASH_WEEKLY SEARCH_DIR OVERFIT_DIR OUT_DIR

The holdout, run once. Candidates fixed before it (reports/optimize/PREREG.md):
  A  the configuration with the best 2010-24 score         B  the per-dimension consensus configuration
  C  the average of the 20 best-scoring configurations     X  the x80 round-2 book
Each is rebuilt on the full data (2004..2026-11), which must reproduce the search's pre-holdout weeks exactly, then
levered to 30% / 50% / 80% target vol with the intraweek wipe-out check (daily sleeve returns inside each week).
Also, for every candidate: a block bootstrap of 8-year paths from its 2010-24 weeks (drawdown and ruin odds), and the
same book financed through futures where that is possible (non-crypto funded sleeves at T-bill + 0.3% instead of + 1%).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import fields

import numpy as np
import pandas as pd

from src.backtest.levscore.ladder import bootstrap
from src.backtest.optimize import HOLDOUT
from src.backtest.optimize.engine import Config, Pool, lever, run_book
from src.backtest.optimize.search import universes

TVS = (0.3, 0.5, 0.8)
PERIODS = {"2010-24": ("2010-01-01", "2024-12-31"), "2018-24": ("2018-01-01", "2024-12-31"),
           "holdout 2025-26": (HOLDOUT, "2030-01-01")}


def to_config(d: dict) -> Config:
    typed = {f.name: f.type for f in fields(Config)}
    kw = {}
    for f in fields(Config):
        v = d[f.name]
        kw[f.name] = int(v) if typed[f.name] in ("int", int) else float(v) if typed[f.name] in ("float", float) else str(v)
    return Config(**kw)


def intraweek_trough(daily: pd.DataFrame, idx: pd.DatetimeIndex, W: np.ndarray) -> np.ndarray:
    """Worst cumulative return of the week's held weights inside each week (buy and hold within the week)."""
    D = daily.to_numpy(float)
    days = daily.index.values
    out = np.zeros(len(idx))
    for i, t in enumerate(idx):
        w = W[i]
        if w.sum() <= 0:
            continue
        lo = np.searchsorted(days, np.datetime64(t - pd.Timedelta(days=7)), side="right")
        hi = np.searchsorted(days, np.datetime64(t), side="right")
        if hi <= lo:
            continue
        j = np.flatnonzero(w > 0)
        g = np.cumprod(1 + np.nan_to_num(D[lo:hi][:, j]), axis=0) - 1
        out[i] = min(0.0, float((g @ w[j]).min()))
    return out


def perf(r: pd.Series, cash: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    ex = r - cash.reindex(r.index).fillna(0.0)
    eq = (1 + r).cumprod()
    yrs = len(r) / 52
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0, "total": float(eq.iloc[-1] - 1),
            "sharpe": float(ex.mean() / ex.std() * np.sqrt(52)), "vol": float(r.std() * np.sqrt(52)),
            "max_dd": float((eq / eq.cummax() - 1).min()), "worst_week": float(r.min()), "weeks": len(r)}


def main() -> None:
    pool_dir, cash_path, sdir, odir, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    w = pd.read_parquet(os.path.join(pool_dir, "pool_weekly.parquet"))
    d = pd.read_parquet(os.path.join(pool_dir, "pool_daily.parquet"))
    m = pd.read_csv(os.path.join(pool_dir, "pool_meta.csv"), index_col=0)
    cash = pd.read_parquet(cash_path)["cash"]
    uni = universes(list(w.columns))
    pool = Pool(w, cash, m, uni)
    m_fut = m.copy()
    m_fut.loc[~m_fut["crypto"] & ~m_fut["unfunded"], "spread"] = 0.003
    pool_fut = Pool(w, cash, m_fut, uni)
    with open(os.path.join(odir, "chosen.json")) as fh:
        ch = json.load(fh)
    met = pd.read_csv(os.path.join(sdir, "metrics.csv"), index_col=0)
    cfgs = pd.read_csv(os.path.join(sdir, "configs.csv"), index_col=0)
    r80_search = pd.read_parquet(os.path.join(sdir, "returns_tv80.parquet"))
    top20 = met["sharpe_2010-24_tv80"].sort_values(ascending=False).index[:20]
    cands = {"A max-Sharpe": [ch["A_max_sharpe"]["id"]], "B consensus": None, "C top-20 average": list(top20), "X x80 book": ["c00000"]}
    cand_cfg = {"B consensus": to_config(ch["B_consensus"])}
    idx = pool.idx
    rows, series, boot = [], {}, []
    for name, ids in cands.items():
        members = [(cid, to_config(cfgs.loc[cid].to_dict())) for cid in ids] if ids else [("B", cand_cfg[name])]
        acc = {tv: [] for tv in TVS}
        acc_fut = {tv: [] for tv in TVS}
        for cid, cfg in members:
            bk = run_book(pool, cfg, keep_weights=True)
            bkf = run_book(pool_fut, cfg)
            tr = intraweek_trough(d, idx, bk["weights"])
            if cid in r80_search.columns:                                   # the pre-holdout weeks must be the search's
                pre = r80_search[cid].astype(float)
                nt, _ = lever(bk, pool.cash, 0.8, cfg.lev_band)
                gap = (pd.Series(nt, index=idx).reindex(pre.index) - pre).abs().max()
                assert gap < 1e-5, (cid, gap)
            for tv in TVS:
                acc[tv].append(pd.Series(lever(bk, pool.cash, tv, cfg.lev_band, trough=tr)[0], index=idx))
                acc_fut[tv].append(pd.Series(lever(bkf, pool_fut.cash, tv, cfg.lev_band, trough=tr)[0], index=idx))
        for tv in TVS:
            rs = pd.concat(acc[tv], axis=1).mean(axis=1, skipna=False)
            rf = pd.concat(acc_fut[tv], axis=1).mean(axis=1, skipna=False)
            series[f"{name} | tv{int(tv * 100)}"] = rs
            series[f"{name} | tv{int(tv * 100)} | futures financing"] = rf
            for lab, (a, z) in PERIODS.items():
                for fin_lab, s_ in (("x80 financing", rs), ("futures financing", rf)):
                    rows.append({"candidate": name, "target_vol": tv, "period": lab, "financing": fin_lab,
                                 **perf(s_[(s_.index >= a) & (s_.index <= z)], cash)})
            pre = rs[(rs.index >= "2010-01-01") & (rs.index < HOLDOUT)].dropna().to_numpy()
            boot.append({"candidate": name, "target_vol": tv, **bootstrap(pre)})
        print(name, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "holdout.csv"), index=False)
    pd.DataFrame(boot).to_csv(os.path.join(out, "bootstrap.csv"), index=False)
    pd.DataFrame(series).to_parquet(os.path.join(out, "candidate_series.parquet"))
    show = res[res["financing"] == "x80 financing"].pivot_table(index=["candidate", "target_vol"], columns="period",
                                                                 values=["cagr", "max_dd", "sharpe"])
    print(show.round(3).to_string())
    print(pd.DataFrame(boot).round(3).to_string())


if __name__ == "__main__":
    main()
