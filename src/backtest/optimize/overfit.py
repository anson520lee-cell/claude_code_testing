"""python -m src.backtest.optimize.overfit SEARCH_DIR CASH_WEEKLY OUT_DIR

How much of the search's best result is real? Everything here uses the search output only (data before the holdout).

  walk_forward   each January 2013..2024 re-run the optimiser on 2010..last December and hold its pick for the year:
                 best score so far / best of the last 3 years / average of the top 20 / best CAGR so far; against the
                 x80 book and the average of all configurations. Also the pick's percentile among all configurations in
                 the year it is held (50% = picking had no skill).
  cscv           probability of backtest overfitting (Bailey, Borwein, Lopez de Prado & Zhu 2017): 16 blocks of
                 2010-24, every 8/8 split; PBO = share of splits in which the in-sample winner ranks in the bottom half
                 out of sample. Also the out-of-sample Sharpe of the in-sample winner.
  deflated       deflated Sharpe ratio of the best configuration and of its excess over x80, for this search's count
                 of configurations and for that count plus the ~1,500 versions of the two earlier x80 rounds.
  one_step       x80 with one setting changed: Sharpe in 2010-17 and 2018-24 and the Newey-West t of the difference.
  dimensions     median score of the random configurations by option of each dimension, in both halves.
  consensus      the configuration built from each dimension's best median option (candidate B).
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from dataclasses import fields

import numpy as np
import pandas as pd

from src.backtest.levscore.study import hac_t
from src.backtest.optimize.engine import BASELINE, Config
from src.backtest.optimize.checkpoint import require_complete
from src.backtest.optimize.search import OPTIONS
from src.backtest.x80.common import deflated_sharpe

START, END = "2010-01-01", "2024-12-31"
WF_YEARS = range(2013, 2025)
PRIOR_TRIALS = 1500                                                     # x80 rounds 1 + 2 (~1,470 versions)


def sharpe(x: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.nanmean(x, axis=axis)
    s = np.nanstd(x, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return m / s * np.sqrt(52)


def perf(r: pd.Series, cash: pd.Series) -> dict:
    r = r.dropna()
    ex = r - cash.reindex(r.index).fillna(0.0)
    eq = (1 + r).cumprod()
    yrs = len(r) / 52
    return {"sharpe": float(ex.mean() / ex.std() * np.sqrt(52)), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0,
            "max_dd": float((eq / np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:] - 1).min()), "weeks": len(r)}


def walk_forward(R: pd.DataFrame, cash: pd.Series, base_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ex = R.sub(cash.reindex(R.index).fillna(0.0), axis=0)
    rules = {"best score so far": ("exp", "sharpe", 1), "best score, last 3 years": ("3y", "sharpe", 1),
             "top-20 average so far": ("exp", "sharpe", 20), "best CAGR so far": ("exp", "cagr", 1)}
    picks, track = [], {k: [] for k in rules}
    for y in WF_YEARS:
        hold = (R.index >= f"{y}-01-01") & (R.index <= f"{y}-12-31")
        oos_sr = pd.Series(sharpe(ex.loc[hold].to_numpy()), index=R.columns)
        for name, (win, crit, k) in rules.items():
            a = START if win == "exp" else f"{y - 3}-01-01"
            ins = (R.index >= a) & (R.index < f"{y}-01-01")
            if crit == "sharpe":
                score = pd.Series(sharpe(ex.loc[ins].to_numpy()), index=R.columns)
            else:
                eq = (1 + R.loc[ins].fillna(0.0)).prod()
                score = eq ** (52 / ins.sum()) - 1
            top = score.sort_values(ascending=False).index[:k]
            track[name].append(R.loc[hold, top].mean(axis=1))
            pct = float((oos_sr < oos_sr[top].mean()).mean()) if k > 1 else float((oos_sr < oos_sr[top[0]]).mean())
            picks.append({"year": y, "rule": name, "pick": ",".join(top[:3]), "in_sample_score": float(score[top].mean()),
                          "held_year_sharpe": float(oos_sr[top].mean()), "held_year_percentile": pct,
                          "x80_held_year_sharpe": float(oos_sr[base_id])})
    rows = []
    series = {k: pd.concat(v) for k, v in track.items()}
    series["x80 book (fixed)"] = R.loc[(R.index >= "2013-01-01") & (R.index <= END), base_id]
    series["average of all configurations"] = R.loc[(R.index >= "2013-01-01") & (R.index <= END)].mean(axis=1)
    for name, s in series.items():
        for lab, (a, z) in {"2013-24": ("2013-01-01", END), "2018-24": ("2018-01-01", END)}.items():
            rows.append({"rule": name, "period": lab, **perf(s[(s.index >= a) & (s.index <= z)], cash)})
    pk = pd.DataFrame(picks)
    summ = pk.groupby("rule")[["held_year_percentile"]].mean()
    return pd.DataFrame(rows), pk, summ


def cscv(R: pd.DataFrame, cash: pd.Series, n_blocks: int = 16, chunk: int = 500) -> dict:
    sel = (R.index >= START) & (R.index <= END)
    X = R.loc[sel].sub(cash.reindex(R.index[sel]).fillna(0.0), axis=0).fillna(0.0).to_numpy(np.float64)
    T, N = X.shape
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    s1 = np.array([X[a:b].sum(0) for a, b in itertools.pairwise(edges)])
    s2 = np.array([(X[a:b] ** 2).sum(0) for a, b in itertools.pairwise(edges)])
    cnt = np.diff(edges).astype(float)
    combos = np.array([[1.0 if b in c else 0.0 for b in range(n_blocks)] for c in itertools.combinations(range(n_blocks), n_blocks // 2)])
    lam, is_best, oos_best, oos_median = [], [], [], []
    for k in range(0, len(combos), chunk):
        C = combos[k:k + chunk]
        out = []
        for M in (C, 1 - C):
            n = M @ cnt
            m = (M @ s1) / n[:, None]
            v = (M @ s2) / n[:, None] - m ** 2
            out.append(m / np.sqrt(np.maximum(v, 1e-18)) * np.sqrt(52))
        sr_is, sr_oos = out
        star = np.argmax(sr_is, axis=1)
        chosen = sr_oos[np.arange(len(C)), star]
        rank = (sr_oos < chosen[:, None]).sum(axis=1) + 1
        w = rank / (N + 1)
        lam.append(np.log(w / (1 - w)))
        is_best.append(sr_is[np.arange(len(C)), star])
        oos_best.append(chosen)
        oos_median.append(np.median(sr_oos, axis=1))
    lam, is_best, oos_best, oos_median = (np.concatenate(a) for a in (lam, is_best, oos_best, oos_median))
    slope = float(np.polyfit(is_best, oos_best, 1)[0])
    return {"splits": len(lam), "configs": N, "pbo": float((lam <= 0).mean()), "median_logit": float(np.median(lam)),
            "is_winner_sharpe_median": float(np.median(is_best)), "oos_winner_sharpe_median": float(np.median(oos_best)),
            "oos_all_median_sharpe": float(np.median(oos_median)), "p_oos_winner_below_zero": float((oos_best < 0).mean()),
            "degradation_slope": slope}


def effective_trials(R: pd.DataFrame, cash: pd.Series, max_cols: int = 3000, seed: int = 0) -> float:
    """Effective number of independent configurations: (sum of eigenvalues)^2 / sum of squared eigenvalues of the
    correlation matrix of their weekly returns (a random subset of up to ``max_cols`` columns, scaled up)."""
    sel = (R.index >= START) & (R.index <= END)
    X = R.loc[sel].sub(cash.reindex(R.index[sel]).fillna(0.0), axis=0).fillna(0.0)
    cols = X.columns
    if len(cols) > max_cols:
        cols = np.random.default_rng(seed).choice(cols, max_cols, replace=False)
    C = np.corrcoef(X[cols].to_numpy().T)
    C = np.nan_to_num(C)
    ev = np.linalg.eigvalsh(C)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def deflated(R: pd.DataFrame, cash: pd.Series, best_id: str, base_id: str, n_eff: float) -> pd.DataFrame:
    sel = (R.index >= START) & (R.index <= END)
    X = R.loc[sel].sub(cash.reindex(R.index[sel]).fillna(0.0), axis=0).fillna(0.0)
    sr_all = X.mean() / X.std()
    rows = []
    for lab, x, srs in (("best configuration", X[best_id], sr_all),
                        ("best minus x80", X[best_id] - X[base_id], (X.sub(X[base_id], axis=0).mean() / X.sub(X[base_id], axis=0).std()).drop(base_id))):
        sr = float(x.mean() / x.std())
        sk = float(((x - x.mean()) ** 3).mean() / x.std(ddof=0) ** 3)
        ku = float(((x - x.mean()) ** 4).mean() / x.std(ddof=0) ** 4)
        for trials_lab, n in (("this search", len(R.columns)), ("this search + earlier rounds", len(R.columns) + PRIOR_TRIALS),
                              ("effective (independent) trials", max(n_eff, 2))):
            rows.append({"series": lab, "trials": trials_lab, "n_trials": round(float(n), 1), "annual_sharpe": sr * math.sqrt(52),
                         "skew": sk, "kurtosis": ku, "weeks": len(x),
                         "dsr_cross_var": deflated_sharpe(sr, float(srs.var()), round(n), len(x), sk, ku),
                         "dsr_null_var": deflated_sharpe(sr, None, round(n), len(x), sk, ku)})
    return pd.DataFrame(rows)


def one_step(R: pd.DataFrame, cfg: pd.DataFrame, cash: pd.Series, base_id: str) -> pd.DataFrame:
    names = [f.name for f in fields(Config)]
    b = pd.Series({f.name: getattr(BASELINE, f.name) for f in fields(Config)})
    ex = R.sub(cash.reindex(R.index).fillna(0.0), axis=0)
    rows = []
    for cid, row in cfg.iterrows():
        diff = [n for n in names if str(row[n]) != str(b[n])]
        if len(diff) != 1:
            continue
        d = ex[cid] - ex[base_id]
        r = {"change": f"{diff[0]} = {row[diff[0]]}", "id": cid}
        for lab, (a, z) in {"2010-17": ("2010-01-01", "2017-12-31"), "2018-24": ("2018-01-01", END)}.items():
            m = (R.index >= a) & (R.index <= z)
            r[f"sharpe_{lab}"] = float(sharpe(ex.loc[m, cid].to_numpy()))
            r[f"x80_{lab}"] = float(sharpe(ex.loc[m, base_id].to_numpy()))
            r[f"diff_pp_{lab}"] = float(d[m].mean() * 52 * 100)
            r[f"t_{lab}"] = hac_t(d[m].dropna())[1]
        rows.append(r)
    return pd.DataFrame(rows)


def dimensions(met: pd.DataFrame, n_struct: int) -> tuple[pd.DataFrame, Config]:
    rnd = met.iloc[n_struct:]
    rows, best = [], {}
    dims = {"universe": None, **OPTIONS}
    for dim in dims:
        opts = rnd[dim].astype(str)
        if dim == "universe":
            opts = opts.where(~opts.str.startswith("subset"), "random subset")
        g = rnd.groupby(opts)
        t = pd.DataFrame({"n": g.size(), "median_2010-24": g["sharpe_2010-24_tv80"].median(),
                          "median_2010-17": g["sharpe_2010-17_tv80"].median(), "median_2018-24": g["sharpe_2018-24_tv80"].median(),
                          "share_beating_x80_2010-24": g["sharpe_2010-24_tv80"].apply(lambda s: (s > met.iloc[0]["sharpe_2010-24_tv80"]).mean())})
        t.insert(0, "dimension", dim)
        rows.append(t.reset_index(names="option"))
        top = t["median_2010-24"].idxmax()
        if dim == "universe" and top == "random subset":
            top = t.drop("random subset")["median_2010-24"].idxmax()
        best[dim] = top
    typed = {f.name: f.type for f in fields(Config)}
    kw = {}
    for k, v in best.items():
        kw[k] = int(v) if typed[k] in ("int", int) else float(v) if typed[k] in ("float", float) else v
    return pd.concat(rows, ignore_index=True), Config(**kw)


def main() -> None:
    sdir, cash_path, out = sys.argv[1:4]
    require_complete(sdir)
    os.makedirs(out, exist_ok=True)
    cash = pd.read_parquet(cash_path)["cash"]
    R = pd.read_parquet(os.path.join(sdir, "returns_tv80.parquet")).astype(np.float64)
    met = pd.read_csv(os.path.join(sdir, "metrics.csv"), index_col=0)
    cfg = pd.read_csv(os.path.join(sdir, "configs.csv"), index_col=0)
    base_id = "c00000"
    n_struct = 1 + sum(1 for c in cfg.index[1:60] if sum(str(cfg.loc[c, f.name]) != str(getattr(BASELINE, f.name)) for f in fields(Config)) == 1)
    best_id = met["sharpe_2010-24_tv80"].idxmax()
    print("configurations", len(R.columns), "structured", n_struct, "best", best_id, flush=True)

    wf, picks, pct = walk_forward(R, cash, base_id)
    wf.to_csv(os.path.join(out, "walk_forward.csv"), index=False)
    picks.to_csv(os.path.join(out, "walk_forward_picks.csv"), index=False)
    print(wf.round(3).to_string(), "\n", pct.round(3).to_string(), flush=True)

    cs = cscv(R, cash)
    print("CSCV", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in cs.items()}), flush=True)
    n_eff = effective_trials(R, cash)
    print("effective trials", round(n_eff, 1), flush=True)
    ds = deflated(R, cash, best_id, base_id, n_eff)
    ds.to_csv(os.path.join(out, "deflated.csv"), index=False)
    print(ds.round(3).to_string(), flush=True)

    os_ = one_step(R, cfg, cash, base_id)
    os_.to_csv(os.path.join(out, "one_step.csv"), index=False)
    print(os_.round(3).to_string(), flush=True)

    dims, cons = dimensions(met, n_struct)
    dims.to_csv(os.path.join(out, "dimensions.csv"), index=False)
    print(dims.round(3).to_string(), flush=True)
    chosen = {"A_max_sharpe": {"id": best_id, **{f.name: cfg.loc[best_id, f.name] for f in fields(Config)}},
              "B_consensus": {f.name: getattr(cons, f.name) for f in fields(Config)},
              "X_x80": {f.name: getattr(BASELINE, f.name) for f in fields(Config)},
              "cscv": cs, "effective_trials": n_eff, "n_configs": len(R.columns), "n_structured": n_struct}
    with open(os.path.join(out, "chosen.json"), "w") as fh:
        json.dump(chosen, fh, indent=1, default=str)
    print(json.dumps(chosen, indent=1, default=str))


if __name__ == "__main__":
    main()
