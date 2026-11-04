"""python -m src.backtest.levscore.ladder X80_OUT RESULT_DIR : how much leverage, if no score can time it?
Reads the book that study.py saved (RESULT_DIR/book.parquet) and the sleeves' cash series.

1. History: fixed vol targets 10%..100% through the x80 rule (target / the book's whole-history vol, cap 10x), with
   costs and the intraweek wipe-out check, per period, plus where the worst drawdown sat.
2. Spread of outcomes: a moving-block bootstrap (13-week blocks) of the 2010-26 book weeks into 2,000 eight-year
   paths at a FIXED leverage of target / the book's 2010-26 vol (a hindsight scale, not a strategy): median and
   10th-percentile CAGR, and the chance of a 30% / 50% / 80% drawdown and of losing everything.
3. Kelly: the growth-optimal leverage mean / variance of the book's net weekly excess return, per period.
4. The hindsight what-if of study.py again with no re-leveraging cost, to split forecast value from trading cost.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from src.backtest.levscore import book as bk
from src.backtest.levscore import study as sd
from src.backtest.x80 import meta

TARGETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
SEED = 11


def weekly_levered(base: pd.Series, fin: pd.DataFrame, cash_w: pd.Series, L: float, trough: pd.Series) -> pd.Series:
    """One week's account return at a constant leverage L, every week on its own (no ruin carried forward)."""
    lev = pd.Series(L, index=base.index).where(base.notna())
    c = cash_w.reindex(base.index).fillna(0.0)
    ret = lev * base + c + lev * fin["unfunded"] * c - (lev * fin["funded"] - 1).clip(lower=0) * fin["spread"] / 52
    ret = ret - meta.trading_cost(lev.fillna(0.0), base.fillna(0.0), fin, ret.fillna(0.0))
    ret = ret.mask((1 + lev * trough.reindex(base.index).fillna(0.0)) <= 0, -1.0)
    return ret.dropna().clip(lower=-1.0)


def bootstrap(R: np.ndarray, years: int = 8, n: int = 2000, block: int = 13, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    T = years * 52
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, len(R) - block + 1, size=(n, nb))
    pos = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    eq = np.cumprod(1 + R[pos], axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n, 1)), eq], axis=1), axis=1)[:, 1:]
    dd = (eq / peak - 1).min(axis=1)
    cagr = np.where(eq[:, -1] > 0, eq[:, -1] ** (1 / years) - 1, -1.0)
    return {"cagr_median": float(np.median(cagr)), "cagr_p10": float(np.quantile(cagr, 0.1)), "cagr_p90": float(np.quantile(cagr, 0.9)),
            "maxdd_median": float(np.median(dd)), "p_dd30": float((dd <= -0.3).mean()), "p_dd50": float((dd <= -0.5).mean()),
            "p_dd80": float((dd <= -0.8).mean()), "p_ruin": float((eq[:, -1] <= 0).mean()), "p_loss_8y": float((eq[:, -1] < 1).mean())}


def main() -> None:
    x80_out, res = sys.argv[1:3]
    bkf = pd.read_parquet(os.path.join(res, "book.parquet"))
    base, fin, trough = bkf["base"], bkf[["funded", "spread", "unfunded", "turn"]], bkf["trough"]
    cash_w = pd.read_parquet(os.path.join(x80_out, "cash_weekly.parquet"))["cash"]

    hist = []
    for tg in TARGETS:
        lev = bk.vol_leverage(base, tg)
        r = bk.lever_path(base, fin, cash_w, lev, trough=trough)
        for lab, (a, b) in sd.PERIODS.items():
            rr = r[(r.index >= a) & (r.index <= b)]
            p = bk.perf(rr, cash_w)
            eq = (1 + rr).cumprod()
            dd = eq / eq.cummax() - 1
            low = dd.idxmin()
            hist.append({"target": tg, "period": lab, "avg_lev": float(lev[(lev.index >= a) & (lev.index <= b)].mean()), **p,
                         "dd_peak": str(eq[:low].idxmax().date()), "dd_trough": str(low.date())})
    hist = pd.DataFrame(hist)
    hist.to_csv(os.path.join(res, "ladder_history.csv"), index=False)

    sel = (base.index >= "2010-01-01") & base.notna()
    vol = float(base[sel].std() * np.sqrt(52))
    boot = []
    for tg in TARGETS:
        L = min(tg / vol, bk.CAP)
        R = weekly_levered(base, fin, cash_w, L, trough)
        R = R[R.index >= "2010-01-01"].to_numpy()
        boot.append({"target": tg, "leverage": L, **bootstrap(R)})
    boot = pd.DataFrame(boot)
    boot.to_csv(os.path.join(res, "ladder_bootstrap.csv"), index=False)

    c = cash_w.reindex(base.index).fillna(0.0)
    net = base - fin["funded"] * fin["spread"] / 52 + fin["unfunded"] * c          # per unit of leverage, above T-bills
    kel = []
    for lab, (a, b) in sd.PERIODS.items():
        x = net[(net.index >= a) & (net.index <= b)].dropna()
        k = float(x.mean() / x.var())
        kel.append({"period": lab, "ann_excess_per_1x": float(x.mean() * 52), "ann_vol_per_1x": float(x.std() * np.sqrt(52)),
                    "sharpe": float(x.mean() / x.std() * np.sqrt(52)), "kelly_leverage": k, "kelly_vol": float(k * x.std() * np.sqrt(52)),
                    "half_kelly_leverage": k / 2})
    kel = pd.DataFrame(kel)
    kel.to_csv(os.path.join(res, "ladder_kelly.csv"), index=False)

    old = meta.META_COST
    meta.META_COST = 0.0
    try:
        wi = sd.whatif(base, fin, cash_w, bkf[["trough"]], 0.3, "2018-01-01", "2030-01-01", rhos=(0.03, 0.05, 0.1, 0.2))
    finally:
        meta.META_COST = old
    pd.DataFrame(wi).to_csv(os.path.join(res, "whatif_nocost.csv"), index=False)

    pd.set_option("display.width", 250)
    print(hist[["target", "period", "avg_lev", "cagr", "max_dd", "worst_year", "sharpe", "dd_peak", "dd_trough"]].round(3).to_string(index=False))
    print(boot.round(3).to_string(index=False))
    print(kel.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
