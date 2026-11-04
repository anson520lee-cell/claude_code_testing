"""python -m src.backtest.x80.vrp OUT_CSV : the volatility risk premium - short short-term VIX futures (SVXY-style).

Series: a synthetic "-1x short-term VIX futures, collateral in T-bills" account return:
* until 2018-02-27 SVXY itself (it was -1x then);
* from 2018-02-28 SVXY became -0.5x, so 2 x SVXY financed with one turn of margin at T-bill + 1.5%/yr
  (pays the 0.95% fee twice, the cost of really holding it that way).

Rules, decided on the close of day t (VIX / VIX3M closes) and held over day t+1 ("lag 1") or one day later
("lag 2", for anyone who cannot trade the 16:00 close with the 16:15 VIX print):
always short; short only when VIX/VIX3M < 1.00 / 0.95 / 0.90 (contango); short only when VIX exceeds 10-day
realised S&P volatility (a premium exists) and the curve is in contango. 5 bp per side on every change.
In-sample 2011-10..2017, out-of-sample 2018-26 (contains Feb 2018, Mar 2020, 2022, Aug 2024).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, load, stats

COST = 0.0005
MARGIN_SPREAD = 0.015
PERIODS = {"IS 2011-17": ("2011-10-01", "2017-12-31"), "OOS 2018-26": ("2018-01-01", "2030-01-01"), "2018": ("2018-01-01", "2018-12-31"),
           "2019-21": ("2019-01-01", "2021-12-31"), "2022-23": ("2022-01-01", "2023-12-31"), "2024-26": ("2024-01-01", "2030-01-01"),
           "2022-26": ("2022-01-01", "2030-01-01")}


def short_vol_series(px: dict) -> tuple[pd.Series, pd.Series]:
    r = px["SVXY"]["adj_close"].pct_change().dropna()
    cash = cash_daily(px, r.index)
    late = r.index >= "2018-02-28"
    s = r.copy()
    s[late] = 2 * r[late] - (cash[late] + MARGIN_SPREAD / 252)
    return s, cash


def account(s: pd.Series, cash: pd.Series, expo: pd.Series, lag: int = 1) -> pd.Series:
    e = expo.shift(lag).reindex(s.index).fillna(0.0)
    turn = e.diff().abs().fillna(e.abs())
    ret = cash + e * (s - cash) - (e - 1).clip(lower=0) * MARGIN_SPREAD / 252 - turn * COST
    ret = ret.clip(lower=-1.0)
    dead = (ret <= -1.0).cumsum().shift(1).fillna(0) > 0
    return ret.mask(dead, 0.0)


def main() -> None:
    out_path = sys.argv[1]
    px = load()
    s, cash = short_vol_series(px)
    vix, vix3m = px["^VIX"]["close"], px["^VIX3M"]["close"]
    spx = px["^GSPC"]["close"].pct_change()
    rv10 = spx.rolling(10).std() * np.sqrt(252) * 100
    ts = (vix / vix3m).reindex(s.index)
    one = pd.Series(1.0, index=s.index)
    rules = {"always": one, "ts<1.00": (ts < 1.0).astype(float), "ts<0.95": (ts < 0.95).astype(float),
             "ts<0.90": (ts < 0.90).astype(float),
             "vrp>0 & ts<1": ((vix - rv10).reindex(s.index) > 0).astype(float) * (ts < 1.0).astype(float)}
    rows = []
    for name, ex in rules.items():
        for lag in (1, 2):
            for lev in (0.5, 1.0, 1.5, 2.0):
                ret = account(s, cash, ex * lev, lag)
                for lab, (a, b) in PERIODS.items():
                    rr = ret[(ret.index >= a) & (ret.index <= b)]
                    st = stats(rr, cash)
                    if st:
                        rows.append({"rule": name, "lag": lag, "lev": lev, "period": lab, "time_in": float(ex.mean()), **st})
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    pd.set_option("display.width", 250)
    y = t[t["lev"].isin([1.0, 2.0])].pivot_table(index=["rule", "lag", "lev"], columns="period", values="cagr")
    print("CAGR %\n", (y * 100).round(0).to_string())
    y = t[t["lev"].isin([1.0, 2.0])].pivot_table(index=["rule", "lag", "lev"], columns="period", values="max_dd")
    print("max DD %\n", (y * 100).round(0).to_string())
    worst = account(s, cash, rules["ts<1.00"], 1)
    print("worst days ts<1 lag1:\n", worst.sort_values().head(8).round(3).to_string())


if __name__ == "__main__":
    main()
