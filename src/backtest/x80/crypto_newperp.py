"""python -m src.backtest.x80.crypto_newperp PERP_1D FUNDING OUT_CSV : short every new Binance perpetual for its first weeks.

New coins often slide after they list (airdrop and unlock selling). Every USDT-M perpetual Binance ever listed is in the
data (delisted ones too); the listing day is its first daily kline. Short each one from day k1 to day k2 after its
listing (decided at the previous close), 10% of equity per coin at most (so the book is fully short only with 10 or more
coins in their window), rebalanced daily; optionally hedged with a long BTC perpetual of the same size. Shorts receive
funding when it is positive and pay it when negative (new perps often have deeply negative funding); 0.15% of notional
per side; the margin earns nothing; a day that loses the equity ends the account. Coins must have traded >= $2M the day
before. k1 in (1, 3, 7), k2 in (14, 30, 60), hedged or not: 18 rules, picked on 2020-22, read on 2023-26.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import (
    cash_daily,
    deflated_sharpe,
    funding_daily,
    load,
    stats,
)
from src.backtest.x80.crypto_factors import simulate

COST = 0.0015
PPY = 365
CAP = 0.10
PERIODS = {"IS 2020-22": ("2020-01-01", "2022-12-31"), "OOS 2023-26": ("2023-01-01", "2030-01-01"), "2023-24": ("2023-01-01", "2024-12-31"),
           "2025-26": ("2025-01-01", "2030-01-01"), "all": ("2020-01-01", "2030-01-01")}


def main() -> None:
    k_path, f_path, out_path = sys.argv[1:4]
    k = pd.read_parquet(k_path)
    close = k.pivot_table(index="date", columns="symbol", values="close").sort_index().asfreq("D")
    qv = k.pivot_table(index="date", columns="symbol", values="quote_volume").sort_index().reindex(close.index)
    fund = funding_daily(pd.read_parquet(f_path)).reindex(index=close.index, columns=close.columns)
    r = close.pct_change(fill_method=None).where(close.shift(1).notna() & close.notna())
    first = close.apply(lambda s: s.first_valid_index())
    first = first.where(first > close.index[0])                            # already trading when the data start: not new
    age = pd.DataFrame(np.subtract.outer(close.index.values, first.values).astype("timedelta64[D]").astype(float),
                       index=close.index, columns=close.columns)
    idx = close.index
    start = np.where(idx >= "2020-01-01")[0][0]
    reb = np.arange(start, len(idx))
    R, FU = r.fillna(0.0).values, fund.fillna(0.0).values
    jb = list(close.columns).index("BTCUSDT")
    px = load()
    cash = cash_daily(px, idx, PPY)
    print("perpetuals listed after 2020-01-01:", int(first.notna().sum()), "| by year:", first.dropna().dt.year.value_counts().sort_index().to_dict())
    rows, series, turns = [], {}, {}
    for k1 in (1, 3, 7):
        for k2 in (14, 30, 60):
            act = (age >= k1) & (age <= k2) & (qv.shift(0) >= 2e6) & close.notna()
            act["BTCUSDT"] = False
            A = act.values
            for hedge in (False, True):
                W = np.zeros((len(reb), close.shape[1]))
                for i, s in enumerate(reb):
                    a = A[s - 1]                                           # known at the previous close
                    n = int(a.sum())
                    if n == 0:
                        continue
                    w = min(CAP, 1.0 / n)
                    W[i, a] = -w
                    if hedge:
                        W[i, jb] = w * n
                tl: list = []
                d = simulate(W, reb, R, FU, cost=COST, turnover=tl)
                name = f"short day {k1}-{k2}{', BTC hedge' if hedge else ''}"
                ret = pd.Series(d, index=idx).iloc[start:]
                series[name] = ret
                turns[name] = float(np.mean(tl))
                for lab, (a_, b_) in PERIODS.items():
                    st = stats(ret[(ret.index >= a_) & (ret.index <= b_)], cash, PPY)
                    if st:
                        rows.append({"rule": name, "period": lab, **st})
    t = pd.DataFrame(rows)
    is_ = t[t["period"] == "IS 2020-22"].set_index("rule")["sharpe"]
    pick = is_.idxmax()
    o = t[(t["period"] == "OOS 2023-26")].set_index("rule")
    x = series[pick][series[pick].index <= "2022-12-31"].dropna()
    dsr = deflated_sharpe(is_.max() / np.sqrt(PPY), None, len(is_), len(x), float(x.skew()), float(x.kurt() + 3))
    print(f"picked on 2020-22: {pick} (Sharpe {is_.max():.2f}, deflated-Sharpe probability {dsr:.3f}); "
          f"2023-26: Sharpe {o.loc[pick, 'sharpe']:.2f}, CAGR {o.loc[pick, 'cagr'] * 100:.0f}%, max DD {o.loc[pick, 'max_dd'] * 100:.0f}%")
    t.to_csv(out_path, index=False)
    pd.DataFrame(series).to_parquet(out_path.replace(".csv", "_series.parquet"))
    with open(out_path.replace(".csv", "_summary.json"), "w") as fh:
        json.dump({"pick": pick, "dsr_lenient": dsr, "turnover": turns}, fh, indent=1)
    pd.set_option("display.width", 250)
    print("Sharpe\n", t.pivot_table(index="rule", columns="period", values="sharpe")[list(PERIODS)].round(2).to_string())
    print("CAGR %\n", (t.pivot_table(index="rule", columns="period", values="cagr")[list(PERIODS)] * 100).round(0).to_string())
    print("max DD %\n", (t.pivot_table(index="rule", columns="period", values="max_dd")[list(PERIODS)] * 100).round(0).to_string())


if __name__ == "__main__":
    main()
