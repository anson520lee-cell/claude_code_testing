"""python -m src.backtest.x80.ls_neutral GRID_DIR PRICES OUT_CSV : market-neutral long/short on the point-in-time S&P 500.

Every 5 sessions, rank the eligible members on each signal (features of the previous session, so FINRA files and
filings are public before the trade at the close), go long the best 10% and short the worst 10%, equal weight.
Account return = long leg - short leg + T-bill on the short proceeds - 0.5%/yr borrow - 5 bp per side traded.
Dollar-neutral and beta-neutral (legs sized so the betas cancel) versions. Signals:
low 5-day short-volume ratio, low idiosyncratic vol, 1-week reversal, 12-1 momentum, low MAX, gross
profitability, low asset growth, low issuance, book/market, earnings/price, low accruals; an equal-weight
composite of all, and a composite of only the signals that worked in 2010-17 (read on 2018-26).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, kelly, load, stats

SIGNALS = {"low_svr5": ("svr5", -1), "low_ivol": ("ivol60", -1), "reversal_1w": ("ret5", -1), "mom12_1": ("mom12_1", 1),
           "low_max": ("max20", -1), "gross_prof": ("gpa", 1), "low_asset_growth": ("asset_growth", -1),
           "low_issuance": ("issuance", -1), "value_bm": ("bm", 1), "value_ep": ("ep", 1), "low_accruals": ("accruals", -1)}
HOLD, FRAC, COST, BORROW = 5, 0.10, 0.0005, 0.005
PERIODS = {"IS 2010-17": ("2010-01-01", "2017-12-31"), "OOS 2018-26": ("2018-01-01", "2030-01-01"), "2022-26": ("2022-01-01", "2030-01-01"),
           "all": ("2010-01-01", "2030-01-01")}


def leg_returns(rank: pd.DataFrame, r: pd.DataFrame, beta: pd.DataFrame, neutral: str) -> pd.Series:
    """Daily account return of the decile long/short on composite ``rank`` (dates x symbols, higher = long)."""
    dates = rank.index
    reb = dates[::HOLD]
    w_prev = pd.Series(dtype=float)
    out = []
    for i, d in enumerate(reb):
        x = rank.loc[d].dropna()
        if len(x) < 100:
            continue
        n = max(int(len(x) * FRAC), 10)
        longs, shorts = x.nlargest(n).index, x.nsmallest(n).index
        wl, ws = 1.0, 1.0
        if neutral == "beta":
            bl, bs = beta.loc[d, longs].mean(), beta.loc[d, shorts].mean()
            if np.isfinite(bl) and np.isfinite(bs) and bl > 0.2 and bs > 0.2:
                wl, ws = 2 * bs / (bl + bs), 2 * bl / (bl + bs)
        w = pd.concat([pd.Series(wl / n, index=longs), pd.Series(-ws / n, index=shorts)])
        turn = w.sub(w_prev, fill_value=0).abs().sum()
        w_prev = w
        end = reb[i + 1] if i + 1 < len(reb) else dates[-1] + pd.Timedelta(days=1)
        span = r.loc[(r.index > d) & (r.index <= end)]
        if span.empty:
            continue
        rr = span.reindex(columns=w.index).fillna(0.0)
        day = rr.mul(w, axis=1).sum(axis=1)
        day.iloc[0] -= turn * COST
        out.append(day)
    return pd.concat(out)


def main() -> None:
    grid_dir, prices_path, out_path = sys.argv[1:4]
    g = pd.read_parquet(f"{grid_dir}/grid.parquet", columns=["symbol", "date", "beta", "ivol60", "ret5", "mom12_1", "max20"])
    s = pd.read_parquet(f"{grid_dir}/grid_src.parquet", columns=["symbol", "date", "svr5", "gpa", "asset_growth", "issuance", "bm", "ep", "accruals"])
    g = g.merge(s, on=["symbol", "date"], how="left")
    p = pd.read_parquet(prices_path, columns=["symbol", "date", "adj_close"])
    px = p.pivot_table(index="date", columns="symbol", values="adj_close").sort_index()
    r = px.pct_change(fill_method=None).clip(-0.9, 3.0)
    feats = {}
    for col in {c for c, _ in SIGNALS.values()} | {"beta"}:
        m = g.pivot_table(index="date", columns="symbol", values=col).sort_index()
        feats[col] = m.shift(1)                                       # observed the session before the trade
    ranks = {name: feats[col].rank(axis=1, pct=True) * sign for name, (col, sign) in SIGNALS.items()}
    ranks = {k: v.add(0.5 if SIGNALS[k][1] < 0 else 0) for k, v in ranks.items()}          # map both signs to (0,1]
    mx = load()
    rows, series = [], {}
    for name, rk in ranks.items():
        for neutral in ("dollar", "beta"):
            series[(name, neutral)] = leg_returns(rk, r, feats["beta"], neutral)
    comp = pd.concat(ranks.values(), keys=ranks.keys()).groupby(level=1).mean()
    cnt = pd.concat([v.notna() for v in ranks.values()], keys=ranks.keys()).groupby(level=1).sum()
    comp = comp.where(cnt >= len(ranks) // 2).sort_index()
    for neutral in ("dollar", "beta"):
        series[("composite_all", neutral)] = leg_returns(comp, r, feats["beta"], neutral)
    for (name, neutral), ret in series.items():
        cash = cash_daily(mx, ret.index)
        ret = ret + cash - BORROW / 252                                 # rebate on the short proceeds, minus borrow
        series[(name, neutral)] = ret
        for lab, (a, b) in PERIODS.items():
            rr = ret[(ret.index >= a) & (ret.index <= b)]
            st = stats(rr, cash)
            if st:
                rows.append({"signal": name, "neutral": neutral, "period": lab, **st, **kelly(rr, cash)})
    t = pd.DataFrame(rows)
    good = t[(t["period"] == "IS 2010-17") & (t["neutral"] == "beta") & (t["sharpe"] > 0.3) & (~t["signal"].str.startswith("composite"))]["signal"].tolist()
    if good:
        sel = pd.concat([ranks[k] for k in good], keys=good).groupby(level=1).mean().sort_index()
        for neutral in ("dollar", "beta"):
            ret = leg_returns(sel, r, feats["beta"], neutral)
            cash = cash_daily(mx, ret.index)
            ret = ret + cash - BORROW / 252
            for lab, (a, b) in PERIODS.items():
                rr = ret[(ret.index >= a) & (ret.index <= b)]
                rows.append({"signal": "composite_IS_selected", "neutral": neutral, "period": lab, "members": "+".join(good), **stats(rr, cash), **kelly(rr, cash)})
            series[("composite_IS_selected", neutral)] = ret
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    pd.DataFrame({f"{a}|{b}": v for (a, b), v in series.items()}).to_parquet(out_path.replace(".csv", "_series.parquet"))
    pd.set_option("display.width", 250)
    y = t.pivot_table(index=["signal", "neutral"], columns="period", values=["cagr", "sharpe"])
    print(y.round(2).to_string())
    print("IS-selected members:", good)


if __name__ == "__main__":
    main()
