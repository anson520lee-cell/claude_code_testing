"""python -m src.backtest.x80.kelly_table : realised Sharpe and the Kelly ceiling of every asset / strategy."""
from __future__ import annotations

import json
import sys

import pandas as pd

from src.backtest.x80.common import cash_daily, kelly, load, required_sharpe, stats


def main() -> None:
    out_path = sys.argv[1]
    px = load()
    rows = []
    for s in ["SPY", "QQQ", "^GSPC", "^NDX", "TQQQ", "UPRO", "SOXL", "TECL", "SVXY", "BTC-USD", "ETH-USD", "SOL-USD", "GLD", "TLT"]:
        x = px[s]
        col = "adj_close" if s[0] != "^" else "close"
        r = x[col].pct_change().dropna()
        ppy = 365 if s.endswith("-USD") else 252
        cash = cash_daily(px, r.index, ppy)
        for lab, (a, b) in {"all": ("1900", "2100"), "2010-17": ("2010-01-01", "2017-12-31"), "2018-26": ("2018-01-01", "2026-12-31"),
                            "2022-26": ("2022-01-01", "2026-12-31")}.items():
            rr = r[(r.index >= a) & (r.index <= b)]
            if len(rr) < 200:
                continue
            rows.append({"asset": s, "period": lab, **stats(rr, cash, ppy), **kelly(rr, cash, ppy)})
    # the account curves of the live strategies (edge hunt) - monthly-booked P&L, so use daily changes of the curve
    for name, path, cols in (("SP500 PIT", "/home/user/research_data/v2/acct/account_curves.parquet", ["live_like_only", "live_like_filter"]),
                             ("SP400/600", "/home/user/research_data/rep/acct/account_curves.parquet", ["live_like_only", "live_like_filter"]),
                             ("SP500 PIT", "/home/user/research_data/v2/acct/account_setups_curves.parquet", ["crash_squeeze", "crash_squeeze_filter"]),
                             ("SP400/600", "/home/user/research_data/rep/acct/account_setups_curves.parquet", ["crash_squeeze", "crash_squeeze_filter"])):
        c = pd.read_parquet(path)
        for col in cols:
            r = c[col].dropna().pct_change().dropna()
            cash = cash_daily(px, r.index)
            rows.append({"asset": f"{name} {col}", "period": "2010-26", **stats(r, cash), **kelly(r, cash)})
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(t[["asset", "period", "cagr", "vol", "sharpe", "max_dd", "kelly_leverage", "kelly_growth", "half_kelly_growth"]].round(3).to_string())
    print(json.dumps({f"S needed for 80% at Kelly fraction {f}": round(required_sharpe(0.8, 0.02, f), 2) for f in (1.0, 0.5)}))


if __name__ == "__main__":
    main()
