"""Trade-journal check: turns a chronological fills CSV into per-position results and discipline metrics.

    python backtest/journal.py fills.csv --equity 9000

fills.csv columns: symbol,side,qty,price   (side = B or S, oldest first; filled quantity only).
Keep the file private (the repo is public) - e.g. in backtest/private/, which is git-ignored.

A "campaign" is the time from the first buy of a symbol until the position is flat again. Sells that
have no known opening buy in the file are reported separately and excluded from the P&L.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import pandas as pd


def campaigns(fills: pd.DataFrame) -> pd.DataFrame:
    pos, cost = defaultdict(float), defaultdict(float)
    cur: dict[str, dict] = {}
    done, orphan = [], defaultdict(float)
    for r in fills.itertuples():
        s, q, p = r.symbol, float(r.qty), float(r.price)
        if r.side == "B":
            c = cur.setdefault(s, dict(symbol=s, pnl=0.0, peak_cost=0.0, buys=0, bought=0.0))
            pos[s] += q; cost[s] += q * p; c["buys"] += 1; c["bought"] += q * p
            c["peak_cost"] = max(c["peak_cost"], cost[s])
        else:
            if s not in cur or pos[s] <= 0:
                orphan[s] += q; continue
            k = min(q, pos[s]); avg = cost[s] / pos[s]
            cur[s]["pnl"] += k * (p - avg); cost[s] -= k * avg; pos[s] -= k
            orphan[s] += q - k
            if pos[s] <= 1e-9:
                done.append(cur.pop(s)); pos[s] = cost[s] = 0.0
    for s, c in cur.items():
        c["open"] = True; done.append(c)
    df = pd.DataFrame(done)
    df["ret"] = df["pnl"] / df["bought"]
    return df, dict(orphan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fills"); ap.add_argument("--equity", type=float, required=True, help="account equity in USD")
    a = ap.parse_args()
    df, orphan = campaigns(pd.read_csv(a.fills))
    closed = df[df.get("open").isna()] if "open" in df else df
    w, l = closed[closed.pnl > 0], closed[closed.pnl <= 0]
    print(f"closed campaigns: {len(closed)}   winners: {len(w)} ({len(w)/max(len(closed),1):.0%})")
    print(f"net P&L: {closed.pnl.sum():+.0f}   gross win: {w.pnl.sum():+.0f}   gross loss: {l.pnl.sum():+.0f}")
    print(f"avg win: {w.ret.mean():+.1%}   avg loss: {l.ret.mean():+.1%}   payoff ratio: "
          f"{(w.pnl.mean() / -l.pnl.mean()) if len(w) and len(l) else float('nan'):.2f}")
    if len(w) and len(l):
        wr = len(w) / len(closed)
        print(f"expectancy per campaign: {closed.pnl.mean():+.0f}  (break-even win rate at this payoff: "
              f"{1/(1+w.pnl.mean()/-l.pnl.mean()):.0%}, actual {wr:.0%})")
    big = closed.sort_values("pnl")
    print(f"3 largest losses = {big.pnl.head(3).sum():+.0f}; largest win = {big.pnl.iloc[-1]:+.0f}")
    print(f"largest position cost vs equity: {closed.peak_cost.max()/a.equity:.0%}; "
          f"median {closed.peak_cost.median()/a.equity:.0%}; campaigns >100% of equity: {(closed.peak_cost > a.equity).sum()}")
    print(f"campaigns built with >=3 buys (averaging in): {(closed.buys >= 3).sum()}; of those, losers: "
          f"{((closed.buys >= 3) & (closed.pnl <= 0)).sum()}")
    print("\n" + closed.sort_values("pnl")[["symbol", "pnl", "ret", "peak_cost", "buys"]]
          .assign(ret=lambda d: d.ret.map("{:+.1%}".format), pnl=lambda d: d.pnl.round(0), peak_cost=lambda d: d.peak_cost.round(0))
          .to_string(index=False))
    orphan = {k: v for k, v in orphan.items() if v > 0}
    if orphan:
        print("\nsells without a known opening buy (excluded):", orphan)
    if "open" in df:
        print("still open:", ", ".join(df[df.open == True].symbol))  # noqa: E712


if __name__ == "__main__":
    main()
