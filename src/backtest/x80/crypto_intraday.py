"""python -m src.backtest.x80.crypto_intraday PERP_1H FUNDING OUT_CSV : hour-by-hour rules on BTC / ETH / SOL perpetuals.

Binance USDT-M 1-hour closes. A position decided at the close of hour h is held over hour h+1 (no look-ahead); it pays
funding when it is open at a funding time (00/08/16 UTC, later every 4h for some coins) and 0.06% of notional per side
(taker fee 0.05% + slippage; 0.02% maker as a check). Positions are -1 / 0 / +1 x equity per coin; the margin earns
nothing. 2020-22 is the early period every choice is made on; 2023-26 is out of sample.
A. Time of day: per coin, the hours whose 2020-22 mean return has t > 1.5 (long only), or long the best 4 / short the
   worst 4 hours. B. Momentum: sign of the past N-hour return (N = 1..48), held H hours (decided every H hours), long
   and short. C. Reversal: after an hour that moves more than 2.5 trailing (30-day) standard deviations, take the other
   side for H hours. D. Day of week: long on the weekdays whose 2020-22 mean return has t > 1.5.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import deflated_sharpe

COST = 0.0006
IS_END = "2022-12-31 23:00"
PERIODS = {"IS 2020-22": ("2020-01-01", IS_END), "OOS 2023-26": ("2023-01-01", "2030-01-01"), "2023-24": ("2023-01-01", "2024-12-31 23:00"),
           "2025-26": ("2025-01-01", "2030-01-01")}


def load_coin(k: pd.DataFrame, f: pd.DataFrame, sym: str) -> pd.DataFrame:
    x = k[k["symbol"] == sym].set_index("time").sort_index()
    idx = pd.date_range(x.index.min(), x.index.max(), freq="h")
    close = x["close"].reindex(idx).ffill()
    fr = f[f["symbol"] == sym].copy()
    fr["h"] = pd.to_datetime(fr["time"]).dt.floor("h")
    fund = fr.groupby("h")["rate"].sum().reindex(idx).fillna(0.0)
    return pd.DataFrame({"close": close, "r": close.pct_change().fillna(0.0), "fund": fund})


def book(d: pd.DataFrame, pos: np.ndarray, cost: float) -> pd.Series:
    """pos[h] decided at the close of hour h, earns hour h+1 (and pays the funding stamped at its start)."""
    p = np.concatenate([[0.0], pos[:-1]])
    turn = np.abs(np.diff(np.concatenate([[0.0], p])))
    return pd.Series(p * d["r"].values - p * d["fund"].values - turn * cost, index=d.index)


def hourly_stats(ret: pd.Series) -> dict:
    """Annualised from hourly returns, compounded on daily sums (the account is marked daily)."""
    day = ret.resample("D").sum()
    if len(day) < 60:
        return {}
    eq = (1 + day).cumprod()
    yrs = len(day) / 365
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0, "vol": float(day.std() * np.sqrt(365)),
            "sharpe": float(day.mean() / day.std() * np.sqrt(365)) if day.std() > 0 else np.nan,
            "max_dd": float((eq / eq.cummax() - 1).min()), "days": len(day),
            "exposure": float((ret != 0).mean())}


def rules(d: pd.DataFrame) -> dict[str, np.ndarray]:
    r = d["r"]
    n = len(d)
    is_ = d.index <= IS_END
    hour = d.index.hour.values
    out = {}
    # A. time of day (statistics from 2020-22 only); the position for hour h+1 is set at the close of hour h
    g = r[is_].groupby(r[is_].index.hour)
    t = g.mean() / g.std() * np.sqrt(g.count())
    nxt = (hour + 1) % 24
    out["time of day: long hours with t > 1.5"] = np.where(t.reindex(nxt).values > 1.5, 1.0, 0.0)
    best, worst = set(t.nlargest(4).index), set(t.nsmallest(4).index)
    out["time of day: long best 4 / short worst 4 hours"] = np.where(np.isin(nxt, list(best)), 1.0, np.where(np.isin(nxt, list(worst)), -1.0, 0.0))
    # D. day of week (the hour that starts a day decides for the whole day)
    dd = r[is_].groupby(r[is_].index.dayofweek)
    td = dd.mean() / dd.std() * np.sqrt(dd.count())
    nxt_dow = ((d.index + pd.Timedelta(hours=1)).dayofweek).values
    out["day of week: long days with t > 1.5"] = np.where(td.reindex(nxt_dow).values > 1.5, 1.0, 0.0)
    # B. momentum
    lc = np.log(d["close"].values)
    for N in (1, 3, 6, 12, 24, 48):
        past = np.full(n, np.nan)
        past[N:] = lc[N:] - lc[:-N]
        for H in (1, 4, 12, 24):
            pos = np.zeros(n)
            sgn = np.sign(np.nan_to_num(past))
            for s in range(0, n, H):
                pos[s:s + H] = sgn[s]
            out[f"momentum {N}h, hold {H}h"] = pos
    # C. reversal after a big hour
    sd = r.rolling(24 * 30, min_periods=24 * 10).std().shift(1).values
    z = r.values / sd
    for H in (1, 4, 12):
        pos = np.zeros(n)
        for i in np.where(np.abs(np.nan_to_num(z)) > 2.5)[0]:
            pos[i:i + H] = -np.sign(z[i])
        out[f"reversal after 2.5-sd hour, hold {H}h"] = pos
    return out


def main() -> None:
    k_path, f_path, out_path = sys.argv[1:4]
    k = pd.read_parquet(k_path)
    f = pd.read_parquet(f_path)
    rows, daily = [], {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        d = load_coin(k, f, sym)
        bh = book(d, np.ones(len(d)), 0.0)
        for lab, (a, b) in PERIODS.items():
            st = hourly_stats(bh[(bh.index >= a) & (bh.index <= b)])
            if st:
                rows.append({"coin": sym[:-4], "rule": "buy and hold", "cost": 0.0, "period": lab, **st})
        for name, pos in rules(d).items():
            for cost in (COST, 0.0002):
                ret = book(d, pos, cost)
                if cost == COST:
                    daily[f"{sym[:-4]} | {name}"] = ret.resample("D").sum()
                for lab, (a, b) in PERIODS.items():
                    st = hourly_stats(ret[(ret.index >= a) & (ret.index <= b)])
                    if st:
                        rows.append({"coin": sym[:-4], "rule": name, "cost": cost, "period": lab, **st})
    t = pd.DataFrame(rows)
    base = t[(t["cost"] == COST) & (t["rule"] != "buy and hold")]
    # choose on 2020-22 only: per coin the best rule; then read 2023-26
    is_ = base[base["period"] == "IS 2020-22"]
    oos = base[base["period"] == "OOS 2023-26"].set_index(["coin", "rule"])
    picks = {}
    for coin, g in is_.groupby("coin"):
        best = g.sort_values("sharpe").iloc[-1]
        o = oos.loc[(coin, best["rule"])]
        picks[coin] = {"rule": best["rule"], "is_sharpe": float(best["sharpe"]), "oos_sharpe": float(o["sharpe"]), "oos_cagr": float(o["cagr"])}
    n_trials = int(is_.shape[0])
    sr_var = float((is_["sharpe"] / np.sqrt(365)).var())
    dsr = {}
    for coin, p in picks.items():
        x = daily[f"{coin} | {p['rule']}"]
        x = x[x.index <= IS_END]
        args = (n_trials, len(x), float(x.skew()), float(x.kurt() + 3))
        dsr[coin] = {"strict": deflated_sharpe(p["is_sharpe"] / np.sqrt(365), sr_var, *args),
                     "lenient": deflated_sharpe(p["is_sharpe"] / np.sqrt(365), None, *args)}
    pos_is = is_[is_["sharpe"] > 1.0][["coin", "rule"]]
    surv = oos.loc[[tuple(v) for v in pos_is.values]] if len(pos_is) else oos.iloc[:0]
    print("rules with 2020-22 Sharpe > 1 after costs:", len(pos_is), "of", n_trials, "; their 2023-26 Sharpe median",
          round(float(surv["sharpe"].median()), 2) if len(surv) else None)
    print("in-sample picks and what happened next:", json.dumps(picks, indent=1))
    print("deflated Sharpe of each in-sample pick (over", n_trials, "trials):", {k: {a: round(b, 3) for a, b in v.items()} for k, v in dsr.items()})
    t.to_csv(out_path, index=False)
    pd.DataFrame(daily).to_parquet(out_path.replace(".csv", "_series.parquet"))
    with open(out_path.replace(".csv", "_summary.json"), "w") as fh:
        json.dump({"picks": picks, "dsr_in_sample": dsr, "n_trials": n_trials}, fh, indent=1)
    pd.set_option("display.width", 250)
    for coin in ("BTC", "ETH", "SOL"):
        x = t[(t["coin"] == coin) & ((t["cost"] == COST) | (t["rule"] == "buy and hold"))]
        print(f"--- {coin}: Sharpe after 0.06% per side")
        print(x.pivot_table(index="rule", columns="period", values="sharpe")[list(PERIODS)].round(2).to_string())
    x = t[(t["cost"] == 0.0002)]
    print("--- maker cost 0.02% per side: best 10 by 2020-22 Sharpe, with 2023-26")
    pv = x.pivot_table(index=["coin", "rule"], columns="period", values="sharpe")[list(PERIODS)]
    print(pv.sort_values("IS 2020-22", ascending=False).head(10).round(2).to_string())


if __name__ == "__main__":
    main()
