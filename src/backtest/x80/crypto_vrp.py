"""python -m src.backtest.x80.crypto_vrp DVOL_CSV BINANCE_SPOT OUT_CSV : sell bitcoin / ether volatility (Deribit DVOL).

Proxy for systematically selling 30-day options and delta-hedging them: a ladder of short 30-day variance swaps. Each
day at the 00:00 UTC close we sell a new tranche (1/30 of the book) with strike = DVOL (Deribit's 30-day implied vol
index, model-free like the VIX) minus a haircut for spreads and hedging (2 vol points; 0 and 4 as checks), sized as a
share of current equity. Every open tranche is marked daily: realised variance so far (365 x squared daily log
returns of the Binance close) for the elapsed part, today's DVOL for the rest, so an implied-vol spike shows up as a
loss at once. A tranche settles after 30 days. Margin earns nothing; a day that loses the equity ends the account.
Rules: always sell; sell only when DVOL is above the trailing 30-day realised vol (or 10-day); skip after a 7-day
fall of more than 10%. Size = total variance notional per $1 of equity (0.1 to 0.5). DVOL starts 2021-03-24:
2021-04..2022 is the early period (LUNA, FTX), 2023-26 is read out of sample.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, load, stats

PPY = 365
TENOR = 30
PERIODS = {"IS 2021-22": ("2021-04-01", "2022-12-31"), "OOS 2023-26": ("2023-01-01", "2030-01-01"), "2022": ("2022-01-01", "2022-12-31"),
           "2024-26": ("2024-01-01", "2030-01-01"), "all": ("2021-04-01", "2030-01-01")}


def inputs(dvol_csv: str, spot: str) -> dict[str, pd.DataFrame]:
    dv = pd.read_csv(dvol_csv)
    dv["date"] = pd.to_datetime(dv["time_ms"], unit="ms").dt.floor("D")
    s = pd.read_parquet(spot, columns=["date", "symbol", "close"])
    out = {}
    for cur in ("BTC", "ETH"):
        iv = dv[dv["currency"] == cur].set_index("date")["close"] / 100
        px = s[s["symbol"] == f"{cur}USDT"].set_index("date")["close"].sort_index()
        idx = pd.date_range(max(iv.index.min(), px.index.min()), min(iv.index.max(), px.index.max()), freq="D")
        out[cur] = pd.DataFrame({"iv": iv.reindex(idx).ffill(), "px": px.reindex(idx).ffill()})
    return out


def short_var_ladder(d: pd.DataFrame, size: float, haircut: float, rule: str) -> pd.Series:
    x = np.log(d["px"]).diff().fillna(0.0).values
    iv = d["iv"].values
    n = len(d)
    rv30 = pd.Series(x).rolling(30).std().values * np.sqrt(PPY)
    rv10 = pd.Series(x).rolling(10).std().values * np.sqrt(PPY)
    r7 = d["px"].pct_change(7).values
    eq = 1.0
    open_: list[list] = []                                                 # [notional, strike^2, sum x^2, days elapsed, last value]
    out = np.zeros(n)
    for t in range(1, n):
        pnl = 0.0
        keep = []
        for sw in open_:
            sw[2] += x[t] ** 2
            sw[3] += 1
            e = sw[3]
            rv2 = PPY * sw[2] / e
            val = sw[0] * (e / TENOR * (sw[1] - rv2) + (TENOR - e) / TENOR * (sw[1] - iv[t] ** 2))
            pnl += val - sw[4]
            sw[4] = val
            if e < TENOR:
                keep.append(sw)
        open_ = keep
        r = pnl / eq if eq > 0 else 0.0
        if eq + pnl <= 0:
            out[t] = -1.0
            break
        out[t] = r
        eq += pnl
        # decide at today's close whether to sell a new tranche (it starts earning tomorrow)
        go = np.isfinite(iv[t])
        if rule == "iv>rv30":
            go = go and np.isfinite(rv30[t]) and iv[t] > rv30[t]
        elif rule == "iv>rv10":
            go = go and np.isfinite(rv10[t]) and iv[t] > rv10[t]
        elif rule == "no crash":
            go = go and not (np.isfinite(r7[t]) and r7[t] < -0.10)
        if go:
            k = max(iv[t] - haircut, 0.05)
            notional = size / TENOR * eq
            open_.append([notional, k * k, 0.0, 0, notional * (k * k - iv[t] ** 2)])
            out[t] += notional * (k * k - iv[t] ** 2) / eq                # the spread is paid on the day of the sale
            eq += notional * (k * k - iv[t] ** 2)
    return pd.Series(out, index=d.index)


def main() -> None:
    dvol_csv, spot, out_path = sys.argv[1:4]
    data = inputs(dvol_csv, spot)
    px = load()
    rows, series = [], {}
    for cur, d in data.items():
        d = d[d.index >= "2021-03-25"]
        cash = cash_daily(px, d.index, PPY)
        rv = np.log(d["px"]).diff().rolling(30).std().mean() * np.sqrt(PPY) * 100
        print(cur, f"mean DVOL {d['iv'].mean() * 100:.1f}, mean realised 30d vol {rv:.1f}")
        for rule in ("always", "iv>rv30", "iv>rv10", "no crash"):
            for size in (0.1, 0.2, 0.3, 0.5):
                for hc in (0.0, 0.02, 0.04):
                    if hc != 0.02 and size != 0.2:
                        continue
                    ret = short_var_ladder(d, size, hc, rule)
                    name = f"{cur} short vol, {rule}, size {size}, haircut {int(hc * 100)}"
                    series[name] = ret
                    for lab, (a, b) in PERIODS.items():
                        st = stats(ret[(ret.index >= a) & (ret.index <= b)], cash, PPY)
                        if st:
                            rows.append({"rule": name, "asset": cur, "filter": rule, "size": size, "haircut": hc, "period": lab, **st})
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    pd.DataFrame(series).to_parquet(out_path.replace(".csv", "_series.parquet"))
    pd.set_option("display.width", 250)
    base = t[(t["haircut"] == 0.02)]
    print("Sharpe (haircut 2 vol points)\n", base.pivot_table(index="rule", columns="period", values="sharpe")[list(PERIODS)].round(2).to_string())
    print("CAGR %\n", (base.pivot_table(index="rule", columns="period", values="cagr")[list(PERIODS)] * 100).round(0).to_string())
    print("max DD %\n", (base.pivot_table(index="rule", columns="period", values="max_dd")[list(PERIODS)] * 100).round(0).to_string())
    hc = t[(t["size"] == 0.2)]
    print("haircut check, size 0.2, Sharpe\n", hc.pivot_table(index=["asset", "filter", "haircut"], columns="period", values="sharpe")[list(PERIODS)].round(2).to_string())


if __name__ == "__main__":
    main()
