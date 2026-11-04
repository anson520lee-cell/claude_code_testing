"""python -m src.backtest.edge_hunt.grid --prices P --hist H --signals signals_v2.parquet --out DIR

Builds the point-in-time panel once and writes
  grid.parquet      every eligible stock-day 2009-06..2026-10: self-made indicators, beta / volatility,
                    multi-strategy confluence features
  zoo_rows.parquet  every signal of every online strategy (long and short): R of the live exit, excess over
                    same-day random entries, forward returns 1-20 sessions (gross) and their same-day excess
  sig_fwd.parquet   forward returns for the live-setup / random rows of signals_v2 (diagnostics)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

from src.backtest.edge_hunt import confluence as cf, zoo
from src.backtest.edge_hunt.panel import load_prices, universe
from src.backtest.swing42 import core
from src.backtest.swing42 import strategies as st

START = "2009-06-01"


def wide_features(p: core.Panel, spy: pd.Series) -> dict[str, np.ndarray]:
    C, O = p.C, p.O
    I = st.I
    lc, lo = np.log(C), np.log(O)
    r1 = C / core.shift(C, 1) - 1
    r5 = C / core.shift(C, 5) - 1
    r3 = C / core.shift(C, 3) - 1
    s = spy.reindex(p.dates).ffill().to_numpy()
    s1 = np.r_[np.nan, s[1:] / s[:-1] - 1]
    s5 = np.r_[[np.nan] * 5, s[5:] / s[:-5] - 1]
    R1 = pd.DataFrame(r1)
    S1 = pd.Series(s1)
    beta = (R1.rolling(120, min_periods=100).cov(S1) / S1.rolling(120, min_periods=100).var().to_numpy()[:, None]).to_numpy()
    hv60 = R1.rolling(60, min_periods=60).std().to_numpy() * math.sqrt(252)
    tot5 = lc - core.shift(lc, 5)
    on = lo - core.shift(lc, 1)
    on5 = core.roll(on, 5, "sum")
    dn = tot5 < -0.02
    a = I(p, "atr", 14)
    e21 = I(p, "ema", 21)
    ext = (C - e21) / a
    event = (r5 <= -0.20) | (r3 <= -0.15) | (ext <= -3)
    recent = core.roll(core.shift(event.astype(float), 1), 10, "max", minp=1) > 0
    first = event & ~recent
    resid = r1 - np.nan_to_num(beta, nan=1.0) * s1[:, None]
    f = {
        "beta": beta, "hv60": hv60, "ret1": r1, "ret5": r5, "ext_atr": ext,
        "sys_share": np.where(dn & (r5 < -0.02), beta * s5[:, None] / r5, np.nan),
        "on_share5": np.where(dn, on5 / tot5, np.nan),
        "speed": np.where(dn, (lc - core.shift(lc, 1)) / tot5, np.nan),
        "ivol60": pd.DataFrame(resid).rolling(60, min_periods=50).std().to_numpy() * math.sqrt(252),
        "max20": core.roll(core.shift(r1, 1), 20, "max", minp=15),
        "mom12_1": core.shift(C, 21) / core.shift(C, 252) - 1,
        "crash_cnt": core.roll(core.shift(first.astype(float), 11), 241, "sum", minp=100),
        "crash_first": first.astype(float),
    }
    # same-calendar-month seasonality: mean return of this month over the previous 1-10 years (>= 5)
    m = pd.DataFrame(C, index=p.dates).resample("ME").last().pct_change()
    lagged = [m.shift(12 * k) for k in range(1, 11)]
    stack = np.stack([x.to_numpy() for x in lagged])
    cnt = np.isfinite(stack).sum(0)
    seas_m = np.where(cnt >= 5, np.nanmean(np.where(np.isfinite(stack), stack, np.nan), axis=0), np.nan)
    seas_m = pd.DataFrame(seas_m, index=m.index)
    f["seas"] = seas_m.reindex(p.dates + pd.offsets.MonthEnd(0)).to_numpy()
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chosen", default="reports/swing42/results_A.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    px = load_prices(a.prices)
    cal = px["SPY"].index
    cal = cal[cal >= "2006-01-01"]
    mem = {s: m for s, m in universe(a.hist, px, cal).items() if s in px}
    p = cf.panel_from(px, mem, cal)
    first_valid = np.argmax(np.isfinite(p.C), axis=0)
    hist_ok = np.arange(p.D)[:, None] >= first_valid[None, :] + 260
    p.elig &= hist_ok
    print("panel", p.D, p.S, round(time.time() - t0), flush=True)
    W = wide_features(p, px["SPY"]["adj_close"])
    print("wide features", round(time.time() - t0), flush=True)
    chosen = cf.chosen_params(a.chosen)
    longs = zoo.long_signals(p, chosen)
    shorts = zoo.short_signals(p)
    print("zoo signals", {k: int(v.sum()) for k, v in {**longs, **shorts}.items()}, round(time.time() - t0), flush=True)
    weights = json.load(open(os.path.join(os.path.dirname(a.signals), "confluence_weights.json")))["weights"]
    conf = cf.features(p, {int(k[1:]): v for k, v in longs.items() if k != "L42"}, {int(k): v for k, v in weights.items()})
    out = zoo.outcomes(p)
    print("outcomes", round(time.time() - t0), flush=True)
    start_i = int(np.searchsorted(cal.values, np.datetime64(START)))
    E = p.elig.copy()
    E[:start_i] = False

    def daymean(x):
        v = np.where(E & np.isfinite(x), x, np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(v, axis=1)

    dm = {k: daymean(v) for k, v in out.items()}
    # ---------------- grid
    di, sj = np.nonzero(E)
    g = pd.DataFrame({"symbol": np.array(p.tickers)[sj], "date": cal[di]})
    for k, v in {**W, **conf}.items():
        g[k] = np.asarray(v, dtype=float)[di, sj].astype("float32")
    g.to_parquet(os.path.join(a.out, "grid.parquet"))
    print("grid", g.shape, round(time.time() - t0), flush=True)
    del g
    # ---------------- zoo rows
    rows = []
    for name, sig in {**longs, **shorts}.items():
        side = 1 if name.startswith("L") else -1
        y = out["RL"] if side > 0 else out["RS"]
        m = sig & E & np.isfinite(y)
        d_i, s_j = np.nonzero(m)
        r = pd.DataFrame({"strategy": name, "side": side, "symbol": np.array(p.tickers)[s_j], "date": cal[d_i],
                          "y": y[d_i, s_j], "exc": y[d_i, s_j] - (dm["RL"] if side > 0 else dm["RS"])[d_i]})
        for k in (1, 3, 5, 10, 20):
            fw = out[f"fwd{k}"]
            r[f"fx{k}"] = side * (fw[d_i, s_j] - dm[f"fwd{k}"][d_i])
        rows.append(r)
    z = pd.concat(rows, ignore_index=True)
    z.to_parquet(os.path.join(a.out, "zoo_rows.parquet"))
    print("zoo rows", z.shape, round(time.time() - t0), flush=True)
    # ---------------- forward returns for the live-setup rows
    sg = pd.read_parquet(a.signals, columns=["symbol", "date", "setup"])
    sj_map = {s: j for j, s in enumerate(p.tickers)}
    jj = sg["symbol"].map(sj_map)
    ii = np.searchsorted(cal.values, sg["date"].values.astype("datetime64[ns]"))
    ok = jj.notna().to_numpy() & (ii < len(cal))
    f = sg[ok].copy()
    ii, jj = ii[ok], jj[ok].astype(int).to_numpy()
    for k in (1, 3, 5, 10, 20):
        f[f"fx{k}"] = out[f"fwd{k}"][ii, jj] - dm[f"fwd{k}"][ii]
    f.to_parquet(os.path.join(a.out, "sig_fwd.parquet"))
    pd.DataFrame({"date": cal, **{f"dm_{k}": v for k, v in dm.items() if k in ("RL", "RS")}}).to_parquet(os.path.join(a.out, "daymeans.parquet"))
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
