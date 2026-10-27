"""Reproducible backtest of the SOXL position-sizing / trend rules described in STRATEGY.md.

    python backtest/fetch_data.py        # once: daily bars -> backtest/data/*.csv
    python backtest/run_backtest.py      # prints the tables and writes backtest/results.md

No look-ahead: every signal uses data up to the close of day t, the order is filled at the open of t+1.
Costs: 5 bp per side, 4% on idle cash, 10% on borrowed money (weight > 1). Assumptions, not broker quotes.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import CASH_YIELD, COST_BPS, MARGIN_RATE, load, rsi, run, stats  # noqa: E402

SPLIT = "2024-07-01"           # first half = "in-sample" (where rules were looked at), second half = out-of-sample
soxl, smh, spy = load("SOXL"), load("SMH"), load("SPY")
idx, c = soxl.index, soxl.close
rv20 = c.pct_change().rolling(20).std() * 252 ** 0.5
sma = lambda s, n: s.rolling(n).mean()  # noqa: E731



def to_md(df: pd.DataFrame, index: bool = False) -> str:
    """Tiny markdown table writer (avoids the optional `tabulate` dependency)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def weight(target_vol: float, gates: list[pd.Series], cap: float = 1.0) -> pd.Series:
    w = (target_vol / rv20).clip(upper=cap)
    for g in gates:
        w = w * g.astype(float)
    return w.fillna(0.0)


def run_breaker(w: pd.Series, half_dd=-0.12, stop_dd=-0.20, cooldown=10) -> pd.Series:
    """Account circuit breaker: from the strategy's own equity peak, -12% -> half size, -20% -> flat for
    `cooldown` sessions, then the peak is reset to the current equity."""
    o = soxl.open
    ret = (o.shift(-2) / o.shift(-1) - 1).reindex(idx)
    out, eq, peak, flat_until, prev = [], 1.0, 1.0, -1, 0.0
    for i, t in enumerate(idx):
        if np.isnan(ret.iloc[i]) or np.isnan(w.iloc[i]):
            out.append(0.0); continue
        dd = eq / peak - 1
        k = 1.0
        if i <= flat_until: k = 0.0
        elif dd <= stop_dd: k, flat_until = 0.0, i + cooldown
        elif dd <= half_dd: k = 0.5
        we = w.iloc[i] * k
        r = we * ret.iloc[i] + (1 - min(we, 1)) * CASH_YIELD / 252 - abs(we - prev) * COST_BPS / 1e4
        prev = we
        eq *= 1 + r; peak = max(peak, eq)
        if i == flat_until: peak = eq        # reset after cooldown
        out.append(r)
    return pd.Series(out, index=idx)


def row(name, r):
    f, a, b = stats(r), stats(r[:SPLIT]), stats(r[SPLIT:])
    return {"Strategy": name, "CAGR": f["cagr"], "Vol": f["vol"], "Sharpe": f["sharpe"], "MaxDD": f["maxdd"],
            "CAGR IS": a["cagr"], "MaxDD IS": a["maxdd"], "CAGR OOS": b["cagr"], "MaxDD OOS": b["maxdd"]}


def md(df, pct=("CAGR", "Vol", "MaxDD", "CAGR IS", "MaxDD IS", "CAGR OOS", "MaxDD OOS")):
    d = df.copy()
    for k in pct:
        if k in d: d[k] = d[k].map(lambda x: f"{x:+.0%}")
    if "Sharpe" in d: d["Sharpe"] = d["Sharpe"].map(lambda x: f"{x:.2f}")
    return to_md(d)


out = [f"# Backtest results\n\nData: daily bars {idx[0].date()} .. {idx[-1].date()} ({len(idx)} sessions). "
       f"IS = before {SPLIT}, OOS = after. Costs {COST_BPS} bp/side, cash {CASH_YIELD:.0%}, margin {MARGIN_RATE:.0%} "
       "(assumptions). Past results of a 4.8-year window that contains a semiconductor bull market; not a forecast.\n"]
ones = pd.Series(1.0, index=idx)
smh200, smh100, smh50 = smh.close > sma(smh.close, 200), smh.close > sma(smh.close, 100), smh.close > sma(smh.close, 50)
soxl50 = c > sma(c, 50)

# 1. baselines and the main candidates -------------------------------------------------------------
rows = [row("Buy & hold SPY", run(spy, ones)), row("Buy & hold SMH (semis, 1x)", run(smh, ones)),
        row("Buy & hold SOXL (3x semis)", run(soxl, ones)),
        row("SOXL 127% of equity, margin (current style)", run(soxl, ones * 1.27)),
        row("SOXL when SMH > SMA200, else cash", run(soxl, smh200.astype(float)))]
for tv in (0.2, 0.3, 0.4):
    rows.append(row(f"SOXL vol-target {tv:.0%}", run(soxl, weight(tv, []))))
for tv in (0.2, 0.3, 0.4):
    rows.append(row(f"SOXL vol-target {tv:.0%} + SMH>SMA200", run(soxl, weight(tv, [smh200]))))
for tv in (0.2, 0.3, 0.4):
    rows.append(row(f"SOXL vol-target {tv:.0%} + SMH>SMA200 + SOXL>SMA50", run(soxl, weight(tv, [smh200, soxl50]))))
base = weight(0.30, [smh200])
rows.append(row("A: vol-target 30% + SMH>SMA200 + circuit breaker", run_breaker(base)))
rows.append(row("A2: same + SOXL>SMA50 + circuit breaker", run_breaker(weight(0.30, [smh200, soxl50]))))
t1 = pd.DataFrame(rows)
out += ["\n## 1. Candidates vs buy & hold\n", md(t1)]

# 2. robustness: gate length -----------------------------------------------------------------------
rob = []
for n in (100, 125, 150, 175, 200, 225, 250):
    r = run(soxl, weight(0.30, [smh.close > sma(smh.close, n)]))
    s = stats(r); rob.append({"SMH SMA length": n, "CAGR": s["cagr"], "Sharpe": s["sharpe"], "MaxDD": s["maxdd"]})
out += ["\n## 2. Robustness: length of the SMH trend gate (vol-target 30%)\n", md(pd.DataFrame(rob))]

# 3. by calendar year ------------------------------------------------------------------------------
yr = {}
for name, r in (("SOXL buy&hold", run(soxl, ones)), ("SMH buy&hold", run(smh, ones)),
                ("A (30% VT + SMH>SMA200 + breaker)", run_breaker(base))):
    yr[name] = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
out += ["\n## 3. Calendar years (return)\n", to_md(pd.DataFrame(yr).map(lambda x: f"{x:+.0%}"), index=True)]

# 4. block bootstrap of 1-year outcomes ------------------------------------------------------------
rA = run_breaker(base).dropna().to_numpy()
rng = np.random.default_rng(7)
def boot(r, n=252, block=20, sims=5000):
    res = []
    for _ in range(sims):
        path = []
        while len(path) < n:
            s = rng.integers(0, len(r) - block); path.extend(r[s:s + block])
        eq = np.cumprod(1 + np.array(path[:n]))
        res.append((eq[-1] - 1, (eq / np.maximum.accumulate(eq) - 1).min()))
    return np.array(res)
rows = []
for name, r in (("A", rA), ("SOXL buy&hold", run(soxl, ones).dropna().to_numpy()),
                ("SOXL 127% margin", run(soxl, ones * 1.27).dropna().to_numpy())):
    b = boot(r)
    rows.append({"Strategy": name, "median 1y": f"{np.median(b[:,0]):+.0%}", "5th pct 1y": f"{np.percentile(b[:,0],5):+.0%}",
                 "P(1y loss)": f"{(b[:,0]<0).mean():.0%}", "median maxDD": f"{np.median(b[:,1]):.0%}",
                 "95th pct worst DD": f"{np.percentile(b[:,1],5):.0%}", "P(DD worse than -30%)": f"{(b[:,1]<-0.30).mean():.0%}"})
out += ["\n## 4. Block-bootstrap (20-day blocks, 5000 one-year paths, resampled from this same window)\n",
        to_md(pd.DataFrame(rows))]

# 5. short-term swing setups on SOXL (event studies) -----------------------------------------------
o, h = soxl.open, soxl.high
r2 = rsi(c, 2)
e21 = c.ewm(span=21, adjust=False).mean()
down3 = (c < c.shift(1)) & (c.shift(1) < c.shift(2)) & (c.shift(2) < c.shift(3))
def trades(entry, exit_fn, maxhold, stop):
    res, i, n = [], 0, len(c)
    while i < n - 2:
        if entry.iloc[i]:
            ei, j = i + 1, i + 1; ep = o.iloc[ei]
            while not (c.iloc[j] / ep - 1 <= -stop or exit_fn(j) or j - ei + 1 >= maxhold or j >= n - 2): j += 1
            xi = min(j + 1, n - 1); res.append((c.index[ei], o.iloc[xi] / ep - 1)); i = xi
        else: i += 1
    return pd.DataFrame(res, columns=["date", "ret"])
def summ(name, t):
    w, l = t.ret[t.ret > 0], t.ret[t.ret <= 0]
    isn, oosn = t[t.date < SPLIT], t[t.date >= SPLIT]
    return {"Setup": name, "N": len(t), "Win%": f"{len(w)/len(t):.0%}", "Avg": f"{t.ret.mean():+.1%}", "Median": f"{t.ret.median():+.1%}",
            "Avg win": f"{w.mean():+.1%}", "Avg loss": f"{l.mean():+.1%}", "Worst": f"{t.ret.min():+.1%}",
            "Avg IS (N)": f"{isn.ret.mean():+.1%} ({len(isn)})", "Avg OOS (N)": f"{oosn.ret.mean():+.1%} ({len(oosn)})"}
prev_high_exit = lambda j: c.iloc[j] > h.iloc[j - 1]  # noqa: E731
rows = []
rows.append(summ("RSI2<10, SMH>SMA200 (exit > prior high / 5d / -10%)", trades((r2 < 10) & smh200, prev_high_exit, 5, 0.10)))
rows.append(summ("RSI2<20, SMH>SMA200 (same exit)", trades((r2 < 20) & smh200, prev_high_exit, 5, 0.10)))
rows.append(summ("3 down closes, SMH>SMA200 (same exit)", trades(down3 & smh200, prev_high_exit, 5, 0.10)))
rows.append(summ("20-day-high breakout, SMH>SMA200 (exit < EMA21 / 20d / -12%)",
                 trades((c >= c.rolling(20).max()) & smh200, lambda j: c.iloc[j] < e21.iloc[j], 20, 0.12)))
rng2 = np.random.default_rng(0)
rows.append(summ("Random entries, same exit as RSI2 setups (control)",
                 trades(pd.Series(rng2.random(len(c)) < 0.06, index=c.index), prev_high_exit, 5, 0.10)))
out += ["\n## 5. Short-term SOXL setups (one position at a time, full position, per-trade return)\n",
        to_md(pd.DataFrame(rows))]

# 6. today's state ---------------------------------------------------------------------------------
last = idx[-1]
state = {"date": str(last.date()), "SOXL close": round(c.iloc[-1], 2), "SOXL 20d vol": f"{rv20.iloc[-1]:.0%}",
         "SMH>SMA200": bool(smh200.iloc[-1]), "SMH>SMA100": bool(smh100.iloc[-1]), "SMH>SMA50": bool(smh50.iloc[-1]),
         "SOXL>SMA50": bool(soxl50.iloc[-1]),
         "weight A (share of equity in SOXL)": f"{base.iloc[-1]:.0%}"}
out += ["\n## 6. Signal state at the last bar\n", "\n".join(f"- {k}: {v}" for k, v in state.items())]

text = "\n".join(out) + "\n"
Path(__file__).with_name("results.md").write_text(text)
print(text)
