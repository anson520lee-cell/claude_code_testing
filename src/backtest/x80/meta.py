"""python -m src.backtest.x80.meta OUT_DIR : walk-forward ways to combine and lever every sleeve.

Sleeves: the 12 of frontier.py (the two funding-carry books stay out, see frontier.EXCLUDE) plus this round's
versions, each picked on its own early period: crypto funding long/short (design picked on 2020-21), BTC and ETH
trend as the average of 28 rules (method picked on BTC 2016-19), multi-coin trend (2018-19), selling BTC volatility
(Deribit DVOL ladder, filter picked on 2021-22), concentrated S&P 500 momentum (2010-17). Weekly excess returns over
T-bills. Every allocator only sees the weeks before the one it trades:
  inverse vol (all sleeves) | inverse vol of the sleeves whose trailing mean is > 0 ("strategy momentum") |
  Sharpe-weighted (mean / variance, negatives dropped) | shrunk mean-variance (covariance half-way to its diagonal,
  means half-way to their cross-sleeve average, long only, at most 40% in one sleeve) | equal risk contribution;
  lookbacks 26 / 52 / 104 weeks.
Leverage on top: target vol 10%..100% from the book's trailing 13-, 26-, 52- or 104-week vol or its whole history so far
(the window is one more choice, so it is picked on 2010-17: the highest median Sharpe over the 15 books), or Kelly (mean /
variance of the book's trailing 104 weeks, full or half); cap 10x. A drawdown brake (Grossman-Zhou: leverage scaled by
the cushion above (1 - D) x the running peak, fresh account per period) is tried on the picked rule. The allocator and
its lookback are then also picked on 2010-17 (highest Sharpe with the picked window), and that rule is re-run under
harsher assumptions (robustness()).
Funded sleeves (spot, stocks, ETFs) borrow beyond 100% at T-bill + 1%
(crypto T-bill + 4%); the market-neutral perpetual books (UNFUNDED) need only margin, so they use no capital and their
leverage costs nothing beyond the funding already in their returns (the margin is assumed to earn T-bills, e.g. a
T-bill-backed collateral token).
Re-weighting the sleeves and resetting the leverage each week costs 0.10% of the notional traded (META_COST; the sleeves'
own trading costs are already in their returns).
Multiple testing: deflated Sharpe over every allocator x lookback x leverage-rule trial.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from src.backtest.x80.common import cash_daily, deflated_sharpe, load
from src.backtest.x80.frontier import CRYPTO, EXCLUDE, perf, sleeves, weekly_excess

UNFUNDED = {"crypto funding L/S", "BTC vs alts"}
NEW_CRYPTO = {"crypto funding L/S", "BTC trend (28-rule avg)", "ETH trend (28-rule avg)", "multi-coin trend", "BTC short vol (DVOL)"}
METHODS = ("inverse vol", "inverse vol, positive only", "Sharpe-weighted", "shrunk mean-variance", "equal risk")
LOOKS = (26, 52, 104)
TARGETS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00)
VOL_WINDOWS = (26, 13, 52, 104, 0)                                         # weeks; 0 = the whole history so far
META_COST = 0.001                                                          # per unit of notional traded when the mix or its leverage changes
PERIODS = {"2010-17": ("2010-01-01", "2017-12-31"), "2018-26": ("2018-01-01", "2030-01-01"), "2022-26": ("2022-01-01", "2030-01-01")}


def win_label(vw: int) -> str:
    return "(all)" if vw == 0 else f"({vw}w)"


def spread_of(name: str) -> float:
    return 0.04 if name in CRYPTO or name in NEW_CRYPTO else 0.01


def new_sleeves(px: dict, out: str) -> pd.DataFrame:
    s = {}
    cf = pd.read_parquet(f"{out}/crypto_factors_series.parquet")["funding L/S (chosen design)"].dropna()
    s["crypto funding L/S"] = weekly_excess(cf, cash_daily(px, cf.index, 365))
    im = pd.read_parquet(f"{out}/improve_series.parquet")
    for col, name, ppy in (("BTC trend, walk-forward pick", "BTC trend (28-rule avg)", 365), ("ETH trend, walk-forward pick", "ETH trend (28-rule avg)", 365),
                           ("multi-coin trend, chosen", "multi-coin trend", 365), ("momentum, chosen", "S&P momentum top20", 252)):
        x = im[col].dropna()
        x = x[x.index >= x.ne(0).idxmax()]                                 # from the first day the rule is live
        s[name] = weekly_excess(x, cash_daily(px, x.index, ppy))
    vr = pd.read_csv(f"{out}/crypto_vrp.csv")
    vr = vr[(vr["asset"] == "BTC") & (vr["haircut"] == 0.02) & (vr["period"] == "IS 2021-22")]
    best = vr.sort_values("sharpe").iloc[-1]["rule"]
    v = pd.read_parquet(f"{out}/crypto_vrp_series.parquet")[best].dropna()
    s["BTC short vol (DVOL)"] = weekly_excess(v, cash_daily(px, v.index, 365))
    print("BTC short-vol rule picked on 2021-22:", best)
    return pd.DataFrame(s)


def cap_weights(w: np.ndarray, cap: float) -> np.ndarray:
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        extra = (w[over] - cap).sum()
        w[over] = cap
        under = ~over & (w > 0)
        if not under.any():
            break
        w[under] += extra * w[under] / w[under].sum()
    return w


def equal_risk_weights(S: np.ndarray, tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    """Long-only weights with equal risk contributions w_i (S w)_i: cyclical coordinate descent on
    0.5 w'Sw - sum(ln w_i) / n, whose minimum has w_i (S w)_i = 1/n for every i; each step solves a quadratic with a
    positive root, so weights stay positive even when a sleeve hedges the others (negative covariance)."""
    n = len(S)
    d = np.maximum(np.diag(S), 1e-12)
    w = 1 / np.sqrt(d)
    for _ in range(max_iter):
        w_old = w.copy()
        for i in range(n):
            s = S[i] @ w - S[i, i] * w[i]
            w[i] = (-s + np.sqrt(s * s + 4 * d[i] / n)) / (2 * d[i])
        if np.max(np.abs(w - w_old)) < tol * np.max(w):
            break
    return w / w.sum()


def allocate(x: pd.DataFrame, method: str) -> pd.Series:
    """Long-only weights summing to 1 (or all zero = stay in T-bills) from a window of weekly excess returns."""
    ok = x.columns[x.notna().sum() >= max(13, len(x) // 2)]
    w = pd.Series(0.0, index=x.columns)
    if len(ok) == 0:
        return w
    y = x[ok].fillna(0.0)
    mu, sd = y.mean(), y.std().replace(0, np.nan)
    if method == "inverse vol":
        v = 1 / sd
    elif method == "inverse vol, positive only":
        v = (1 / sd).where(mu > 0, 0.0)
    elif method == "Sharpe-weighted":
        v = (mu / sd ** 2).clip(lower=0)
    elif method == "shrunk mean-variance":
        S = y.cov().values
        sig = 0.5 * S + 0.5 * np.diag(np.diag(S))
        m = 0.5 * mu.values + 0.5 * mu.mean()
        raw = np.clip(np.linalg.solve(sig + 1e-12 * np.eye(len(m)), m), 0, None)
        if raw.sum() > 0:
            raw = cap_weights(raw / raw.sum(), 0.4)
        v = pd.Series(raw, index=ok)
    elif method == "equal risk":
        live = sd.notna()                                                  # a flat sleeve has no risk to budget
        v = pd.Series(0.0, index=ok)
        v[live] = equal_risk_weights(y.loc[:, live].cov().values)
    else:
        raise ValueError(method)
    v = v.fillna(0.0)
    if v.sum() <= 0:
        return w
    w[ok] = (v / v.sum()).values
    return w


def run(w_ex: pd.DataFrame, cash_w: pd.Series, method: str, look: int, unfunded: set = UNFUNDED) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Unlevered book (excess return); per week the funded share, its weighted borrowing spread, the unfunded share and
    the weight turnover (sum of |new weight - last week's weight after it drifted with the sleeves' returns|);
    and the weights used each week."""
    idx = w_ex.index
    base = pd.Series(np.nan, index=idx)
    fin = pd.DataFrame(0.0, index=idx, columns=["funded", "spread", "unfunded", "turn"])
    sp = pd.Series({c: spread_of(c) for c in w_ex.columns})
    unf = pd.Series({c: c in unfunded for c in w_ex.columns})
    W = {}
    drifted = pd.Series(0.0, index=w_ex.columns)
    for i in range(26, len(idx)):
        win = w_ex.iloc[max(0, i - look):i]
        w = allocate(win, method)
        W[idx[i]] = w
        row = w_ex.iloc[i]
        live = row.notna()
        if w[live].sum() <= 0:
            base.iloc[i] = 0.0
            fin.iloc[i, 3] = float(drifted.abs().sum())
            drifted[:] = 0.0
            continue
        ww = w.where(live, 0.0)
        ww = ww / ww.sum()                                                 # a sleeve that has stopped reporting hands its weight on
        base.iloc[i] = float((ww * row.fillna(0.0)).sum())
        f = ww[~unf].sum()
        fin.iloc[i] = [f, float((ww[~unf] * sp[~unf]).sum() / f) if f > 0 else 0.0, float(ww[unf].sum()), float((ww - drifted).abs().sum())]
        drifted = ww * (1 + row.fillna(0.0)) / (1 + base.iloc[i])
    return base, fin, pd.DataFrame(W).T


def trading_cost(lev: pd.Series, base: pd.Series, fin: pd.DataFrame, gross: pd.Series, cost: float | None = None) -> pd.Series:
    """META_COST per unit of notional traded each week: the weight changes at the new leverage, plus the trade that takes
    last week's exposure (which drifted to lev x (1 + book) / (1 + levered return)) to this week's leverage."""
    turn = fin["turn"] if "turn" in fin else pd.Series(0.0, index=base.index)
    prev = (lev * (1 + base) / (1 + gross)).shift(1).fillna(0.0)
    return (META_COST if cost is None else cost) * (lev * turn + (lev - prev).abs())


def lever(base: pd.Series, fin: pd.DataFrame, cash_w: pd.Series, how: str, x: float, cap: float = 10.0, vol_win: int = 26,
          cost: float | None = None) -> pd.Series:
    """cash + lev x book + the T-bill the unfunded books' margin earns - spread on what the funded books borrow."""
    if how == "vol":
        roll = base.expanding(min_periods=52) if vol_win == 0 else base.rolling(vol_win, min_periods=max(8, vol_win // 2))
        bvol = roll.std().shift(1) * np.sqrt(52)
        lev = (x / bvol).clip(upper=cap)
    else:                                                                  # Kelly fraction x of trailing mean / variance
        m = base.rolling(104, min_periods=52).mean().shift(1)
        v = base.rolling(104, min_periods=52).var().shift(1)
        lev = (x * m / v).clip(lower=0, upper=cap)
    c = cash_w.reindex(base.index).fillna(0.0)
    ret = lev * base + c + lev * fin["unfunded"] * c - (lev * fin["funded"] - 1).clip(lower=0) * fin["spread"] / 52
    ret = ret - trading_cost(lev.fillna(0.0), base.fillna(0.0), fin, ret.fillna(0.0), cost)
    ret = ret.dropna().clip(lower=-1.0)
    dead = (ret <= -1.0).cumsum().shift(1).fillna(0) > 0
    return ret.mask(dead, 0.0), lev


def dd_brake(base: pd.Series, fin: pd.DataFrame, cash_w: pd.Series, lev: pd.Series, dmax: float, start: str, end: str) -> pd.Series:
    """A fresh account from `start`: each week's leverage is lev x cushion / dmax, cushion = 1 - (1 - dmax) x peak / equity
    (so full leverage at a new high, none at a drawdown of dmax), from the equity known at the previous close."""
    c = cash_w.reindex(base.index).fillna(0.0)
    idx = base.index[(base.index >= start) & (base.index <= end) & base.notna() & lev.notna()]
    eq, peak, out, held = 1.0, 1.0, [], 0.0
    turn = fin["turn"] if "turn" in fin else pd.Series(0.0, index=base.index)
    for t in idx:
        L = float(lev[t]) * max(0.0, 1 - (1 - dmax) * peak / eq) / dmax
        g = L * base[t] + c[t] + L * fin.at[t, "unfunded"] * c[t] - max(L * fin.at[t, "funded"] - 1, 0.0) * fin.at[t, "spread"] / 52
        r = max(g - META_COST * (L * float(turn[t]) + abs(L - held)), -1.0)
        held = L * (1 + base[t]) / (1 + g) if g > -1 else 0.0
        out.append(r)
        eq *= 1 + r
        if eq <= 0:
            break
        peak = max(peak, eq)
    return pd.Series(out, index=idx[:len(out)])


def kelly_math(base: pd.Series, fin: pd.DataFrame, cash_w: pd.Series, start: str) -> dict:
    """Growth of a constant leverage L on the book: log growth ~ r + s + L k sigma - (L sigma)^2 / 2 with k the Sharpe net
    of the financing mix (r T-bill, s spread, f funded share, u unfunded share). Best L = k / sigma; also the realised
    CAGR of that constant leverage, and the book Sharpe that 80% a year would need (at the best L, and at 30% vol)."""
    b = base[base.index >= start].dropna()
    f = fin.loc[b.index]
    c = cash_w.reindex(b.index).fillna(0.0)
    mu, sd, r = b.mean() * 52, b.std() * np.sqrt(52), c.mean() * 52
    fu, un = float(f["funded"].mean()), float(f["unfunded"].mean())
    sp = float((f["funded"] * f["spread"]).sum() / f["funded"].sum())
    k = (mu + un * r - fu * sp) / sd
    L = k / sd
    g = r + sp + k ** 2 / 2
    x = L * b + c + L * f["unfunded"] * c - (L * f["funded"] - 1).clip(lower=0) * f["spread"] / 52
    x = x - trading_cost(pd.Series(L, index=b.index), b, f, x)
    p = perf(x.clip(lower=-1.0))
    need = np.log(1.8) - r - sp
    return {"excess": mu, "vol": sd, "sharpe": mu / sd, "tbill": r, "spread": sp, "funded": fu, "unfunded": un, "net_sharpe": k,
            "best_lev": L, "theory_cagr": float(np.exp(g) - 1), "best_lev_vol": L * sd, "realised_cagr": p["cagr"], "realised_dd": p["max_dd"],
            "sharpe_needed_80_at_best_lev": float(np.sqrt(2 * need) + (fu * sp - un * r) / sd),
            "sharpe_needed_80_at_30vol": float((need + 0.3 ** 2 / 2) / 0.3 + (fu * sp - un * r) / sd)}


def robustness(w_ex: pd.DataFrame, cash_w: pd.Series, method: str, look: int, vw: int) -> pd.DataFrame:
    """The picked rule under harsher assumptions: double the re-weighting cost, a 5x leverage cap, the perpetual books
    financed like everything else, and without the live swing sleeve / this round's new sleeves / any crypto sleeve."""
    crypto = CRYPTO | NEW_CRYPTO
    cases = {"as reported": (w_ex, UNFUNDED, {}), "re-weighting cost 0.20%": (w_ex, UNFUNDED, {"cost": 2 * META_COST}),
             "leverage cap 5x": (w_ex, UNFUNDED, {"cap": 5.0}), "perpetual books financed at T-bill + 4%": (w_ex, set(), {}),
             "without live swing": (w_ex.drop(columns=["live swing (crash+squeeze)"]), UNFUNDED, {}),
             "without this round's 6 sleeves": (w_ex.drop(columns=[c for c in w_ex if c in NEW_CRYPTO or c == "S&P momentum top20"]), UNFUNDED, {}),
             "without any crypto sleeve": (w_ex.drop(columns=[c for c in w_ex if c in crypto]), UNFUNDED, {})}
    rows = []
    for case, (x, unf, kw) in cases.items():
        base, fin, _ = run(x, cash_w, method, look, unfunded=unf)
        for tv in (0.3, 0.6, 0.8):
            r, _ = lever(base, fin, cash_w, "vol", tv, vol_win=vw, **kw)
            for lab, (a, b) in PERIODS.items():
                rr = r[(r.index >= a) & (r.index <= b)]
                ex = rr - cash_w.reindex(rr.index).fillna(0)
                rows.append({"case": case, "target_vol": tv, "period": lab, **perf(rr), "sharpe": float(ex.mean() / ex.std() * np.sqrt(52))})
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = sys.argv[1]
    px = load()
    old = sleeves(px)
    new = new_sleeves(px, out_dir)
    w_ex = pd.concat([old, new], axis=1)
    w_ex = w_ex.drop(columns=[c for c in EXCLUDE if c in w_ex])
    w_ex = w_ex[w_ex.index >= "2004-01-01"]
    cash_w = (1 + cash_daily(px, pd.date_range("2004-01-01", "2026-11-04"), 365)).resample("W-FRI").prod().reindex(w_ex.index).fillna(0) - 1
    w_ex.to_parquet(os.path.join(out_dir, "meta_sleeves_weekly.parquet"))
    sh = {}
    for c in w_ex:
        for lab, (a, b) in PERIODS.items():
            y = w_ex[c][(w_ex.index >= a) & (w_ex.index <= b)].dropna()
            sh[(c, lab)] = y.mean() / y.std() * np.sqrt(52) if len(y) > 26 else np.nan
    print("sleeve Sharpe:\n", pd.Series(sh).unstack()[list(PERIODS)].round(2).to_string())
    print("correlation 2022-26 of the new sleeves with the rest:\n",
          w_ex[w_ex.index >= "2022-01-01"].corr()[list(new.columns)].round(2).to_string())
    rows, trials, last_w, keep, books = [], [], {}, {}, {}
    for method in METHODS:
        for look in LOOKS:
            base, fin, W = run(w_ex, cash_w, method, look)
            books[(method, look)] = (base, fin)
            last_w[f"{method} {look}w"] = W.iloc[-1].round(3).to_dict()
            b18 = base[base.index >= "2018-01-01"].dropna()
            b22 = base[base.index >= "2022-01-01"].dropna()
            trials.append({"method": method, "look": look, "lev": "none", "sr_w": float(b18.mean() / b18.std()), "n": len(b18),
                           "skew": float(b18.skew()), "kurt": float(b18.kurt() + 3), "sr_2022": float(b22.mean() / b22.std() * np.sqrt(52))})
            for vw in VOL_WINDOWS:
                for tv in TARGETS:
                    r, lv = lever(base, fin, cash_w, "vol", tv, vol_win=vw)
                    if tv in (0.3, 0.6, 0.8, 1.0):
                        keep[f"{method} {look}w @vol {int(tv * 100)}% {win_label(vw)}"] = r
                    if tv == 0.3:
                        x = (r - cash_w.reindex(r.index).fillna(0))[r.index >= "2018-01-01"]
                        trials.append({"method": method, "look": look, "lev": f"vol target, {win_label(vw)}", "sr_w": float(x.mean() / x.std()), "n": len(x),
                                       "skew": float(x.skew()), "kurt": float(x.kurt() + 3)})
                    for lab, (a, b) in PERIODS.items():
                        rr = r[(r.index >= a) & (r.index <= b)]
                        p = perf(rr)
                        if p:
                            ex = rr - cash_w.reindex(rr.index).fillna(0)
                            rows.append({"method": method, "look": look, "lev": f"vol {int(tv * 100)}% {win_label(vw)}", "period": lab, **p,
                                         "sharpe": float(ex.mean() / ex.std() * np.sqrt(52)), "avg_lev": float(lv[(lv.index >= a) & (lv.index <= b)].mean())})
            for f in (0.5, 1.0):
                r, lev = lever(base, fin, cash_w, "kelly", f)
                x = (r - cash_w.reindex(r.index).fillna(0))[r.index >= "2018-01-01"]
                trials.append({"method": method, "look": look, "lev": f"Kelly x{f}", "sr_w": float(x.mean() / x.std()), "n": len(x),
                               "skew": float(x.skew()), "kurt": float(x.kurt() + 3)})
                for lab, (a, b) in PERIODS.items():
                    p = perf(r[(r.index >= a) & (r.index <= b)])
                    if p:
                        rows.append({"method": method, "look": look, "lev": f"Kelly x{f}", "period": lab,
                                     "avg_lev": float(lev[(lev.index >= a) & (lev.index <= b)].mean()), **p})
    t = pd.DataFrame(rows)
    tr = pd.DataFrame(trials)
    # the vol window is picked on 2010-17 only: highest median Sharpe over the 15 books at 30% vol
    early = t[(t["period"] == "2010-17") & t["lev"].str.startswith("vol 30% ")]
    by_win = early.groupby(early["lev"].str.extract(r"(\(.*\))")[0])["sharpe"].median()
    win_pick = str(by_win.idxmax())
    print("median 2010-17 Sharpe by vol window (30% vol):", by_win.round(3).to_dict(), "-> picked", win_pick)
    vw = 0 if win_pick == "(all)" else int(win_pick.strip("(w)"))
    # then the allocator and its lookback, also on 2010-17 only: the highest Sharpe at 30% vol with the picked window
    e17 = t[(t["period"] == "2010-17") & (t["lev"] == f"vol 30% {win_pick}")].sort_values("sharpe", ascending=False)
    pick = (str(e17.iloc[0]["method"]), int(e17.iloc[0]["look"]))
    print("2010-17 Sharpe of the allocators (30% vol):", e17[["method", "look", "sharpe"]].head(5).round(3).values.tolist(), "-> picked", pick)
    rob = robustness(w_ex, cash_w, pick[0], pick[1], vw)
    # drawdown brake on the picked rule (a fresh account in each period)
    base, fin = books[pick]
    dd_rows = []
    for tv in (0.6, 0.8, 1.0):
        _, lev = lever(base, fin, cash_w, "vol", tv, vol_win=vw)
        for dmax in (None, 0.6, 0.5, 0.4, 0.3):
            for lab, (a, b) in PERIODS.items():
                r = dd_brake(base, fin, cash_w, lev, dmax, a, b) if dmax else lever(base, fin, cash_w, "vol", tv, vol_win=vw)[0]
                p = perf(r[(r.index >= a) & (r.index <= b)])
                dd_rows.append({"target_vol": tv, "dmax": dmax if dmax else "none", "period": lab, **p})
    dd = pd.DataFrame(dd_rows)
    km = {lab: kelly_math(base, fin, cash_w, a) for lab, (a, b) in PERIODS.items() if lab != "2010-17"}
    print(f"constant-leverage arithmetic, {pick}:", json.dumps(km, indent=1))
    best = tr.sort_values("sr_w").iloc[-1]
    dsr_strict = deflated_sharpe(best["sr_w"], float(tr["sr_w"].var()), len(tr), int(best["n"]), best["skew"], best["kurt"])
    dsr_len = deflated_sharpe(best["sr_w"], None, len(tr), int(best["n"]), best["skew"], best["kurt"])
    print(f"best of {len(tr)} combination trials on 2018-26: {best['method']} {best['look']}w {best['lev']}, "
          f"Sharpe {best['sr_w'] * np.sqrt(52):.2f}; deflated Sharpe probability strict {dsr_strict:.3f} / lenient {dsr_len:.3f}")
    # hindsight upper bound with every sleeve: long-only max-Sharpe weights fitted on the period itself
    hind = []
    for lab, a in (("2018-26", "2018-01-01"), ("2022-26", "2022-01-01")):
        x = w_ex[w_ex.index >= a]
        x = x.dropna(axis=1, thresh=int(0.9 * len(x))).fillna(0.0)
        mu, cov = x.mean().values, x.cov().values
        w = np.clip(np.linalg.pinv(cov) @ mu, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(mu)) / len(mu)
        for _ in range(300):
            g = mu / np.sqrt(w @ cov @ w) - (w @ mu) * (cov @ w) / (w @ cov @ w) ** 1.5
            w = np.clip(w + 0.05 * g / (np.abs(g).max() + 1e-12), 0, None)
            w = w / w.sum()
        bw = x.values @ w
        sr = bw.mean() / bw.std() * np.sqrt(52)
        unf = np.array([c in UNFUNDED for c in x.columns])
        f, u = float(w[~unf].sum()), float(w[unf].sum())
        sp = float(w[~unf] @ np.array([spread_of(c) for c in x.columns])[~unf]) / f if f > 0 else 0.0
        c = cash_w.reindex(x.index).fillna(0.0)
        drift = (w * (1 + x.values)) / (1 + bw)[:, None]                  # fixed weights are traded back every week
        hfin = pd.DataFrame({"turn": np.r_[1.0, np.abs(w - drift[:-1]).sum(axis=1)]}, index=x.index)
        for tv in TARGETS:
            lev = tv / (bw.std() * np.sqrt(52))
            r = pd.Series(lev * bw, index=x.index) + c + lev * u * c - max(lev * f - 1, 0) * sp / 52
            r = r - trading_cost(pd.Series(lev, index=x.index), pd.Series(bw, index=x.index), hfin, r)
            hind.append({"period": lab, "target_vol": tv, "sharpe": sr, **perf(r.clip(lower=-1))})
        print(f"hindsight max-Sharpe {lab}: Sharpe {sr:.2f}", {c: round(float(v), 2) for c, v in zip(x.columns, w) if v > 0.005})
    t.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    pd.DataFrame(keep).to_parquet(os.path.join(out_dir, "meta_series.parquet"))
    tr.to_csv(os.path.join(out_dir, "meta_trials.csv"), index=False)
    pd.DataFrame(hind).to_csv(os.path.join(out_dir, "meta_hindsight.csv"), index=False)
    dd.to_csv(os.path.join(out_dir, "meta_ddbrake.csv"), index=False)
    rob.to_csv(os.path.join(out_dir, "meta_robust.csv"), index=False)
    summary = {"best_trial": {k: (v if isinstance(v, str) else float(v)) for k, v in best.items()}, "dsr_strict": dsr_strict, "dsr_lenient": dsr_len,
               "n_trials": len(tr), "vol_window_pick": win_pick, "vol_window_sharpe_2010_17": by_win.to_dict(), "kelly_math": km,
               "allocator_pick": {"method": pick[0], "look": pick[1]}, "allocator_sharpe_2010_17": e17[["method", "look", "sharpe"]].values.tolist(),
               "last_weights": last_w}
    with open(os.path.join(out_dir, "meta_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    pd.set_option("display.width", 250)
    for lab in ("2018-26", "2022-26"):
        x = t[t["period"] == lab]
        print(f"--- {lab}: CAGR % (rows: method, lookback; columns: leverage rule)")
        order = [f"vol {int(tv * 100)}% {win_label(vw)}" for vw in VOL_WINDOWS for tv in TARGETS] + ["Kelly x0.5", "Kelly x1.0"]
        print((x.pivot_table(index=["method", "look"], columns="lev", values="cagr")[order] * 100).round(0).to_string())
        print(f"--- {lab}: max DD %")
        print((x.pivot_table(index=["method", "look"], columns="lev", values="max_dd")[order] * 100).round(0).to_string())
    print("robustness of the picked rule (CAGR / max DD):\n",
          rob.assign(v=lambda d: (d["cagr"] * 100).round(0).astype(str) + " / " + (d["max_dd"] * 100).round(0).astype(str))
          .pivot_table(index=["case", "target_vol"], columns="period", values="v", aggfunc="first").to_string())
    print("drawdown brake on", pick, win_pick, "(CAGR / max DD):\n",
          dd.assign(v=lambda d: (d["cagr"] * 100).round(0).astype(str) + " / " + (d["max_dd"] * 100).round(0).astype(str))
          .pivot_table(index=["target_vol", "dmax"], columns="period", values="v", aggfunc="first").to_string())
    print("Sharpe of the unlevered books (2018-26, 2022-26):\n", tr[tr["lev"] == "none"].assign(sharpe=lambda d: d["sr_w"] * np.sqrt(52))[
        ["method", "look", "sharpe", "sr_2022"]].round(2).to_string())


if __name__ == "__main__":
    main()
