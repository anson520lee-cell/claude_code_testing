"""python -m src.backtest.x80.crypto_factors BINANCE_DIR OUT_CSV : market-neutral crypto factor long/short (Binance perps).

Universe each week, at the previous 00:00 UTC close: coins with a USDT-M perpetual (a funding print in the last 3 days
and >= 14 days of funding history), >= 60 days of spot history, 30-day median spot turnover >= $5M, the top 100 by
turnover (BTC and ETH included). Every coin Binance later delisted is in the data; a position in a coin that stops
trading is frozen at its last close until the next rebalance.
13 factors, each as of that close: yesterday's return (reversed), 7- and 28-day momentum, 84-day momentum skipping the
last week, 30-day volatility, 90-day beta and idiosyncratic volatility against BTC, 30-day max daily return, 7-day
funding, days since listing, 30-day median turnover, 7-day / 90-day turnover (volume shock), distance from the 90-day high.
Portfolio: long the top fifth, short the bottom fifth, equal weight, weekly; dollar neutral (gross 2) or beta neutral
(each leg scaled to beta 1, then gross 2). Both legs are perpetuals: no borrowing, the long side pays funding and the
short side receives it; positions drift with prices inside the week (a short that rallies grows); 0.12% of notional per
side. The margin earns nothing (no T-bill). P&L uses spot closes (the perp basis is not modelled).
Signs are read on 2020-21 only; 2022-26 is out of sample. Walk-forward composites re-pick factors every week from
the trailing 52 weeks of factor returns only. Leverage scales the weights (perps need no loan); a day that loses the
whole margin ends the account.
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
from src.backtest.x80.crypto_xs import panel

COST = 0.0012
PPY = 365
START = "2020-01-01"
TOP = 100
QUANT = 0.2
PERIODS = {"IS 2020-21": ("2020-01-01", "2021-12-31"), "2022-23": ("2022-01-01", "2023-12-31"), "2024-26": ("2024-01-01", "2030-01-01"),
           "OOS 2022-26": ("2022-01-01", "2030-01-01"), "all": ("2020-01-01", "2030-01-01")}


def spot_name(perp: str) -> str:
    """Binance lists cheap coins' perps per 1,000 or 1,000,000 units (1000PEPEUSDT): map them to the spot symbol."""
    for p in ("1000000", "1000", "1M"):
        if perp.startswith(p) and len(perp) > len(p) + 4 and not perp[len(p)].isdigit():
            return perp[len(p):]
    return perp


def inputs(bdir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close, qv = panel(f"{bdir}/spot.parquet")
    close = close[close.index >= "2019-06-01"]
    qv = qv.reindex(close.index)
    f = pd.read_parquet(f"{bdir}/funding.parquet")
    f["spot"] = [s if s in close.columns else spot_name(s) for s in f["symbol"]]
    f = f[f["spot"].isin(close.columns)]
    fund = funding_daily(f, "spot").reindex(index=close.index, columns=close.columns)
    r = close.pct_change(fill_method=None).where(close.shift(1).notna() & close.notna())
    return close, qv, r, fund


def btc_moments(r: pd.DataFrame, win: int = 90, minp: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling beta and idiosyncratic vol of every coin against BTC, on the days both traded."""
    b = pd.DataFrame(np.repeat(r["BTCUSDT"].values[:, None], r.shape[1], axis=1), index=r.index, columns=r.columns).where(r.notna())
    roll = lambda x: x.rolling(win, min_periods=minp).mean()
    mx, mb = roll(r), roll(b)
    cov = roll(r * b) - mx * mb
    vb = roll(b * b) - mb * mb
    vx = roll(r * r) - mx * mx
    beta = cov / vb
    ivol = np.sqrt((vx - cov * cov / vb).clip(lower=0))
    return beta, ivol


def factors(close: pd.DataFrame, qv: pd.DataFrame, r: pd.DataFrame, fund: pd.DataFrame) -> dict[str, pd.DataFrame]:
    beta, ivol = btc_moments(r)
    return {
        "rev1": -r,
        "mom7": close / close.shift(7) - 1,
        "mom28": close / close.shift(28) - 1,
        "mom84skip7": close.shift(7) / close.shift(84) - 1,
        "vol30": r.rolling(30, min_periods=20).std(),
        "beta90": beta,
        "ivol90": ivol,
        "max30": r.rolling(30, min_periods=20).max(),
        "fund7": fund.rolling(7, min_periods=5).sum(),
        "age": close.notna().cumsum().where(close.notna()),
        "size": qv.rolling(30, min_periods=20).median(),
        "vshock": qv.rolling(7, min_periods=5).mean() / qv.rolling(90, min_periods=60).mean(),
        "high90": close / close.rolling(90, min_periods=60).max(),
    }


def universe(close: pd.DataFrame, qv: pd.DataFrame, fund: pd.DataFrame, top: int = TOP, min_qv: float = 5e6) -> pd.DataFrame:
    age = close.notna().cumsum()
    liq = qv.rolling(30, min_periods=20).median()
    perp = fund.notna().rolling(3, min_periods=1).max().astype(bool) & (fund.notna().cumsum() >= 14)
    ok = (age >= 60) & (liq >= min_qv) & close.notna() & perp
    return ok & (liq.where(ok).rank(axis=1, ascending=False) <= top)


def rebalance_days(index: pd.DatetimeIndex) -> np.ndarray:
    return np.where(index >= START)[0][::7]


def ls_weights(score: np.ndarray, beta: np.ndarray | None, q: float = QUANT) -> np.ndarray:
    """Long the top ``q`` of ``score`` (NaN = not eligible), short the bottom ``q``; gross 2, net 0 (or beta 0)."""
    w = np.zeros_like(score)
    ok = np.isfinite(score)
    n = int(ok.sum())
    if n < 20:
        return w
    k = max(int(n * q), 3)
    order = np.argsort(np.where(ok, score, np.nan))[:n]                   # ascending, NaNs last
    lo, hi = order[:k], order[n - k:]
    gl = gs = 1.0
    if beta is not None:
        bl = np.clip(np.nanmean(np.where(np.isfinite(beta[hi]), beta[hi], 1.0)), 0.25, 4.0)
        bs = np.clip(np.nanmean(np.where(np.isfinite(beta[lo]), beta[lo], 1.0)), 0.25, 4.0)
        gl, gs = 1.0 / bl, 1.0 / bs
        s = 2.0 / (gl + gs)
        gl, gs = gl * s, gs * s
    w[hi] = gl / k
    w[lo] = -gs / k
    return w


def band_weights(score: np.ndarray, held_long: np.ndarray, held_short: np.ndarray, q_in: float, q_out: float) -> tuple:
    """Like ``ls_weights`` but a coin already held stays until it leaves the outer band (less trading)."""
    w = np.zeros_like(score)
    ok = np.isfinite(score)
    if ok.sum() < 20:
        return w, np.zeros(len(score), bool), np.zeros(len(score), bool)
    pct = pd.Series(score).rank(pct=True).values
    longs = ok & ((pct > 1 - q_in) | (held_long & (pct > 1 - q_out)))
    shorts = ok & ((pct <= q_in) | (held_short & (pct <= q_out)))
    w[longs] = 1.0 / longs.sum()
    w[shorts] = -1.0 / shorts.sum()
    return w, longs, shorts


def simulate(W: np.ndarray, reb: np.ndarray, r: np.ndarray, fund: np.ndarray, cost: float = COST,
             turnover: list | None = None) -> np.ndarray:
    """Daily returns of a book that sets weights ``W[i]`` (fraction of equity) at the open of day ``reb[i]`` and lets the
    positions drift with prices until the next rebalance. Long positions pay funding, shorts receive it.
    A row of NaN keeps the drifted positions untouched (no trade). ``turnover`` collects sum |trade| per rebalance."""
    n = r.shape[0]
    out = np.full(n, np.nan)
    w_old = np.zeros(r.shape[1])
    dead = False
    for i, s in enumerate(reb):
        e = reb[i + 1] if i + 1 < len(reb) else n
        if dead:
            out[s:e] = 0.0
            continue
        target = w_old if np.isnan(W[i]).all() else W[i]
        turn = np.abs(target - w_old).sum()
        if turnover is not None:
            turnover.append(turn)
        eq0 = 1.0
        eq = 1.0 - turn * cost
        pos = target * eq
        for t in range(s, e):
            pnl = pos @ r[t] - pos @ fund[t]
            new = eq + pnl
            out[t] = new / eq0 - 1
            if new <= 0:
                out[t] = -1.0
                out[t + 1:e] = 0.0
                dead = True
                break
            pos = pos * (1 + r[t])
            eq0 = eq = new
        if dead:
            continue
        w_old = pos / eq
    return out


def block_returns(daily: np.ndarray, reb: np.ndarray) -> np.ndarray:
    """Compounded return of each rebalance block (the week a set of weights was held)."""
    out = np.full(len(reb), np.nan)
    for i, s in enumerate(reb):
        e = reb[i + 1] if i + 1 < len(reb) else len(daily)
        x = daily[s:e]
        if np.isfinite(x).all() and len(x):
            out[i] = np.prod(1 + x) - 1
    return out


def vol_target_scale(daily: np.ndarray, reb: np.ndarray, target: float, cap: float = 5.0, win: int = 60) -> np.ndarray:
    """Leverage set at each rebalance from the 1x book's trailing ``win``-day vol (days before the rebalance only)."""
    lev = np.zeros(len(reb))
    for i, s in enumerate(reb):
        x = daily[max(0, s - win):s]
        x = x[np.isfinite(x)]
        if len(x) >= 30 and x.std() > 0:
            lev[i] = min(target / (x.std() * np.sqrt(PPY)), cap)
    return lev


def main() -> None:
    bdir, out_path = sys.argv[1:3]
    close, qv, r, fund = inputs(bdir)
    F = factors(close, qv, r, fund)
    uni = universe(close, qv, fund)
    idx = close.index
    reb = rebalance_days(idx)
    prev = reb - 1                                                         # decide on the previous close
    R = r.fillna(0.0).values
    FU = fund.fillna(0.0).values
    U = uni.values
    beta = F["beta90"].values
    print("universe size by year:", uni.iloc[reb].sum(axis=1).groupby(idx[reb].year).mean().round(0).to_dict())
    print("coins ever in the universe:", int(uni.any().sum()), "of which stopped trading:", int((uni.any() & close.iloc[-30:].isna().all()).sum()))
    px = load()
    cash = cash_daily(px, idx, PPY)

    weights, series = {}, {}
    for name, fac in F.items():
        V = fac.values
        for neutral in ("dollar", "beta"):
            W = np.zeros((len(reb), r.shape[1]))
            for i, p in enumerate(prev):
                sc = np.where(U[p], V[p], np.nan)
                W[i] = ls_weights(sc, beta[p] if neutral == "beta" else None)
            key = f"{name}|{neutral}"
            weights[key] = W
            series[key] = simulate(W, reb, R, FU)
    # in-sample signs (2020-21) and t-stats of every single factor
    blocks = {k: block_returns(v, reb) for k, v in series.items()}
    reb_dates = idx[reb]
    is_mask = reb_dates <= "2021-12-31"
    is_t = {}
    for k, b in blocks.items():
        x = b[is_mask]
        x = x[np.isfinite(x)]
        is_t[k] = float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 10 and x.std() > 0 else 0.0

    def composite_score(i: int, picks: dict[str, float], neutral: str) -> np.ndarray:
        p = prev[i]
        tot = np.zeros(r.shape[1])
        cnt = np.zeros(r.shape[1])
        for name, sign in picks.items():
            v = np.where(U[p], F[name].values[p], np.nan)
            ok = np.isfinite(v)
            if ok.sum() < 20:
                continue
            rk = pd.Series(v[ok]).rank(pct=True).values - 0.5
            tot[ok] += sign * rk
            cnt[ok] += 1
        sc = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        return ls_weights(np.where(U[p], sc, np.nan), beta[p] if neutral == "beta" else None)

    for neutral in ("dollar", "beta"):
        names = [k.split("|")[0] for k in blocks if k.endswith("|" + neutral)]
        # 1. in-sample picks: every factor with |t| >= 2 on 2020-21, its sign fixed from then on (judge on 2022-26 only)
        picks = {n: float(np.sign(is_t[f"{n}|{neutral}"])) for n in names if abs(is_t[f"{n}|{neutral}"]) >= 2}
        print(neutral, "in-sample picks:", picks)
        W = np.array([composite_score(i, picks, neutral) if picks else np.zeros(r.shape[1]) for i in range(len(reb))])
        series[f"IS-picked composite score|{neutral}"] = simulate(W, reb, R, FU)
        Wm = sum(weights[f"{n}|{neutral}"] * s for n, s in picks.items()) / max(len(picks), 1)
        series[f"IS-picked factor mix|{neutral}"] = simulate(Wm, reb, R, FU)
        # 2. walk-forward: each week keep the factors whose trailing 52-week t-stat is > 1, with that sign
        Ws, Wf = np.zeros((len(reb), r.shape[1])), np.zeros((len(reb), r.shape[1]))
        for i in range(len(reb)):
            if i < 26:
                continue
            pk, inv = {}, {}
            for n in names:
                x = blocks[f"{n}|{neutral}"][max(0, i - 52):i]
                x = x[np.isfinite(x)]
                if len(x) < 26 or x.std() == 0:
                    continue
                t = x.mean() / x.std() * np.sqrt(len(x))
                if abs(t) > 1:
                    pk[n] = float(np.sign(t))
                    inv[n] = 1.0 / x.std()
            if pk:
                Ws[i] = composite_score(i, pk, neutral)
                tot = sum(inv.values())
                Wf[i] = sum(weights[f"{n}|{neutral}"][i] * pk[n] * inv[n] / tot for n in pk)
        series[f"walk-forward composite score|{neutral}"] = simulate(Ws, reb, R, FU)
        series[f"walk-forward factor mix|{neutral}"] = simulate(Wf, reb, R, FU)
        weights[f"walk-forward composite score|{neutral}"] = Ws
        weights[f"walk-forward factor mix|{neutral}"] = Wf
        weights[f"IS-picked composite score|{neutral}"] = W
        weights[f"IS-picked factor mix|{neutral}"] = Wm

    # funding factor designs (the one factor that passed in sample): long low funding / short high funding. Each design is
    # a trial; the design is chosen on 2020-21 Sharpe only.
    def fund_score(look: int) -> pd.DataFrame:
        return -fund.rolling(look, min_periods=max(1, look - 2)).sum()

    designs = {}
    for look in (1, 3, 7, 14, 28):
        designs[f"funding {look}d, fifths, weekly"] = (look, "q", 0.2, 1)
    designs["funding 7d, tenths, weekly"] = (7, "q", 0.1, 1)
    designs["funding 7d, thirds, weekly"] = (7, "q", 1 / 3, 1)
    designs["funding 7d, band 20/40, weekly"] = (7, "band", 0.2, 1)
    designs["funding 28d, band 20/40, weekly"] = (28, "band", 0.2, 1)
    designs["funding 7d, fifths, every 2 weeks"] = (7, "q", 0.2, 2)
    designs["funding 7d, fifths, every 4 weeks"] = (7, "q", 0.2, 4)
    turn_avg = {}
    for name, (look, kind, q, every) in designs.items():
        V = fund_score(look).values
        W = np.full((len(reb), r.shape[1]), np.nan)
        hl, hs = np.zeros(r.shape[1], bool), np.zeros(r.shape[1], bool)
        for i, p in enumerate(prev):
            if i % every:
                continue                                                   # NaN row: keep the positions
            sc = np.where(U[p], V[p], np.nan)
            if kind == "band":
                W[i], hl, hs = band_weights(sc, hl, hs, q, 2 * q)
            else:
                W[i] = ls_weights(sc, None, q)
        tl: list = []
        series[name] = simulate(W, reb, R, FU, turnover=tl)
        weights[name] = W
        turn_avg[name] = float(np.mean(tl)) if tl else np.nan
        b = block_returns(series[name], reb)[is_mask]
        b = b[np.isfinite(b)]
        is_t[name] = float(b.mean() / b.std() * np.sqrt(len(b))) if len(b) > 10 and b.std() > 0 else 0.0
    best_design = max(designs, key=lambda k: is_t[k])
    print("funding designs, in-sample t / average turnover per rebalance:",
          {k: (round(is_t[k], 2), round(turn_avg[k], 2)) for k in designs}, "-> chosen on 2020-21:", best_design)
    # robustness of the chosen design (not new trials): dearer trading, the top-50 universe only
    look, kind, q, every = designs[best_design]
    W = weights[best_design]
    series[f"{best_design} @0.25% cost"] = simulate(W, reb, R, FU, cost=0.0025)
    U50 = universe(close, qv, fund, top=50).values
    V = fund_score(look).values
    W50 = np.full((len(reb), r.shape[1]), np.nan)
    hl, hs = np.zeros(r.shape[1], bool), np.zeros(r.shape[1], bool)
    for i, p in enumerate(prev):
        if i % every:
            continue
        sc = np.where(U50[p], V[p], np.nan)
        if kind == "band":
            W50[i], hl, hs = band_weights(sc, hl, hs, q, 2 * q)
        else:
            W50[i] = ls_weights(sc, None, q)
    series[f"{best_design}, top-50 coins"] = simulate(W50, reb, R, FU)
    zero_fund = simulate(W, reb, R, np.zeros_like(FU))
    series[f"{best_design}, price part only (no funding)"] = zero_fund
    weights["funding L/S (chosen design)"] = W
    series["funding L/S (chosen design)"] = series[best_design]

    rows = []
    for name, d in series.items():
        ret = pd.Series(d, index=idx).iloc[reb[0]:]
        for lab, (a, b) in PERIODS.items():
            st = stats(ret[(ret.index >= a) & (ret.index <= b)], cash, PPY)
            if st:
                rows.append({"rule": name, "lev": "1x", "period": lab, "is_t": is_t.get(name, np.nan), **st})
    # leverage on the walk-forward composites (pre-chosen: these are the only ones that never saw their test period)
    lev_series = {}
    for name in [k for k in series if k.startswith("walk-forward")] + ["funding L/S (chosen design)"]:
        base = series[name]
        for tv in (0.2, 0.4, 0.6, 0.8):
            lev = vol_target_scale(base, reb, tv)
            d = simulate(weights[name] * lev[:, None], reb, R, FU)
            lev_series[f"{name} @vol {int(tv * 100)}%"] = d
            ret = pd.Series(d, index=idx).iloc[reb[0]:]
            for lab, (a, b) in PERIODS.items():
                st = stats(ret[(ret.index >= a) & (ret.index <= b)], cash, PPY)
                if st:
                    rows.append({"rule": name, "lev": f"vol {int(tv * 100)}%", "period": lab, **st})
    t = pd.DataFrame(rows)
    # multiple testing: Deflated Sharpe of the best 1x rule over every rule tried, on the out-of-sample period
    checks = ("@0.25% cost", "top-50 coins", "price part only", "(chosen design)")
    oos = t[(t["lev"] == "1x") & (t["period"] == "OOS 2022-26") & ~t["rule"].str.contains("|".join(x.replace("(", r"\(").replace(")", r"\)") for x in checks))]
    daily_sr = oos["sharpe"] / np.sqrt(PPY)
    dsr = {}
    for rule in (oos.sort_values("sharpe").iloc[-1]["rule"], best_design):
        sr = float(oos.loc[oos["rule"] == rule, "sharpe"].iloc[0])
        bret = pd.Series(series[rule], index=idx)
        bret = bret[(bret.index >= "2022-01-01")].dropna()
        args = (len(oos), len(bret), float(bret.skew()), float(bret.kurt() + 3))
        dsr[rule] = {"oos_sharpe": sr, "dsr_strict": deflated_sharpe(sr / np.sqrt(PPY), float(daily_sr.var()), *args),
                     "dsr_lenient": deflated_sharpe(sr / np.sqrt(PPY), None, *args)}
    best = {"rule": max(dsr, key=lambda k: dsr[k]["oos_sharpe"]), "sharpe": max(v["oos_sharpe"] for v in dsr.values())}
    print(f"deflated Sharpe over {len(oos)} rules (strict = cross-trial variance, lenient = zero-skill sampling variance):", dsr)
    t.to_csv(out_path, index=False)
    pd.DataFrame({**series, **lev_series}, index=idx).iloc[reb[0]:].to_parquet(out_path.replace(".csv", "_series.parquet"))
    with open(out_path.replace(".csv", "_summary.json"), "w") as fh:
        json.dump({"is_t": is_t, "best_oos": best["rule"], "best_oos_sharpe": float(best["sharpe"]), "deflated_sharpe": dsr, "n_rules": len(oos),
                   "chosen_funding_design": best_design, "turnover": turn_avg}, fh, indent=1)
    pd.set_option("display.width", 250)
    one = t[t["lev"] == "1x"]
    print("out of sample 2022-26, 1x:\n", one[one["period"] == "OOS 2022-26"].sort_values("sharpe", ascending=False)[
        ["rule", "cagr", "vol", "sharpe", "max_dd", "worst_year"]].round(3).head(20).to_string())
    print("Sharpe 1x\n", one.pivot_table(index="rule", columns="period", values="sharpe")[list(PERIODS)].round(2).to_string())
    print("CAGR % 1x\n", (one.pivot_table(index="rule", columns="period", values="cagr")[list(PERIODS)] * 100).round(1).to_string())
    lv = t[t["lev"] != "1x"]
    print("levered walk-forward composites, CAGR % / max DD %\n",
          (lv.pivot_table(index=["rule", "lev"], columns="period", values="cagr")[list(PERIODS)] * 100).round(0).to_string(), "\n",
          (lv.pivot_table(index=["rule", "lev"], columns="period", values="max_dd")[list(PERIODS)] * 100).round(0).to_string())


if __name__ == "__main__":
    main()
