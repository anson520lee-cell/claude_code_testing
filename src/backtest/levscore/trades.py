"""python -m src.backtest.levscore.trades SIGNALS FLAGS PRICES X80_RAW RESULT_DIR : per-trade leverage score for the live
swing set-ups (crash rebound + squeeze, point-in-time S&P 500, live exits, next-open entries, 10 bp slippage a side).

Question: can what is known at the signal close (the trade's own shape, the stock, FINRA short volume, the market)
tell which trades win more, so that risk per trade goes up for those and down for the rest?
1. 2010-17 only: per input, Spearman correlation with the trade's R, win rate / mean R by thirds, t-stat clustered
   by month (trades on the same days are not independent). Inputs with |t| >= 2 and the same sign in 2010-13 and
   2014-17 form the score (signs fixed there).
2. Risk per trade = 1% x multiplier: 0.5 / 1 / 1.5 by the score's percentile among all EARLIER trades (expanding);
   also a walk-forward logistic model (refit every January on earlier trades) of the chance to win.
3. Account (10 slots, 25% cap, crash > squeeze priority, 5 seeds), marked to market daily as x80.live_mtm, compared
   with the same average risk per trade spread evenly. Variants: the live configuration (live-like stocks + FINRA
   low short-volume filter) and every S&P 500 member.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from src.backtest.edge_hunt.account import PRIORITY
from src.backtest.x80.common import load

START, END = "2010-01-01", "2026-10-31"
PERIODS = {"2010-17": (START, "2017-12-31"), "2018-26": ("2018-01-01", END), "2022-26": ("2022-01-01", END)}
TRADE_FEATS = ["beta", "hv60", "dv20", "ext_atr", "ret1", "ret5", "spy_r5", "on_share5", "speed", "sys_share", "ivol60", "max20",
               "crash_cnt", "mom12_1", "vr", "clv", "above200", "from52", "svr5_pct"]
MKT_FEATS = ["vix", "vix_ts", "spy_trend", "spy_rv20", "same_day"]
LABEL = {"beta": "beta", "hv60": "60 日波動", "dv20": "成交額", "ext_atr": "偏離 EMA21（ATR）", "ret1": "當日回報", "ret5": "5 日回報",
         "spy_r5": "SPY 5 日回報", "on_share5": "跌幅中隔夜佔比", "speed": "跌速（當日佔 5 日）", "sys_share": "跌幅中大市佔比",
         "ivol60": "特異波動", "max20": "20 日最大單日升幅", "crash_cnt": "過去一年暴跌次數", "mom12_1": "12-1 月動量",
         "vr": "成交量比", "clv": "收市位置", "above200": "企 200 日線上", "from52": "距 52 週高位", "svr5_pct": "FINRA 5 日沽空量比（百分位）",
         "vix": "VIX", "vix_ts": "VIX/VIX3M", "spy_trend": "SPY 對 200 日線", "spy_rv20": "SPY 20 日波動", "same_day": "同日訊號數目"}


def market_daily(px: dict) -> pd.DataFrame:
    vix, vix3m, spy = px["^VIX"]["close"], px["^VIX3M"]["close"], px["SPY"]["adj_close"]
    return pd.DataFrame({"vix": vix, "vix_ts": vix / vix3m.reindex(vix.index), "spy_trend": spy / spy.rolling(200).mean() - 1,
                         "spy_rv20": spy.pct_change().rolling(20).std() * np.sqrt(252)}).ffill()


def cluster_t(y: pd.Series, x: pd.Series, groups: pd.Series) -> float:
    import statsmodels.api as sm

    d = pd.concat({"y": y, "x": x, "g": groups}, axis=1).dropna()
    if len(d) < 50 or d["x"].std() == 0:
        return np.nan
    X = sm.add_constant((d["x"] - d["x"].mean()) / d["x"].std())
    r = sm.OLS(d["y"], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["g"])[0]})
    return float(r.tvalues["x"])


def feature_table(t: pd.DataFrame, feats: list[str], a: str, b: str, cuts_from: pd.DataFrame | None = None) -> pd.DataFrame:
    s = t[(t["date"] >= a) & (t["date"] <= b)]
    q = cuts_from if cuts_from is not None else s
    mid = s["date"].min() + (s["date"].max() - s["date"].min()) / 2
    rows = []
    for f in feats:
        x = s[f].astype(float)
        if x.notna().sum() < 100 or x.nunique() < 2:
            continue
        lo, hi = q[f].astype(float).quantile([1 / 3, 2 / 3])
        row = {"feature": f, "n": int(x.notna().sum()), "ic": x.corr(s["r"], method="spearman"),
               "t_cluster": cluster_t(s["r"], x, s["month"]),
               "ic_h1": x[s["date"] <= mid].corr(s["r"][s["date"] <= mid], method="spearman"),
               "ic_h2": x[s["date"] > mid].corr(s["r"][s["date"] > mid], method="spearman")}
        groups = (("low", x <= lo), ("mid", (x > lo) & (x <= hi)), ("high", x > hi)) if lo < hi else (("low", x <= lo), ("high", x > lo))
        for lab, m in groups:
            y = s["r"][m]
            row[f"{lab}_n"], row[f"{lab}_win"], row[f"{lab}_meanR"] = len(y), float((y > 0).mean()), float(y.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def expanding_pct_by_date(t: pd.DataFrame, col: str, min_n: int = 300) -> pd.Series:
    """Percentile of each trade's value among all trades signalled on EARLIER dates (same-day trades excluded)."""
    t = t.sort_values("date")
    out = pd.Series(np.nan, index=t.index)
    hist = np.array([])
    for d, g in t.groupby("date", sort=True):
        v = g[col].to_numpy(float)
        if len(hist) >= min_n:
            h = np.sort(hist)
            out[g.index] = np.searchsorted(h, v, side="right") / len(h)
        hist = np.concatenate([hist, v[np.isfinite(v)]])
    return out


def expanding_z_by_date(t: pd.DataFrame, col: str, min_n: int = 300) -> pd.Series:
    t = t.sort_values("date")
    x = t[col].astype(float)
    day = t["date"]
    s1 = x.fillna(0).groupby(day).sum().cumsum().shift(1)
    s2 = (x.fillna(0) ** 2).groupby(day).sum().cumsum().shift(1)
    n = x.notna().astype(float).groupby(day).sum().cumsum().shift(1)
    mu = (s1 / n).reindex(day).to_numpy()
    sd = np.sqrt((s2 / n - (s1 / n) ** 2).clip(lower=0)).reindex(day).to_numpy()
    z = (x.to_numpy() - mu) / sd
    z[(n.reindex(day).to_numpy() < min_n) | ~np.isfinite(z)] = np.nan
    return pd.Series(z, index=t.index).clip(-3, 3)


def tiers(p: pd.Series, mults=(0.5, 1.0, 1.5)) -> pd.Series:
    m = pd.Series(mults[1], index=p.index)
    m[p <= 1 / 3] = mults[0]
    m[p > 2 / 3] = mults[2]
    return m.where(p.notna(), 1.0)


def walk_forward_logit(t: pd.DataFrame, feats: list[str], first_year: int = 2012, min_train: int = 500, min_n: int = 300) -> pd.Series:
    from sklearn.linear_model import LogisticRegression

    t = t.sort_values("date")
    Z = pd.DataFrame({f: expanding_z_by_date(t, f, min_n) for f in feats}).fillna(0.0)
    out = pd.Series(np.nan, index=t.index)
    for year in range(first_year, t["date"].max().year + 1):
        a, b = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
        known = (t["date"] + pd.to_timedelta(t["days"] * 1.5 + 3, unit="D")) < a     # the trade had closed before January
        rows = (t["date"] >= a) & (t["date"] < b)
        if known.sum() < min_train or not rows.any():
            continue
        mdl = LogisticRegression(C=0.1, max_iter=1000).fit(Z[known].to_numpy(), (t["r"][known] > 0).astype(int).to_numpy())
        out[rows] = mdl.predict_proba(Z[rows].to_numpy())[:, 1]
    return out


def simulate(trades: pd.DataFrame, cal: pd.DatetimeIndex, A: np.ndarray, cols: dict, risk: float = 0.01, slots: int = 10,
             cap: float = 0.25, seed: int = 0, mult_col: str | None = None) -> pd.Series:
    """x80.live_mtm.simulate_mtm with risk per trade = risk x trade[mult_col] (marked to market daily)."""
    rng = np.random.default_rng(seed)
    t = trades.copy()
    t["risk_pct"] = (t["ret"] / t["r"]).where(t["r"].abs() > 1e-6)
    t = t.dropna(subset=["risk_pct"])
    t = t[t["risk_pct"] > 0]
    pos = np.searchsorted(cal.values, t["date"].values.astype("datetime64[ns]"))
    t["entry_i"] = pos + 1
    t["exit_i"] = t["entry_i"] + t["days"].astype(int) - 1
    t["prio"] = t["setup"].map(PRIORITY) + rng.random(len(t)) * 0.5
    t = t.sort_values(["entry_i", "prio"])
    mult = t[mult_col].to_numpy(float) if mult_col else np.ones(len(t))
    t["mmult"] = mult
    eq = 1.0
    open_pos: list = []
    curve = np.full(len(cal), np.nan)
    by_day = {i: g for i, g in t.groupby("entry_i")}
    for i in range(len(cal)):
        keep = []
        for ex, size, ret, j, ei in open_pos:
            if ex <= i:
                eq += size * ret
            else:
                keep.append((ex, size, ret, j, ei))
        open_pos = keep
        g = by_day.get(i)
        if g is not None:
            for row in g.itertuples():
                if len(open_pos) >= slots:
                    break
                j = cols.get(row.symbol)
                if j is None or row.mmult <= 0:
                    continue
                size = min(risk * row.mmult / row.risk_pct, cap) * eq
                open_pos.append((int(row.exit_i), size, float(row.ret), j, i))
        unreal = 0.0
        for ex, size, ret, j, ei in open_pos:
            ref = A[ei - 1, j] if ei >= 1 else np.nan
            now = A[i, j]
            if np.isfinite(ref) and np.isfinite(now) and ref > 0:
                unreal += size * (now / ref - 1)
        curve[i] = eq + unreal
        if curve[i] <= 0:
            curve[i:] = 0.0
            break
    return pd.Series(curve, index=cal)


def curve_stats(c: pd.Series, a: str, b: str) -> dict:
    x = c[(c.index >= a) & (c.index <= b)].dropna()
    x = x[x > 0]
    if len(x) < 60:
        return {}
    r = x.pct_change().dropna()
    yrs = len(r) / 252
    return {"cagr": float((x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1), "max_dd": float((x / x.cummax() - 1).min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan}


def run_accounts(t: pd.DataFrame, rules: dict[str, str], cal, A, cols, seeds: int = 5) -> pd.DataFrame:
    """Each rule's multiplier column vs a constant multiplier equal to the rule's mean over the trades it could take.
    Each account path is simulated once per seed over 2010-26 and then cut into the periods."""
    rows, cache = [], {}

    def curves(col: str | None, cm: float | None = None) -> list[pd.Series]:
        key = (col, None) if cm is None else ("const", round(cm, 9))
        if key == ("const", 1.0):
            key = (None, None)
        if key not in cache:
            if key[0] == "const":
                t["cmult"] = cm
                cache[key] = [simulate(t, cal, A, cols, seed=s, mult_col="cmult") for s in range(seeds)]
            else:
                cache[key] = [simulate(t, cal, A, cols, seed=s, mult_col=key[0]) for s in range(seeds)]
        return cache[key]

    for name, col in rules.items():
        for lab, (a, b) in PERIODS.items():
            sel = (t["date"] >= a) & (t["date"] <= b)
            cm = float(t.loc[sel, col].mean()) if col else 1.0
            res = {"rule": name, "period": lab, "mean_mult": cm}
            for kind, cs in (("rule", curves(col)), ("matched", curves(col, cm) if col else curves(None))):
                st = [curve_stats(c, a, b) for c in cs]
                for k in ("cagr", "max_dd", "sharpe"):
                    res[f"{kind}_{k}"] = float(np.mean([x[k] for x in st if x]))
            rows.append(res)
            print(name, lab, {k: round(v, 3) for k, v in res.items() if isinstance(v, float)}, flush=True)
    return pd.DataFrame(rows)


def load_trades(sig: str, flags: str, x80_raw: str) -> pd.DataFrame:
    """Crash-rebound and squeeze trades of point-in-time S&P 500 members, 2010-26, with their FINRA and market inputs."""
    s = pd.read_parquet(sig)
    s = s[s["member"] & s["setup"].isin(["crash_rebound", "squeeze"]) & (s["date"] >= START) & (s["date"] <= END)].dropna(subset=["r", "ret", "days"])
    fl = pd.read_parquet(flags)[["symbol", "date", "svr5_pct", "svr5_good"]]
    s = s.merge(fl, on=["symbol", "date"], how="left")
    px = load(os.path.join(x80_raw, "prices.parquet"))
    mk = market_daily(px)
    s = s.merge(mk, left_on="date", right_index=True, how="left")
    s["same_day"] = s.groupby(["date", "setup"])["symbol"].transform("count").astype(float)
    s["above200"] = s["above200"].astype(float)
    s["month"] = s["date"].dt.to_period("M").astype(str)
    s["win"] = (s["r"] > 0).astype(float)
    return s.sort_values("date").reset_index(drop=True)


def price_matrix(prices: str) -> tuple[pd.DatetimeIndex, np.ndarray, dict]:
    p = pd.read_parquet(prices, columns=["symbol", "date", "adj_close"])
    p["date"] = pd.to_datetime(p["date"])
    adj = p.pivot_table(index="date", columns="symbol", values="adj_close").sort_index()
    cal = adj.index[(adj.index >= START) & (adj.index <= END)]
    a = adj.reindex(index=cal).ffill()
    return cal, a.to_numpy(), {c: j for j, c in enumerate(a.columns)}


def live_mask(s: pd.DataFrame) -> pd.Series:
    return s["live_like"] & (s["svr5_good"] == 1)


def score_column(sub: pd.DataFrame, signs: dict, min_n: int) -> pd.Series:
    """Mean of the signed expanding z-scores (earlier dates only) of the chosen inputs."""
    z = pd.DataFrame({f: expanding_z_by_date(sub, f, min_n) * sg for f, sg in signs.items()})
    return z.mean(axis=1).where(z.notna().sum(axis=1) >= max(1, len(signs) // 2))


def main() -> None:
    sig, flags, prices, x80_raw, res = sys.argv[1:6]
    os.makedirs(res, exist_ok=True)
    s = load_trades(sig, flags, x80_raw)
    feats = TRADE_FEATS + MKT_FEATS
    cal, A, cols = price_matrix(prices)
    out = {}
    for variant in ("all S&P 500 members", "live (live-like + FINRA filter)"):
        sub = (s if variant.startswith("all") else s[live_mask(s)]).copy()
        T_is = feature_table(sub, feats, *PERIODS["2010-17"])
        T_oos = feature_table(sub, feats, *PERIODS["2018-26"], cuts_from=sub[(sub["date"] >= START) & (sub["date"] <= "2017-12-31")])
        pick = T_is[(T_is["t_cluster"].abs() >= 2) & (np.sign(T_is["ic_h1"]) == np.sign(T_is["ic_h2"])) & (np.sign(T_is["ic"]) == np.sign(T_is["t_cluster"]))]
        signs = {r.feature: int(np.sign(r.ic)) for r in pick.itertuples()}
        print(variant, "trades", len(sub), "inputs chosen on 2010-17:", signs, flush=True)
        small = len(sub) < 3000                                           # the live subset: ~600 trades in 17 years
        mn, mt = (100, 250) if small else (300, 500)
        rules = {"constant 1% (live)": None}
        if signs:
            sub["score"] = score_column(sub, signs, mn)
            sub["pct"] = expanding_pct_by_date(sub, "score", mn)
            sub["m_score"] = tiers(sub["pct"])
            sub["m_score_wide"] = tiers(sub["pct"], (0.25, 1.0, 1.75))
            rules["score 0.5/1/1.5"] = "m_score"
            rules["score 0.25/1/1.75"] = "m_score_wide"
        sub["p_win"] = walk_forward_logit(sub, feats, min_train=mt, min_n=mn)
        sub["m_logit"] = tiers(expanding_pct_by_date(sub, "p_win", mn))
        rules["walk-forward logistic (win prob.)"] = "m_logit"
        if variant.startswith("all"):                                     # keep the big-sample scores for the live subset
            for c in ("m_score", "m_score_wide", "m_logit"):
                if c in sub:
                    s[c + "_all"] = sub[c]
        else:                                                             # live trades sized by what ALL member trades taught
            for name, c in (("all-member score 0.5/1/1.5", "m_score_all"), ("all-member score 0.25/1/1.75", "m_score_wide_all"),
                            ("all-member logistic", "m_logit_all")):
                if c in sub:
                    sub[c] = sub[c].fillna(1.0)
                    rules[name] = c
        acc = run_accounts(sub, rules, cal, A, cols)
        # what the score's thirds actually did out of sample (win rate / mean R per tier)
        tier_rows = []
        for col in [c for c in ("m_score", "m_logit", "m_score_all", "m_logit_all") if c in sub]:
            for lab, (a_, b_) in PERIODS.items():
                x = sub[(sub["date"] >= a_) & (sub["date"] <= b_)]
                for m_, g in x.groupby(col):
                    tier_rows.append({"rule": col, "period": lab, "mult": m_, "n": len(g), "win": float(g["win"].mean()), "meanR": float(g["r"].mean())})
        out[variant] = {"signs": signs, "n": len(sub), "features_is": T_is.to_dict("records"), "features_oos": T_oos.to_dict("records"),
                        "accounts": acc.to_dict("records"), "tiers": tier_rows}
        tag = "live" if variant.startswith("live") else "all"
        T_is.to_csv(os.path.join(res, f"trade_features_{tag}_2010_17.csv"), index=False)
        T_oos.to_csv(os.path.join(res, f"trade_features_{tag}_2018_26.csv"), index=False)
        acc.to_csv(os.path.join(res, f"trade_accounts_{tag}.csv"), index=False)
        pd.DataFrame(tier_rows).to_csv(os.path.join(res, f"trade_tiers_{tag}.csv"), index=False)
    with open(os.path.join(res, "trades.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
