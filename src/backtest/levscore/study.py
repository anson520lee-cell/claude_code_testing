"""python -m src.backtest.levscore.study X80_OUT RESULT_DIR : can a score decide how much leverage the x80 book should use?

Protocol (fixed before looking at 2018-26):
1. The book is the x80 round-2 rule (16 sleeves, equal risk contribution over 104 weeks, leverage = target vol / the
   book's whole-history vol, cap 10x). Its weekly excess return is the thing a score has to predict.
2. Candidate inputs (features.py) are judged on 2010-17 ONLY: Spearman correlation with the next week's and the next
   four weeks' book return, a Newey-West t-stat (4 lags) of the four-week return on the input, and the sign in both
   halves (2010-13, 2014-17). An input enters the composite score when |t| >= 2 and both halves agree; its sign is
   fixed there.
3. Rules (rules.py) turn a score into a multiplier m[t] on the x80 leverage, using only data up to week t-1:
   tiers 0.5 / 1 / 1.5 by the score's expanding tercile, a wider 0.25 / 1 / 1.75, a continuous version, walk-forward
   ridge (four-week mean) and logistic (probability of a winning week) models refitted every January, Neo's literal
   rule (trailing 26-week win rate), shorter vol windows, a stress switch (VIX >= VIX3M -> half), the x80 drawdown
   brake, and a sleeve-level momentum tilt.
4. Every rule is compared with the SAME AVERAGE LEVERAGE spread evenly (constant multiplier = the rule's mean
   leverage / the x80 leverage in that period), so a rule cannot win just by borrowing more. Costs as meta.lever;
   the intraweek trough from daily data counts a wipe-out inside a week.
5. Significance: Newey-West t of the weekly return difference, and a placebo (the rule's own multipliers shuffled
   in 13-week blocks, 500 times) for the CAGR and drawdown differences.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from src.backtest.levscore import book as bk
from src.backtest.levscore import features as ft
from src.backtest.levscore import rules as rl
from src.backtest.x80 import meta
from src.backtest.x80.common import load

IS = ("2010-01-01", "2017-12-31")
PERIODS = {"2010-17": IS, "2018-26": ("2018-01-01", "2030-01-01"), "2022-26": ("2022-01-01", "2030-01-01")}
TARGETS = (0.3, 0.6, 0.8)
SEED = 7


def hac_t(y: pd.Series, x: pd.Series | None = None, lags: int = 4) -> tuple[float, float]:
    """(coefficient, Newey-West t) of y on [1, x] (x None: of the mean of y)."""
    import statsmodels.api as sm

    d = pd.concat({"y": y, "x": x if x is not None else pd.Series(1.0, index=y.index)}, axis=1).dropna()
    if len(d) < 30:
        return np.nan, np.nan
    X = sm.add_constant(d["x"]) if x is not None else d[["x"]]
    r = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    k = "x"
    return float(r.params[k]), float(r.tvalues[k])


def feature_table(F: pd.DataFrame, base: pd.Series, a: str, b: str, q_from: tuple[str, str] | None = None) -> pd.DataFrame:
    """Per input: correlation with the next week / next 4 weeks, Newey-West t, and the next week's book return,
    win rate and Sharpe in the input's low / middle / high third (thirds cut on q_from, default the same period)."""
    y1 = base.shift(-1)
    y4 = base.rolling(4).sum().shift(-4)
    rows = []
    sel = (F.index >= a) & (F.index <= b)
    qa, qb = q_from or (a, b)
    qsel = (F.index >= qa) & (F.index <= qb)
    for c in F.columns:
        x = F[c][sel]
        if x.notna().sum() < 100:
            continue
        z = (x - x.mean()) / x.std()
        ic1 = x.corr(y1[sel], method="spearman")
        ic4 = x.corr(y4[sel], method="spearman")
        _, t4 = hac_t(y4[sel], z, 4)
        _, t1 = hac_t(y1[sel], z, 1)
        mid = pd.Timestamp(a) + (pd.Timestamp(min(pd.Timestamp(b), F.index.max())) - pd.Timestamp(a)) / 2
        h1, h2 = x.index <= mid, x.index > mid
        ic_h1 = x[h1].corr(y4[sel][h1], method="spearman")
        ic_h2 = x[h2].corr(y4[sel][h2], method="spearman")
        lo, hi = F[c][qsel].quantile([1 / 3, 2 / 3])
        row = {"feature": c, "n": int(x.notna().sum()), "ic1": ic1, "ic4": ic4, "t1": t1, "t4": t4, "ic4_h1": ic_h1, "ic4_h2": ic_h2}
        for lab, m in (("low", x <= lo), ("mid", (x > lo) & (x <= hi)), ("high", x > hi)):
            yy = y1[sel][m].dropna()
            row[f"{lab}_ann"] = float(yy.mean() * 52) if len(yy) else np.nan
            row[f"{lab}_hit"] = float((yy > 0).mean()) if len(yy) else np.nan
            row[f"{lab}_sharpe"] = float(yy.mean() / yy.std() * np.sqrt(52)) if len(yy) > 5 else np.nan
            row[f"{lab}_n"] = len(yy)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(name: str, m: pd.Series, base, fin, cash_w, trough, target: float, cap: float = bk.CAP,
             lev0: pd.Series | None = None) -> tuple[list[dict], dict]:
    """Rule vs the same average leverage spread evenly, per period."""
    L0 = bk.vol_leverage(base, target) if lev0 is None else lev0
    lev = (L0 * m.reindex(base.index).fillna(1.0)).clip(lower=0, upper=cap)
    tr = trough["trough"]
    r = bk.lever_path(base, fin, cash_w, lev, trough=tr)
    rows, series = [], {"rule": r, "lev": lev}
    for lab, (a, b) in PERIODS.items():
        sel = (base.index >= a) & (base.index <= b) & L0.notna()
        if sel.sum() < 52:
            continue
        c = float(lev[sel].mean() / L0[sel].mean())
        lm = (L0 * c).clip(upper=cap)
        rm = bk.lever_path(base, fin, cash_w, lm, trough=tr)
        rr, mm = r[(r.index >= a) & (r.index <= b)], rm[(rm.index >= a) & (rm.index <= b)]
        pr, pm = bk.perf(rr, cash_w), bk.perf(mm, cash_w)
        if not pr or not pm:
            continue
        d = (rr - mm).dropna()
        dlog = (np.log1p(rr.clip(lower=-0.999)) - np.log1p(mm.clip(lower=-0.999))).dropna()
        _, td = hac_t(d, None, 4)
        _, tl = hac_t(dlog, None, 4)
        tw = trough.reindex(rr.index)["trough"]
        rows.append({"rule": name, "target": target, "period": lab, "avg_lev": float(lev[sel].mean()), "avg_lev_x80": float(L0[sel].mean()),
                     "match_mult": c, **{k: v for k, v in pr.items()}, **{f"m_{k}": v for k, v in pm.items()},
                     "d_ann": float(d.mean() * 52), "t_d": td, "d_log_ann": float(dlog.mean() * 52), "t_log": tl,
                     "intraweek_worst": float((lev.reindex(tw.index) * tw).min()),
                     "weeks_intraweek_loss_gt50": int(((lev.reindex(tw.index) * tw) <= -0.5).sum()),
                     "m_intraweek_worst": float((lm.reindex(tw.index) * tw).min())})
        series[f"matched {lab}"] = rm
    return rows, series


def placebo(m: pd.Series, base, fin, cash_w, trough, target: float, a: str, b: str, n: int = 500, block: int = 13) -> dict:
    """Shuffle the rule's multipliers inside [a, b] in blocks of ``block`` weeks; the share of shuffles that beat the
    real rule's CAGR / drawdown edge over its matched constant is the p-value (timing skill vs luck)."""
    rng = np.random.default_rng(SEED)
    L0 = bk.vol_leverage(base, target)
    sel = (base.index >= a) & (base.index <= b) & L0.notna()
    idx = base.index[sel]
    mv = m.reindex(idx).fillna(1.0).to_numpy()

    def edge(mm: np.ndarray) -> tuple[float, float]:
        mfull = m.copy().reindex(base.index).fillna(1.0)
        mfull.loc[idx] = mm
        lev = (L0 * mfull).clip(lower=0, upper=bk.CAP)
        r = bk.lever_path(base, fin, cash_w, lev, trough=trough["trough"])
        c = float(lev[sel].mean() / L0[sel].mean())
        rm = bk.lever_path(base, fin, cash_w, (L0 * c).clip(upper=bk.CAP), trough=trough["trough"])
        pr, pm = bk.perf(r[(r.index >= a) & (r.index <= b)]), bk.perf(rm[(rm.index >= a) & (rm.index <= b)])
        return pr["cagr"] - pm["cagr"], pr["max_dd"] - pm["max_dd"]

    real = edge(mv)
    nb = int(np.ceil(len(mv) / block))
    blocks = [mv[i * block:(i + 1) * block] for i in range(nb)]
    sims = []
    for _ in range(n):
        order = rng.permutation(nb)
        sims.append(edge(np.concatenate([blocks[j] for j in order])[: len(mv)]))
    s = np.array(sims)
    return {"d_cagr": real[0], "d_maxdd": real[1], "p_cagr": float((s[:, 0] >= real[0]).mean()),
            "p_maxdd": float((s[:, 1] >= real[1]).mean()), "placebo_d_cagr_median": float(np.median(s[:, 0])),
            "placebo_d_cagr_p95": float(np.quantile(s[:, 0], 0.95)), "n": n}


def whatif(base, fin, cash_w, trough, target: float, a: str, b: str, rhos=(0.03, 0.05, 0.1, 0.2, 1.0), draws: int = 40) -> list[dict]:
    """HINDSIGHT what-if, not a strategy: a made-up score whose correlation with NEXT week's book return is rho, turned
    into the same 0.5 / 1 / 1.5 tiers. Shows how good a win-rate forecast would have to be to matter."""
    rng = np.random.default_rng(SEED)
    y = base.shift(-1)
    z = (y - y.mean()) / y.std()
    out = []
    for rho in rhos:
        res = []
        for _ in range(draws if rho < 1 else 1):
            noise = pd.Series(rng.standard_normal(len(z)), index=z.index)
            sc = rho * z.fillna(0.0) + np.sqrt(max(1 - rho ** 2, 0.0)) * noise
            m = rl.tiers(rl.expanding_rank(sc)).shift(1).fillna(1.0)
            rows, _ = evaluate("what-if", m, base, fin, cash_w, trough, target)
            r = [x for x in rows if x["period"] == ("2018-26" if a.startswith("2018") else "2022-26")]
            if r:
                res.append((r[0]["cagr"] - r[0]["m_cagr"], r[0]["max_dd"] - r[0]["m_max_dd"], r[0]["sharpe"] - r[0]["m_sharpe"]))
        res = np.array(res)
        out.append({"rho": rho, "target": target, "period": a[:4] + "-26", "d_cagr_median": float(np.median(res[:, 0])),
                    "d_maxdd_median": float(np.median(res[:, 1])), "d_sharpe_median": float(np.median(res[:, 2])),
                    "d_cagr_p10": float(np.quantile(res[:, 0], 0.1)), "d_cagr_p90": float(np.quantile(res[:, 0], 0.9))})
        print("what-if", out[-1], flush=True)
    return out


def sleeve_momentum_book(w_ex: pd.DataFrame, cash_w: pd.Series, W: pd.DataFrame, look: int = 52, hi: float = 1.5, lo: float = 0.5):
    """The same ERC weights tilted by each sleeve's own trailing ``look``-week excess return (known at t-1), renormalised;
    accounting as meta.run (live sleeves only, drifted turnover, funded / unfunded shares)."""
    mom = w_ex.rolling(look, min_periods=look // 2).sum().shift(1)
    tilt = pd.DataFrame(np.where(mom > 0, hi, lo), index=w_ex.index, columns=w_ex.columns).where(mom.notna(), 1.0)
    W2 = (W * tilt.reindex(W.index)).fillna(0.0)
    sp = pd.Series({c: meta.spread_of(c) for c in w_ex.columns})
    unf = pd.Series({c: c in meta.UNFUNDED for c in w_ex.columns})
    base = pd.Series(np.nan, index=w_ex.index)
    fin = pd.DataFrame(0.0, index=w_ex.index, columns=["funded", "spread", "unfunded", "turn"])
    drifted = pd.Series(0.0, index=w_ex.columns)
    for t in W2.index:
        row = w_ex.loc[t]
        live = row.notna()
        w = W2.loc[t].where(live, 0.0)
        if w.sum() <= 0:
            base[t] = 0.0
            fin.loc[t, "turn"] = float(drifted.abs().sum())
            drifted[:] = 0.0
            continue
        w = w / w.sum()
        base[t] = float((w * row.fillna(0.0)).sum())
        f = w[~unf].sum()
        fin.loc[t] = [f, float((w[~unf] * sp[~unf]).sum() / f) if f > 0 else 0.0, float(w[unf].sum()), float((w - drifted).abs().sum())]
        drifted = w * (1 + row.fillna(0.0)) / (1 + base[t])
    return base, fin, W2


def main() -> None:
    x80_out, res = sys.argv[1:3]
    os.makedirs(res, exist_ok=True)
    w_ex = pd.read_parquet(os.path.join(x80_out, "sleeves_weekly.parquet"))
    daily = pd.read_parquet(os.path.join(x80_out, "sleeves_daily.parquet"))
    cash_w = pd.read_parquet(os.path.join(x80_out, "cash_weekly.parquet"))["cash"]
    base, fin, W = bk.base_book(w_ex, cash_w)
    trough = bk.intraweek_trough(daily, W, w_ex)
    px = load()
    ex = pd.read_parquet("/home/user/research_data/x80/raw/prices_extra.parquet")
    ex["date"] = pd.to_datetime(ex["date"])
    for s, g in ex.groupby("symbol"):
        px[s] = g.set_index("date").sort_index()[["open", "high", "low", "close", "adj_close", "volume"]]
    F = ft.all_features(px, base, w_ex, W)
    F.to_parquet(os.path.join(res, "features.parquet"))
    pd.DataFrame({"base": base}).join(fin).join(trough).to_parquet(os.path.join(res, "book.parquet"))
    W.to_parquet(os.path.join(res, "weights.parquet"))

    # ---- 0. the baseline: reproduce x80's numbers
    base_rows = []
    for tv in TARGETS:
        L0 = bk.vol_leverage(base, tv)
        r = bk.lever_path(base, fin, cash_w, L0)
        rt = bk.lever_path(base, fin, cash_w, L0, trough=trough["trough"])
        x, _ = meta.lever(base, fin, cash_w, "vol", tv, vol_win=0)
        for lab, (a, b) in PERIODS.items():
            sl = lambda s, a=a, b=b: s[(s.index >= a) & (s.index <= b)]
            base_rows.append({"target": tv, "period": lab, "x80_lever": bk.perf(sl(x), cash_w), "weekly": bk.perf(sl(r), cash_w),
                              "with_intraweek": bk.perf(sl(rt), cash_w), "avg_lev": float(sl(L0).mean())})
    with open(os.path.join(res, "baseline.json"), "w") as fh:
        json.dump(base_rows, fh, indent=1, default=float)
    for row in base_rows:
        print("baseline", row["target"], row["period"], {k: round(v, 3) for k, v in row["weekly"].items() if k in ("cagr", "max_dd", "sharpe")},
              "intraweek:", {k: round(v, 3) for k, v in row["with_intraweek"].items() if k in ("cagr", "max_dd")}, "lev", round(row["avg_lev"], 2))

    # ---- 1. which inputs predict the book's next weeks? (2010-17 only)
    T_is = feature_table(F, base, *IS)
    T_oos = feature_table(F, base, "2018-01-01", "2030-01-01", q_from=IS)
    T_is.to_csv(os.path.join(res, "features_2010_17.csv"), index=False)
    T_oos.to_csv(os.path.join(res, "features_2018_26.csv"), index=False)
    chosen = T_is[(T_is["t4"].abs() >= 2.0) & (np.sign(T_is["ic4_h1"]) == np.sign(T_is["ic4_h2"])) & (np.sign(T_is["ic4"]) == np.sign(T_is["t4"]))]
    signs = {r.feature: int(np.sign(r.t4)) for r in chosen.itertuples()}
    print("inputs chosen on 2010-17:", signs)
    with open(os.path.join(res, "chosen.json"), "w") as fh:
        json.dump({"signs": signs}, fh, indent=1)

    # ---- 2. rules
    rules: dict[str, pd.Series] = {"x80 (constant vol target)": pd.Series(1.0, index=base.index)}
    for r_ in T_is.itertuples():                                           # each input alone, direction fixed on 2010-17
        if np.isfinite(r_.ic4) and r_.ic4 != 0:
            rules[f"single: {r_.feature}"] = rl.single(F[r_.feature], int(np.sign(r_.ic4)))
    if signs:
        score = rl.composite(F, signs)
        rules["composite 0.5/1/1.5"] = rl.from_score(score)
        rules["composite 0.25/1/1.75"] = rl.from_score(score, (0.25, 1.0, 1.75))
        rules["composite continuous"] = rl.continuous(score)
        score.to_frame("score").to_parquet(os.path.join(res, "score.parquet"))
    y4 = base.rolling(4).sum().shift(-4)
    y1 = base.shift(-1)
    pr = rl.walk_forward_model(F, y4, 4, 2010, "ridge")
    rules["walk-forward ridge (4-week mean)"] = rl.tiers(rl.expanding_rank(pr, 52)).shift(1).fillna(1.0)
    pl = rl.walk_forward_model(F, y1, 1, 2010, "logit")
    rules["walk-forward logistic (win prob.)"] = rl.tiers(rl.expanding_rank(pl, 52)).shift(1).fillna(1.0)
    pd.DataFrame({"ridge": pr, "logit": pl}).to_parquet(os.path.join(res, "model_predictions.parquet"))
    rules["Neo: trailing 26-week win rate"] = rl.single(F["b_hit26"], +1)
    rules["stress switch: VIX >= VIX3M -> 0.5x"] = (1 - 0.5 * (F["vix_ts"] >= 1.0).astype(float)).shift(1).fillna(1.0)
    rules["strategy momentum: 26w < 0 -> 0.5x"] = (1 - 0.5 * (F["b_mom26"] < 0).astype(float)).shift(1).fillna(1.0)

    # ---- 3. evaluate
    rows, keep, plac = [], {}, []
    tr2 = None
    for tv in TARGETS:
        L0 = bk.vol_leverage(base, tv)
        for name, m in rules.items():
            rws, ser = evaluate(name, m, base, fin, cash_w, trough, tv)
            rows += rws
            if tv == 0.8:
                keep[name] = ser["rule"]
                keep[f"{name} | lev"] = ser["lev"]
        for vw in (13, 26, 52):                                            # risk score: shorter vol windows (x80 picked 'all' on 2010-17)
            Lv = bk.vol_leverage(base, tv, vol_win=vw)
            rws, ser = evaluate(f"vol window {vw}w", Lv / L0, base, fin, cash_w, trough, tv)
            rows += rws
        bm, fm, W2 = sleeve_momentum_book(w_ex, cash_w, W)
        tr2 = bk.intraweek_trough(daily, W2, w_ex) if tv == TARGETS[0] else tr2
        Lm = bk.vol_leverage(bm, tv)
        rm_ = bk.lever_path(bm, fm, cash_w, Lm, trough=tr2["trough"])
        rb_ = bk.lever_path(base, fin, cash_w, L0, trough=trough["trough"])
        for lab, (a, b) in PERIODS.items():
            sl = lambda x, a=a, b=b: x[(x.index >= a) & (x.index <= b)]
            pr, pm = bk.perf(sl(rm_), cash_w), bk.perf(sl(rb_), cash_w)
            d = (sl(rm_) - sl(rb_)).dropna()
            _, td = hac_t(d, None, 4)
            rows.append({"rule": "sleeve momentum tilt 1.5/0.5 (52w)", "target": tv, "period": lab, "avg_lev": float(sl(Lm).mean()),
                         "avg_lev_x80": float(sl(L0).mean()), "match_mult": np.nan, **pr, **{f"m_{k}": v for k, v in pm.items()},
                         "d_ann": float(d.mean() * 52), "t_d": td, "note": "vs x80 at the same vol target (not matched leverage)"})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(res, "rules.csv"), index=False)
    pd.DataFrame(keep).to_parquet(os.path.join(res, "rules_series_80.parquet"))
    pd.DataFrame(rules).to_parquet(os.path.join(res, "multipliers.parquet"))
    # placebo for the pre-registered rules (not the single-input ones)
    main_rules = [k for k in rules if not k.startswith("single:") and k != "x80 (constant vol target)"]
    for tv in (0.3, 0.8):
        for name in main_rules:
            for lab in ("2018-26", "2022-26"):
                a, b = PERIODS[lab]
                p = placebo(rules[name], base, fin, cash_w, trough, tv, a, b, n=300)
                plac.append({"rule": name, "target": tv, "period": lab, **p})
                print("placebo", name, tv, lab, {k: round(v, 3) for k, v in p.items() if k != "n"}, flush=True)
    pd.DataFrame(plac).to_csv(os.path.join(res, "placebo.csv"), index=False)
    wi = []
    for tv in (0.3, 0.8):
        wi += whatif(base, fin, cash_w, trough, tv, "2018-01-01", "2030-01-01")
    pd.DataFrame(wi).to_csv(os.path.join(res, "whatif.csv"), index=False)
    pd.set_option("display.width", 250)
    show = t[t["target"] == 0.8][["rule", "period", "avg_lev", "cagr", "max_dd", "sharpe", "m_cagr", "m_max_dd", "m_sharpe", "d_ann", "t_d"]]
    print(show.round(3).to_string())


if __name__ == "__main__":
    main()
