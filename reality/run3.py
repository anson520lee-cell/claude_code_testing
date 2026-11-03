from reality.study import *
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor
out = {}
# ------------------------------------------------ rolling windows / calendar years
eqs = {u: pickle.load(open(f"/tmp/rc/eq_{u}.pkl", "rb")) for u in ("PIT", "SURV", "HIND")}
cal = {}
for u, r in eqs.items():
    y = (1 + r).groupby(r.index.year).prod() - 1
    cal[u] = {int(k): float(v) for k, v in y.items() if k >= 2005}
cal["SPY"] = {int(k): float(v) for k, v in ((1 + ret["SPY"].loc["2005":]).groupby(ret["SPY"].loc["2005":].index.year).prod() - 1).items()}
out["calendar"] = cal
roll = {}
for u, r in eqs.items():
    r = r.loc["2005-06":]
    e = (1 + r).cumprod()
    for yrs in (1, 3, 5):
        k = 252 * yrs
        c = (e / e.shift(k)) ** (1 / yrs) - 1
        c = c.dropna()
        roll[f"{u}|{yrs}y"] = {"share_ge90": float((c >= .9).mean()), "share_ge40": float((c >= .4).mean()), "median": float(c.median()), "max": float(c.max()), "min": float(c.min())}
out["rolling"] = roll
# ------------------------------------------------ hindsight winners persistence
yr_end = {y: dates[dates.year == y][-1] for y in range(2005, 2027) if (dates.year == y).any()}
ps = []
for y in range(2006, 2026):
    a, b, c = yr_end[y - 1], yr_end[y], yr_end[y + 1]
    mem = pit.loc[a] & close.loc[a].notna() & close.loc[b].notna() & close.loc[c].notna()
    names = mem[mem].index
    r1 = close.loc[b, names] / close.loc[a, names] - 1
    r2 = close.loc[c, names] / close.loc[b, names] - 1
    top = r1.nlargest(10).index
    ps.append({"year": y, "n": len(names), "top10_ret": float(r1[top].mean()), "top10_next": float(r2[top].mean()), "univ_next": float(r2.mean()),
               "share_ge90": float((r1 >= .9).mean()), "next_rank_pct": float(r2.rank(pct=True)[top].mean())})
out["persist"] = ps
# 5y windows: how many PIT stocks ever had >=90% CAGR over 5y?
cnt = []
for y in range(2005, 2022):
    a, b = yr_end[y], yr_end[y + 5]
    mem = pit.loc[a] & close.loc[a].notna() & close.loc[b].notna()
    r = (close.loc[b, mem[mem].index] / close.loc[a, mem[mem].index]) ** .2 - 1
    cnt.append({"start": y, "n": int(mem.sum()), "ge90": int((r >= .9).sum()), "ge40": int((r >= .4).sum())})
out["five_year"] = cnt

# ------------------------------------------------ overfit sweep on a hindsight basket
cm = [s for s in cur_members if close[s].loc["2014-12-31"] > 0 and close[s].first_valid_index() <= pd.Timestamp("2014-12-01")]
tot = {s: close[s].loc["2020-12-31"] / close[s].loc["2014-12-31"] for s in cm if pd.notna(close[s].loc["2020-12-31"])}
basket = sorted(tot, key=tot.get, reverse=True)[:10]
out["basket"] = basket
IS = ("2015-01-01", "2020-12-31"); OOS = ("2021-01-01", "2026-12-31")
res = []
px = close[basket]; rr = px.pct_change(fill_method=None).fillna(0)
for fast in range(3, 61, 3):
    for slow in range(30, 301, 10):
        if fast >= slow: continue
        sf = px.rolling(fast).mean(); ss = px.rolling(slow).mean()
        pos = (sf > ss).shift(1).fillna(False).astype(float)
        sw = pos.diff().abs().fillna(0)
        pr = (pos * rr - sw * 0.001).mean(axis=1)
        res.append((fast, slow, stats(pr.loc[IS[0]:IS[1]])["cagr"], stats(pr.loc[OOS[0]:OOS[1]])["cagr"]))
df = pd.DataFrame(res, columns=["fast", "slow", "is", "oos"])
bh = rr.mean(axis=1)
best = df.loc[df["is"].idxmax()]
out["sweep"] = {"n": len(df), "best": best.to_dict(), "is_median": float(df["is"].median()), "oos_median": float(df["oos"].median()),
                "oos_of_top10_is": float(df.nlargest(10, "is")["oos"].mean()), "corr": float(df["is"].corr(df["oos"])),
                "bh_is": stats(bh.loc[IS[0]:IS[1]])["cagr"], "bh_oos": stats(bh.loc[OOS[0]:OOS[1]])["cagr"],
                "spy_is": stats(ret["SPY"].loc[IS[0]:IS[1]])["cagr"], "spy_oos": stats(ret["SPY"].loc[OOS[0]:OOS[1]])["cagr"],
                "share_oos_gt_spy": float((df["oos"] > stats(ret["SPY"].loc[OOS[0]:OOS[1]])["cagr"]).mean())}
print(out["sweep"])

# ------------------------------------------------ "AI": gradient-boosting stock picker, walk-forward
feat = {}
r1 = close / close.shift(21) - 1; r3 = close / close.shift(63) - 1; r6 = close / close.shift(126) - 1
r5 = close / close.shift(5) - 1; vol60 = ret.rolling(60).std(); hi = close / close.rolling(252).max()
ldv = np.log(dvol)
fut = close.shift(-21) / close - 1          # label-only forward return (not a feature)
rows = []
for d in ME:
    i = dates.get_loc(d)
    if i + 22 >= len(dates): fwd_ok = False
    ok = (pit.loc[d] | survivor.loc[d]) & close.loc[d].notna() & mom.loc[d].notna() & vol60.loc[d].notna()
    idx = ok[ok].index
    X = pd.DataFrame({"r1": r1.loc[d, idx], "r3": r3.loc[d, idx], "r6": r6.loc[d, idx], "mom": mom.loc[d, idx], "r5": r5.loc[d, idx],
                      "vol": vol60.loc[d, idx], "hi": hi.loc[d, idx], "ldv": ldv.loc[d, idx]})
    X = X.replace([np.inf, -np.inf], np.nan)
    # cross-sectional ranks as features
    Xr = X.rank(pct=True); Xr.columns = [c + "_rk" for c in X.columns]
    X = pd.concat([X, Xr], axis=1)
    X["date"] = d; X["sym"] = idx; X["fwd"] = fut.loc[d, idx].values; X["pit"] = pit.loc[d, idx].values
    rows.append(X)
P = pd.concat(rows, ignore_index=True)
P["y"] = P["fwd"] - P.groupby("date")["fwd"].transform("median")
fcols = [c for c in P.columns if c not in ("date", "sym", "fwd", "pit", "y")]
preds = []
for yr in range(2011, 2027):
    t0 = pd.Timestamp(f"{yr}-01-01")
    train = P[(P.date < t0 - pd.Timedelta(days=35)) & P.y.notna() & P.pit]       # train only on rows whose label ended before test
    test = P[(P.date >= t0) & (P.date < pd.Timestamp(f"{yr+1}-01-01"))]
    if len(train) < 5000 or test.empty: continue
    m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=150, min_samples_leaf=200, l2_regularization=5.0, random_state=0)
    m.fit(train[fcols], train.y)
    tt = test.copy(); tt["score"] = m.predict(test[fcols]); preds.append(tt)
PR = pd.concat(preds, ignore_index=True)
def ml_port(pitonly, n, bps):
    rows = {}
    for d, g in PR.groupby("date"):
        g = g[g.pit] if pitonly else g
        top = g.nlargest(n, "score")
        w = pd.Series(0.0, index=close.columns); w[list(top.sym)] = 1.0 / n; rows[d] = w
    W = pd.DataFrame(rows).T
    r, _ = run_weights(ret, W, bps); return r.loc["2011":]
ml = {}
for lab, po, n, bps in (("AI選股 前20名 (成分股, 10bps)", True, 20, 10), ("AI選股 前20名 (成分股, 0成本)", True, 20, 0), ("AI選股 前10名 (成分股, 10bps)", True, 10, 10), ("AI選股 前20名 (含今日成分股偏差, 10bps)", False, 20, 10)):
    r = ml_port(po, n, bps)
    ml[lab] = {"periods": {k: stats(r.loc[a:b]) for k, (a, b) in PERIODS.items() if a >= "2010-01-01"}, "full": stats(r.loc["2011":])}
# benchmarks over the same OOS months
base = {"SPY": ret["SPY"].loc["2011":]}
ml["SPY"] = {"periods": {k: stats(base["SPY"].loc[a:b]) for k, (a, b) in PERIODS.items() if a >= "2010-01-01"}, "full": stats(base["SPY"])}
# information coefficient
ic = PR[PR.pit].groupby("date").apply(lambda g: g.score.corr(g.y, method="spearman"))
out["ml"] = ml; out["ml_ic"] = {"mean": float(ic.mean()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic))), "n": int(len(ic))}
# random top-20 on the same PIT rows
rng = np.random.default_rng(1); rr_ = []
for s in range(25):
    rows = {}
    for d, g in PR[PR.pit].groupby("date"):
        top = g.sample(20, random_state=int(rng.integers(1e9)))
        w = pd.Series(0.0, index=close.columns); w[list(top.sym)] = 0.05; rows[d] = w
    r, _ = run_weights(ret, pd.DataFrame(rows).T, 10); rr_.append(stats(r.loc["2011":])["cagr"])
out["ml_random_cagr"] = {"mean": float(np.mean(rr_)), "p95": float(np.percentile(rr_, 95))}
pickle.dump(out, open("/tmp/rc/out3.pkl", "wb"))
for k, v in ml.items():
    print(k.ljust(34), "full11-26 %.1f%%/%.0f%%" % (v["full"]["cagr"]*100, v["full"]["mdd"]*100), " ".join("%s %.1f%%/%.0f%%" % (kk[:6], p["cagr"] * 100, p["mdd"] * 100) for kk, p in v["periods"].items()))
print(out["ml_ic"], out["ml_random_cagr"])
print(roll); print(cal["PIT"]); print(cal["SURV"])
print(pd.DataFrame(ps).round(3).to_string()); print(pd.DataFrame(cnt).to_string())
