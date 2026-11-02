from reality.study import *
import pickle
fr = pd.read_pickle("/tmp/rc/proj.pkl")
pc = pd.DataFrame({s: f["Close"] for s, f in fr.items()}).sort_index().reindex(close.index)
# put project names onto the common panel (same dates), compute everything on that sub-universe
import reality.study as S
S.close = pc; S.ret = pc.pct_change(fill_method=None)
S.mom = pc.shift(21) / pc.shift(252) - 1
S.dvol = (pc * pd.DataFrame({s: f["Volume"] for s, f in fr.items()}).sort_index().reindex(close.index)).rolling(60, min_periods=40).mean()
proj = pd.DataFrame(True, index=close.index, columns=pc.columns)
res = {}
for n in (5, 10):
    W = S.weights(proj, S.mom, n); r, _ = run_weights(S.ret, W, 10)
    rr = []
    for sd in range(40):
        Wr = S.weights(proj, S.mom, n, rng=np.random.default_rng(sd)); x, _ = run_weights(S.ret, Wr, 10); rr.append(by_period(x))
    res[f"momentum|{n}"] = by_period(r); res[f"random|{n}"] = rr
# equal weight everything monthly
rows = {}
for d in S.ME:
    ok = proj.loc[d] & pc.loc[d].notna() & S.mom.loc[d].notna()
    if ok.sum() < 30: continue
    w = pd.Series(0.0, index=pc.columns); w[ok[ok].index] = 1 / ok.sum(); rows[d] = w
r, _ = run_weights(S.ret, pd.DataFrame(rows).T, 10); res["equal_weight"] = by_period(r)
pickle.dump(res, open("/tmp/rc/out4.pkl", "wb"))
for k in ("momentum|5", "momentum|10", "equal_weight"):
    p = res[k]; print(k, " ".join("%s %.1f%%/%.0f%%" % (kk[:6], p[kk]["cagr"] * 100, p[kk]["mdd"] * 100) for kk in p))
for k in ("random|5", "random|10"):
    a = [x for x in res[k]]
    print(k, " ".join("%s mean %.1f%% p95 %.1f%%" % (kk[:6], np.mean([x[kk]["cagr"] for x in a]) * 100, np.percentile([x[kk]["cagr"] for x in a], 95) * 100) for kk in a[0]))
