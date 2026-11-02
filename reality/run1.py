from reality.study import *
import pickle
out = {}
spy = ret["SPY"]; qqq = ret["QQQ"]
out["SPY"] = by_period(spy.loc["2005":])
out["QQQ"] = by_period(qqq.loc["2005":])

# hindsight universe: today's S&P members that turned out to be the 50 best performers 2005-2026
tot = (close.ffill().iloc[-1] / close.bfill().iloc[0]).reindex(list(cur_members)).dropna()
first_ok = close.apply(lambda s: s.first_valid_index())
long_hist = [s for s in tot.index if first_ok[s] <= pd.Timestamp("2005-03-01")]
winners = tot[long_hist].nlargest(50).index
hind = pd.DataFrame(False, index=dates, columns=close.columns); hind[list(winners)] = True
out["hindsight_names"] = list(winners[:20])
U = {"PIT": pit, "SURV": survivor, "HIND": hind}
res = {}
for uname, u in U.items():
    for n in (5, 10, 20):
        for bps in (0, 10, 25):
            if uname == "HIND" and n > 10: continue
            W = weights(u, mom, n)
            if W.empty: continue
            r, t = run_weights(ret, W, bps)
            res[f"{uname}|{n}|{bps}"] = {"periods": by_period(r), "turn": float(t.sum() / (len(t) / 252))}
            if bps == 10 and n in (5, 10):
                rr = []
                for s in range(40):
                    Wr = weights(u, mom, n, rng=np.random.default_rng(s))
                    x, _ = run_weights(ret, Wr, bps); rr.append(by_period(x))
                res[f"{uname}|{n}|{bps}|rand"] = rr
                if n == 10:
                    pickle.dump(r, open(f"/tmp/rc/eq_{uname}.pkl", "wb"))
out["mom"] = res
pickle.dump(out, open("/tmp/rc/out1.pkl", "wb"))
for k, v in res.items():
    if k.endswith("rand"):
        a = np.array([x["全期 2005-26"]["cagr"] for x in v]); b = np.array([x["樣本外 2010-19"]["cagr"] for x in v])
        print(k, "rand full mean %.3f p95 %.3f | 10-19 mean %.3f p95 %.3f" % (a.mean(), np.percentile(a, 95), b.mean(), np.percentile(b, 95)))
    else:
        p = v["periods"]; print(k, " ".join("%s %.1f%%/%.0f%%" % (kk[:4], p[kk]["cagr"] * 100, p[kk]["mdd"] * 100) for kk in p))
print("SPY", {k: round(v["cagr"], 3) for k, v in out["SPY"].items()})
