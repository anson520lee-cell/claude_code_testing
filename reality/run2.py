from reality.study import *
import pickle
out = {}
CASH = {2003:.011,2004:.014,2005:.032,2006:.048,2007:.044,2008:.014,2009:.0015,2010:.0014,2011:.0005,2012:.0009,2013:.0006,2014:.0003,
        2015:.0005,2016:.003,2017:.009,2018:.019,2019:.021,2020:.004,2021:.0005,2022:.02,2023:.051,2024:.05,2025:.042,2026:.039}
cash = pd.Series([CASH[d.year] for d in dates], index=dates)
cd = cash / TRADING_DAYS

def timing(und_ret, sig_close, lev, n=200, cost=0.001):
    """lev x underlying when yesterday's close > SMA(n), else cash. cost per switch on notional."""
    sma = sig_close.rolling(n).mean()
    on = (sig_close > sma).shift(1).fillna(False)
    levr = leveraged(und_ret, lev, cash=cash)
    r = np.where(on, levr, cd)
    sw = (on != on.shift(1)).astype(float).fillna(0)
    r = pd.Series(r, index=dates) - sw * cost * lev
    return r, int(sw.sum())

eq = {}
rows = {}
def add(name, r, **kw):
    r = r.loc["2005":]
    rows[name] = {"periods": by_period(r), **kw}; eq[name] = r
qr, sr, mr = ret["QQQ"], ret["SPY"], ret["SMH"]
add("SPY 買入持有", sr); add("QQQ 買入持有", qr)
for L in (2, 3):
    add(f"QQQ {L}x 買入持有(模擬)", leveraged(qr, L, cash=cash))
add("SPY 3x 買入持有(模擬,≈UPRO)", leveraged(sr, 3, cash=cash))
add("SMH 3x 買入持有(模擬,≈SOXL)", leveraged(mr, 3, cash=cash))
for t in ("TQQQ", "UPRO", "SOXL"):
    if t in ret: add(f"{t} 真實", ret[t].dropna())
r, nsw = timing(qr, close["QQQ"], 3); add("QQQ 3x + 200日線進出", r, switches=nsw)
r, nsw = timing(mr, close["SMH"], 3); add("SMH 3x + 200日線進出", r, switches=nsw)
# validate synthetic vs real over common window
for t, und in (("TQQQ", qr), ("SOXL", mr), ("UPRO", sr)):
    if t in ret:
        a = ret[t].dropna(); s = leveraged(und, int(3), cash=cash).reindex(a.index)
        out[f"valid_{t}"] = {"real": stats(a)["cagr"], "synthetic": stats(s)["cagr"], "corr": float(a.corr(s))}
out["lev"] = rows
# worst stats
out["lev_extra"] = {}
for k in ("QQQ 3x 買入持有(模擬)", "SMH 3x 買入持有(模擬,≈SOXL)", "TQQQ 真實", "SOXL 真實"):
    if k in eq:
        e = (1 + eq[k]).cumprod(); dd = e / e.cummax() - 1
        under = (dd < -1e-9); 
        # longest time under water in years
        grp = (~under).cumsum(); run = under.groupby(grp).sum().max() / 252
        out["lev_extra"][k] = {"mdd": float(dd.min()), "worst_day": float(eq[k].min()), "years_underwater": float(run), "mdd_date": str(dd.idxmin().date())}

# ---- costs: daily short-term reversal on liquid PIT names
liq = pit.copy()
rev = -ret.rolling(2).sum()   # buy the biggest 2-day losers
rows_w = {}
for d in dates[300:]:
    ok = liq.loc[d] & close.loc[d].notna() & rev.loc[d].notna()
    cand = dvol.loc[d][ok].nlargest(200).index
    pick = rev.loc[d][cand].nlargest(10).index
    w = pd.Series(0.0, index=close.columns); w[list(pick)] = 0.1; rows_w[d] = w
Wd = pd.DataFrame(rows_w).T
rv = {}
for lab, delay, bps in (("紙上績效: 收盤進出、0成本", 1, 0), ("加 10bps 成本", 1, 10), ("多等1天成交 + 10bps", 2, 10), ("多等1天成交 + 25bps", 2, 25)):
    r, t = run_weights(ret, Wd, bps, delay=delay); r = r.loc["2005":]
    rv[lab] = {"periods": by_period(r), "turn_per_year": float(t.sum() / (len(t) / 252))}
out["reversal"] = rv
pickle.dump(out, open("/tmp/rc/out2.pkl", "wb")); pickle.dump(eq, open("/tmp/rc/eq2.pkl", "wb"))
for k, v in rows.items():
    p = v["periods"]; print(k.ljust(32), " ".join("%s %.0f%%/%.0f%%" % (kk[:6], p[kk]["cagr"] * 100, p[kk]["mdd"] * 100) for kk in p))
print(out["lev_extra"]); print({k: v for k, v in out.items() if k.startswith("valid")})
for k, v in rv.items():
    p = v["periods"]; print(k.ljust(28), "turn/yr %.0f" % v["turn_per_year"], " ".join("%s %.0f%%" % (kk[:6], p[kk]["cagr"] * 100) for kk in p))
