import sys, json, numpy as np, pandas as pd
from src.backtest.edge_hunt.account_run import run_variant, START, END
sig, prices, flags, out = sys.argv[1:5]
s = pd.read_parquet(sig, columns=["symbol","date","setup","member","live_like","r","ret","days"])
s = s[s["member"] & s["setup"].isin(["crash_rebound","squeeze","breakout"]) & (s["date"]>=START) & (s["date"]<=END)].dropna(subset=["r","ret","days"])
s = s.merge(pd.read_parquet(flags)[["symbol","date","svr5_good"]], on=["symbol","date"], how="left")
px = pd.read_parquet(prices, filters=[("symbol","==","SPY")]); px["date"]=pd.to_datetime(px["date"])
cal = px.set_index("date").sort_index().index; cal = cal[(cal>=START)&(cal<=END)]
L = s[s["live_like"]]
V = {"crash_only": L[L.setup=="crash_rebound"], "crash_only_filter": L[(L.setup=="crash_rebound")&(L.svr5_good==1)],
     "crash_squeeze": L[L.setup!="breakout"], "crash_squeeze_filter": L[(L.setup!="breakout")&(L.svr5_good==1)]}
res, curves = {}, {}
for k, t in V.items():
    c, st = run_variant(t, cal); res[k] = st; curves[k] = c
    print(k, {p: {m: round(x,3) for m,x in d.items()} for p,d in st.items() if isinstance(d, dict)}, round(st["trades_per_year"]))
json.dump(res, open(out + "/account_setups.json","w"), indent=1)
pd.DataFrame(curves).to_parquet(out + "/account_setups_curves.parquet")
