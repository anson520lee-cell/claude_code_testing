"""Yahoo per-ticker event history: analyst rating changes (upgrades_downgrades, ~2012+) and earnings dates
with EPS estimate / reported / surprise (get_earnings_dates(limit=100), ~2002+)."""
import time, pandas as pd, yfinance as yf
RAW = "/home/user/research_data/raw"
syms = sorted(pd.read_parquet(f"{RAW}/prices.parquet", columns=["symbol"])["symbol"].unique())
syms = [s for s in syms if not s.startswith("^")]
ud, ed, n = [], [], 0
for s in syms:
    for a in range(4):
        try:
            t = yf.Ticker(s)
            u = t.upgrades_downgrades
            if u is not None and len(u):
                u = u.reset_index(); u["symbol"] = s; ud.append(u)
            e = t.get_earnings_dates(limit=100)
            if e is not None and len(e):
                e = e.reset_index(); e["symbol"] = s; ed.append(e)
            break
        except Exception as ex:
            msg = str(ex)
            if "Rate" in msg or "429" in msg or "Too Many" in msg:
                time.sleep(30 * (a + 1))
            else:
                break
    n += 1
    if n % 50 == 0:
        print(n, len(ud), len(ed), flush=True)
    time.sleep(0.4)
pd.concat(ud, ignore_index=True).to_parquet(f"{RAW}/yahoo_ratings.parquet")
E = pd.concat(ed, ignore_index=True)
E.columns = [str(c) for c in E.columns]
E.to_parquet(f"{RAW}/yahoo_earnings.parquet")
print("saved", len(ud), len(ed))
