"""Replication universe: current S&P 400 + 600 members that were NEVER in our S&P 500 sample. Yahoo prices + FINRA."""
import io, time, concurrent.futures as cf, pandas as pd, requests, yfinance as yf
R = "/home/user/research_data/rep/raw"
sp = set(pd.read_parquet("/home/user/research_data/raw/prices.parquet", columns=["symbol"])["symbol"].unique())
tick = [t for t in open(f"{R}/rep_tickers.txt").read().split() if t not in sp]
print("tickers", len(tick), flush=True)
frames = []
for i in range(0, len(tick), 80):
    ch = tick[i:i + 80]
    d = yf.download(ch + (["SPY"] if i == 0 else []), start="2006-01-01", end="2026-11-03", auto_adjust=False, progress=False, threads=True, group_by="ticker")
    for s in ch + (["SPY"] if i == 0 else []):
        try:
            x = d[s].dropna(how="all")
        except Exception:
            continue
        if len(x) < 300:
            continue
        x = x.rename(columns=str.lower).rename(columns={"adj close": "adj_close"}); x["symbol"] = s
        frames.append(x.reset_index().rename(columns={"Date": "date"}))
    time.sleep(1)
p = pd.concat(frames, ignore_index=True); p.to_parquet(f"{R}/prices.parquet")
keep = set(p["symbol"].unique())
print("prices", len(keep), flush=True)
S = requests.Session()
def fetch(day):
    ds = day.strftime("%Y%m%d")
    feeds = ["CNMS"] if day >= pd.Timestamp("2018-08-01") else ["FNSQ", "FNYX", "FORF"]
    out = []
    for f in feeds:
        r = None
        for a in range(3):
            try:
                r = S.get(f"https://cdn.finra.org/equity/regsho/daily/{f}shvol{ds}.txt", timeout=30); break
            except Exception:
                time.sleep(2 * (a + 1))
        if r is None or r.status_code != 200 or len(r.content) < 200:
            continue
        try:
            d = pd.read_csv(io.StringIO(r.text), sep="|", dtype={"Symbol": str})
        except Exception:
            continue
        if not {"Date", "Symbol", "ShortVolume", "TotalVolume"} <= set(d.columns):
            continue
        d = d[pd.to_numeric(d["Date"], errors="coerce").notna()]
        d["Symbol"] = d["Symbol"].astype(str).str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
        d = d[d["Symbol"].isin(keep)]
        out.append(d[["Date", "Symbol", "ShortVolume", "TotalVolume"]])
    return pd.concat(out) if out else None
fr = []
with cf.ThreadPoolExecutor(12) as ex:
    for res in ex.map(fetch, pd.bdate_range("2009-08-03", "2026-10-30")):
        if res is not None: fr.append(res)
f = pd.concat(fr, ignore_index=True)
f["Date"] = pd.to_datetime(f["Date"].astype(int).astype(str))
f.groupby(["Date", "Symbol"], as_index=False)[["ShortVolume", "TotalVolume"]].sum().to_parquet(f"{R}/finra_shvol.parquet")
print("finra saved", flush=True)
