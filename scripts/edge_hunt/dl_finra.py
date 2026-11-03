"""FINRA Reg SHO daily short-sale volume (off-exchange TRF/ORF prints), 2009-08 .. 2026-10.
Before 2018-08-01: sum of FNSQ (Nasdaq TRF Carteret), FNYX (NYSE TRF), FORF (ORF); FNQC not public.
From 2018-08-01: CNMS (consolidated). Keeps only symbols in our ticker set. Output parquet."""
import io, sys, time, concurrent.futures as cf, pandas as pd, requests
RAW = "/home/user/research_data/raw"
h = pd.read_csv(f"{RAW}/sp500_hist.csv", parse_dates=["date"])
keep = {t.strip().replace(".", "-") for row in h[h["date"] >= "2008-01-01"]["tickers"] for t in row.split(",")}
import yaml
cfg = yaml.safe_load(open("/home/user/claude_code_testing/config/terminal.yaml"))
keep |= set(map(str, (cfg.get("universe") or {}).keys()))
days = pd.bdate_range("2009-08-03", "2026-10-30")
S = requests.Session()
def fetch(day):
    ds = day.strftime("%Y%m%d")
    feeds = ["CNMS"] if day >= pd.Timestamp("2018-08-01") else ["FNSQ", "FNYX", "FORF"]
    out = []
    for f in feeds:
        url = f"https://cdn.finra.org/equity/regsho/daily/{f}shvol{ds}.txt"
        for a in range(3):
            try:
                r = S.get(url, timeout=30)
                break
            except Exception:
                time.sleep(2 * (a + 1)); r = None
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
        out.append(d[["Date", "Symbol", "ShortVolume", "TotalVolume"]].assign(feed=f))
    return pd.concat(out) if out else None
frames, n = [], 0
with cf.ThreadPoolExecutor(12) as ex:
    for res in ex.map(fetch, days):
        n += 1
        if res is not None:
            frames.append(res)
        if n % 250 == 0:
            print(n, len(frames), flush=True)
p = pd.concat(frames, ignore_index=True)
p["Date"] = pd.to_datetime(p["Date"].astype(int).astype(str))
p = p.groupby(["Date", "Symbol"], as_index=False)[["ShortVolume", "TotalVolume"]].sum()
p.to_parquet(f"{RAW}/finra_shvol.parquet")
print("saved", len(p), p["Date"].nunique(), p["Symbol"].nunique())
