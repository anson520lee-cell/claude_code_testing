import time, pandas as pd, yfinance as yf
RAW = "/home/user/research_data/raw"
syms = sorted(s for s in pd.read_parquet(f"{RAW}/prices.parquet", columns=["symbol"])["symbol"].unique() if not s.startswith("^"))
out = []
for i in range(0, len(syms), 100):
    ch = syms[i:i + 100]
    d = yf.download(ch, start="2006-01-01", end="2026-11-03", actions=True, auto_adjust=False, progress=False, group_by="ticker")
    for s in ch:
        try:
            x = d[s]["Stock Splits"].dropna()
        except Exception:
            continue
        x = x[x > 0]
        for dt, v in x.items():
            out.append((s, dt, float(v)))
    time.sleep(1)
pd.DataFrame(out, columns=["symbol", "date", "ratio"]).to_parquet(f"{RAW}/splits.parquet")
print("saved", len(out))
