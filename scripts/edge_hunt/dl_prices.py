"""Download daily OHLCV (unadjusted + Adj Close) from Yahoo for every ticker that was in the S&P 500
on or after 2006-06-01, the live terminal universe, and context ETFs/indices. Output: one parquet panel."""
import sys, time, yaml, pandas as pd, yfinance as yf
sys.path.insert(0, "/home/user/claude_code_testing")
RAW = "/home/user/research_data/raw"
h = pd.read_csv(f"{RAW}/sp500_hist.csv", parse_dates=["date"])
h = h[h["date"] >= "2006-06-01"]
sp = sorted({t.strip().replace(".", "-") for row in h["tickers"] for t in row.split(",") if t.strip()})
cfg = yaml.safe_load(open("/home/user/claude_code_testing/config/terminal.yaml"))
live = list((cfg.get("universe") or {}).keys())
try:
    auto = yaml.safe_load(open("/home/user/claude_code_testing/config/universe_auto.default.yaml"))
    auto = list((auto.get("universe") or auto).keys()) if isinstance(auto, dict) else []
except Exception:
    auto = []
ctx = ["SPY", "QQQ", "IWM", "DIA", "MDY", "RSP", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
       "XLC", "SMH", "XBI", "KRE", "XHB", "XRT", "IGV", "TLT", "IEF", "SHY", "HYG", "LQD", "GLD", "USO", "UUP",
       "^VIX", "^VIX3M", "^VIX9D", "^VVIX", "^SKEW", "^TNX", "^IRX", "^GSPC", "SPHB", "SPLV", "MTUM", "QUAL"]
tick = sorted(set(sp) | set(map(str, live)) | set(map(str, auto)) | set(ctx))
print("tickers", len(tick), "sp", len(sp), "live", len(live), "auto", len(auto), flush=True)
frames = []
for i in range(0, len(tick), 80):
    chunk = tick[i:i + 80]
    for attempt in range(4):
        try:
            d = yf.download(chunk, start="2006-01-01", end="2026-11-03", auto_adjust=False, progress=False,
                            threads=True, group_by="ticker")
            break
        except Exception as e:
            print("retry", e, flush=True); time.sleep(5 * (attempt + 1))
    for s in chunk:
        try:
            x = d[s].dropna(how="all")
        except Exception:
            continue
        if len(x) < 100:
            continue
        x = x.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
        x["symbol"] = s
        frames.append(x.reset_index().rename(columns={"Date": "date"}))
    print(i + len(chunk), "done", len(frames), flush=True)
    time.sleep(1)
p = pd.concat(frames, ignore_index=True)
p.to_parquet(f"{RAW}/prices.parquet")
print("saved", p["symbol"].nunique(), len(p))
