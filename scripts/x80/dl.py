"""Long-history ETF / index / crypto daily bars for the 80%-a-year search (Yahoo via yfinance)."""
import pandas as pd, yfinance as yf
SYMS = ["SPY", "QQQ", "^GSPC", "^NDX", "^IXIC", "^IRX", "^VIX", "^VIX3M", "TLT", "IEF", "GLD", "SHY",
        "TQQQ", "UPRO", "SSO", "QLD", "SOXL", "TECL", "SVXY", "VXX", "UVXY", "TMF",
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "LTC-USD", "ADA-USD"]
frames = []
for s in SYMS:
    d = yf.download(s, start="1985-01-01", end="2026-11-04", auto_adjust=False, progress=False, threads=False)
    if d is None or d.empty:
        print("empty", s, flush=True); continue
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    d["symbol"] = s
    frames.append(d.reset_index().rename(columns={"Date": "date"}))
    print(s, d.index.min().date(), d.index.max().date(), len(d), flush=True)
p = pd.concat(frames, ignore_index=True)
p.to_parquet("/home/user/research_data/x80/raw/prices.parquet")
print("saved", p.shape)
