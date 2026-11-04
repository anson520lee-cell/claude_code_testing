"""Extra Yahoo daily bars (yfinance): inverse 3x ETFs and a multi-asset ETF set for trend following."""
import pandas as pd, yfinance as yf
SYMS = ["SQQQ", "SPXU", "EFA", "EEM", "IWM", "DBC", "UUP", "VNQ", "HYG", "LQD", "SLV", "USO", "FXE", "FXY", "TIP", "EWJ", "DIA", "XLE", "XLK", "XLF"]
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
p.to_parquet("/home/user/research_data/x80/raw/prices_extra.parquet")
print("saved", p.shape)
