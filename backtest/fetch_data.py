"""Download the daily bars used by run_backtest.py into backtest/data/<SYMBOL>.csv (needs internet).

    python backtest/fetch_data.py            # SOXL SMH SPY (+ optional extras given on the command line)

The committed results.md was produced from broker daily bars (2022-01-13 .. 2026-10-26, split adjusted);
Yahoo data can differ slightly, so re-running may move the numbers a little.
"""
from __future__ import annotations
import sys
from pathlib import Path
import yfinance as yf

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
symbols = sys.argv[1:] or ["SOXL", "SMH", "SPY"]
for s in symbols:
    d = yf.download(s, period="5y", interval="1d", auto_adjust=True, progress=False)
    d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
    d.index.name = "date"
    d[["open", "high", "low", "close", "volume"]].to_csv(OUT / f"{s}.csv")
    print(s, len(d), d.index[0].date(), d.index[-1].date())
