"""G10 FX carry inputs: daily spot from Yahoo (yfinance) and monthly 3-month interbank rates from FRED (OECD MEI).

    python -I scripts/optimize/fx_dl.py OUT_DIR      ->  OUT_DIR/fx_spot.parquet, OUT_DIR/fx_rates.parquet
Spot is quoted as USD per unit of the currency (USDJPY etc. are inverted), so a rise means the currency gained.
"""
import io
import sys
import time
import urllib.request

import pandas as pd
import yfinance as yf

OUT = sys.argv[1]
PAIRS = {"AUD": ("AUDUSD=X", False), "NZD": ("NZDUSD=X", False), "GBP": ("GBPUSD=X", False), "EUR": ("EURUSD=X", False),
         "JPY": ("USDJPY=X", True), "CHF": ("USDCHF=X", True), "CAD": ("USDCAD=X", True), "NOK": ("USDNOK=X", True),
         "SEK": ("USDSEK=X", True)}
RATES = {"USD": "IR3TIB01USM156N", "AUD": "IR3TIB01AUM156N", "NZD": "IR3TIB01NZM156N", "GBP": "IR3TIB01GBM156N",
         "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N", "CHF": "IR3TIB01CHM156N", "CAD": "IR3TIB01CAM156N",
         "NOK": "IR3TIB01NOM156N", "SEK": "IR3TIB01SEM156N"}

spot = {}
for ccy, (sym, inv) in PAIRS.items():
    d = yf.download(sym, start="2000-01-01", end="2026-11-05", auto_adjust=False, progress=False, threads=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    c = d["Close"].dropna()
    c = c[c > 0]
    spot[ccy] = 1 / c if inv else c
    print(ccy, sym, c.index.min().date(), c.index.max().date(), len(c), flush=True)
    time.sleep(1)
pd.DataFrame(spot).to_parquet(f"{OUT}/fx_spot.parquet")

rates = {}
for ccy, sid in RATES.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    raw = urllib.request.urlopen(url, timeout=60).read().decode()
    x = pd.read_csv(io.StringIO(raw))
    x.columns = ["date", "rate"]
    x["date"] = pd.to_datetime(x["date"])
    x["rate"] = pd.to_numeric(x["rate"], errors="coerce")
    s = x.set_index("date")["rate"].dropna()
    rates[ccy] = s
    print(ccy, sid, s.index.min().date(), s.index.max().date(), len(s), flush=True)
    time.sleep(0.5)
pd.DataFrame(rates).to_parquet(f"{OUT}/fx_rates.parquet")
print("saved")
