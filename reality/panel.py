from __future__ import annotations
import numpy as np, pandas as pd

def load(path="/tmp/rc/prices.pkl", tabpath="/tmp/rc/tab.pkl"):
    fr = pd.read_pickle(path); tab = pd.read_pickle(tabpath)
    tab = tab.assign(sym=tab.symbol.str.replace(".", "-", regex=False))
    close = pd.DataFrame({s: f["Close"] for s, f in fr.items()}).sort_index()
    vol = pd.DataFrame({s: f["Volume"] for s, f in fr.items()}).sort_index()
    close = close[close.index >= "2003-06-01"].dropna(thresh=100)
    return close, vol.reindex(close.index), tab

def pit_mask(tab, dates, cols):
    """bool DataFrame dates x cols: S&P 500 member on that date."""
    m = pd.DataFrame(False, index=dates, columns=cols)
    for r in tab.itertuples():
        if r.sym in m.columns:
            m.loc[(m.index >= r.start) & (m.index < r.end), r.sym] = True
    return m
