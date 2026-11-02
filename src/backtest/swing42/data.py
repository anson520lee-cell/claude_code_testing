"""Builds and caches the panels and side data used by the 42-strategy study."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from .core import Panel, load_prices, membership_matrix

INDEX = ["SPY", "QQQ", "IWM", "DIA", "MDY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
         "SMH", "XBI", "KRE", "XHB", "XRT", "TLT", "IEF", "SHY", "GLD", "SLV", "USO", "UUP", "HYG", "LQD", "EEM", "EFA",
         "FXE", "FXY", "DBC", "^VIX", "^VIX3M", "^VIX9D", "^TNX", "^IRX", "^GSPC", "^NDX", "^RUT", "RSP", "MTUM", "SPLV",
         "ARKK", "BTC-USD", "MNA", "CSD", "IPO", "SPHB", "QUAL", "IWD", "IWF", "IWN", "IWO", "HG=F", "GC=F", "CL=F", "^SPX"]


def build(px_dir: str, hist_csv: str, cache: str, start: str = "1999-01-01", end: str = "2026-10-31") -> dict:
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    spy = pd.read_pickle(os.path.join(px_dir, "SPY.pkl"))
    cal = spy.index[(spy.index >= start) & (spy.index <= end)]
    cal = pd.DatetimeIndex(cal.tz_localize(None) if cal.tz is not None else cal)
    files = sorted(f[:-4] for f in os.listdir(px_dir) if f.endswith(".pkl"))
    stocks = [t for t in files if t not in INDEX]
    idx = [t for t in INDEX if t in files]
    pa = load_prices(px_dir, stocks, cal)
    mem = membership_matrix(hist_csv, cal, stocks)
    elig = mem & (pa["C"] >= 5) & np.isfinite(pa["O"])
    A = Panel(cal, stocks, pa["O"], pa["H"], pa["L"], pa["C"], pa["V"], elig)
    pi = load_prices(px_dir, idx, cal)
    X = Panel(cal, idx, pi["O"], pi["H"], pi["L"], pi["C"], pi["V"], np.isfinite(pi["C"]) & np.isfinite(pi["O"]))
    # coverage of point-in-time membership by available price data, per year
    d = pd.read_csv(hist_csv, parse_dates=["date"])
    cov = {}
    have = set(stocks)
    for y in range(2000, 2027):
        row = d[d["date"] <= f"{y}-06-30"].iloc[-1]
        names = [t.replace(".", "-") for t in row["tickers"].split(",")]
        cov[y] = (len(names), sum(t in have for t in names))
    out = {"A": A, "X": X, "coverage": cov}
    with open(cache, "wb") as f:
        pickle.dump(out, f, protocol=4)
    return out


def context(X: Panel, A: Panel) -> dict:
    """Market context per date: SPY trend, VIX level / term structure, breadth of universe A."""
    c = X.df("C")
    spy = c["SPY"]
    vix = c.get("^VIX")
    vix3m = c.get("^VIX3M")
    s50 = pd.DataFrame(A.C).rolling(50).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        above = np.where(A.elig, A.C > s50, np.nan)
    breadth = pd.Series(np.nanmean(above, axis=1), index=A.dates)
    ctx = pd.DataFrame({
        "spy": spy, "spy_200": spy.rolling(200).mean(), "vix": vix, "vix3m": vix3m, "breadth": breadth,
    })
    ctx["bull"] = ctx["spy"] > ctx["spy_200"]
    ctx["ts"] = ctx["vix"] / ctx["vix3m"]
    return ctx


FILTERS = {
    "SPY>200MA": lambda c: c["bull"],
    "SPY<200MA": lambda c: ~c["bull"],
    "VIX<20": lambda c: c["vix"] < 20,
    "VIX 20-30": lambda c: (c["vix"] >= 20) & (c["vix"] < 30),
    "VIX>=25": lambda c: c["vix"] >= 25,
    "VIX contango (VIX<VIX3M)": lambda c: c["ts"] < 1,
    "VIX backwardation (VIX>=VIX3M)": lambda c: c["ts"] >= 1,
    "Breadth>60%": lambda c: c["breadth"] > 0.6,
    "Breadth<40%": lambda c: c["breadth"] < 0.4,
}
