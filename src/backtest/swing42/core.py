"""Core of the 42-strategy swing study: panel data, exit simulation, statistics.

Design (see docs/SWING42.md):
* Prices are Yahoo daily bars (split/dividend adjusted), 1999-2026, aligned on SPY's calendar into
  wide matrices (dates x tickers).
* Universe A = point-in-time S&P 500 members (fja05680/sp500 history). A stock is eligible on a day only
  if it was a member that day, has a price >= $5 and 60 sessions of history. Delisted members that Yahoo
  no longer serves are missing, so some survivorship bias remains (coverage is reported).
* A signal is computed from data up to the close of day d; the trade enters at the open of d+1.
* Every exit is pre-computed for EVERY stock-day once (outcome matrix), so a strategy's trades are a
  lookup and the random-entry benchmark is simply the average over all eligible stock-days with the
  same exit.
* Costs: 15 bps round trip on longs (10 commission/spread + 5 slippage); shorts 25 bps + 2%/yr borrow.
* In-sample 2000-2014 (choose parameters), out-of-sample 2015-2026 (judge).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

IS_END = pd.Timestamp("2014-12-31")
START = pd.Timestamp("2000-01-01")
COST_LONG = 0.0015
COST_SHORT = 0.0025
BORROW = 0.02


# ====================================================================== data
@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: list[str]
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    elig: np.ndarray  # bool (D,S): may trade this stock on this day
    cache: dict = field(default_factory=dict)

    @property
    def D(self) -> int:
        return len(self.dates)

    @property
    def S(self) -> int:
        return len(self.tickers)

    def df(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(getattr(self, name), index=self.dates, columns=self.tickers)

    def sub(self, tickers: list[str]) -> "Panel":
        ix = [self.tickers.index(t) for t in tickers if t in self.tickers]
        return Panel(self.dates, [self.tickers[i] for i in ix], self.O[:, ix], self.H[:, ix], self.L[:, ix],
                     self.C[:, ix], self.V[:, ix], self.elig[:, ix])


def load_prices(px_dir: str, tickers: list[str], calendar: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    D, S = len(calendar), len(tickers)
    out = {k: np.full((D, S), np.nan, dtype=np.float64) for k in "OHLCV"}
    for j, t in enumerate(tickers):
        p = os.path.join(px_dir, f"{t}.pkl")
        if not os.path.exists(p):
            continue
        x = pd.read_pickle(p)
        x = x[~x.index.duplicated()].reindex(calendar)
        o, h, l, c, v = (x[k].to_numpy(float, copy=True) for k in ["Open", "High", "Low", "Close", "Volume"])
        bad = ~(o > 0) | ~(h > 0) | ~(l > 0) | ~(c > 0)
        # repair inconsistent bars, drop absurd prints (> 60% one-day jump that fully reverses next day)
        h = np.fmax(h, np.fmax(o, c))
        l = np.fmin(l, np.fmin(o, c))
        for arr in (o, h, l, c, v):
            arr[bad] = np.nan
        out["O"][:, j], out["H"][:, j], out["L"][:, j], out["C"][:, j], out["V"][:, j] = o, h, l, c, v
    return out


def membership_matrix(hist_csv: str, calendar: pd.DatetimeIndex, tickers: list[str]) -> np.ndarray:
    d = pd.read_csv(hist_csv, parse_dates=["date"]).sort_values("date")
    ix = {t: j for j, t in enumerate(tickers)}
    rows = np.zeros((len(d), len(tickers)), dtype=bool)
    for i, s in enumerate(d["tickers"]):
        for t in s.split(","):
            j = ix.get(t.replace(".", "-"))
            if j is not None:
                rows[i, j] = True
    pos = np.searchsorted(d["date"].to_numpy(), calendar.to_numpy(), side="right") - 1
    m = rows[np.clip(pos, 0, None)]
    m[pos < 0] = False
    return m


# ====================================================================== indicators (numpy / pandas, wide)
def shift(a: np.ndarray, k: int) -> np.ndarray:
    """shift(a, 1)[d] = a[d-1]; shift(a, -1)[d] = a[d+1]."""
    out = np.full_like(a, np.nan, dtype=np.float64)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def roll(a: np.ndarray, n: int, fn: str, minp: int | None = None) -> np.ndarray:
    r = pd.DataFrame(a).rolling(n, min_periods=minp or n)
    return getattr(r, fn)().to_numpy()


def sma(a, n):
    return roll(a, n, "mean")


def ema(a, n):
    return pd.DataFrame(a).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def wilder(a, n):
    return pd.DataFrame(a).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def atr(p: Panel, n: int = 14) -> np.ndarray:
    key = ("atr", n)
    if key not in p.cache:
        pc = shift(p.C, 1)
        tr = np.fmax(p.H - p.L, np.fmax(np.abs(p.H - pc), np.abs(p.L - pc)))
        p.cache[key] = wilder(tr, n)
    return p.cache[key]


def rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(c, axis=0, prepend=np.nan)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    up[np.isnan(d)] = np.nan
    dn[np.isnan(d)] = np.nan
    au, ad = wilder(up, n), wilder(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + au / ad)


def fresh(sig: np.ndarray, cooldown: int) -> np.ndarray:
    """True only where sig is True and was False on each of the previous `cooldown` days."""
    s = np.nan_to_num(sig.astype(float))
    prior = roll(shift(s, 1), cooldown, "max", minp=1)
    return (s > 0) & ~(np.nan_to_num(prior) > 0)


def cached(p: Panel, key, fn):
    if key not in p.cache:
        p.cache[key] = fn()
    return p.cache[key]


# ====================================================================== exits (pre-computed for every stock-day)
def _paths(p: Panel):
    cff = pd.DataFrame(p.C).ffill().to_numpy()
    return cff


def exit_hold(p: Panel, n: int, side: int = 1) -> np.ndarray:
    """Enter next open, exit at the close of the n-th session (n=1: same-day close). Net return."""
    cff = cached(p, "cff", lambda: _paths(p))
    entry = shift(p.O, -1)
    ex = shift(cff, -n)
    r = side * (ex / entry - 1)
    return r - (COST_LONG if side > 0 else COST_SHORT + BORROW * n / 252)


def exit_path(p: Panel, kind: str, side: int = 1, maxh: int = 20, stop_k: float = 1.5, trail_k: float = 3.0,
              stop_pct: float = 0.08, ma_n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Path-dependent exits. Returns (net return, sessions held).

    kind="live":  stop = entry - stop_k*ATR, chandelier trail = highest high since entry - trail_k*ATR
                  (ATR frozen at the signal day), max `maxh` sessions (the terminal's live exit).
    kind="trend": stop entry*(1-stop_pct), exit at the close of the first close below EMA21, max maxh.
    kind="mr":    exit at the close of the first close above SMA5 (below for shorts), max maxh, no stop.
    A bar that gaps through the stop fills at the open. Stocks that stop trading exit at the last close.
    """
    a = atr(p)
    cff = cached(p, "cff", lambda: _paths(p))
    entry = shift(p.O, -1)
    D, S = p.O.shape
    res = np.full((D, S), np.nan)
    held = np.full((D, S), np.nan)
    done = ~np.isfinite(entry) | ~np.isfinite(a)
    stop0 = entry - side * stop_k * a if kind == "live" else entry * (1 - side * stop_pct)
    ext = entry.copy()  # highest high (long) / lowest low (short) so far
    e21 = cached(p, ("ema", 21), lambda: ema(p.C, 21))
    s5 = cached(p, ("sma", ma_n), lambda: sma(p.C, ma_n))
    for k in range(1, maxh + 1):
        o, h, l = shift(p.O, -k), shift(p.H, -k), shift(p.L, -k)
        c = shift(cff, -k)
        if kind == "live":
            level = np.fmax(stop0, ext - trail_k * a) if side > 0 else np.fmin(stop0, ext + trail_k * a)
        else:
            level = stop0
        if kind in ("live", "trend"):
            if side > 0:
                gap = (o <= level) & (k > 1)
                hit = l <= level
            else:
                gap = (o >= level) & (k > 1)
                hit = h >= level
            g = ~done & gap
            res[g] = (o[g] / entry[g] - 1)
            held[g] = k
            done |= g
            hh = ~done & hit
            res[hh] = (level[hh] / entry[hh] - 1)
            held[hh] = k
            done |= hh
        if kind == "trend":
            ce = shift(e21, -k)
            x = ~done & ((c < ce) if side > 0 else (c > ce))
            res[x] = c[x] / entry[x] - 1
            held[x] = k
            done |= x
        if kind == "mr":
            cs = shift(s5, -k)
            x = ~done & ((c > cs) if side > 0 else (c < cs))
            res[x] = c[x] / entry[x] - 1
            held[x] = k
            done |= x
        if side > 0:
            ext = np.fmax(ext, h)
        else:
            ext = np.fmin(ext, l)
        if k == maxh:
            x = ~done
            res[x] = c[x] / entry[x] - 1
            held[x] = k
    res = side * res
    cost = COST_LONG if side > 0 else COST_SHORT + BORROW * np.nan_to_num(held) / 252
    return res - cost, held


def exit_level(p: Panel, level: np.ndarray, side: int, n: int) -> np.ndarray:
    """Same-day stop entry: enter on day d when price trades through `level` (open if it gaps through),
    exit at the close of the n-th session counted from the entry day (n=1: same-day close)."""
    cff = cached(p, "cff", lambda: _paths(p))
    if side > 0:
        trig = p.H >= level
        fill = np.where(p.O > level, p.O, level)
    else:
        trig = p.L <= level
        fill = np.where(p.O < level, p.O, level)
    ex = shift(cff, -(n - 1)) if n > 1 else cff
    r = side * (ex / fill - 1)
    r = np.where(trig, r, np.nan)
    return r - (COST_LONG if side > 0 else COST_SHORT + BORROW * n / 252)


# ====================================================================== statistics
def _split_masks(dates: pd.DatetimeIndex):
    is_ = (dates >= START) & (dates <= IS_END)
    oos = dates > IS_END
    return {"IS": is_, "OOS": oos, "ALL": dates >= START}


def evaluate(p: Panel, sig: np.ndarray, out: np.ndarray, base_mask: np.ndarray | None = None,
             unit: np.ndarray | None = None, bench: np.ndarray | None = None) -> dict:
    """Trades = sig & eligible & finite outcome. Benchmark = random entries: every eligible stock-day with
    the same exit (unconditional) and the same-day cross-section (date-matched).
    Returns per split: n, mean, win, edge (vs unconditional random), edge_dm (vs same-day random),
    t (edge, standard error clustered by calendar month), pos_years (share of years with edge > 0)."""
    if unit is not None:
        unit = np.where(unit > 1e-4, unit, np.nan)
        out = out / unit
    bench = out if bench is None else bench
    base = (p.elig if base_mask is None else base_mask) & np.isfinite(bench)
    trades = sig & (p.elig if base_mask is None else base_mask) & np.isfinite(out)
    months = (p.dates.year * 12 + p.dates.month).to_numpy()
    years = p.dates.year.to_numpy()
    o0 = np.where(base, bench, 0.0)
    nb = base.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        day_mean = o0.sum(1) / nb
    res = {}
    for name, m in _split_masks(p.dates).items():
        rows = np.where(m)[0]
        bm = base[rows]
        mu = o0[rows].sum() / max(bm.sum(), 1)
        ti, tj = np.where(trades[rows])
        if len(ti) == 0:
            res[name] = {"n": 0}
            continue
        x = out[rows][ti, tj]
        exc = x - mu
        exc_dm = x - day_mean[rows][ti]
        g = months[rows][ti]
        # cluster-robust t of the mean excess (clusters = calendar months)
        ug, inv = np.unique(g, return_inverse=True)
        sums = np.bincount(inv, weights=exc - exc.mean())
        se = np.sqrt((sums ** 2).sum()) / len(exc)
        sums_dm = np.bincount(inv, weights=exc_dm - exc_dm.mean())
        se_dm = np.sqrt((sums_dm ** 2).sum()) / len(exc)
        yy = years[rows][ti]
        ys = pd.Series(exc).groupby(yy).mean()
        ys_dm = pd.Series(exc_dm).groupby(yy).mean()
        res[name] = {
            "n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "win": float((x > 0).mean()),
            "base": float(mu), "edge": float(exc.mean()), "t": float(exc.mean() / se) if se > 0 else np.nan,
            "edge_dm": float(exc_dm.mean()), "t_dm": float(exc_dm.mean() / se_dm) if se_dm > 0 else np.nan,
            "pos_years": float((ys > 0).mean()), "pos_years_dm": float((ys_dm > 0).mean()), "n_years": int(len(ys)),
            "trades_per_year": float(len(x) / max(len(np.unique(yy)), 1)),
        }
    return res
