"""Trendline-touch and EMA200/250-retest long entries versus random entries (research only).

Question (Neo, 2026-11-02): buying near a hand-drawn daily / weekly trendline, or on the first / second
retest of the 200/250-day average after a big rally, "almost always bounces" - so try wider stops.
This module turns both ideas into objective rules and tests them.

Definitions (all causal: a level only uses bars that had closed at the signal close)
* Pivot low: the bar's low is the lowest of +-k bars (daily k=5, weekly k=3 weeks). It is known k bars later.
* Trendline: through two pivot lows a < b (a is one of the 4 pivots before b; 10-150 sessions apart for
  daily lines, 15-600 for weekly), in log price, slope >= -5%/yr, no low between the two pivots more than
  0.5 ATR below the line. The line is known from the confirmation of b. It dies at the first close more
  than 0.5 ATR (weekly lines 1 ATR) under it. A *touch* = low <= line + 0.25 ATR (weekly 0.5) while the
  line is alive; consecutive touch days are one episode (no touch in the previous 5 sessions). The two
  anchors are touches 1 and 2, so the first signal is the 3rd touch ("3 points make a trendline").
  The signal fires on the first day of every episode. Several lines on one day: the one with most touches.
  A stock cannot signal again within 10 sessions.
* Major leg: close more than L (30 / 50 / 80%) above the 200- (or 250-) day EMA on some day in the last
  250 sessions. Retest: a touch of the EMA (low <= EMA + 0.25 ATR, close >= EMA - 0.5 ATR) coming from
  above (no touch in the previous 5 sessions, all those closes above). Touch number = episodes since the last
  day the leg condition held (1 = first retest, 2 = second ...).
* Entry at the first open after the signal close, 10 bps slippage each side, exits simulated on daily bars
  (stop beats target on a double touch, gaps fill at the open).
* Baselines: every stock-day of the universe (same-day mean, same horizon), a dip-matched control (same
  month, same drawdown-from-10d-high bucket in ATR, same side of the 200-day EMA) and, for exits, random
  entries (every 3rd session of every stock) run through the very same exit, averaged over the same day +-1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

W = 41                    # bars loaded per entry for the exit simulations (entry bar + 40)
HZS = (5, 10, 20, 40, 60)
PERIODS = {"gfc": ("2000-01-01", "2009-12-31"), "hold": ("2010-01-01", "2019-12-31"),
           "is": ("2020-01-01", "2024-06-30"), "oos": ("2024-07-01", "2100-01-01")}
SLOPE_BUCKETS = {"flat": (-0.05, 0.15), "rising": (0.15, 0.8), "steep": (0.8, 99.0)}
DD_EDGES = (0.0, 1.0, 2.0, 3.0, 4.5, 6.5, 1e9)           # drawdown from the 10-day high, in ATR
STOPS = (2.0, 3.0, 4.0, 6.0, 10.0, "struct")
TRAILS = (None, 4.0, 8.0)
EXIT_HZ = (20, 40)
BOUNCE_K = (1.0, 2.0, 3.0)                              # reach +k ATR before -k ATR within 10 sessions


# ====================================================================== pivots and lines
def pivot_lows(low: np.ndarray, k: int) -> np.ndarray:
    """Positions p whose low is below the k bars before and not above the k bars after (ties: the first)."""
    n = len(low)
    if n < 2 * k + 1:
        return np.array([], dtype=int)
    win = np.lib.stride_tricks.sliding_window_view(low, 2 * k + 1)          # window centred on p = i + k
    centre = win[:, k]
    left = win[:, :k].min(1)
    right = win[:, k + 1:].min(1)
    ok = (centre < left) & (centre <= right)
    return np.flatnonzero(ok) + k


def weekly_pivots(idx: pd.DatetimeIndex, low: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Weekly pivot lows -> (position of the week's lowest daily low, session on which the pivot is known:
    the last session of the k-th following week)."""
    wk = idx.to_period("W").asi8
    df = pd.DataFrame({"wk": wk, "low": low, "pos": np.arange(len(low))})
    g = df.groupby("wk", sort=True)
    wlow = g["low"].min().to_numpy()
    wlast = g["pos"].max().to_numpy()
    first = g["pos"].min().to_numpy()
    wpos = np.empty(len(wlow), dtype=int)
    for i, (a, b) in enumerate(zip(first, wlast)):
        wpos[i] = a + int(np.argmin(low[a:b + 1]))
    piv = pivot_lows(wlow, k)
    piv = piv[piv + k < len(wlow)]
    return wpos[piv], wlast[piv + k]


def _dedupe(df: pd.DataFrame, cooldown: int) -> pd.DataFrame:
    """One line per day (the one with most touches, then the newest), then at most one signal per `cooldown` sessions."""
    if df.empty:
        return df.reset_index(drop=True)
    df = df.sort_values(["d", "nt", "pb"], ascending=[True, False, False]).drop_duplicates("d")
    keep, last = [], -10 ** 9
    for i, dd in enumerate(df["d"].to_numpy()):
        if dd - last >= cooldown:
            keep.append(i)
            last = dd
    return df.iloc[keep].reset_index(drop=True)


LINE_COLS = ["d", "nt", "ann", "level", "pa", "pb", "age", "sep", "held", "fill", "lim_px"]


def find_lines(low: np.ndarray, close: np.ndarray, atr: np.ndarray, p_pos: np.ndarray,
               p_conf: np.ndarray, *, min_sep: int, max_sep: int, max_age: int, touch_tol: float, break_tol: float,
               max_prev: int = 4, min_slope: float = -0.05, cooldown: int = 10, lim_tol: float = 0.1) -> dict[str, pd.DataFrame]:
    """Touch events of every rising / flat support line through pairs of pivot lows (see the module doc).

    Returns {"held": signals confirmed at the close (the touch day closed no more than break_tol ATR under the line;
    entry at the next open), "lim": buy-limit signals (a limit at line + lim_tol x ATR of the previous session was
    filled during the touch day, whatever the close; entry at the limit)}. A line that closes through is dead from
    the next session on, but its last touch day is kept in "lim" (those are the limit orders that lose)."""
    n = len(low)
    ly = np.log(low)
    rel = 0.5 * atr / close                                           # tolerance for lows between the anchors
    rows: list[tuple] = []
    for bi in range(1, len(p_pos)):
        pb, cb = int(p_pos[bi]), int(p_conf[bi])
        if cb >= n - 2:
            continue
        for ai in range(max(0, bi - max_prev), bi):
            pa = int(p_pos[ai])
            sep = pb - pa
            if sep < min_sep or sep > max_sep:
                continue
            slope = (ly[pb] - ly[pa]) / sep
            ann = slope * 252
            if ann < min_slope:
                continue
            seg = np.arange(pa, pb + 1)
            if np.any(ly[seg] - (ly[pa] + slope * (seg - pa)) < -rel[seg]):
                continue
            end = min(n - 2, pb + max_age)
            d = np.arange(pb + 1, end + 1)
            if len(d) == 0:
                continue
            line = np.exp(ly[pa] + slope * (d - pa))
            dead = close[d] < line - break_tol * atr[d]
            if dead.any():
                cut = int(np.argmax(dead)) + 1                           # keep the day that closed through the line
                d, line, dead = d[:cut], line[:cut], dead[:cut]
            touch = low[d] <= line + touch_tol * atr[d]
            # episode start: a touch with no touch in the previous 5 sessions (the anchor pivot counts as a touch)
            t_all = np.r_[True, touch]                                   # t_all[i] = the session before d[i]
            cs = np.cumsum(t_all)
            i = np.arange(len(touch))
            prev5 = cs[i] - np.where(i >= 5, cs[np.maximum(i - 5, 0)], 0)
            ep = np.flatnonzero(touch & (prev5 == 0))
            lim_px = line + lim_tol * atr[d - 1]
            for j, e in enumerate(ep):
                if d[e] <= cb:                                           # line not yet known on that day
                    continue
                rows.append((int(d[e]), 3 + j, ann, float(line[e]), pa, pb, int(d[e]) - pb, sep, not bool(dead[e]),
                             bool(low[d[e]] <= lim_px[e]), float(lim_px[e])))
    df = pd.DataFrame(rows, columns=LINE_COLS)
    # the touch number counts every episode after the second anchor (the anchors are touches 1 and 2)
    return {"held": _dedupe(df[df["held"].astype(bool)], cooldown), "lim": _dedupe(df[df["fill"].astype(bool)], cooldown)}


def ema_retests(close: np.ndarray, low: np.ndarray, atr: np.ndarray, ma: np.ndarray, legs: tuple[float, ...]) -> pd.DataFrame:
    """First / second / ... retest of an average after a major leg (see the module doc). One row per episode
    start with the touch number for each leg size (0 = the leg condition is not met), `held` (the day closed no
    more than 0.5 ATR under the average) and `fill` / `lim_px` (a buy limit at the average + 0.1 ATR of the
    previous session was filled during the day)."""
    n = len(close)
    ok_ma = np.isfinite(ma) & np.isfinite(atr)
    touch = ok_ma & (low <= ma + 0.25 * atr)
    above = ok_ma & (close > ma)
    ts = pd.Series(touch.astype(int))
    ab = pd.Series(above.astype(int))
    no_touch_prev = (ts.shift(1).rolling(5, min_periods=5).sum() == 0).to_numpy()
    all_above_prev = (ab.shift(1).rolling(5, min_periods=5).sum() == 5).to_numpy()
    start = touch & no_touch_prev & all_above_prev
    cs = np.cumsum(start)
    ext = np.where(ok_ma, close / ma - 1, -np.inf)
    idx = np.arange(n)
    ds = np.flatnonzero(start)
    cols = ["d"] + [f"no{int(L * 100)}" for L in legs] + ["rising", "ext_peak", "held", "fill", "lim_px"]
    if len(ds) == 0:
        return pd.DataFrame(columns=cols)
    out = {"d": ds}
    rising = ma > np.r_[np.full(20, np.nan), ma[:-20]]
    for L in legs:
        # last session BEFORE the retest day on which the leg condition held (never the touch day's own close)
        last_ext = np.maximum.accumulate(np.where(ext >= L, idx, -1))
        last = np.r_[-1, last_ext[:-1]]
        age = idx - last
        valid = (last >= 0) & (age <= 250)
        num = cs - np.where(last >= 0, cs[np.maximum(last, 0)], 0)
        out[f"no{int(L * 100)}"] = np.where(valid, num, 0)[ds]
    df = pd.DataFrame(out)
    df["rising"] = rising[ds]
    pk = pd.Series(np.where(np.isfinite(ext), ext, np.nan)).shift(1).rolling(250, min_periods=1).max().to_numpy()
    df["ext_peak"] = pk[ds]
    df["held"] = close[ds] >= ma[ds] - 0.5 * atr[ds]
    prev = np.maximum(ds - 1, 0)
    lim_px = ma[ds] + 0.1 * atr[prev]
    df["lim_px"] = lim_px
    df["fill"] = low[ds] <= lim_px
    leg_cols = [c for c in df.columns if c.startswith("no")]
    return df[(df[leg_cols].to_numpy() > 0).any(axis=1) & (df["d"] <= n - 2)][cols].reset_index(drop=True)


# ====================================================================== panel features
def stock_features(f: pd.DataFrame) -> dict[str, np.ndarray]:
    """Arrays used everywhere (index = sessions of one stock)."""
    c = f["close"].to_numpy(dtype=float)
    h = f["high"].to_numpy(dtype=float)
    l = f["low"].to_numpy(dtype=float)
    a = f["atr"].to_numpy(dtype=float)
    hi10 = pd.Series(h).rolling(10, min_periods=10).max().to_numpy()
    ext200 = c / f["ema200"].to_numpy(dtype=float) - 1
    leg30 = pd.Series(ext200).rolling(250, min_periods=1).max().to_numpy() >= 0.3          # a +30% leg within the last year
    return {"dd10": (hi10 - c) / a, "ext200": ext200, "leg30": leg30,
            "e200": f["ema200"].to_numpy(dtype=float), "e250": f["ema250"].to_numpy(dtype=float)}


def collect_signals(sym: str, f: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All trendline / retest signal tables of one stock (index positions refer to f).
    Keys: tl_d, tl_w, ma200, ma250 (close-confirmed, next-open entry) and the same with _lim (buy limit at the level)."""
    c = f["close"].to_numpy(dtype=float)
    l = f["low"].to_numpy(dtype=float)
    a = f["atr"].to_numpy(dtype=float)
    ok = np.isfinite(a) & (a > 0)
    an = np.where(ok, a, np.nan)
    out: dict[str, pd.DataFrame] = {}
    pp = pivot_lows(l, 5)
    pc = pp + 5
    keep = pc < len(l)
    d_lines = find_lines(l, c, an, pp[keep], pc[keep], min_sep=10, max_sep=150, max_age=250, touch_tol=0.25, break_tol=0.5)
    wp, wc = weekly_pivots(f.index, l, 3)
    w_lines = find_lines(l, c, an, wp, wc, min_sep=15, max_sep=600, max_age=500, touch_tol=0.5, break_tol=1.0)
    out["tl_d"], out["tl_d_lim"] = d_lines["held"], d_lines["lim"]
    out["tl_w"], out["tl_w_lim"] = w_lines["held"], w_lines["lim"]
    legs = (0.3, 0.5, 0.8)
    for name, col in (("ma200", "ema200"), ("ma250", "ema250")):
        r = ema_retests(c, l, a, f[col].to_numpy(dtype=float), legs)
        out[name] = r[r["held"].astype(bool)].reset_index(drop=True)
        out[name + "_lim"] = r[r["fill"].astype(bool)].reset_index(drop=True)
    for k, df in list(out.items()):
        df = df[ok[df["d"].to_numpy(dtype=int)]] if len(df) else df
        df = df.copy()
        df.insert(0, "sym", sym)
        out[k] = df.reset_index(drop=True)
    return out


# ====================================================================== simulation
def simulate(P: dict[str, np.ndarray], slip: float, stopd: np.ndarray, hz: int, trail: float | None = None,
             atr: np.ndarray | None = None, tgtd: np.ndarray | None = None, entry_px: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """One exit rule on every entry. P = O,H,L,C paths (N x W; bar 0 = entry day). stopd/tgtd: price distance
    from the entry fill. trail: chandelier distance in ATR from the highest high (needs atr). Returns ret (net of
    costs), r (in units of the initial risk), eq (account fraction: 1% risk, 25% cap), stopped, target flags.
    entry_px: fill price before slippage (default: the open of bar 0). With a limit fill on bar 0 the stop is
    tested against that bar's low (the low always comes after the fill that it triggers) but the bar's high gets no
    credit (it may have come before the fill)."""
    O, H, L, C = P["O"], P["H"], P["L"], P["C"]
    n = len(O)
    entry = (O[:, 0] if entry_px is None else entry_px) * (1 + slip)
    ok = np.isfinite(entry) & np.isfinite(stopd) & (stopd > 0) & np.isfinite(C[:, hz - 1])
    stop = entry - np.where(ok, stopd, 1.0)
    tgt = entry + tgtd if tgtd is not None else np.full(n, np.inf)
    exit_px = np.full(n, np.nan)
    live = ok.copy()
    stopped = np.zeros(n, dtype=bool)
    hit_tgt = np.zeros(n, dtype=bool)
    hh = np.full(n, -np.inf)
    for j in range(hz):
        o, h, l, c = O[:, j], H[:, j], L[:, j], C[:, j]
        if j == 0 and entry_px is not None:
            h = np.minimum(h, entry_px)                       # the day's high may have come before the limit fill: no credit
        hit_s = live & (l <= stop)
        px_s = np.minimum(o, stop) if j > 0 else stop
        hit_t = live & ~hit_s & (h >= tgt)
        px_t = np.maximum(o, tgt) if j > 0 else tgt
        exit_px = np.where(hit_s, px_s, np.where(hit_t, px_t, exit_px))
        stopped |= hit_s
        hit_tgt |= hit_t
        live &= ~(hit_s | hit_t)
        hh = np.maximum(hh, np.where(live, h, -np.inf))
        if trail is not None:
            stop = np.where(live, np.maximum(stop, hh - trail * atr), stop)
    exit_px = np.where(live, C[:, hz - 1], exit_px) * (1 - slip)
    pnl = exit_px - entry
    risk_pct = np.where(ok, stopd, 1.0) / entry
    ret = pnl / entry
    return {"ret": np.where(ok, ret, np.nan), "r": np.where(ok, pnl / np.where(ok, stopd, 1.0), np.nan),
            "eq": np.where(ok, ret * np.minimum(0.01 / np.maximum(risk_pct, 1e-6), 0.25), np.nan),
            "stopped": stopped & ok, "target": hit_tgt & ok}


def gather_paths(arrs: dict[str, dict[str, np.ndarray]], sym_ids: np.ndarray, d: np.ndarray, syms: list[str],
                 offset: int = 1) -> dict[str, np.ndarray]:
    """Paths of the W bars starting `offset` sessions after day d (1 = the open after the signal close, 0 = day d
    itself for same-day limit fills); NaN beyond the end of the data."""
    N = len(d)
    out = {k: np.full((N, W), np.nan, dtype=np.float32) for k in "OHLC"}
    names = {"O": "open", "H": "high", "L": "low", "C": "close"}
    for si in np.unique(sym_ids):
        sel = np.flatnonzero(sym_ids == si)
        a = arrs[syms[si]]
        n = len(a["close"])
        e = d[sel] + offset
        j = e[:, None] + np.arange(W)[None, :]
        valid = j < n
        jj = np.minimum(j, n - 1)
        for k, nm in names.items():
            v = a[nm][jj].astype(np.float32)
            v[~valid] = np.nan
            out[k][sel] = v
    return out


# ====================================================================== statistics
def cluster_stat(x: np.ndarray, cl: np.ndarray) -> tuple[float, float, int]:
    """(mean, t, n) of x with standard errors clustered by entry month."""
    m = np.isfinite(x)
    x, cl = x[m], cl[m]
    n = len(x)
    if n < 30:
        return (float(x.mean()) if n else float("nan")), float("nan"), n
    mu = x.mean()
    _, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, weights=x - mu)
    g = len(s)
    if g < 5:
        return float(mu), float("nan"), n
    se = math.sqrt((s ** 2).sum() * g / (g - 1)) / n
    return float(mu), (float(mu / se) if se > 0 else float("nan")), n


def month_id(dates: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return (d.year * 12 + d.month).to_numpy()


def half_id(dates: pd.Series) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return (d.year * 2 + (d.month > 6)).to_numpy()


def period_masks(dates: pd.Series) -> dict[str, np.ndarray]:
    return {p: ((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))).to_numpy() for p, (lo, hi) in PERIODS.items()}


def day_baseline(dates: np.ndarray, vals: np.ndarray, day_index: pd.DatetimeIndex, window: int = 1) -> pd.Series:
    """Mean of vals over entries on the same session +-window, returned as a series indexed by session."""
    s = pd.Series(vals).groupby(dates).agg(["sum", "count"]).reindex(day_index).fillna(0)
    sm = s["sum"].rolling(2 * window + 1, center=True, min_periods=1).sum()
    cn = s["count"].rolling(2 * window + 1, center=True, min_periods=1).sum()
    return (sm / cn.where(cn > 0)).rename("base")


def bin_dd(dd: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(dd, DD_EDGES[1:-1]), 0, len(DD_EDGES) - 2)


def control_means(mon: np.ndarray, ddb: np.ndarray, up: np.ndarray, vals: np.ndarray) -> pd.Series:
    """Mean of vals by (month, drawdown bucket, above/below the 200-day) and counts."""
    key = pd.MultiIndex.from_arrays([mon, ddb, up.astype(int)])
    g = pd.Series(vals, index=key).groupby(level=[0, 1, 2]).agg(["mean", "count"])
    return g
