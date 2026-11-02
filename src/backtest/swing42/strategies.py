"""Mechanical definitions of Neo's 42 swing strategies (research only, nothing here feeds the live scanner).

Every strategy is reduced to rules on DAILY bars that use only data up to the signal day's close.
Where the textbook version needs data we do not have (intraday bars, options flow, short interest, deal
lists) the closest honest proxy is used and flagged in `proxy`, or the strategy is marked untestable.

Each entry of STRATEGIES:
    id, zh (Chinese name), family, side (+1 long / -1 short or from params),
    exits: textbook exit keys (see run.EXITS) - the live exit "live" is always added,
    grid: list of parameter dicts (small on purpose, the whole grid is reported),
    fn(p, ctx, **params) -> bool matrix (D, S) of signal days,
    universe: "A" (point-in-time S&P 500), "IDX" (index ETFs), or a custom evaluator name,
    proxy: text describing any approximation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from . import nb
from .core import Panel, atr, cached, ema, fresh, roll, rsi, shift, sma, wilder


# ====================================================================== shared indicators (cached on the panel)
def I(p: Panel, name: str, n: int = 0):
    c = p.cache
    key = (name, n)
    if key in c:
        return c[key]
    if name == "sma":
        v = sma(p.C, n)
    elif name == "ema":
        v = ema(p.C, n)
    elif name == "vsma":
        v = sma(p.V, n)
    elif name == "hh":  # highest high of the last n bars INCLUDING today
        v = roll(p.H, n, "max")
    elif name == "ll":
        v = roll(p.L, n, "min")
    elif name == "hc":
        v = roll(p.C, n, "max")
    elif name == "std":
        v = roll(p.C, n, "std")
    elif name == "ret":
        v = p.C / shift(p.C, n) - 1
    elif name == "rsi":
        v = rsi(p.C, n)
    elif name == "atr":
        v = atr(p, n)
    elif name == "pivots":
        v = nb.pivots(p.H, p.L, n)
    elif name == "adx":
        v = _adx(p, n)
    else:
        raise KeyError(name)
    c[key] = v
    return v


def _adx(p: Panel, n: int):
    up = p.H - shift(p.H, 1)
    dn = shift(p.L, 1) - p.L
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[np.isnan(up)] = np.nan
    mdm[np.isnan(dn)] = np.nan
    a = atr(p, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100 * wilder(pdm, n) / a
        mdi = 100 * wilder(mdm, n) / a
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return wilder(dx, n)


def trend_template(p: Panel) -> np.ndarray:
    """Minervini stage-2 trend template."""
    def f():
        c, s50, s150, s200 = p.C, I(p, "sma", 50), I(p, "sma", 150), I(p, "sma", 200)
        return ((c > s50) & (s50 > s150) & (s150 > s200) & (s200 > shift(s200, 20))
                & (c >= 1.3 * I(p, "ll", 252)) & (c >= 0.75 * I(p, "hh", 252)))
    return cached(p, "tt", f)


def up_trend(p: Panel) -> np.ndarray:
    return cached(p, "uptrend", lambda: (p.C > I(p, "sma", 200)))


def vol_ratio(p: Panel) -> np.ndarray:
    return cached(p, "vr", lambda: p.V / shift(I(p, "vsma", 50), 1))


# ====================================================================== numba pattern kernels
@njit(parallel=True, cache=True)
def _zone_retest(lo, hi, L, C, life, need_below_close):
    """Zones [lo, hi] created on rows where lo is finite; active for `life` rows from the next row. Signal on
    the first row whose low trades into the zone (L <= hi) and closes inside or above it (C >= lo).
    A close below lo kills the zone. need_below_close: price must first have closed above hi (left the zone)."""
    D, S = L.shape
    sig = np.zeros((D, S), dtype=np.bool_)
    for s in prange(S):
        zl = np.nan
        zh = np.nan
        start = -1
        left = False
        for d in range(D):
            if start >= 0 and d > start and d - start <= life and not np.isnan(zl):
                if C[d, s] < zl:
                    zl = np.nan
                elif L[d, s] <= zh and (left or not need_below_close):
                    sig[d, s] = True
                    zl = np.nan
                elif C[d, s] > zh:
                    left = True
            if not np.isnan(lo[d, s]):
                zl = lo[d, s]
                zh = hi[d, s]
                start = d
                left = False
    return sig


@njit(parallel=True, cache=True)
def _double_bottom(H, L, C, pl, pli, tol, min_gap, max_gap, min_rise, prior_drop, max_wait, ihs):
    """Double bottom (ihs=False) / inverse head-and-shoulders (ihs=True) from confirmed pivot lows.
    Signal: first close above the neckline (max high between the bottoms) within max_wait rows of the last low."""
    D, S = H.shape
    sig = np.zeros((D, S), dtype=np.bool_)
    for s in prange(S):
        vals = np.full(3, np.nan)
        idx = np.full(3, -1)
        neck = np.nan
        last_low = -1
        armed = False
        for d in range(1, D):
            if not np.isnan(pl[d, s]):
                vals[2] = vals[1]
                vals[1] = vals[0]
                idx[2] = idx[1]
                idx[1] = idx[0]
                vals[0] = pl[d, s]
                idx[0] = int(pli[d, s])
                armed = False
                i2, i1 = idx[0], idx[1]
                ok = False
                if not ihs and i1 >= 0:
                    gap = i2 - i1
                    if min_gap <= gap <= max_gap and abs(vals[0] / vals[1] - 1) <= tol:
                        a, b = i1, i2
                        ok = True
                if ihs and idx[2] >= 0:
                    head = vals[1]
                    if head < vals[0] and head < vals[2] and abs(vals[0] / vals[2] - 1) <= tol \
                            and min_gap <= idx[0] - idx[2] <= max_gap:
                        a, b = idx[2], idx[0]
                        ok = True
                if ok:
                    nk = -1e18
                    for j in range(a, b + 1):
                        if H[j, s] > nk:
                            nk = H[j, s]
                    base = min(vals[0], vals[1]) if not ihs else vals[1]
                    hi_before = -1e18
                    for j in range(max(0, a - 120), a + 1):
                        if H[j, s] > hi_before:
                            hi_before = H[j, s]
                    if nk >= base * (1 + min_rise) and hi_before >= base * (1 + prior_drop) and C[d, s] <= nk:
                        neck = nk
                        last_low = b
                        armed = True
            if armed:
                if d - last_low > max_wait:
                    armed = False
                elif C[d, s] > neck and C[d - 1, s] <= neck:
                    sig[d, s] = True
                    armed = False
    return sig


@njit(parallel=True, cache=True)
def _structure(H, L, C, ph, pl, phi, pli):
    """Market structure from confirmed pivots. Returns (bos, choch, sweep_choch, last_ph, last_pl_before):
    bos: close crosses above the last swing high while the last two highs and lows were rising.
    choch: close crosses above the last swing high while the last two highs and lows were falling.
    sweep_choch: choch where the latest swing low undercut the previous one (liquidity taken first).
    leg_low: the last swing low at the time of the break (for OTE legs)."""
    D, S = H.shape
    bos = np.zeros((D, S), dtype=np.bool_)
    choch = np.zeros((D, S), dtype=np.bool_)
    swc = np.zeros((D, S), dtype=np.bool_)
    leg_low = np.full((D, S), np.nan)
    brk_level = np.full((D, S), np.nan)
    for s in prange(S):
        h1 = np.nan
        h2 = np.nan
        l1 = np.nan
        l2 = np.nan
        used = True
        for d in range(1, D):
            if not np.isnan(ph[d, s]):
                h2 = h1
                h1 = ph[d, s]
                used = False
            if not np.isnan(pl[d, s]):
                l2 = l1
                l1 = pl[d, s]
            if used or np.isnan(h1) or np.isnan(h2) or np.isnan(l1) or np.isnan(l2):
                continue
            if C[d, s] > h1 and C[d - 1, s] <= h1:
                used = True
                leg_low[d, s] = l1
                brk_level[d, s] = h1
                if h1 > h2 and l1 > l2:
                    bos[d, s] = True
                elif h1 < h2 and l1 < l2:
                    choch[d, s] = True
                    swc[d, s] = True
                elif h1 < h2:
                    choch[d, s] = True
    return bos, choch, swc, leg_low, brk_level


@njit(parallel=True, cache=True)
def _ote(H, L, C, start, leg_low, lo_f, hi_f, life):
    """After a break of structure on row r (leg from leg_low to the running high), signal the first row within
    `life` rows whose low reaches the [hi_f, lo_f] retracement band (e.g. 62%-79%) and closes inside or above it.
    The leg is void once price closes below leg_low."""
    D, S = H.shape
    sig = np.zeros((D, S), dtype=np.bool_)
    for s in prange(S):
        r = -1
        lw = np.nan
        top = np.nan
        for d in range(D):
            if start[d, s]:
                r = d
                lw = leg_low[d, s]
                top = H[d, s]
                continue
            if r < 0 or np.isnan(lw):
                continue
            if d - r > life or C[d, s] < lw:
                r = -1
                continue
            rng = top - lw
            if rng > 0:
                upper = top - lo_f * rng
                lower = top - hi_f * rng
                if L[d, s] <= upper and C[d, s] >= lower:
                    sig[d, s] = True
                    r = -1
                    continue
            if H[d, s] > top:
                top = H[d, s]
    return sig


@njit(parallel=True, cache=True)
def _wyckoff(H, L, C, V, vavg, n_range, max_width, prior_drop, mode):
    """Accumulation: a trading range of n_range rows (width <= max_width) after a decline >= prior_drop.
    mode 0 = spring (undercut of the range low that closes back inside), mode 1 = sign of strength (close above the
    range high on volume >= 1.5x average), mode 2 = distribution upthrust (mirror of the spring after a rally)."""
    D, S = H.shape
    sig = np.zeros((D, S), dtype=np.bool_)
    for s in prange(S):
        for d in range(n_range + 250, D):
            hi = -1e18
            lo = 1e18
            bad = False
            for j in range(d - n_range, d):
                if np.isnan(H[j, s]):
                    bad = True
                    break
                if H[j, s] > hi:
                    hi = H[j, s]
                if L[j, s] < lo:
                    lo = L[j, s]
            if bad or lo <= 0 or hi / lo - 1 > max_width:
                continue
            pre_hi = -1e18
            pre_lo = 1e18
            for j in range(d - 250, d - n_range):
                if H[j, s] > pre_hi:
                    pre_hi = H[j, s]
                if L[j, s] < pre_lo:
                    pre_lo = L[j, s]
            if mode == 0:
                if pre_hi >= hi * (1 + prior_drop) and L[d, s] < lo and C[d, s] > lo:
                    sig[d, s] = True
            elif mode == 1:
                if pre_hi >= hi * (1 + prior_drop) and C[d, s] > hi and C[d - 1, s] <= hi and V[d, s] >= 1.5 * vavg[d, s]:
                    sig[d, s] = True
            else:
                if pre_lo * (1 + prior_drop) <= lo and H[d, s] > hi and C[d, s] < hi:
                    sig[d, s] = True
    return sig


@njit(parallel=True, cache=True)
def _anchor_index(cond):
    """Most recent row where cond is True (inclusive), else -1."""
    D, S = cond.shape
    out = np.full((D, S), -1.0)
    for s in prange(S):
        last = -1
        for d in range(D):
            if cond[d, s]:
                last = d
            out[d, s] = last
    return out


def _avwaps(p: Panel) -> dict[str, np.ndarray]:
    def f():
        tp = (p.H + p.L + p.C) / 3
        v = np.nan_to_num(p.V)
        lo_ix = nb.rolling_argext(p.L, 252, False)
        hi_ix = nb.rolling_argext(p.H, 252, True)
        gap = (p.O / shift(p.C, 1) - 1 >= 0.04) & (vol_ratio(p) >= 2)
        ev_ix = _anchor_index(gap)
        yr = p.dates.year.to_numpy()
        first = np.r_[True, yr[1:] != yr[:-1]]
        ytd = _anchor_index(np.repeat(first[:, None], p.S, axis=1))
        return {"low52": nb.avwap(tp, v, lo_ix), "high52": nb.avwap(tp, v, hi_ix), "event": nb.avwap(tp, v, ev_ix),
                "ytd": nb.avwap(tp, v, ytd)}
    return cached(p, "avwaps", f)


def _vp(p: Panel, n: int):
    return cached(p, ("vp", n), lambda: nb.volume_profile(p.H, p.L, p.C, np.nan_to_num(p.V), n, 24, 1))


# ====================================================================== 1. momentum & price action
def s01_vcp(p, ctx, tight=0.5, pivot=20, vm=1.4):
    rng10 = I(p, "hh", 10) - I(p, "ll", 10)
    rng50 = I(p, "hh", 50) - I(p, "ll", 50)
    tight_ok = shift(rng10 / rng50, 1) <= tight
    dry = shift(I(p, "vsma", 10) / I(p, "vsma", 50), 1) < 0.8
    brk = (p.C > shift(I(p, "hh", pivot), 1)) & (vol_ratio(p) >= vm)
    return trend_template(p) & tight_ok & dry & brk


def s02_htf(p, ctx, gain=0.9, depth=0.25):
    peak = shift(I(p, "hh", 15), 1)
    pole_lo = shift(I(p, "ll", 40), 16)
    flag_lo = shift(I(p, "ll", 10), 1)
    cond = (peak >= pole_lo * (1 + gain)) & (flag_lo >= peak * (1 - depth)) & (shift(I(p, "hh", 3), 1) < peak)
    return cond & (p.C > peak) & (vol_ratio(p) >= 1.0)


def s03_pocket_pivot(p, ctx, ext=0.05, trend="sma50"):
    dn_vol = np.where(p.C < shift(p.C, 1), p.V, 0.0)
    max_dn = shift(roll(dn_vol, 10, "max"), 1)
    s10, s50 = I(p, "sma", 10), I(p, "sma", 50)
    ok = (p.C > shift(p.C, 1)) & (p.V > max_dn) & (p.C > s50) & (p.C > s10) & (p.C / s10 - 1 <= ext)
    if trend == "tt":
        ok &= trend_template(p)
    return ok


def s04_flat_base(p, ctx, n=30, depth=0.15, vm=1.3):
    hi = shift(I(p, "hh", n), 1)
    lo = shift(I(p, "ll", n), 1)
    return (hi / lo - 1 <= depth) & (p.C > hi) & (vol_ratio(p) >= vm) & up_trend(p)


def s05_cup_handle(p, ctx, variant="handle", max_depth=0.35):
    lip = shift(I(p, "hh", 130), 16)
    low = shift(I(p, "ll", 100), 16)
    depth = 1 - low / lip
    ok_depth = (depth >= 0.12) & (depth <= max_depth)
    right = shift(I(p, "hh", 15), 1)
    if variant == "handle":
        hl = shift(I(p, "ll", 10), 1)
        hh10 = shift(I(p, "hh", 10), 1)
        handle = (right >= 0.95 * lip) & (hl >= right * 0.88) & (hl <= right * 0.97)
        return ok_depth & handle & (p.C > hh10) & (p.C >= 0.97 * lip) & (vol_ratio(p) >= 1.2) & up_trend(p)
    # cheat / low-cheat: pause in the middle third of the cup, buy the break of the pause high
    pos = (p.C - low) / (lip - low)
    pause = (shift(I(p, "hh", 5), 1) / shift(I(p, "ll", 5), 1) - 1) <= 0.05
    third = (pos >= 0.33) & (pos <= 0.75) if variant == "cheat" else (pos >= 0.2) & (pos <= 0.5)
    return ok_depth & pause & third & (p.C > shift(I(p, "hh", 5), 1)) & (p.C > I(p, "sma", 50))


def s07_3wt(p, ctx, tol=0.015):
    # weekly bars from daily: last trading day of each ISO week
    wk = pd.Series(p.dates.to_period("W-FRI"), index=p.dates)
    last_of_week = (wk != wk.shift(-1)).to_numpy()
    C = pd.DataFrame(p.C, index=p.dates)
    H = pd.DataFrame(p.H, index=p.dates)
    wc = C[last_of_week]
    wh = H.groupby(wk.to_numpy()).max()
    wh.index = wc.index
    tight = ((wc / wc.shift(1) - 1).abs() <= tol) & ((wc.shift(1) / wc.shift(2) - 1).abs() <= tol)
    pivot = wh.rolling(3).max().where(tight)
    s50 = pd.DataFrame(I(p, "sma", 50), index=p.dates)[last_of_week]
    pivot = pivot.where(wc > s50)
    # carry the pivot forward for 10 sessions after the tight week closes
    piv_d = pivot.reindex(p.dates).shift(1).ffill(limit=10).to_numpy()
    sig = (p.C > piv_d) & (shift(p.C, 1) <= piv_d)
    return sig


def s08_holy_grail(p, ctx, adx=30):
    a = I(p, "adx", 14)
    e20 = I(p, "ema", 20)
    touched = roll((p.L <= e20).astype(float), 3, "max", 1) > 0
    return (a > adx) & (a >= shift(a, 5)) & (e20 > shift(e20, 5)) & shift(touched, 1).astype(bool) & (p.C > shift(p.H, 1)) & (p.C > e20)


def s09_double_bottom(p, ctx, pattern="db", tol=0.03):
    ph, pl, phi, pli = I(p, "pivots", 5)
    return _double_bottom(p.H, p.L, p.C, pl, pli, tol, 10, 80, 0.10, 0.20, 40, pattern == "ihs")


# ====================================================================== 2. smart money / ICT
def s10_undercut_rally(p, ctx, n=50, trend=True):
    sup = shift(I(p, "ll", n), 1)
    sig = (p.L < sup) & (p.C > sup) & (p.C > (p.H + p.L) / 2)
    if trend:
        sig &= up_trend(p)
    return sig


def s11_order_block(p, ctx, m=2.0, life=30):
    a = I(p, "atr", 14)
    impulse = ((p.C - shift(p.C, 1)) >= m * shift(a, 1)) & (p.C > shift(I(p, "hh", 20), 1))
    lo = np.full(p.C.shape, np.nan)
    hi = np.full(p.C.shape, np.nan)
    found = np.zeros(p.C.shape, dtype=bool)
    for k in range(1, 6):  # last down candle within the 5 bars before the impulse
        down = shift(p.C, k) < shift(p.O, k)
        take = impulse & down & ~found
        lo[take] = shift(p.L, k)[take]
        hi[take] = shift(p.H, k)[take]
        found |= take
    return _zone_retest(lo, hi, p.L, p.C, life, True)


def s12_fvg(p, ctx, min_gap=0.25, trend=False, life=20):
    a = I(p, "atr", 14)
    top, bot = p.L, shift(p.H, 2)
    gap = (top - bot) >= min_gap * a
    lo = np.where(gap, bot, np.nan)
    hi = np.where(gap, top, np.nan)
    sig = _zone_retest(lo, hi, p.L, p.C, life, False)
    if trend:
        sig &= p.C > I(p, "sma", 50)
    return sig


def _struct(p, k):
    def f():
        ph, pl, phi, pli = I(p, "pivots", k)
        return _structure(p.H, p.L, p.C, ph, pl, phi, pli)
    return cached(p, ("struct", k), f)


def s13_bos_choch(p, ctx, kind="choch", k=3):
    bos, choch, swc, leg_low, lvl = _struct(p, k)
    return bos if kind == "bos" else choch


def s14_breaker(p, ctx, kind="breaker", k=3, life=20):
    bos, choch, swc, leg_low, lvl = _struct(p, k)
    ev = swc if kind == "breaker" else (choch & ~swc)
    a = I(p, "atr", 14)
    lo = np.where(ev, lvl - 0.5 * a, np.nan)
    hi = np.where(ev, lvl + 0.1 * a, np.nan)
    return _zone_retest(lo, hi, p.L, p.C, life, True)


def s15_ote(p, ctx, k=3, band="62-79", life=30):
    bos, choch, swc, leg_low, lvl = _struct(p, k)
    lo_f, hi_f = (0.62, 0.79) if band == "62-79" else (0.5, 0.618)
    return _ote(p.H, p.L, p.C, bos | choch, leg_low, lo_f, hi_f, life)


def s16_wyckoff(p, ctx, mode=0, width=0.25, n_range=60):
    return _wyckoff(p.H, p.L, p.C, np.nan_to_num(p.V), np.nan_to_num(shift(I(p, "vsma", 50), 1)), n_range, width, 0.25, mode)


def s17_sweep(p, ctx, side=1, k=5, tol_atr=0.5):
    ph, pl, phi, pli = I(p, "pivots", k)
    a = I(p, "atr", 14)
    if side > 0:
        last = nb.last_pivots(pl, 2)
        l1, l2 = last[:, :, 0], last[:, :, 1]
        equal = np.abs(l1 - l2) <= tol_atr * a
        pool = np.fmin(l1, l2)
        return equal & (p.L < pool) & (p.C > pool) & (shift(p.L, 1) >= pool)
    last = nb.last_pivots(ph, 2)
    h1, h2 = last[:, :, 0], last[:, :, 1]
    equal = np.abs(h1 - h2) <= tol_atr * a
    pool = np.fmax(h1, h2)
    return equal & (p.H > pool) & (p.C < pool) & (shift(p.H, 1) <= pool)


# ====================================================================== 3. volume & market profile
def s18_avwap_bounce(p, ctx, anchor="low52"):
    v = _avwaps(p)[anchor]
    above_before = roll((p.C > v).astype(float), 5, "min") > 0
    return shift(above_before, 1).astype(bool) & (p.L <= v * 1.005) & (p.C > v) & (p.C > I(p, "sma", 50))


def s19_avwap_pinch(p, ctx, width=1.0):
    a = _avwaps(p)
    stack = np.stack([a["ytd"], a["high52"], a["low52"], a["event"]])
    hi = np.nanmax(stack, axis=0)
    lo = np.nanmin(stack, axis=0)
    pinch = (hi - lo) <= width * I(p, "atr", 14)
    return shift(pinch, 1).astype(bool) & (p.C > hi) & (shift(p.C, 1) <= shift(hi, 1))


def s20_poc_shift(p, ctx, n=60, m=1.0):
    poc, vah, val, thin = _vp(p, n)
    a = I(p, "atr", 14)
    shift_up = (poc - shift(poc, 10)) >= m * a
    return fresh(shift_up & (p.C > poc) & (p.C > I(p, "sma", 50)), 20)


def s21_lvn(p, ctx, n=60, thin_max=0.3):
    poc, vah, val, thin = _vp(p, n)
    return (p.C > vah) & (shift(p.C, 1) <= shift(vah, 1)) & (thin <= thin_max)


def s22_value_area(p, ctx, kind="reject", n=20):
    poc, vah, val, thin = _vp(p, n)
    pval, pvah = shift(val, 1), shift(vah, 1)
    if kind == "reject":  # 80% rule: open below value, close back inside
        return (p.O < pval) & (p.C > pval) & (p.C < pvah)
    return (p.C > pvah) & (shift(p.C, 1) <= pvah) & (vol_ratio(p) >= 1.3)


def s23_hvb_flip(p, ctx, n=120):
    ix = nb.rolling_argext(shift(np.nan_to_num(p.V, nan=-1.0), 5), n, True)
    rows = np.clip(ix.astype(int) - 5, 0, None)
    cols = np.arange(p.S)[None, :]
    hvb_hi = p.H[rows, cols]
    hvb_hi[ix < 0] = np.nan
    a = I(p, "atr", 14)
    was_above = shift(roll((p.C > hvb_hi).astype(float), 5, "min"), 1) > 0
    return was_above & (p.L <= hvb_hi + 0.25 * a) & (p.C > hvb_hi) & up_trend(p)


# ====================================================================== 4. statistical / mean reversion
def s25_squeeze(p, ctx, n=6, kc=1.5, side=1):
    m = I(p, "sma", 20)
    sd = I(p, "std", 20)
    a = cached(p, ("atr", 20), lambda: atr(p, 20))
    sq = (m + 2 * sd < m + kc * a) & (m - 2 * sd > m - kc * a)
    run = roll(sq.astype(float), n, "min") > 0
    fire = shift(run, 1).astype(bool) & ~sq
    mom = p.C - (I(p, "hh", 20) + I(p, "ll", 20) + 2 * m) / 4
    return fire & ((mom > 0) & (mom > shift(mom, 1)) if side > 0 else (mom < 0) & (mom < shift(mom, 1)))


def s26_rsi2(p, ctx, x=5, trend=True):
    sig = I(p, "rsi", 2) < x
    if trend:
        sig &= up_trend(p)
    return sig


def s27_zscore(p, ctx, z=2.0, trend=True):
    zz = (p.C - I(p, "sma", 20)) / I(p, "std", 20)
    sig = zz < -z
    if trend:
        sig &= up_trend(p)
    return sig


def s28_parabolic_short(p, ctx, n=5, gain=0.4, first_red=True):
    run = shift(I(p, "ret", n), 1) >= gain
    if first_red:
        red = (p.C < p.O) & (p.C < shift(p.C, 1))
        return fresh(run & red, 10)
    return fresh(I(p, "ret", n) >= gain, 10)


# ====================================================================== 5. events
def s30_episodic_pivot(p, ctx, gap=0.08, vm=3.0):
    g = p.O / shift(p.C, 1) - 1
    return (g >= gap) & (vol_ratio(p) >= vm) & (p.C > (p.H + p.L) / 2)


def s31_squeeze_proxy(p, ctx, dd=0.5, gap=0.10):
    beaten = shift(p.C / I(p, "hh", 252), 1) <= 1 - dd
    g = p.C / shift(p.C, 1) - 1
    return beaten & (g >= gap) & (vol_ratio(p) >= 3) & (p.C >= p.H - 0.25 * (p.H - p.L))


def s33_uoa_proxy(p, ctx, vm=3.0, move=0.01):
    ret = np.abs(p.C / shift(p.C, 1) - 1)
    return (vol_ratio(p) >= vm) & (ret <= move) & (p.C > I(p, "sma", 50))


# ====================================================================== 6. intermarket / RS
def s38_rs_leaders(p, ctx, top=0.9, near_high=0.85):
    r = 0.4 * I(p, "ret", 63) + 0.2 * I(p, "ret", 126) + 0.2 * I(p, "ret", 189) + 0.2 * I(p, "ret", 252)
    r = np.where(p.elig, r, np.nan)
    rk = pd.DataFrame(r).rank(axis=1, pct=True).to_numpy()
    month_start = np.r_[True, p.dates.month[1:] != p.dates.month[:-1]]
    pick = (rk >= top) & (p.C >= near_high * I(p, "hh", 252))
    # signal on the last session of each month (enter first session of the next month)
    month_end = np.r_[month_start[1:], False]
    return pick & month_end[:, None]


def s42_tax_loss(p, ctx, bottom=0.1, entry_day=15):
    """Mid-December: buy the YTD losers (bottom decile), hold into January."""
    d = p.dates
    yr_start = pd.Series(p.C.tolist(), index=d)  # placeholder to keep shape logic simple
    del yr_start
    first = np.r_[True, d.year[1:] != d.year[:-1]]
    ytd_anchor = _anchor_index(np.repeat(first[:, None], p.S, axis=1)).astype(int)
    cols = np.arange(p.S)[None, :]
    c0 = p.C[np.clip(ytd_anchor, 0, None), cols]
    ytd = p.C / c0 - 1
    ytd = np.where(p.elig, ytd, np.nan)
    rk = pd.DataFrame(ytd).rank(axis=1, pct=True).to_numpy()
    is_entry = np.zeros(len(d), dtype=bool)
    for y in np.unique(d.year):
        m = np.where((d.year == y) & (d.month == 12) & (d.day >= entry_day))[0]
        if len(m):
            is_entry[m[0]] = True
    return (rk <= bottom) & is_entry[:, None]


# ====================================================================== 0. reference: the live crash rebound
def s00_crash(p, ctx, kind="atr3"):
    """Reference only (already live): crash events as in src/backtest/crash_study.py."""
    if kind == "atr3":
        return p.C <= I(p, "ema", 21) - 3 * I(p, "atr", 14)
    if kind == "drop5_20":
        return I(p, "ret", 5) <= -0.20
    if kind == "drop3_15":
        return I(p, "ret", 3) <= -0.15
    r1 = I(p, "ret", 1)
    sd = roll(r1, 60, "std")
    return r1 <= -3 * shift(sd, 1)
