"""Numba kernels for path-dependent indicators (pivots, anchored VWAP, volume profile, rolling arg-extremes).
All kernels work column by column on (D, S) float arrays and only use data up to each row (no look-ahead),
except where stated: pivots are reported on the day they are CONFIRMED (k bars after the extreme)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def pivots(H, L, k):
    """Confirmed swing points. ph[d] = value of the swing high that became confirmed on day d (k bars after it),
    else NaN; same for pl. phi/pli = row index of that extreme."""
    D, S = H.shape
    ph = np.full((D, S), np.nan)
    pl = np.full((D, S), np.nan)
    phi = np.full((D, S), -1.0)
    pli = np.full((D, S), -1.0)
    for s in prange(S):
        for d in range(2 * k, D):
            c = d - k
            hv = H[c, s]
            lv = L[c, s]
            if np.isnan(hv):
                continue
            okh = True
            okl = True
            for j in range(c - k, d + 1):
                if j == c:
                    continue
                if np.isnan(H[j, s]):
                    okh = False
                    okl = False
                    break
                if (j < c and H[j, s] >= hv) or (j > c and H[j, s] > hv):
                    okh = False
                if (j < c and L[j, s] <= lv) or (j > c and L[j, s] < lv):
                    okl = False
            if okh:
                ph[d, s] = hv
                phi[d, s] = c
            if okl:
                pl[d, s] = lv
                pli[d, s] = c
    return ph, pl, phi, pli


@njit(parallel=True, cache=True)
def last_pivots(pv, n):
    """For each day, the last n confirmed pivot values (most recent first) -> (D, S, n)."""
    D, S = pv.shape
    out = np.full((D, S, n), np.nan)
    for s in prange(S):
        buf = np.full(n, np.nan)
        for d in range(D):
            if not np.isnan(pv[d, s]):
                for j in range(n - 1, 0, -1):
                    buf[j] = buf[j - 1]
                buf[0] = pv[d, s]
            for j in range(n):
                out[d, s, j] = buf[j]
    return out


@njit(parallel=True, cache=True)
def rolling_argext(X, n, want_max):
    """Row index of the rolling max (want_max) / min of X over the last n rows (inclusive)."""
    D, S = X.shape
    out = np.full((D, S), -1.0)
    for s in prange(S):
        for d in range(n - 1, D):
            best = np.nan
            bi = -1
            for j in range(d - n + 1, d + 1):
                v = X[j, s]
                if np.isnan(v):
                    continue
                if bi < 0 or (want_max and v >= best) or ((not want_max) and v <= best):
                    best = v
                    bi = j
            out[d, s] = bi
    return out


@njit(parallel=True, cache=True)
def avwap(TP, V, anchor):
    """Anchored VWAP from row anchor[d, s] (inclusive) to d. anchor < 0 -> NaN."""
    D, S = TP.shape
    out = np.full((D, S), np.nan)
    for s in prange(S):
        cpv = np.zeros(D + 1)
        cv = np.zeros(D + 1)
        for d in range(D):
            pv = TP[d, s] * V[d, s]
            if np.isnan(pv):
                pv = 0.0
            vv = V[d, s]
            if np.isnan(vv):
                vv = 0.0
            cpv[d + 1] = cpv[d] + pv
            cv[d + 1] = cv[d] + vv
        for d in range(D):
            a = int(anchor[d, s])
            if a < 0 or a > d:
                continue
            vol = cv[d + 1] - cv[a]
            if vol > 0:
                out[d, s] = (cpv[d + 1] - cpv[a]) / vol
    return out


@njit(parallel=True, cache=True)
def volume_profile(H, L, C, V, n, bins, stride):
    """Rolling n-day volume profile (each bar's volume spread evenly over its high-low range).
    Computed every `stride` days and carried forward (stride=1: daily). Returns POC, VAH, VAL (70%) and
    `thin_above`: average bin volume in [C, C + (window range)/bins*3] relative to the POC bin."""
    D, S = H.shape
    poc = np.full((D, S), np.nan)
    vah = np.full((D, S), np.nan)
    val = np.full((D, S), np.nan)
    thin = np.full((D, S), np.nan)
    for s in prange(S):
        last = -1
        for d in range(n - 1, D):
            if (d - (n - 1)) % stride != 0 and last >= 0:
                poc[d, s] = poc[d - 1, s]
                vah[d, s] = vah[d - 1, s]
                val[d, s] = val[d - 1, s]
            else:
                lo = 1e18
                hi = -1e18
                ok = True
                for j in range(d - n + 1, d + 1):
                    if np.isnan(H[j, s]) or np.isnan(V[j, s]):
                        ok = False
                        break
                    if L[j, s] < lo:
                        lo = L[j, s]
                    if H[j, s] > hi:
                        hi = H[j, s]
                if not ok or hi <= lo:
                    continue
                w = (hi - lo) / bins
                prof = np.zeros(bins)
                for j in range(d - n + 1, d + 1):
                    a = int((L[j, s] - lo) / w)
                    b = int((H[j, s] - lo) / w)
                    if a >= bins:
                        a = bins - 1
                    if b >= bins:
                        b = bins - 1
                    share = V[j, s] / (b - a + 1)
                    for q in range(a, b + 1):
                        prof[q] += share
                pi = 0
                for q in range(bins):
                    if prof[q] > prof[pi]:
                        pi = q
                tot = prof.sum()
                acc = prof[pi]
                a = pi
                b = pi
                while acc < 0.7 * tot:
                    up = prof[b + 1] if b + 1 < bins else -1.0
                    dn = prof[a - 1] if a - 1 >= 0 else -1.0
                    if up < 0 and dn < 0:
                        break
                    if up >= dn:
                        b += 1
                        acc += up
                    else:
                        a -= 1
                        acc += dn
                poc[d, s] = lo + (pi + 0.5) * w
                vah[d, s] = lo + (b + 1) * w
                val[d, s] = lo + a * w
                # thinness of the 3 bins directly above today's close
                cq = int((C[d, s] - lo) / w)
                if cq < bins - 1:
                    m = 0.0
                    cnt = 0
                    for q in range(cq + 1, min(cq + 4, bins)):
                        m += prof[q]
                        cnt += 1
                    thin[d, s] = (m / cnt) / prof[pi]
                else:
                    thin[d, s] = 0.0
                last = d
    return poc, vah, val, thin
